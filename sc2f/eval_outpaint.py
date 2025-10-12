# eval_runner.py
import os, csv, json, datetime, argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Iterable, Tuple

# for optional NMS
try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from einops import rearrange
import math

# add these near the other stdlib / third-party imports:
import glob, random
import cv2

# ---- your project imports (paths the same as your current code) ----
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'yoloh')))
from data.transforms import ValTransforms
from evaluator.coco_evaluator_multishot import COCOAPIEvaluator  # multishot-capable AP evaluator
from data.coco_fb_diff import COCODataset_FB_Diff  # noqa: F401 (for side-effects)
from utils.CocoEval_with_Difficulty import COCOeval as COCOeval_with_Difficulty
from config.yoloh_config_expand import yoloh_config
from models.yoloh import build_model_noargs

# ============================================================
# HeatmapCollector (unchanged)
# ============================================================

def pad_tensor_image_3(image):
    h, w = image.shape[1], image.shape[2]
    pad = (w, w, h, h)
    image = torch.nn.functional.pad(image, pad, value=0)
    return image


def convert_bbox_heatmap_per_img_tensor(img, bboxes, num_classes, down_ratio=1, radius_down_ratio=1):
    c, h, w  = img.shape
    h, w = h // down_ratio, w // down_ratio
    hm = np.zeros((num_classes, h, w), dtype=np.float32)
    # print(f'In convert_bbox_heatmap_per_img_tensor: h, w = {h}, {w}, num_classes = {num_classes}, bboxes.shape = {bboxes.shape}')
    for bbox in bboxes:
        if bbox.shape[0] == 5:
            bbox_category_id = bbox[4]
            if 'int' not in str(type(bbox_category_id)):
                bbox_category_id = int(bbox_category_id)
        else:
            print('Unknown category id')
            continue
        # print(bbox)
        # print(f'int(bbox): {int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}')
        hm[bbox_category_id, int(bbox[1])//down_ratio:int(bbox[3])//down_ratio, int(bbox[0])//down_ratio:int(bbox[2])//down_ratio] = 1.0
    return torch.tensor(hm)

# ==================== full-res GT heatmap builder (EAD-aware) ====================

def build_fullres_box_heatmap(img_like: torch.Tensor,
                              boxes_with_labels: torch.Tensor,
                              num_classes: int,
                              ead_pad: bool) -> torch.Tensor:
    """
    Returns (C, 3H, 3W) GT heatmap filled with 1s inside GT boxes, 0 elsewhere.
    - If ead_pad=True: canvas is padded to 3H x 3W; boxes are shifted by (+W, +H) to the center.
    - If ead_pad=False: canvas is a 3x 'resized' frame; boxes are scaled by 3x (no shift).
    """
    _, H, W = img_like.shape  # this is the *transformed* model input size (e.g., 320x320)
    Htgt, Wtgt = 3 * H, 3 * W

    hm = torch.zeros((num_classes, Htgt, Wtgt), dtype=torch.float32)

    if boxes_with_labels.numel() == 0:
        return hm

    if boxes_with_labels.shape[-1] == 4:
        # add dummy class 0 if labels not attached
        boxes_with_labels = torch.cat(
            [boxes_with_labels, torch.zeros((boxes_with_labels.shape[0], 1),
                                            dtype=boxes_with_labels.dtype,
                                            device=boxes_with_labels.device)],
            dim=1
        )

    # EAD pad -> shift; no EAD -> scale 3x
    if ead_pad:
        sx = sy = 1.0
        dx = dy = 0.0
    else:
        sx = sy = 3.0
        dx = dy = 0.0

    for b in boxes_with_labels:
        x1, y1, x2, y2, cls = b.tolist()
        cls = int(cls)
        if cls < 0 or cls >= num_classes:
            continue

        xs1 = int(np.floor(x1 * sx + dx)); ys1 = int(np.floor(y1 * sy + dy))
        xs2 = int(np.ceil (x2 * sx + dx)); ys2 = int(np.ceil (y2 * sy + dy))

        xs1 = max(0, min(Wtgt, xs1)); xs2 = max(0, min(Wtgt, xs2))
        ys1 = max(0, min(Htgt, ys1)); ys2 = max(0, min(Htgt, ys2))

        if xs2 > xs1 and ys2 > ys1:
            hm[cls, ys1:ys2, xs1:xs2] = 1.0

    return hm

class HeatmapCollector():
    def __init__(self, num_classes, down_ratio=16, radius_down_ratio=1, EAD_pad=True, *args, **kwargs):
        self.num_classes = num_classes
        self.down_ratio = down_ratio
        self.radius_down_ratio = radius_down_ratio
        self.EAD_pad = EAD_pad

    def __call__(self, batch):
        batch_size = len(batch)
        images, targets, masks, heatmaps = [], [], [], []
        for bid in range(batch_size):
            img = batch[bid][0]                   # (3,H,W) model input size (e.g., 320x320)
            tgt = batch[bid][1]                   # dict with 'boxes','labels' in this frame
            msk = batch[bid][2]

            images.append(img)
            targets.append(tgt)
            masks.append(msk)

            boxes_with_labels = torch.cat(
                (tgt['boxes'], rearrange(tgt['labels'], 'n -> n 1').to(tgt['boxes'].dtype)),
                dim=1
            )
            # build full-res (3H x 3W) GT heatmap: EAD->shift; no EAD->scale*3
            hm_CHW = build_fullres_box_heatmap(img_like=img,
                                               boxes_with_labels=boxes_with_labels,
                                               num_classes=self.num_classes,
                                               ead_pad=bool(self.EAD_pad))
            heatmaps.append(hm_CHW)

        images   = torch.stack(images, 0)
        masks    = torch.stack(masks, 0) if masks[0] is not None else None
        heatmaps = torch.stack(heatmaps, 0)  # (B,C,3H,3W)
        return images, targets, masks, heatmaps

# ============================================================
# CE / SE helpers (center-ignore aware, normalized)
# ============================================================

def _gt_norm_over_space(gt_BSC, eps=1e-12, valid_mask_BS=None):
    if valid_mask_BS is not None:
        gt_BSC = gt_BSC * valid_mask_BS.unsqueeze(-1)
    Z = gt_BSC.clamp_min(0).sum(dim=1, keepdim=True)              # (B,1,C)
    present = (Z.squeeze(1) > eps)                                # (B,C) boolean
    G = (gt_BSC / Z.clamp_min(eps)).clamp_min(eps)                # (B,S,C)
    return G, present

def _pred_spatial_probs_from_logits(pred_BSC, eps=1e-12, valid_mask_BS=None):
    B, S, C = pred_BSC.shape
    pred_BCS = pred_BSC.permute(0,2,1)
    if valid_mask_BS is not None:
        m = valid_mask_BS.unsqueeze(1)
        pred_BCS = pred_BCS.masked_fill(m == 0, float('-inf'))
    probs_BCS = F.softmax(pred_BCS, dim=-1)
    if valid_mask_BS is not None:
        probs_BCS = torch.nan_to_num(probs_BCS, nan=0.0, posinf=0.0, neginf=0.0)
        Z = (probs_BCS * m).sum(dim=-1, keepdim=True).clamp_min(eps)
        probs_BCS = probs_BCS / Z
    return probs_BCS.permute(0,2,1).clamp_min(eps)

def _pred_spatial_probs_from_scores(pred_scores_BSC, eps=1e-12, valid_mask_BS=None):
    if valid_mask_BS is not None:
        pred_scores_BSC = pred_scores_BSC * valid_mask_BS.unsqueeze(-1)
    Z = pred_scores_BSC.clamp_min(0).sum(dim=1, keepdim=True).clamp_min(eps)
    return (pred_scores_BSC / Z).clamp_min(eps)

def _log_cardinality_from_mask(S, valid_mask_BS, dtype, device, eps=1e-12):
    if valid_mask_BS is None:
        return torch.log(torch.tensor(S, dtype=dtype, device=device).clamp_min(1))
    counts = valid_mask_BS.sum(dim=1, keepdim=True).clamp_min(1.0)
    return counts.log()

def _build_center_mask(B, S, spatial_hw, device, dtype):
    H, W = spatial_hw
    assert (H * W) == S, f"S={S} does not match H*W={H*W}"
    mask = torch.ones((H, W), dtype=dtype, device=device)
    h1, h2 = H // 3, 2 * (H // 3)
    w1, w2 = W // 3, 2 * (W // 3)
    mask[h1:h2, w1:w2] = 0
    return mask.reshape(1, S).expand(B, S)

def heatmap_2dce(
    pred: torch.Tensor,   # (B,S,C)
    gt: torch.Tensor,     # (B,S,C)
    reduction: str = 'mean',
    input_type: str = 'pred_logits',  # 'pred_logits'|'pred_probs'|'gt'
    ignore_center: bool = False,
    spatial_hw: tuple[int,int] | None = None
):
    assert pred.shape == gt.shape and pred.dim() == 3
    B, S, C = pred.shape
    device, dtype = pred.device, pred.dtype

    valid_mask_BS = None
    if ignore_center:
        assert spatial_hw is not None, "spatial_hw=(H_full,W_full) required when ignore_center=True"
        valid_mask_BS = _build_center_mask(B, S, spatial_hw, device, dtype)

    G, present = _gt_norm_over_space(gt, valid_mask_BS=valid_mask_BS)
    if input_type == 'pred_logits':
        D = _pred_spatial_probs_from_logits(pred, valid_mask_BS=valid_mask_BS)
    elif input_type == 'pred_probs':
        D = _pred_spatial_probs_from_scores(pred, valid_mask_BS=valid_mask_BS)
    elif input_type == 'gt':
        D, _ = _gt_norm_over_space(pred, valid_mask_BS=valid_mask_BS)
    else:
        raise ValueError("input_type must be 'pred_logits', 'pred_probs', or 'gt'.")

    logX = _log_cardinality_from_mask(S, valid_mask_BS, dtype, device)
    CE_BC = -(G * D.log()).sum(dim=1) / logX
    CE_BC = CE_BC.masked_fill(~present, float('nan'))

    if reduction == 'none':
        return CE_BC
    if reduction == 'sum':
        return torch.nan_to_num(CE_BC, nan=0.0).sum()
    return torch.nanmean(CE_BC)

class Heatmap_CrossEntropy_2D(nn.Module):
    def __init__(self, reduction='mean', input_type='pred_logits',
                 ignore_center: bool = False, spatial_hw: tuple[int,int] | None = None):
        super().__init__()
        self.reduction = reduction
        self.input_type = input_type
        self.ignore_center = ignore_center
        self.spatial_hw = spatial_hw
    def forward(self, pred_hm, tgt_hm):
        return heatmap_2dce(
            pred_hm, tgt_hm,
            reduction=self.reduction,
            input_type=self.input_type,
            ignore_center=self.ignore_center,
            spatial_hw=self.spatial_hw
        )

def heatmap_self_entropy_2d(
    X: torch.Tensor,       # (B,S,C)
    input_type: str = 'pred_logits',    # 'gt' | 'pred_logits' | 'pred_probs'
    reduction: str = 'mean',
    ignore_absent_classes: bool = True,
    class_idx=None,
    ignore_center: bool = False,
    spatial_hw: tuple[int,int] | None = None
):
    assert X.dim() == 3
    B, S, C = X.shape
    device, dtype = X.device, X.dtype

    if class_idx is not None:
        if isinstance(class_idx, int):
            idx = torch.tensor([class_idx], dtype=torch.long, device=X.device)
        elif isinstance(class_idx, (list, tuple)):
            idx = torch.tensor(class_idx, dtype=torch.long, device=X.device)
        else:
            idx = class_idx.to(dtype=torch.long, device=X.device)
        X = X.index_select(dim=-1, index=idx)

    valid_mask_BS = None
    if ignore_center:
        assert spatial_hw is not None, "spatial_hw=(H_full,W_full) required when ignore_center=True"
        valid_mask_BS = _build_center_mask(B, S, spatial_hw, device, dtype)

    if input_type == 'gt':
        P, present = _gt_norm_over_space(X, valid_mask_BS=valid_mask_BS)
        if not ignore_absent_classes:
            present = torch.ones_like(present, dtype=torch.bool, device=X.device)
    elif input_type == 'pred_logits':
        P = _pred_spatial_probs_from_logits(X, valid_mask_BS=valid_mask_BS)
        present = torch.ones((X.size(0), X.size(-1)), dtype=torch.bool, device=X.device)
    elif input_type == 'pred_probs':
        P = _pred_spatial_probs_from_scores(X, valid_mask_BS=valid_mask_BS)
        present = torch.ones((X.size(0), X.size(-1)), dtype=torch.bool, device=X.device)
    else:
        raise ValueError("input_type must be 'gt','pred_logits','pred_probs'.")

    logX = _log_cardinality_from_mask(S, valid_mask_BS, dtype=X.dtype, device=X.device)
    H_BC = -(P * P.log()).sum(dim=1) / logX
    H_BC = H_BC.masked_fill(~present, float('nan'))

    if reduction == 'none':
        return H_BC
    if reduction == 'sum':
        return torch.nan_to_num(H_BC, nan=0.0).sum()
    return torch.nanmean(H_BC)

class Heatmap_SelfEntropy_2D(nn.Module):
    def __init__(self, input_type='pred_logits', reduction='mean',
                 ignore_absent_classes=True, class_idx=None,
                 ignore_center: bool = False, spatial_hw: tuple[int,int] | None = None):
        super().__init__()
        self.input_type = input_type
        self.reduction = reduction
        self.ignore_absent_classes = ignore_absent_classes
        self.class_idx = class_idx
        self.ignore_center = ignore_center
        self.spatial_hw = spatial_hw
    def forward(self, X):
        return heatmap_self_entropy_2d(
            X,
            input_type=self.input_type,
            reduction=self.reduction,
            ignore_absent_classes=self.ignore_absent_classes,
            class_idx=self.class_idx,
            ignore_center=self.ignore_center,
            spatial_hw=self.spatial_hw
        )

# ============================================================
# mIoU / mIoU_int / pixelAP (vectorized, center-ignore aware)
# ============================================================

# ---- spatial unification: always evaluate on the EAD-on grid ----
HEATMAP_STRIDE = 16  # your convert_bbox_heatmap_per_img_tensor down_ratio

def _ead_target_hw(Ht: int, Wt: int, stride: int = HEATMAP_STRIDE):
    # EAD-on canvas is 3x spatially, so HM becomes (3*Ht)/stride, (3*Wt)/stride
    return ( (3 * Ht) // stride, (3 * Wt) // stride )

@torch.inference_mode()
def _resize_pred_gt_to_target(
    pred_logits_BSC: torch.Tensor,   # (B,S,C) logits
    gt_heat_BCHW: torch.Tensor,      # (B,C,H,W)
    cur_hw: Tuple[int,int],          # (Hc,Wc) currently matched between pred & gt
    target_hw: Tuple[int,int],       # (Htgt,Wtgt)
):
    Hc, Wc = cur_hw
    Htgt, Wtgt = target_hw
    if (Hc, Wc) == (Htgt, Wtgt):
        return pred_logits_BSC, gt_heat_BCHW, target_hw

    # logits: (B,S,C) -> (B,C,Hc,Wc) -> bilinear -> (B,S',C)
    B, S, C = pred_logits_BSC.shape
    pred_BCHW = rearrange(pred_logits_BSC, 'b (h w) c -> b c h w', h=Hc, w=Wc)
    pred_up_BCHW = F.interpolate(pred_BCHW, size=(Htgt, Wtgt), mode='bilinear', align_corners=False)
    pred_up_BSC  = rearrange(pred_up_BCHW, 'b c h w -> b (h w) c')

    # GT: nearest to preserve binary structure; binarization happens later anyway
    gt_up_BCHW = F.interpolate(gt_heat_BCHW, size=(Htgt, Wtgt), mode='nearest')

    return pred_up_BSC, gt_up_BCHW, (Htgt, Wtgt)

HEATMAP_STRIDE = 16  # native stride of the model heatmap

@torch.inference_mode()
def _unify_pred_to_gt_grid(
    pred_logits_BSC: torch.Tensor,   # (B,S,C) logits on the model's native grid
    gt_heat_BCHW: torch.Tensor,      # (B,C,Hgt,Wgt) full-res GT (3H x 3W)
    Ht: int, Wt: int,                # model input size (e.g., 320x320)
    EAD_enabled: bool,               # cfg['EAD']
):
    """
    Upsample pred logits to the GT grid size w/o any stride constants.
    We infer (Hc,Wc) from S and the input aspect ratio (including EAD factor).
    """
    Htgt, Wtgt = int(gt_heat_BCHW.shape[-2]), int(gt_heat_BCHW.shape[-1])
    B, S, C = pred_logits_BSC.shape

    # aspect ratio on the model's native canvas (with/without EAD factor)
    H_canvas = (3*Ht) if EAD_enabled else Ht
    W_canvas = (3*Wt) if EAD_enabled else Wt
    ar = float(H_canvas) / max(W_canvas, 1)

    # factor S ≈ Hc*Wc with Hc/Wc ≈ ar (robust to rounding)
    Wc = max(1, int(round(math.sqrt(S / max(ar, 1e-12)))))
    Hc = max(1, S // Wc)
    if Hc * Wc != S:
        Hc = max(1, int(round(math.sqrt(S * ar))))
        Wc = max(1, S // Hc)
    assert Hc * Wc == S, f"Cannot factor S={S} into HxW (Hc={Hc}, Wc={Wc}, ar={ar:.4f})"

    pred_BCHW = rearrange(pred_logits_BSC, 'b (h w) c -> b c h w', h=Hc, w=Wc)
    pred_up_BCHW = F.interpolate(pred_BCHW, size=(Htgt, Wtgt), mode='bilinear', align_corners=False)
    pred_up_BSC  = rearrange(pred_up_BCHW, 'b c h w -> b (h w) c')
    return pred_up_BSC, gt_heat_BCHW, (Htgt, Wtgt), (Hc, Wc)

# ===============================
# B-mode (single-image) helpers
# ===============================

def _box_iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.size(0), b.size(0)), device=a.device, dtype=a.dtype)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-12)

def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    if tv_nms is not None:
        return tv_nms(boxes, scores, iou_thr)
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        ious = _box_iou_matrix(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious <= iou_thr]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)

def _filter_boxes_posthoc(pred: Dict[str, torch.Tensor],
                          conf_thr: float,
                          nms_thr: float,
                          topk: Optional[int],
                          class_agnostic_nms: bool = True) -> Dict[str, torch.Tensor]:
    boxes = pred['boxes']; scores = pred['scores']; labels = pred['labels']
    if boxes.numel() == 0:
        return {k: v[:0] for k, v in pred.items()}
    keep = scores >= conf_thr
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.numel() == 0:
        return {'boxes': boxes, 'scores': scores, 'labels': labels}
    if class_agnostic_nms:
        keep_idx = _nms(boxes, scores, nms_thr)
        boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
    else:
        final_keep = []
        for c in labels.unique():
            m = (labels == c)
            if m.sum() == 0:
                continue
            idx = _nms(boxes[m], scores[m], nms_thr)
            sel = torch.nonzero(m, as_tuple=False).squeeze(1)[idx]
            final_keep.append(sel)
        if len(final_keep):
            final_keep = torch.cat(final_keep, dim=0)
            boxes, scores, labels = boxes[final_keep], scores[final_keep], labels[final_keep]
    if topk is not None and boxes.size(0) > topk:
        order = scores.argsort(descending=True)[:topk]
        boxes, scores, labels = boxes[order], scores[order], labels[order]
    return {'boxes': boxes, 'scores': scores, 'labels': labels}

def _coerce_pred_to_image_frame(pred: Dict[str, torch.Tensor],
                                img_hw: Tuple[int, int],
                                ead_maybe: bool = True) -> Dict[str, torch.Tensor]:
    H, W = img_hw
    out = {k: (v.clone().detach() if isinstance(v, torch.Tensor) else torch.as_tensor(v))
           for k, v in pred.items()}
    b = out['boxes'].to(torch.float32)
    if b.numel() == 0:
        out['boxes'] = b; return out
    # scale normalized boxes
    if torch.isfinite(b).all() and b.max() <= 1.5:
        scale = torch.tensor([W, H, W, H], device=b.device, dtype=b.dtype)
        b = b * scale
    # map from 3x EAD canvas back to center crop
    if ead_maybe and b.max() > max(H, W) * 1.2 and b.max() < max(H, W) * 3.5:
        b = b / 3.0
        b[:, [0, 2]] -= W
        b[:, [1, 3]] -= H
    # order + clamp
    x1 = torch.minimum(b[:, 0], b[:, 2]); y1 = torch.minimum(b[:, 1], b[:, 3])
    x2 = torch.maximum(b[:, 0], b[:, 2]); y2 = torch.maximum(b[:, 1], b[:, 3])
    b = torch.stack([x1, y1, x2, y2], dim=1)
    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H - 1)
    out['boxes'] = b.to(out['boxes'].dtype)
    # scores: logits → probs
    s = out['scores'].to(torch.float32)
    if s.numel() and (s.min() < 0 or s.max() > 1):
        s = torch.sigmoid(s)
    out['scores'] = s
    out['labels'] = out.get('labels', torch.zeros_like(s, dtype=torch.int64)).to(torch.int64)
    return out

@torch.inference_mode()
def _run_model_for_boxes(model, img_B3HW: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Try a few common APIs; return {'boxes','scores','labels'} tensors on same device as img.
    """
    def _standardize(one):
        if isinstance(one, dict) and all(k in one for k in ('boxes','scores','labels')):
            d = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v, device=img_B3HW.device))
                 for k,v in one.items()}
            return d
        if isinstance(one, (tuple, list)) and len(one) == 3:
            b, s, l = one
            b = b if isinstance(b, torch.Tensor) else torch.tensor(b, device=img_B3HW.device)
            s = s if isinstance(s, torch.Tensor) else torch.tensor(s, device=img_B3HW.device)
            l = l if isinstance(l, torch.Tensor) else torch.tensor(l, device=img_B3HW.device)
            return {'boxes': b, 'scores': s, 'labels': l}
        return None
    out = model(img_B3HW)
    if isinstance(out, list) and len(out) == 1:
        cand = _standardize(out[0]);  
        if cand: return cand
    cand = _standardize(out)
    if cand: return cand
    for fn in ('predict', 'detect', 'inference', 'infer', 'forward_for_debug'):
        if hasattr(model, fn):
            out = getattr(model, fn)(img_B3HW)
            if isinstance(out, list) and len(out) == 1:
                cand = _standardize(out[0])
                if cand: return cand
            cand = _standardize(out)
            if cand: return cand
    raise RuntimeError("Could not parse model outputs into boxes/scores/labels.")
    
def _bmode_single_image_metrics(pred: Dict[str, torch.Tensor],
                                gt_boxes_xyxy: torch.Tensor,
                                iou_thr: float = 0.5) -> Dict[str, float]:
    P = pred['boxes']; G = gt_boxes_xyxy if gt_boxes_xyxy is not None else torch.empty((0,4), device=P.device)
    if P.numel() == 0 and (G is None or G.numel() == 0):
        return {'TP': 0, 'FP': 0, 'FN': 0, 'Precision': 1.0, 'Recall': 1.0, 'F1': 1.0, 'meanIoU_TP': 1.0}
    if P.numel() == 0:
        return {'TP': 0, 'FP': 0, 'FN': int(G.size(0)), 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'meanIoU_TP': 0.0}
    if G.numel() == 0:
        return {'TP': 0, 'FP': int(P.size(0)), 'FN': 0, 'Precision': 0.0, 'Recall': 1.0, 'F1': 0.0, 'meanIoU_TP': 0.0}
    ious = _box_iou_matrix(P, G)
    tp = 0; iou_sum = 0.0
    taken_g = torch.zeros(G.size(0), dtype=torch.bool, device=P.device)
    for i in torch.argsort(pred['scores'], descending=True):
        j = torch.argmax(ious[i])
        if ious[i, j] >= iou_thr and not taken_g[j]:
            taken_g[j] = True
            tp += 1
            iou_sum += float(ious[i, j].item())
    fp = int(P.size(0)) - tp
    fn = int(G.size(0)) - tp
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = (2 * prec * rec) / max(prec + rec, 1e-12)
    mean_iou_tp = (iou_sum / max(tp, 1)) if tp > 0 else 0.0
    return {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': prec, 'Recall': rec, 'F1': f1, 'meanIoU_TP': mean_iou_tp}

@dataclass
class ThresholdGrid:
    kind: str = "prob"
    values: Iterable[float] = tuple(np.round(np.linspace(0.05, 0.95, 19), 2))

@dataclass
class MiouOptions:
    ignore_center: bool = True
    spatial_hw: Optional[Tuple[int,int]] = None
    gt_binarize: str = "relative"     # 'relative'|'percentile'|'gt>0'
    gt_rel_thresh: float = 0.25
    gt_percentile: float = 90.0
    beta: float = 1.0
    class_idx: Optional[int] = None   # for (B,S,C) inputs, choose one class

def _build_center_mask_hw(H_full: int, W_full: int, device=None, dtype=None):
    m = torch.ones((H_full, W_full), dtype=dtype or torch.float32, device=device)
    h1, h2 = H_full // 3, 2 * (H_full // 3)
    w1, w2 = W_full // 3, 2 * (W_full // 3)
    m[h1:h2, w1:w2] = 0
    return m

def _binarize_gt_spatial(gt_BHW: torch.Tensor, opts: MiouOptions) -> torch.Tensor:
    if opts.gt_binarize == "gt>0":
        return (gt_BHW > 0).to(gt_BHW.dtype)
    elif opts.gt_binarize == "relative":
        B, H, W = gt_BHW.shape
        gmax = gt_BHW.view(B, -1).amax(dim=1).clamp_min(1e-8)    # (B,)
        thr = (gmax * opts.gt_rel_thresh).view(B,1,1)
        return (gt_BHW >= thr).to(gt_BHW.dtype)
    elif opts.gt_binarize == "percentile":
        B, H, W = gt_BHW.shape
        flat = gt_BHW.view(B, -1)
        kth = torch.tensor([np.percentile(flat[b].detach().cpu().numpy(), opts.gt_percentile)
                            for b in range(B)], device=gt_BHW.device, dtype=gt_BHW.dtype).view(B,1,1)
        return (gt_BHW >= kth).to(gt_BHW.dtype)
    else:
        raise ValueError("gt_binarize must be one of {'relative','percentile','gt>0'}.")


@torch.inference_mode()
def run_single_sample(cfg, model, device, evaluator, radius_down_ratio, index: int,
                      conf_thr: float, nms_thr: float, topk: Optional[int],
                      iou_thr: float,
                      miou_cfg: MiouOptions,
                      pred_input_type: str = 'pred_logits',
                      class_agnostic_nms: bool = True,
                      raw_topk_max: int = 1200) -> Dict[str, Any]:
    """
    Compute per-image B-mode and H-mode metrics for dataset[index].
    Matches visualize_runner logic (post-hoc conf/NMS; CE/SE/mIoU suite).
    """
    dataset = evaluator.dataset
    dataset.transform = evaluator.transform

    # Pull one transformed sample and build its GT heatmap via the same collector
    collate = HeatmapCollector(num_classes=2, radius_down_ratio=radius_down_ratio, EAD_pad=cfg.get('EAD', True))
    sample = dataset[index]  # (img, target, mask)
    imgs_B3HW, targets_list, masks_B, hm_BCHW = collate([sample])
    img = imgs_B3HW[0].to(device)           # (3,H,W)
    tgt = targets_list[0]                   # dict with 'boxes','labels'
    hm  = hm_BCHW.to(device)                # (1,C,Hh,Ww)

    Hh, Ww = hm.shape[-2], hm.shape[-1]

    # -------- H-mode metrics (single image) ----------
    pred_logits_BSC = model.heatmap(img.unsqueeze(0))  # (1,S,C)
    
    Ht, Wt = int(img.shape[-2]), int(img.shape[-1])
    # in run_single_sample (unchanged logic, just function name/return)
    pred_logits_u, hm_u, (Hu, Wu), (Hc, Wc) = _unify_pred_to_gt_grid(
        pred_logits_BSC, hm, Ht=Ht, Wt=Wt, EAD_enabled=bool(cfg.get('EAD', True))
    )
    gt_BSC_u = rearrange(hm_u, 'b c h w -> b (h w) c')

    ce = heatmap_2dce(
        pred_logits_u, gt_BSC_u,
        input_type=pred_input_type, reduction='none',
        ignore_center=bool(miou_cfg.ignore_center), spatial_hw=(Hu, Wu)
    ).nanmean().item()

    se = heatmap_self_entropy_2d(
        pred_logits_u, input_type='pred_logits', reduction='none',
        class_idx=miou_cfg.class_idx,
        ignore_center=bool(miou_cfg.ignore_center), spatial_hw=(Hu, Wu)
    ).nanmean().item()

    grid = ThresholdGrid(values=tuple(miou_cfg.thresholds))
    miou_opts = MiouOptions(
        ignore_center=miou_cfg.ignore_center, spatial_hw=(Hu, Wu),
        gt_binarize=miou_cfg.gt_binarize, gt_rel_thresh=miou_cfg.gt_rel_thresh,
        gt_percentile=miou_cfg.gt_percentile, beta=miou_cfg.beta,
        class_idx=miou_cfg.class_idx
    )
    mres = miou_over_thresholds_batch(pred_logits_u, hm_u, grid, miou_opts)

    H_metrics = {'CE': ce, 'SE': se, **mres}

    # -------- B-mode metrics (single image, post-hoc conf/NMS) ----------
    # Run model once with permissive thresholds to harvest raw boxes (match visualize_runner)
    # (We assume the current 'model' already exists; if it was built with strict conf/nms,
    # we still try to parse its outputs; for full alignment, build a permissive copy.)
    raw = _run_model_for_boxes(model, img.unsqueeze(0))
    raw_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in raw.items()}
    # map to unpadded image frame (the image here is unpadded 3xH/3xW already)
    im_H, im_W = int(img.shape[1]), int(img.shape[2])
    raw_cpu = _coerce_pred_to_image_frame(raw_cpu, img_hw=(im_H, im_W))

    pred = _filter_boxes_posthoc(raw_cpu, conf_thr, nms_thr, topk, class_agnostic_nms=class_agnostic_nms)
    gt_boxes = tgt['boxes'].detach().cpu().to(torch.float32)

    B_metrics = _bmode_single_image_metrics(
        {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in pred.items()},
        gt_boxes, iou_thr=iou_thr
    )

    return {'index': index, 'B': B_metrics, 'H': H_metrics,
            'conf_thresh': conf_thr, 'nms_thresh': nms_thr, 'topk': (None if topk is None else int(topk))}

@torch.inference_mode()
def miou_over_thresholds_batch(
    pred_logits_BSC: torch.Tensor,   # (B,S,C) logits over space per class
    gt_heat_BCHW: torch.Tensor,      # (B,C,H,W) nonnegative GT heatmaps
    grid,
    opts
) -> Dict[str, float]:
    """
    Compute dataset-mean mIoU / mIoU_int and Pixel-AP for the selected class (opts.class_idx).
    - mIoU / mIoU_int: averaged over images, with the same empty-GT policy as before.
    - Pixel-AP: micro-averaged PR across the whole dataset (sum TP/FP/FN over batch)
      evaluated at the provided thresholds, then AP via the precision envelope.
    """
    B, S, C = pred_logits_BSC.shape
    _, Cg, H, W = gt_heat_BCHW.shape
    assert C == Cg, "C mismatch between pred and gt"
    assert S == H * W, "S must equal H*W"

    cls = 0 if opts.class_idx is None else opts.class_idx

    # pick class, reshape to (B,H,W)
    logits_BHW = pred_logits_BSC[..., cls].reshape(B, H, W)
    gt_BHW = gt_heat_BCHW[:, cls, :, :]

    # prob map
    P = torch.sigmoid(logits_BHW)  # (B,H,W)

    # GT binarization
    Y = _binarize_gt_spatial(gt_BHW, opts)  # (B,H,W)

    # center ignore mask
    if opts.ignore_center:
        m = _build_center_mask_hw(H, W, device=P.device, dtype=P.dtype)  # (H,W)
        Pm = P * m
        Ym = Y * m
    else:
        Pm, Ym = P, Y

    # flatten spatial to speed reductions
    Pm = Pm.reshape(B, -1)  # (B,S)
    Ym = Ym.reshape(B, -1)  # (B,S)

    thr = torch.tensor(list(grid.values), dtype=P.dtype, device=P.device)  # (T,)
    pred_m = (Pm.unsqueeze(-1) >= thr.view(1,1,-1)).to(P.dtype)            # (B,S,T)

    # per-image sums per threshold
    gt_sum  = Ym.sum(dim=1, keepdim=True)                      # (B,1)
    pred_sum = pred_m.sum(dim=1)                               # (B,T)
    tp = (pred_m * Ym.unsqueeze(-1)).sum(dim=1)                # (B,T)
    fp = pred_sum - tp                                         # (B,T)
    fn = gt_sum - tp                                           # (B,T)

    # ---------- mIoU / mIoU_int (same policy as before, averaged over images then thresholds) ----------
    gt_empty = (gt_sum == 0)                                   # (B,1) bool
    if gt_empty.any():
        pred_empty = (pred_sum == 0)                           # (B,T)
        iou_std = torch.where(gt_empty, pred_empty.to(P.dtype), (tp / (tp + fp + fn).clamp_min(1e-12)))
        iou_int = torch.where(gt_empty, pred_empty.to(P.dtype), (tp / (tp + fn).clamp_min(1e-12)))
    else:
        iou_std = tp / (tp + fp + fn).clamp_min(1e-12)
        iou_int = tp / (tp + fn).clamp_min(1e-12)

    mIoU     = iou_std.mean(dim=0).mean().item()
    mIoU_int = iou_int.mean(dim=0).mean().item()

    # ---------- Pixel-AP (micro PR across dataset, then AP via precision envelope) ----------
    # aggregate counts across the batch (dataset micro-averaging)
    TP_T = tp.sum(dim=0)                                       # (T,)
    FP_T = fp.sum(dim=0)                                       # (T,)
    FN_T = fn.sum(dim=0)                                       # (T,)

    total_gt = gt_sum.sum()                                    # scalar
    if total_gt <= 0:
        # No positive pixels anywhere in the dataset:
        # AP = 1 if the model never predicts positives at any threshold; else 0
        any_pred_any_thr = (pred_sum.sum() > 0)
        pixelAP = float(not bool(any_pred_any_thr))
        return {"mIoU": mIoU, "mIoU_int": mIoU_int, "pixelAP": pixelAP}

    prec_T = TP_T / (TP_T + FP_T).clamp_min(1e-12)             # (T,)
    rec_T  = TP_T / (TP_T + FN_T).clamp_min(1e-12)             # (T,)

    # sort by recall ascending
    order = torch.argsort(rec_T)
    r = rec_T[order]
    p = prec_T[order]

    # precision envelope (monotone non-increasing when moving to lower recall)
    for i in range(p.numel() - 2, -1, -1):
        p[i] = torch.maximum(p[i], p[i+1])

    # step-area under PR curve (AP) with [0, r1, r2, ...]
    r_pad = torch.cat([torch.zeros(1, device=r.device, dtype=r.dtype), r], dim=0)
    dr = r_pad[1:] - r_pad[:-1]
    pixelAP = torch.sum(dr * p).item()

    return {"mIoU": mIoU, "mIoU_int": mIoU_int, "pixelAP": pixelAP}

# ============================================================
# Evaluator & DataLoader builders
# ============================================================

def get_evaluator(cfg, device, val_img_folder, val_ann_file, cocoeval_iouthr=None):
    fig_size = cfg.get('input_size', (320, 320))
    print('trans_config:', cfg.get('val_transform', None))
    val_transform = ValTransforms(min_size=fig_size[0], max_size=fig_size[1],
                                  pixel_mean=cfg['pixel_mean'],
                                  pixel_std=cfg['pixel_std'],
                                  trans_config=cfg.get('val_transform', None),
                                  format=cfg['format'])
    evaluator = COCOAPIEvaluator(data_dir=None, EAD=cfg['EAD'],
                                 device=device,
                                 transform=val_transform,
                                 image_folder=val_img_folder,
                                 ann_file=val_ann_file,
                                 iouThr=cocoeval_iouthr)
    return evaluator

@dataclass
class MetricOptions:
    pred_input_type: str = 'pred_logits'   # 'pred_logits'|'pred_probs'
    se_class_idx: Optional[int] = 0
    batch_size: int = 32
    num_workers: int = 2
    ignore_center: bool = True            # default ON
    heat_unify_to_ead_on: bool = True   # <--- NEW

def build_eval_dataloader(cfg, evaluator, num_classes=2, mopts: MetricOptions = MetricOptions(), radius_down_ratio=0.5):
    dataset = evaluator.dataset
    dataset.transform = evaluator.transform
    collate = HeatmapCollector(num_classes, radius_down_ratio=radius_down_ratio, EAD_pad=cfg.get('EAD', True))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=mopts.batch_size,
        shuffle=False,
        num_workers=mopts.num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        collate_fn=collate
    )

# ============================================================
# Metric pass (CE, SE, mIoU suite)
# ============================================================

@dataclass
class MiouCLI:
    thresholds: Iterable[float]
    gt_binarize: str = "gt>0"
    gt_rel_thresh: float = 0.25
    gt_percentile: float = 90.0
    beta: float = 1.0
    class_idx: int = 0
    ignore_center: bool = True

@torch.inference_mode()
def run_metrics_multishot(
    model,
    evaluator,
    device,
    mopts: MetricOptions,
    miou_cfg: MiouCLI,
    radius_down_ratio: float,
    num_per_index: int = 0,
    seed: int = 123,
    candidate_exts: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Multishot heatmap evaluation (no dataset[index]/pull_image use):
      - for each COCO image id, gather shots from image_folder/{id:012d}/* (or flat {id:012d}.ext fallback)
      - optionally subsample with num_per_index (deterministic via seed+id)
      - run model.heatmap on all shots, average predicted *logits*
      - build GT heatmap once from COCO ann (or dataset.pull_ann) scaled to the transformed (H,W), with EAD padding
      - compute CE/SE/mIoU suite against the averaged prediction (parity with single-shot code)
    """
    if candidate_exts is None:
        candidate_exts = ['.jpg', '.jpeg', '.png']

    model.eval().to(device)
    dataset = evaluator.dataset
    transform = evaluator.transform
    EAD = bool(evaluator.EAD)

    # ---- resolve evaluation order of image ids (no image loads) ----
    if hasattr(dataset, 'ids'):
        coco_ids_order = list(map(int, dataset.ids))
    elif hasattr(dataset, 'img_ids'):
        coco_ids_order = list(map(int, dataset.img_ids))
    else:
        coco_ids_order = list(map(int, dataset.coco.getImgIds()))
    N = len(coco_ids_order)

    # accumulators
    sums = {'ce_pred_gt': 0.0, 'se_pred': 0.0, 'mIoU': 0.0, 'mIoU_int': 0.0, 'pixelAP': 0.0}
    n_imgs = 0

    grid = ThresholdGrid(values=tuple(miou_cfg.thresholds))

    def _build_gt_hm_for_id(img_id: int, Ht: int, Wt: int, H0: int, W0: int) -> torch.Tensor:
        """
        Build full-res GT heatmap (1,C,3Ht,3Wt) using *image* size (H0,W0), not annotations.
        EAD=True: shift (+Wt,+Ht). EAD=False: scale boxes by 3×.
        """
        # scale from native (possibly later 3x) into model (Ht,Wt)
        if EAD:
            sx = float(Wt) / max(3*W0, 1)
            sy = float(Ht) / max(3*H0, 1)
        else:
            sx = float(Wt) / max(W0, 1)
            sy = float(Ht) / max(H0, 1)

        # pull boxes/labels (prefer dataset.pull_ann, fallback to COCO anns) — unchanged
        boxes_xyxy, labels = None, None
        if hasattr(dataset, 'pull_ann'):
            try:
                ann = dataset.pull_ann(coco_ids_order.index(img_id))
                if isinstance(ann, dict):
                    boxes_xyxy = ann.get('boxes', ann.get('bboxes', None))
                    labels     = ann.get('labels', ann.get('cls', None))
                elif isinstance(ann, (list, tuple)) and len(ann) >= 2:
                    boxes_xyxy, labels = ann[0], ann[1]
            except Exception:
                pass
        if boxes_xyxy is None or labels is None:
            ann_ids = dataset.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = dataset.coco.loadAnns(ann_ids)
            inv_map = None
            if hasattr(dataset, 'class_ids'):
                inv_map = {int(cat_id): int(i) for i, cat_id in enumerate(dataset.class_ids)}
            bx, lb = [], []
            for a in anns:
                x, y, w, h = a['bbox']
                bx.append([x, y, x + w, y + h])
                lb.append(inv_map[int(a['category_id'])] if inv_map is not None else int(a['category_id']))
            boxes_xyxy = np.array(bx, dtype=np.float32) if len(bx) else np.zeros((0,4), dtype=np.float32)
            labels     = np.array(lb, dtype=np.int64)    if len(lb) else np.zeros((0,), dtype=np.int64)

        boxes_xyxy = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        labels     = torch.as_tensor(labels,     dtype=torch.int64)

        scale = torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
        boxes_xyxy = boxes_xyxy * scale
        labels_col = labels.view(-1, 1).to(dtype=torch.float32)
        boxes_with_labels = torch.cat([boxes_xyxy, labels_col], dim=1)

        dummy = torch.zeros(3, Ht, Wt, dtype=torch.float32)
        hm_CHW = build_fullres_box_heatmap(dummy, boxes_with_labels, num_classes=2, ead_pad=EAD)
        return hm_CHW.unsqueeze(0).to(device)  # (1,C,3Ht,3Wt)

    # per-id loop
    img_root = getattr(dataset, 'img_folder', None)
    if img_root is None:
        raise RuntimeError("dataset.img_folder must be set for multishot heatmap evaluation.")

    num_nan_ce = 0

    for i_idx, img_id in tqdm(list(enumerate(coco_ids_order)), desc="Metric pass (multishot)", leave=False):
        # ---- collect candidate paths: {root}/{id:012d}/*.{ext} (or flat fallback) ----
        subdir = os.path.join(img_root, f"{img_id:012d}")
        cand_paths: List[str] = []
        if os.path.isdir(subdir):
            for e in candidate_exts:
                cand_paths.extend(glob.glob(os.path.join(subdir, f"*{e}")))
        cand_paths = sorted(cand_paths)

        if not cand_paths:
            # flat fallback: {root}/{id:012d}.{ext}
            for e in candidate_exts:
                p = os.path.join(img_root, f"{img_id:012d}{e}")
                if os.path.isfile(p):
                    cand_paths = [p]
                    break

        if not cand_paths:
            # no shots found; skip gracefully
            continue

        # optional subsample (deterministic)
        if num_per_index and len(cand_paths) > num_per_index:
            rng = random.Random(seed + int(img_id))
            cand_paths = rng.sample(cand_paths, num_per_index)

        # ---- get transformed size from the first selected shot ----
        # (keeps parity with your original heatmap grid: resize -> (H,W), optional EAD pad -> /16)
        first_bgr = cv2.imread(cand_paths[0], cv2.IMREAD_COLOR)
        if first_bgr is None:
            continue
        H0, W0 = first_bgr.shape[:2]          # <-- native size from the actual image
        first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        x0 = transform(first_rgb)[0]           # (C,Ht,Wt)
        Ht, Wt = int(x0.shape[-2]), int(x0.shape[-1])

        hm_BCHW = _build_gt_hm_for_id(img_id, Ht, Wt, H0, W0)  # <-- pass H0,W0
        Hh, Ww = int(hm_BCHW.shape[-2]), int(hm_BCHW.shape[-1])
        gt_BSC = rearrange(hm_BCHW, 'b c h w -> b (h w) c')  # (1,S,C)

        # ---- run model.heatmap on ALL selected shots and average logits ----
        pred_logits_sum = None
        shot_count = 0
        # include the first shot already loaded
        x = x0.unsqueeze(0).to(device, non_blocking=False)  # (1,C,Ht,Wt)
        logits_BSC = model.heatmap(x)                       # (1,S,C)
        pred_logits_sum = logits_BSC if pred_logits_sum is None else (pred_logits_sum + logits_BSC)
        shot_count += 1
        del x, logits_BSC

        for p in cand_paths[1:]:
            im_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if im_bgr is None:
                continue
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            x = transform(im_rgb)[0].unsqueeze(0).to(device, non_blocking=False)
            logits_BSC = model.heatmap(x)
            pred_logits_sum = pred_logits_sum + logits_BSC
            shot_count += 1
            del x, logits_BSC

        if shot_count == 0:
            # no valid predictions; zero tensor with GT shape
            pred_logits_avg = torch.zeros_like(gt_BSC, device=device)
        else:
            pred_logits_avg = pred_logits_sum / float(shot_count)
            
        # in run_metrics_multishot
        pred_logits_u, hm_u, (Hu, Wu), (Hc, Wc) = _unify_pred_to_gt_grid(
            pred_logits_avg, hm_BCHW, Ht=Ht, Wt=Wt, EAD_enabled=EAD
        )
        gt_BSC_u = rearrange(hm_u, 'b c h w -> b (h w) c')
        
        # print(f'pred_logits_u shape: {pred_logits_u.shape}, hm_BCHW shape: {hm_BCHW.shape}, hm_u shape: {hm_u.shape}, gt_BSC_u shape: {gt_BSC_u.shape}')

                # ---- CE / SE with center-ignore (ONLY SE uses center-ignore) ----
        if mopts.pred_input_type not in ('pred_logits', 'pred_probs'):
            raise ValueError("pred_input_type must be 'pred_logits' or 'pred_probs'.")

        # Cross-entropy: DO NOT ignore center (avoids all-NaN when GT lies in center)
        ce_tensor = heatmap_2dce(
            pred_logits_u, gt_BSC_u,
            input_type=mopts.pred_input_type,
            reduction='none',
            ignore_center=mopts.ignore_center,           # <--- was mopts.ignore_center
            spatial_hw=(Hu, Wu)
        )
        ce_val = torch.nanmean(ce_tensor)
        
        if not torch.isfinite(ce_val):
            num_nan_ce += 1
        ce_pred_gt = float(ce_val) if torch.isfinite(ce_val) else 0.0
        
        if not torch.isfinite(torch.tensor(ce_pred_gt)):
            print('There are some NaN CE values.')

        # Self-entropy: keep your center-ignore behavior (default ON unless --no-entropy_ignore_center)
        se_input_type = 'pred_logits' if mopts.pred_input_type == 'pred_logits' else 'pred_probs'
        se_in = pred_logits_u if se_input_type == 'pred_logits' else torch.sigmoid(pred_logits_u)
        se_tensor = heatmap_self_entropy_2d(
            se_in,
            input_type=se_input_type,
            reduction='none',
            class_idx=miou_cfg.class_idx,
            ignore_center=mopts.ignore_center,   # <--- SE can use center-ignore
            spatial_hw=(Hu, Wu)
        )
        se_val = torch.nanmean(se_tensor)
        se_pred = float(se_val) if torch.isfinite(se_val) else 0.0

        # ---- mIoU suite (vectorized) ----
        miou_opts = MiouOptions(
            ignore_center=miou_cfg.ignore_center,
            spatial_hw=(Hu, Wu),
            gt_binarize=miou_cfg.gt_binarize,
            gt_rel_thresh=miou_cfg.gt_rel_thresh,
            gt_percentile=miou_cfg.gt_percentile,
            beta=miou_cfg.beta,
            class_idx=miou_cfg.class_idx
        )
        # print(f'pred_logits_u shape: {pred_logits_u.shape}, hm_BCHW shape: {hm_BCHW.shape}, hm_u shape: {hm_u.shape}')
        # print(f'Hu, Wu: {Hu}, {Wu}')
        mres = miou_over_thresholds_batch(pred_logits_u, hm_u, grid, miou_opts)
        # accumulate
        sums['ce_pred_gt'] += ce_pred_gt
        sums['se_pred']    += se_pred
        sums['mIoU']       += mres['mIoU']
        sums['mIoU_int']   += mres['mIoU_int']
        sums['pixelAP']        += mres['pixelAP']
        n_imgs += 1

        # tidy
        del first_bgr, first_rgb, x0, hm_BCHW, gt_BSC, pred_logits_avg, pred_logits_sum, pred_logits_u, gt_BSC_u, hm_u
        torch.cuda.empty_cache()

    result_dict = {k: (v / max(n_imgs, 1)) for k, v in sums.items()}
    
    result_dict['ce_pred_gt'] = sums['ce_pred_gt'] / max(n_imgs - num_nan_ce, 1) if n_imgs > num_nan_ce else 0.0

    print(result_dict)

    return result_dict

@torch.inference_mode()
def visualize_single_index_multishot(
    cfg,
    model,
    device,
    evaluator,
    index: int,
    conf_thr: float,
    nms_thr: float,
    topk: Optional[int],
    iou_thr: float,
    miou_cfg: MiouCLI,
    candidate_exts: Optional[List[str]] = None,
    seed: int = 123,
    radius_down_ratio: float = 0.5,
    class_agnostic_nms: bool = True,
    heat_class_idx: int = 0,
    down_ratio: int = 1,              # heatmap stride used by convert_bbox_heatmap_per_img_tensor
    overlay_alpha: float = 0.8,       # heat overlay strength
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Multishot single-index visualization + metrics.

    - Aggregates detections across ALL shots under val_img_folder/{id:012d}/ (no subsampling).
    - Averages heatmap logits across ALL shots.
    - Overlays both on ONE randomly chosen shot (seeded).
    - Computes AP via pycocotools for ONLY this image id, and B-mode Recall/F1 + H-mode mIoU.
    """
    import glob, random, cv2, tempfile, json, os, numpy as np
    from pycocotools.coco import COCO

    if candidate_exts is None:
        candidate_exts = ['.jpg', '.jpeg', '.png']

    model.eval().to(device)
    dataset = evaluator.dataset
    transform = evaluator.transform
    EAD = bool(evaluator.EAD)

    # ---- resolve image ids in dataset order and pick this one
    if hasattr(dataset, 'ids'):
        coco_ids_order = list(map(int, dataset.ids))
    elif hasattr(dataset, 'img_ids'):
        coco_ids_order = list(map(int, dataset.img_ids))
    else:
        coco_ids_order = list(map(int, dataset.coco.getImgIds()))
    assert 0 <= index < len(coco_ids_order), f"single_index {index} out of range [0,{len(coco_ids_order)-1}]"
    img_id = int(coco_ids_order[index])

    coco: COCO = dataset.coco
    info = coco.loadImgs(img_id)[0]
    W_orig, H_orig = int(info['width']), int(info['height'])

    # ---- gather ALL candidate shots
    img_root = getattr(dataset, 'img_folder', None)
    if img_root is None:
        raise RuntimeError("dataset.img_folder must be set for multishot visualization.")
    subdir = os.path.join(img_root, f"{img_id:012d}")
    cand_paths: List[str] = []
    if os.path.isdir(subdir):
        for e in candidate_exts:
            cand_paths.extend(glob.glob(os.path.join(subdir, f"*{e}")))
    cand_paths = sorted(cand_paths)

    # flat fallback: {root}/{id:012d}.{ext}
    if not cand_paths:
        for e in candidate_exts:
            p = os.path.join(img_root, f"{img_id:012d}{e}")
            if os.path.isfile(p):
                cand_paths = [p]
                break

    if not cand_paths:
        raise FileNotFoundError(f"No candidate shots found for id={img_id} in {subdir} or flat pattern.")

    # ---- choose random canvas for drawing (seeded)
    rng = random.Random(seed + img_id)
    # print(f'cand_paths ({len(cand_paths)}): {cand_paths}')
    canvas_path = rng.choice(cand_paths)
    canvas_bgr = cv2.imread(canvas_path, cv2.IMREAD_COLOR)
    assert canvas_bgr is not None, f"Failed to read canvas image: {canvas_path}"
    H0, W0 = canvas_bgr.shape[:2]   # <-- native size
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    x0 = transform(canvas_rgb)[0]   # (C,Ht,Wt)
    Ht, Wt = int(x0.shape[-2]), int(x0.shape[-1])

    # ---- helper: GT heatmap at (Ht,Wt)
    def _build_gt_hm_for_id(img_id: int, Ht: int, Wt: int, H0: int, W0: int) -> torch.Tensor:
        if EAD:
            sx = float(Wt) / max(3*W0, 1)
            sy = float(Ht) / max(3*H0, 1)
        else:
            sx = float(Wt) / max(W0, 1)
            sy = float(Ht) / max(H0, 1)

        # pull boxes/labels (same logic as above)
        boxes_xyxy, labels = None, None
        if hasattr(dataset, 'pull_ann'):
            try:
                ann = dataset.pull_ann(index)
                if isinstance(ann, dict):
                    boxes_xyxy = ann.get('boxes', ann.get('bboxes', None))
                    labels     = ann.get('labels', ann.get('cls', None))
                elif isinstance(ann, (list, tuple)) and len(ann) >= 2:
                    boxes_xyxy, labels = ann[0], ann[1]
            except Exception:
                pass
        if boxes_xyxy is None or labels is None:
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            inv_map = None
            if hasattr(dataset, 'class_ids'):
                inv_map = {int(cat_id): int(i) for i, cat_id in enumerate(dataset.class_ids)}
            bx, lb = [], []
            for a in anns:
                x, y, w, h = a['bbox']
                bx.append([x, y, x + w, y + h])
                lb.append(inv_map[int(a['category_id'])] if inv_map is not None else int(a['category_id']))
            boxes_xyxy = np.array(bx, dtype=np.float32) if len(bx) else np.zeros((0,4), dtype=np.float32)
            labels     = np.array(lb, dtype=np.int64)    if len(lb) else np.zeros((0,), dtype=np.int64)

        boxes_xyxy = torch.as_tensor(boxes_xyxy, dtype=torch.float32) * torch.tensor([sx, sy, sx, sy])
        labels     = torch.as_tensor(labels,     dtype=torch.int64)
        labels_col = labels.view(-1, 1).to(dtype=torch.float32)
        boxes_with_labels = torch.cat([boxes_xyxy, labels_col], dim=1)

        img_dummy = torch.zeros(3, Ht, Wt, dtype=torch.float32)
        hm_CHW = build_fullres_box_heatmap(img_dummy, boxes_with_labels, num_classes=2, ead_pad=EAD)
        return hm_CHW.unsqueeze(0).to(device)

    hm_BCHW = _build_gt_hm_for_id(img_id, Ht, Wt, H0, W0)
    Hh, Ww = int(hm_BCHW.shape[-2]), int(hm_BCHW.shape[-1])
    gt_BSC = rearrange(hm_BCHW, 'b c h w -> b (h w) c')  # (1,S,C)

    # ---- aggregate detections + average heatmap logits across ALL shots
    scale_vec = torch.tensor([W0, H0, W0, H0], dtype=torch.float32)
    if EAD:
        scale_vec = scale_vec * 3.0

    all_boxes = []
    all_scores = []
    all_labels = []
    pred_logits_sum = None
    shot_count = 0

    for path in cand_paths:
        im_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if im_bgr is None:
            continue
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        x = transform(im_rgb)[0].unsqueeze(0).to(device, non_blocking=False)  # (1,3,Ht,Wt)

        # --- heatmap logits ---
        logits_BSC = model.heatmap(x)  # (1,S,C)
        if logits_BSC.shape[1] == gt_BSC.shape[1]:
            pred_logits_sum = logits_BSC if pred_logits_sum is None else (pred_logits_sum + logits_BSC)
            shot_count += 1

        # --- boxes/scores/labels ---
        out = _run_model_for_boxes(model, x)
        # Ensure tensor types and move to CPU for union
        b = out['boxes']
        s = out['scores']
        l = out['labels']
        if isinstance(b, np.ndarray): b = torch.from_numpy(b).to(device)
        if b.ndim == 1: b = b.unsqueeze(0)

        # scale to original COCO frame (parity with evaluator)
        b_cpu = (b.to(torch.float32).cpu() * scale_vec)  # (N,4) xyxy
        s_cpu = (s if isinstance(s, torch.Tensor) else torch.tensor(s)).to(torch.float32).cpu()
        l_cpu = (l if isinstance(l, torch.Tensor) else torch.tensor(l)).to(torch.int64).cpu()

        all_boxes.append(b_cpu)
        all_scores.append(s_cpu)
        all_labels.append(l_cpu)

        del x, logits_BSC, out, b, s, l

    # averaged logits
    if shot_count == 0:
        pred_logits_avg = torch.zeros_like(gt_BSC, device=device)
    else:
        pred_logits_avg = pred_logits_sum / float(shot_count)

    # union dets
    if len(all_boxes):
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
    else:
        boxes  = torch.zeros((0,4), dtype=torch.float32)
        scores = torch.zeros((0,),  dtype=torch.float32)
        labels = torch.zeros((0,),  dtype=torch.int64)

    # post-hoc filter: conf → (optional per-class) NMS → top-k
    pred_union = {'boxes': boxes, 'scores': scores, 'labels': labels}
    pred_union = _filter_boxes_posthoc(
        pred_union, conf_thr=conf_thr, nms_thr=nms_thr, topk=topk, class_agnostic_nms=class_agnostic_nms
    )

    # ---- B-mode single-image metrics (Recall/F1)
    # build GT boxes in original frame (xyxy)
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    G = []
    for a in anns:
        x, y, w, h = a['bbox']
        G.append([x, y, x + w, y + h])
    gt_boxes_xyxy = torch.tensor(G, dtype=torch.float32) if len(G) else torch.empty((0,4), dtype=torch.float32)

    b_metrics = _bmode_single_image_metrics(
        {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in pred_union.items()},
        gt_boxes_xyxy, iou_thr=iou_thr
    )
    

    # ---- H-mode mIoU suite (using averaged logits)
    grid = ThresholdGrid(values=tuple(miou_cfg.thresholds))
    # in visualize_single_index_multishot
    pred_logits_u, hm_u, (Hu, Wu), (Hc, Wc) = _unify_pred_to_gt_grid(
        pred_logits_avg, hm_BCHW, Ht=Ht, Wt=Wt, EAD_enabled=EAD
    )
    gt_BSC_u = rearrange(hm_u, 'b c h w -> b (h w) c')
    
    if verbose:
        print(f"[VIS] native (H0,W0)=({H0},{W0})  transformed (Ht,Wt)=({Ht},{Wt})")
        print(f"[VIS] pred grid inferred (Hc,Wc)=({Hc},{Wc})  GT grid (Hu,Wu)=({Hu},{Wu})")
    
    # ---- GT occupancy stats (per class) ----
    with torch.no_grad():
        gt_sum_per_class = (hm_u > 0).sum(dim=[0, 2, 3]).cpu().tolist()  # length = C
        total_pixels = int(hm_u.shape[-2] * hm_u.shape[-1])
        print(f"[GT] (Hu,Wu)=({Hu},{Wu}), total_pixels={total_pixels}, per-class positives={gt_sum_per_class}, empty_any={all(s == 0 for s in gt_sum_per_class)}")

    miou_opts = MiouOptions(
        ignore_center=miou_cfg.ignore_center,
        spatial_hw=(Hu, Wu),
        gt_binarize=miou_cfg.gt_binarize,
        gt_rel_thresh=miou_cfg.gt_rel_thresh,
        gt_percentile=miou_cfg.gt_percentile,
        beta=miou_cfg.beta,
        class_idx=miou_cfg.class_idx
    )
    mres = miou_over_thresholds_batch(pred_logits_u, hm_u, grid, miou_opts)

    # ---- COCO AP for ONLY this image id (stream a tiny JSON)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json"); os.close(tmp_fd)
    wrote = False
    with open(tmp_path, 'w') as fjson:
        fjson.write('[')
        for i in range(pred_union['boxes'].size(0)):
            x1, y1, x2, y2 = pred_union['boxes'][i].tolist()
            w_box = x2 - x1; h_box = y2 - y1
            label = dataset.class_ids[int(pred_union['labels'][i].item())] if hasattr(dataset, 'class_ids') else int(pred_union['labels'][i].item())
            det = {
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w_box), float(h_box)],
                "score": float(pred_union['scores'][i].item())
            }
            if wrote: fjson.write(',')
            json.dump(det, fjson, separators=(',', ':'))
            wrote = True
        fjson.write(']')

    cocoDt = coco.loadRes(tmp_path)
    if evaluator.iouThr is not None:
        cocoEval = COCOeval_with_Difficulty(coco, cocoDt, 'bbox', iouThr=evaluator.iouThr)
    else:
        cocoEval = COCOeval_with_Difficulty(coco, cocoDt, 'bbox')
    # evaluate only this image id
    cocoEval.params.imgIds = [img_id]
    cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
    ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]

    try: os.remove(tmp_path)
    except: pass

    # ---- build overlays / dumps ----
    base = cv2.resize(canvas_bgr, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

    # (A) PRED heat overlay (use upsampled logits already on (Hu,Wu))
    with torch.no_grad():
        pred_prob_full = torch.sigmoid(pred_logits_u[..., heat_class_idx]).reshape(1, 1, Hu, Wu)
        pred_prob_for_canvas = F.interpolate(pred_prob_full, size=(Ht, Wt), mode='bilinear', align_corners=False)[0, 0].clamp(0, 1)
        pred_uint8 = (pred_prob_for_canvas.detach().cpu().numpy() * 255.0).astype(np.uint8)

    pred_color = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)
    overlay_pred = cv2.addWeighted(base, 1.0, pred_color, overlay_alpha, 0.0)

    # (B) GT heatmap exports
    gt_full = hm_u[0, heat_class_idx]  # (Hu, Wu), binary (0/1)
    gt_full_uint8 = (gt_full.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    gt_full_color = cv2.applyColorMap(gt_full_uint8, cv2.COLORMAP_JET)

    # also a downsampled GT-on-canvas preview
    gt_for_canvas = cv2.resize(gt_full_uint8, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    gt_color_canvas = cv2.applyColorMap(gt_for_canvas, cv2.COLORMAP_JET)
    overlay_gt = cv2.addWeighted(base, 1.0, gt_color_canvas, overlay_alpha, 0.0)

    # ---- write files ----
    os.makedirs('./viz_multishot', exist_ok=True)
    stamp = f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'
    pred_overlay_png = f'./viz_multishot/overlay_pred_idx{index:06d}_id{img_id:012d}_{stamp}.png'
    gt_overlay_png   = f'./viz_multishot/overlay_gt_idx{index:06d}_id{img_id:012d}_{stamp}.png'
    gt_full_png      = f'./viz_multishot/gt_full_idx{index:06d}_id{img_id:012d}_{stamp}.png'
    gt_full_color_png= f'./viz_multishot/gt_full_color_idx{index:06d}_id{img_id:012d}_{stamp}.png'
    gt_npz           = f'./viz_multishot/gt_full_idx{index:06d}_id{img_id:012d}_{stamp}.npz'
    
    # PRED full-res heatmap on (Hu,Wu)
    pred_prob_full = torch.sigmoid(pred_logits_u[..., heat_class_idx]).reshape(Hu, Wu)
    pred_full_uint8 = (pred_prob_full.clamp(0,1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    pred_full_color = cv2.applyColorMap(pred_full_uint8, cv2.COLORMAP_JET)

    pred_full_png       = f'./viz_multishot/pred_full_idx{index:06d}_id{img_id:012d}_{stamp}.png'
    pred_full_color_png = f'./viz_multishot/pred_full_color_idx{index:06d}_id{img_id:012d}_{stamp}.png'
    pred_npz            = f'./viz_multishot/pred_full_idx{index:06d}_id{img_id:012d}_{stamp}.npz'
    cv2.imwrite(pred_full_png, pred_full_uint8)
    cv2.imwrite(pred_full_color_png, pred_full_color)
    np.savez(pred_npz, pred_logits_u=pred_logits_u.detach().cpu().numpy(), Hu=Hu, Wu=Wu, heat_class_idx=heat_class_idx)

    cv2.imwrite(pred_overlay_png, overlay_pred)
    cv2.imwrite(gt_overlay_png, overlay_gt)
    cv2.imwrite(gt_full_png, gt_full_uint8)
    cv2.imwrite(gt_full_color_png, gt_full_color)
    # save *entire* GT tensor (all classes) for perfect reproducibility
    np.savez(gt_npz, hm_u=hm_u.detach().cpu().numpy(), Hu=Hu, Wu=Wu, heat_class_idx=heat_class_idx)
    if verbose:
        print(f"[B] TP={b_metrics['TP']} FP={b_metrics['FP']} FN={b_metrics['FN']}")

    # pack results
    res = {
        'index': index,
        'image_id': img_id,
        'canvas_path': canvas_path,
        'overlay_path': pred_overlay_png,   # (renamed to pred overlay)
        'gt_overlay_path': gt_overlay_png,  # NEW
        'gt_full_png': gt_full_png,         # NEW
        'gt_full_color_png': gt_full_color_png, # NEW
        'gt_npz': gt_npz,                   # NEW
        'pred_full_png': pred_full_png,
        'pred_full_color_png': pred_full_color_png,
        'pred_npz': pred_npz,
        'B': {
            'TP': b_metrics['TP'], 'FP': b_metrics['FP'], 'FN': b_metrics['FN'],
            'Precision': b_metrics['Precision'], 'Recall': b_metrics['Recall'],
            'F1': b_metrics['F1'], 'meanIoU_TP': b_metrics['meanIoU_TP'],
        },
        'H': {
            'mIoU': mres['mIoU'], 'mIoU_int': mres['mIoU_int'], 'pixelAP': mres['pixelAP']
        },
        'COCO': {
            'AP50_95': ap50_95,
            'AP50': ap50
        },
        'conf_thresh': conf_thr,
        'nms_thresh': nms_thr,
        'topk': (None if topk is None else int(topk)),
    }
    return res

# ============================================================
# CSV upsert
# ============================================================

CSV_HEADER = [
    'config_name','weight_file','date',
    'conf_thresh','nms_thresh','topk',
    'AP50_95','AP50','AP75','AP_Normal','AP_Easy','AP_Medium','AP_Hard', 'AP_Outside',
    'Center_Errors','Center_Errors_normal','Center_Errors_easy','Center_Errors_medium','Center_Errors_hard', 'Center_Errors_outside',
    'SE_Pred','CE_Pred_GT','mIoU','mIoU_int','pixelAP'
]

def load_csv_rows(csv_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        return list(r)

def save_csv_rows(csv_path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def upsert_row(csv_path: str, new_row: Dict[str, Any]):
    rows = load_csv_rows(csv_path)
    key = (new_row['config_name'], new_row['weight_file'], new_row['date'],
           str(new_row['conf_thresh']), str(new_row['nms_thresh']), str(new_row['topk']))
    def row_key(r):
        return (r['config_name'], r['weight_file'], r['date'],
                r['conf_thresh'], r['nms_thresh'], r['topk'])
    updated = False
    for i, r in enumerate(rows):
        if row_key(r) == key:
            rows[i] = new_row
            updated = True
            break
    if not updated:
        rows.append(new_row)
    save_csv_rows(csv_path, rows)

# ============================================================
# Main
# ============================================================

# multishot
def get_evaluator_and_loader(cfg, device, args, radius_down_ratio):
    evaluator = get_evaluator(cfg, device, args.val_img_folder, args.val_ann_file, args.cocoeval_iouthr)

    # NEW: multishot configuration passed to the evaluator (for AP)
    evaluator.num_per_index  = args.num_per_index       # 0 => use all shots
    evaluator.seed           = args.seed
    evaluator.candidate_exts = args.candidate_exts
    evaluator.nms_per_class  = args.nms_per_class

    mopts = MetricOptions(
        pred_input_type=args.pred_input_type,
        se_class_idx=args.se_class_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ignore_center=(not args.no_entropy_ignore_center),
        heat_unify_to_ead_on=bool(args.heat_unify_to_ead_on),
    )
    loader = build_eval_dataloader(cfg, evaluator, num_classes=2, mopts=mopts, radius_down_ratio=radius_down_ratio)
    return evaluator, loader, mopts

def main(args):
    # load model list
    config_names, weight_files = [], []
    for path in args.model_list_json:
        with open(path, 'r') as f:
            model_list = json.load(f)
            config_names.extend(model_list['config_names'])
            weight_files.extend(model_list['weight_files'])
    assert len(config_names) == len(weight_files), "config_names and weight_files length mismatch"

    # thresholds file (conf/nms/topk + hm radius)
    with open(args.thresholds_json, 'r') as f:
        th = json.load(f)
    conf_threshs = args.conf_thresh if args.conf_thresh is not None else [float(th['conf_thresh'])]
    nms_threshs  = args.nms_thresh  if args.nms_thresh  is not None else [float(th['nms_thresh'])]
    topks        = args.topk        if args.topk        is not None else [int(th['topk'])]
    radius_down_ratio = args.radius_down_ratio if args.radius_down_ratio is not None else float(th.get('radius_down_ratio', 0.5))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # mIoU CLI bundle
    miou_cfg = MiouCLI(
        thresholds=(np.round(np.linspace(args.miou_thr_min, args.miou_thr_max, args.miou_thr_num), 3)).tolist(),
        gt_binarize=args.miou_gt_binarize,
        gt_rel_thresh=args.miou_gt_rel,
        gt_percentile=args.miou_gt_pct,
        beta=args.miou_beta,
        class_idx=args.miou_class_idx,
        ignore_center=bool(args.miou_ignore_center)
    )

        # If single-sample mode is requested, we still respect conf/nms/topk sweeps.
    for conf_thresh in conf_threshs:
        for nms_thresh in nms_threshs:
            for topk in topks:
                for config_name, weight_file in zip(config_names, weight_files):

                    cfg = yoloh_config[config_name]
                    if args.img_size is not None:
                        cfg['input_size'] = (args.img_size, args.img_size)
                    else:
                        cfg['input_size'] = (320, 320)

                    # NOTE: For full alignment with visualize_runner, we want permissive thresholds
                    # when extracting raw boxes. We can still use this single 'model' as long as
                    # its forward returns raw boxes; if not, build with conf=0.0,nms=1.0 here.
                    model = build_model_noargs(
                        cfg=cfg, device=device, num_classes=2, trainable=False,
                        coco_pretrained=weight_file, fig_size=cfg['input_size'],
                        conf_thresh=conf_thresh, nms_thresh=nms_thresh, topk=topk,
                    )
                    model.eval().to(device)

                    evaluator, loader, mopts = get_evaluator_and_loader(cfg, device, args, radius_down_ratio)

                    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Evaluating {config_name} | conf={conf_thresh} nms={nms_thresh} topk={topk}")

                    # ---------- SINGLE-SAMPLE PATH ----------
                    # ---------- SINGLE-INDEX MULTISHOT OVERLAY + METRICS ----------
                    if args.single_index is not None:
                        res = visualize_single_index_multishot(
                            cfg=cfg, model=model, device=device, evaluator=evaluator,
                            index=args.single_index,
                            conf_thr=conf_thresh, nms_thr=nms_thresh, topk=topk,
                            iou_thr=args.single_iou_thr,
                            miou_cfg=miou_cfg,
                            candidate_exts=args.candidate_exts,
                            seed=args.seed,
                            radius_down_ratio=radius_down_ratio,
                            class_agnostic_nms=(not args.nms_per_class),
                            heat_class_idx=args.miou_class_idx,
                            down_ratio=1,                   # keep in sync with your heatmap stride
                            overlay_alpha=0.8,
                            verbose=bool(args.verbose)
                        )
                        # pretty print + save JSON
                        print(f"[single-multishot] idx={res['index']} id={res['image_id']} | "
                            f"COCO: AP50_95={res['COCO']['AP50_95']:.4f} AP50={res['COCO']['AP50']:.4f} | "
                            f"B-mode: P={res['B']['Precision']:.3f} R={res['B']['Recall']:.3f} F1={res['B']['F1']:.3f} "
                            f"IoU@TP={res['B']['meanIoU_TP']:.3f} | "
                            f"H-mode: mIoU={res['H']['mIoU']:.3f} mIoU_int={res['H']['mIoU_int']:.3f} pixelAP={res['H']['pixelAP']:.3f} | "
                            f"overlay → {res['overlay_path']}")
                        os.makedirs('./viz_multishot', exist_ok=True)
                        safe_date = f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'
                        out_json = f'./viz_multishot/single_multishot_{config_name.replace("/", "_")}_idx{args.single_index}_conf{conf_thresh}_nms{nms_thresh}_topk{topk}_{safe_date}.json'
                        with open(out_json, 'w') as f:
                            json.dump(res, f, indent=2)
                        # tidy and continue to next (no dataset-wide eval)
                        del model, evaluator, loader
                        torch.cuda.empty_cache()
                        continue

                    # ---------- ORIGINAL DATASET-WIDE PATH ----------
                    # ---------- AP (COCO) ----------
                    if not args.skip_ap_mse:
                        start = datetime.datetime.now()
                        eval_result_dict = evaluator.evaluate(model)
                        print(f"COCO eval time: {datetime.datetime.now() - start}")
                    else:
                        print("Skipping COCO eval")
                        eval_result_dict = {}

                    # ---------- Heatmap metrics ----------
                    start = datetime.datetime.now()

                    # Use multishot heatmap averaging (always enabled; 0 = use all shots; works fine when no subfolder exists)
                    mdict = run_metrics_multishot(
                        model=model,
                        evaluator=evaluator,
                        device=device,
                        mopts=mopts,
                        miou_cfg=miou_cfg,
                        radius_down_ratio=radius_down_ratio,
                        num_per_index=args.num_per_index,
                        seed=args.seed,
                        candidate_exts=args.candidate_exts,
                    )
                    print(f"Metric pass time: {datetime.datetime.now() - start}")

                    # ... (keep your existing packing/saving code below)
                    # pack & save
                    date_str = f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}'
                    safe_date = date_str.replace(' ', '_').replace(':','-')
                    os.makedirs('./result', exist_ok=True)

                    row = {
                        'config_name': config_name,
                        'weight_file': os.path.basename(weight_file),
                        'date': date_str,
                        'conf_thresh': conf_thresh,
                        'nms_thresh': nms_thresh,
                        'topk': topk,
                        'AP50_95': eval_result_dict.get('ap50_95', ''),
                        'AP50': eval_result_dict.get('ap50', ''),
                        'AP75': eval_result_dict.get('ap75', ''),
                        'AP_Normal': eval_result_dict.get('ap_normal', ''),
                        'AP_Easy': eval_result_dict.get('ap_easy', ''),
                        'AP_Medium': eval_result_dict.get('ap_medium', ''),
                        'AP_Hard': eval_result_dict.get('ap_hard', ''),
                        'AP_Outside': eval_result_dict.get('ap_outside', ''),
                        'Center_Errors': eval_result_dict.get('center_errors', ''),
                        'Center_Errors_normal': eval_result_dict.get('center_errors_normal', ''),
                        'Center_Errors_easy': eval_result_dict.get('center_errors_easy', ''),
                        'Center_Errors_medium': eval_result_dict.get('center_errors_medium', ''),
                        'Center_Errors_hard': eval_result_dict.get('center_errors_hard', ''),
                        'Center_Errors_outside': eval_result_dict.get('center_errors_outside', ''),
                        'SE_Pred': mdict.get('se_pred', ''),
                        'CE_Pred_GT': mdict.get('ce_pred_gt', ''),
                        'mIoU': mdict.get('mIoU', ''),
                        'mIoU_int': mdict.get('mIoU_int', ''),
                        'pixelAP': mdict.get('pixelAP', ''),
                    }

                    # date-stamped NPZ (won't overwrite)
                    np.savez(f'./result/{config_name}_{safe_date}_metrics.npz', **row, **eval_result_dict)

                    # CSV upsert
                    csv_out = './result/result.csv'
                    upsert_row(csv_out, {k: ('' if v is None else v) for k, v in row.items()})

                    # tidy
                    del model, evaluator, loader
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_list_json', type=str, required=True, nargs='+',
                    help="JSON file(s) with {'config_names': [...], 'weight_files': [...]}")

    ap.add_argument('--thresholds_json', type=str, required=True, help="JSON with conf_thresh,nms_thresh,topk")
    ap.add_argument('--val_img_folder', type=str, required=True)
    ap.add_argument('--val_ann_file', type=str, required=True)

    # CE/SE
    ap.add_argument('--pred_input_type', type=str, default='pred_logits', choices=['pred_logits','pred_probs'])
    ap.add_argument('--se_class_idx', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=2)

    # model thresholds overrides (optional lists)
    ap.add_argument('--conf_thresh', type=float, default=None, nargs='+')
    ap.add_argument('--nms_thresh', type=float, default=None, nargs='+')
    ap.add_argument('--topk', type=int, default=None, nargs='+')
    ap.add_argument('--radius_down_ratio', type=float, default=None)
    ap.add_argument('--img_size', type=int, default=None)

    # center-ignore (default ON for entropy)
    ap.add_argument('--no-entropy_ignore_center', dest='no_entropy_ignore_center', action='store_true')
    ap.set_defaults(no_entropy_ignore_center=False)

    # COCO/AP skip
    ap.add_argument('--skip_ap_mse', action='store_true')

    # SE/CE skip
    ap.add_argument('--skip_se_ce', action='store_true',
                    help="Skip SE/CE metric pass, only run mIoU suite")

    # mIoU suite options (probability thresholds, GT binarization, center-ignore)
    ap.add_argument('--miou_thr_min', type=float, default=0.05)
    ap.add_argument('--miou_thr_max', type=float, default=0.95)
    ap.add_argument('--miou_thr_num', type=int, default=19)
    ap.add_argument('--miou_gt_binarize', type=str, default='gt>0', choices=['relative','percentile','gt>0'])
    ap.add_argument('--miou_gt_rel', type=float, default=0.25)
    ap.add_argument('--miou_gt_pct', type=float, default=90.0)
    ap.add_argument('--miou_beta', type=float, default=1.0)
    ap.add_argument('--miou_class_idx', type=int, default=0)
    ap.add_argument('--miou_ignore_center', type=int, default=1)  # 1=on, 0=off
    ap.add_argument('--skip_miou', action='store_true')
    
    # cocoeval iouThr
    ap.add_argument('--cocoeval_iouthr', type=float, default=None)
    
    # ---------- single-sample mode ----------
    ap.add_argument('--single_index', type=int, default=None,
                    help='If set, run single-sample metrics for this dataset index and exit.')
    ap.add_argument('--single_iou_thr', type=float, default=0.5,
                    help='IoU threshold for B-mode single-sample matching.')
    ap.add_argument('--single_raw_topk_max', type=int, default=1200,
                    help='If you rebuild a permissive model externally, mirror this value.')
    ap.add_argument('--nms_per_class', action='store_true',
                    help='Use class-wise NMS in single-sample post-hoc filtering.')
    
    # ---------- multishot options ----------
    # multishot controls
    ap.add_argument('--num_per_index', type=int, default=0,
                    help='0 = use all shots per index; otherwise sample this many (canonical + sampled).')
    ap.add_argument('--seed', type=int, default=114514, help='Deterministic subsampling seed.')

    # optional: changeable candidate extensions (defaults to common image types)
    ap.add_argument('--candidate_exts', type=str, nargs='+',
                    default=['.jpg', '.jpeg', '.png'],
                    help='List of file extensions considered for multishot candidates in subfolders.')
    
    # in argparse
    ap.add_argument('--heat_unify_to_ead_on', type=int, default=1,
                help='If 1 and cfg[EAD]==0, upsample pred/gt heatmaps to the equivalent EAD-on grid (3x spatial).')

    ap.add_argument('--verbose', action='store_true',
                help='Verbose logs + save extra GT/pred heatmaps in visualize_single_index.')

    args = ap.parse_args()
    main(args)
