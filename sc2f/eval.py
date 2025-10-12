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

# ---- your project imports (paths the same as your current code) ----
import sys
from data.transforms import ValTransforms
from evaluator.coco_evaluator import COCOAPIEvaluator
from data.coco_fb_diff import COCODataset_FB_Diff  # noqa: F401 (for side-effects)
from config.yoloh_config_expand import yoloh_config
from models.yoloh import build_model_noargs

import cv2
import tempfile
try:
    from pycocotools.cocoeval import COCOeval
except Exception:
    COCOeval = None

# ============================================================
# Heatmap helpers
# ============================================================

def pad_tensor_image_3(image: torch.Tensor) -> torch.Tensor:
    h, w = image.shape[1], image.shape[2]
    pad = (w, w, h, h)
    image = torch.nn.functional.pad(image, pad, value=0)
    return image

def convert_bbox_heatmap_per_img_tensor(img, bboxes, num_classes, down_ratio=1, radius_down_ratio=1):
    c, h, w  = img.shape
    h, w = h // down_ratio, w // down_ratio
    hm = np.zeros((num_classes, h, w), dtype=np.float32)
    for bbox in bboxes:
        if bbox.shape[0] == 5:
            bbox_category_id = bbox[4]
            if 'int' not in str(type(bbox_category_id)):
                bbox_category_id = int(bbox_category_id)
        else:
            continue
        hm[bbox_category_id,
           int(bbox[1])//down_ratio:int(bbox[3])//down_ratio,
           int(bbox[0])//down_ratio:int(bbox[2])//down_ratio] = 1.0
    return torch.tensor(hm)

# ==================== full-res GT heatmap builder (EAD-aware) ====================

def build_fullres_box_heatmap(img_like: torch.Tensor,
                              boxes_with_labels: torch.Tensor,
                              num_classes: int,
                              ead_pad: bool) -> torch.Tensor:
    """
    Returns (C, 3H, 3W) GT heatmap filled with 1s inside GT boxes, 0 elsewhere.
    - If ead_pad=True: canvas is padded to 3H x 3W; boxes are shifted by (+W, +H) to the center.
    - If ead_pad=False: canvas is still 3H x 3W; boxes are scaled by 3x (no shift).
    """
    _, H, W = img_like.shape  # transformed model input size (e.g., 320x320)
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
    def __init__(self, num_classes, down_ratio=1, radius_down_ratio=1, EAD_pad=True, *args, **kwargs):
        self.num_classes = num_classes
        self.down_ratio = down_ratio  # keep for API parity; we use full-res builder now
        self.radius_down_ratio = radius_down_ratio
        self.EAD_pad = EAD_pad

    def __call__(self, batch):
        batch_size = len(batch)
        images, targets, masks, heatmaps = [], [], [], []
        for bid in range(batch_size):
            img = batch[bid][0]      # (3,H,W) transformed input
            tgt = batch[bid][1]      # dict with 'boxes','labels' (transformed coords)
            msk = batch[bid][2]

            images.append(img)
            targets.append(tgt)
            masks.append(msk)

            boxes_with_labels = torch.cat(
                (tgt['boxes'], rearrange(tgt['labels'], 'n -> n 1').to(tgt['boxes'].dtype)),
                dim=1
            )
            hm_CHW = build_fullres_box_heatmap(
                img_like=img,
                boxes_with_labels=boxes_with_labels,
                num_classes=self.num_classes,
                ead_pad=bool(self.EAD_pad)
            )
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

def _build_center_mask_hw(H_full: int, W_full: int, device=None, dtype=None):
    m = torch.ones((H_full, W_full), dtype=dtype or torch.float32, device=device)
    h1, h2 = H_full // 3, 2 * (H_full // 3)
    w1, w2 = W_full // 3, 2 * (W_full // 3)
    m[h1:h2, w1:w2] = 0
    return m

def _binarize_gt_spatial(gt_BHW: torch.Tensor, opts) -> torch.Tensor:
    if opts.gt_binarize == "gt>0":
        return (gt_BHW > 0).to(gt_BHW.dtype)
    elif opts.gt_binarize == "relative":
        B, H, W = gt_BHW.shape
        gmax = gt_BHW.view(B, -1).amax(dim=1).clamp_min(1e-8)
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
# B-mode (single-image) helpers
# ============================================================

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
            if m.sum() == 0: continue
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
        if isinstance(one, dict) and all(k in one for k in ('bboxes','scores','labels')):
            return {'boxes': one['bboxes'], 'scores': one['scores'], 'labels': one['labels']}
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

# ============================================================
# Grid unifier (no fixed stride assumptions)
# ============================================================

@torch.inference_mode()
def _unify_pred_to_gt_grid_simple(
    pred_logits_BSC: torch.Tensor,   # (B,S,C)
    gt_heat_BCHW: torch.Tensor       # (B,C,Htgt,Wtgt)
) -> Tuple[torch.Tensor, Tuple[int,int], Tuple[int,int]]:
    """
    Upsample pred logits to GT grid without assuming any fixed stride.
    We factor S into (Hc,Wc) using the GT aspect ratio.
    Returns: (pred_up_BSC, (Htgt,Wtgt), (Hc,Wc))
    """
    B, S, C = pred_logits_BSC.shape
    _, Cg, Htgt, Wtgt = gt_heat_BCHW.shape
    assert C == Cg, f"C mismatch: pred {C} vs gt {Cg}"
    ar = float(Htgt) / max(Wtgt, 1)

    # Factor S ≈ Hc*Wc with Hc/Wc ≈ ar (robust to rounding)
    Wc = max(1, int(round(math.sqrt(S / max(ar, 1e-12)))))
    Hc = max(1, S // Wc)
    if Hc * Wc != S:
        Hc = max(1, int(round(math.sqrt(S * ar))))
        Wc = max(1, S // Hc)
    assert Hc * Wc == S, f"Cannot factor S={S} into HxW; got Hc={Hc}, Wc={Wc}, ar={ar:.4f}"

    pred_BCHW = rearrange(pred_logits_BSC, 'b (h w) c -> b c h w', h=Hc, w=Wc)
    pred_up_BCHW = F.interpolate(pred_BCHW, size=(Htgt, Wtgt), mode='bilinear', align_corners=False)
    pred_up_BSC  = rearrange(pred_up_BCHW, 'b c h w -> b (h w) c')
    return pred_up_BSC, (Htgt, Wtgt), (Hc, Wc)

# ============================================================
# Single-sample (for quick debugging)
# ============================================================

@dataclass
class ThresholdGrid:
    kind: str = "prob"
    values: Iterable[float] = tuple(np.round(np.linspace(0.05, 0.95, 19), 2))

@dataclass
class MiouOptions:
    ignore_center: bool = True
    spatial_hw: Optional[Tuple[int,int]] = None
    gt_binarize: str = "gt>0"     # 'relative'|'percentile'|'gt>0'
    gt_rel_thresh: float = 0.25
    gt_percentile: float = 90.0
    beta: float = 1.0
    class_idx: Optional[int] = None   # for (B,S,C) inputs, choose one class

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
    """
    dataset = evaluator.dataset
    dataset.transform = evaluator.transform

    # Pull one transformed sample and build its GT heatmap via the same collector
    collate = HeatmapCollector(num_classes=2, down_ratio=1, radius_down_ratio=radius_down_ratio, EAD_pad=cfg.get('EAD', True))
    sample = dataset[index]  # (img, target, mask)
    imgs_B3HW, targets_list, masks_B, hm_BCHW = collate([sample])
    img = imgs_B3HW[0].to(device)           # (3,H,W)
    tgt = targets_list[0]                   # dict with 'boxes','labels'
    hm  = hm_BCHW.to(device)                # (1,C,3H,3W)

    # -------- H-mode metrics (single image) ----------
    pred_logits_raw = model.heatmap(img.unsqueeze(0))  # (1,S,C)
    pred_logits_BSC, (Hh, Ww), (Hc, Wc) = _unify_pred_to_gt_grid_simple(pred_logits_raw, hm)

    ce = heatmap_2dce(
        pred_logits_BSC, rearrange(hm, 'b c h w -> b (h w) c'),
        input_type=pred_input_type, reduction='none',
        ignore_center=bool(miou_cfg.ignore_center), spatial_hw=(Hh, Ww)
    ).nanmean().item()

    se = heatmap_self_entropy_2d(
        pred_logits_BSC, input_type='pred_logits', reduction='none',
        class_idx=miou_cfg.class_idx,
        ignore_center=bool(miou_cfg.ignore_center), spatial_hw=(Hh, Ww)
    ).nanmean().item()

    grid = ThresholdGrid(values=tuple(miou_cfg.thresholds))
    miou_opts = MiouOptions(
        ignore_center=miou_cfg.ignore_center, spatial_hw=(Hh, Ww),
        gt_binarize=miou_cfg.gt_binarize, gt_rel_thresh=miou_cfg.gt_rel_thresh,
        gt_percentile=miou_cfg.gt_percentile, beta=miou_cfg.beta,
        class_idx=miou_cfg.class_idx
    )
    mres = miou_over_thresholds_batch(pred_logits_BSC, hm, grid, miou_opts)

    H_metrics = {'CE': ce, 'SE': se, **mres}

    # -------- B-mode metrics (single image, post-hoc conf/NMS) ----------
    raw = _run_model_for_boxes(model, img.unsqueeze(0))
    raw_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in raw.items()}
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
def visualize_single_index_overlay(
    cfg,
    model,
    device,
    evaluator,
    index: int,
    conf_thr: float,
    nms_thr: float,
    topk: Optional[int],
    iou_thr: float,
    miou_cfg: MiouOptions,
    radius_down_ratio: float = 0.5,
    heat_class_idx: int = 0,
    overlay_alpha: float = 0.8,
    verbose: bool = False,
):
    """
    Single-index overlay + metrics as if the dataset had exactly one image.
    - Canvas = transformed input; if cfg['EAD'] -> 3x padded (EAD) canvas.
    - Draw heatmap (single shot) + post-NMS boxes.
    - Compute AP via COCOeval restricted to this image id.
    - Save pred & GT heatmaps (grayscale, color, and NPZ).
    """
    dataset = evaluator.dataset
    transform = evaluator.transform
    EAD = bool(cfg.get('EAD', True))

    # --- resolve img_id and fetch the original RGB for canvas sizing ---
    img_raw, img_id = dataset.pull_image(index)     # original RGB (H_orig,W_orig,3), COCO id
    H_orig, W_orig = img_raw.shape[0], img_raw.shape[1]

    # --- transformed size (Ht,Wt) ---
    img_t = transform(img_raw)[0]                   # (3,Ht,Wt) torch
    Ht, Wt = int(img_t.shape[-2]), int(img_t.shape[-1])

    # --- build GT heatmap via collector (EAD-aware, on (3H,3W)) ---
    collate = HeatmapCollector(num_classes=2, down_ratio=1, radius_down_ratio=radius_down_ratio, EAD_pad=EAD)
    sample = dataset[index]
    _, targets_list, _, hm_BCHW = collate([sample])    # (1,C,3H,3W)
    Hh, Ww = int(hm_BCHW.shape[-2]), int(hm_BCHW.shape[-1])

    # --- forward (heatmap logits) on transformed image, then unify to GT grid ---
    img_in = img_t.unsqueeze(0).to(device, non_blocking=False)     # (1,3,Ht,Wt)
    logits_raw = model.heatmap(img_in)                             # (1,S,C)
    logits_BSC, (Hh, Ww), (Hc, Wc) = _unify_pred_to_gt_grid_simple(logits_raw, hm_BCHW.to(device))
    gt_BSC = rearrange(hm_BCHW.to(device), 'b c h w -> b (h w) c') # (1,S,C)

    if verbose:
        print(f"[VIS] native (H0,W0)=({H_orig},{W_orig})  transformed (Ht,Wt)=({Ht},{Wt})")
        print(f"[VIS] pred grid inferred (Hc,Wc)=({Hc},{Wc})  GT grid (Hh,Ww)=({Hh},{Ww})")

    # --- CE/SE + mIoU (H-mode) ---
    ce = heatmap_2dce(
        logits_BSC, gt_BSC, input_type='pred_logits', reduction='none',
        ignore_center=bool(miou_cfg.ignore_center), spatial_hw=(Hh, Ww)
    ).nanmean().item()
    se = heatmap_self_entropy_2d(
        logits_BSC, input_type='pred_logits', reduction='none',
        class_idx=miou_cfg.class_idx,
        ignore_center=bool(miou_cfg.ignore_center), spatial_hw=(Hh, Ww)
    ).nanmean().item()
    grid = ThresholdGrid(values=tuple(miou_cfg.thresholds))
    miou_opts = MiouOptions(
        ignore_center=miou_cfg.ignore_center, spatial_hw=(Hh, Ww),
        gt_binarize=miou_cfg.gt_binarize, gt_rel_thresh=miou_cfg.gt_rel_thresh,
        gt_percentile=miou_cfg.gt_percentile, beta=miou_cfg.beta,
        class_idx=miou_cfg.class_idx
    )
    mres = miou_over_thresholds_batch(logits_BSC, hm_BCHW.to(device), grid, miou_opts)

    # --- boxes: parse → map to center-crop frame (Ht,Wt) → post-hoc filter ---
    raw = _run_model_for_boxes(model, img_in)  # device tensors
    raw_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in raw.items()}
    pred_cc = _coerce_pred_to_image_frame(raw_cpu, img_hw=(Ht, Wt))  # center-crop coords

    pred_cc = _filter_boxes_posthoc(
        pred_cc, conf_thr=conf_thr, nms_thr=nms_thr, topk=topk,
        class_agnostic_nms=(not getattr(miou_cfg, 'nms_per_class', False))
    )

    # --- B-mode single-image metrics (Recall/F1) vs GT boxes at center-crop scale ---
    ann_ids = dataset.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
    anns = dataset.coco.loadAnns(ann_ids)
    G = []
    for a in anns:
        x, y, w, h = a['bbox']
        G.append([x, y, x + w, y + h])
    if len(G):
        G = torch.tensor(G, dtype=torch.float32)
        sx = float(Wt) / max(W_orig, 1)
        sy = float(Ht) / max(H_orig, 1)
        G_cc = G * torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
    else:
        G_cc = torch.empty((0,4), dtype=torch.float32)

    b_metrics = _bmode_single_image_metrics(
        {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in pred_cc.items()},
        G_cc, iou_thr=iou_thr
    )
    if verbose:
        print(f"[B] TP={b_metrics['TP']} FP={b_metrics['FP']} FN={b_metrics['FN']}")

    # --- AP (COCO) on THIS image only ---
    if COCOeval is None:
        ap50_95 = ap50 = -1.0
    else:
        sx_inv = float(W_orig) / max(Wt, 1)
        sy_inv = float(H_orig) / max(Ht, 1)
        boxes_orig = pred_cc['boxes'] * torch.tensor([sx_inv, sy_inv, sx_inv, sy_inv], dtype=torch.float32)
        labels = pred_cc['labels']
        scores = pred_cc['scores']
        def to_cat_id(label_idx: int) -> int:
            if hasattr(dataset, 'class_ids'):
                return int(dataset.class_ids[int(label_idx)])
            return int(label_idx)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json"); os.close(tmp_fd)
        with open(tmp_path, 'w') as fjson:
            fjson.write('['); wrote = False
            for i in range(boxes_orig.size(0)):
                x1, y1, x2, y2 = boxes_orig[i].tolist()
                w_box = x2 - x1; h_box = y2 - y1
                det = {
                    "image_id": int(img_id),
                    "category_id": to_cat_id(int(labels[i].item())),
                    "bbox": [float(x1), float(y1), float(w_box), float(h_box)],
                    "score": float(scores[i].item())
                }
                if wrote: fjson.write(',')
                json.dump(det, fjson, separators=(',', ':')); wrote = True
            fjson.write(']')
        cocoDt = dataset.coco.loadRes(tmp_path)
        cocoEval = COCOeval(dataset.coco, cocoDt, 'bbox')
        cocoEval.params.imgIds = [int(img_id)]
        cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
        ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
        try: os.remove(tmp_path)
        except: pass

    # --- build overlay canvas ---
    base_rgb = cv2.resize(img_raw, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    if EAD:
        canvas = np.zeros((Ht * 3, Wt * 3, 3), dtype=base_rgb.dtype)
        canvas[Ht:2*Ht, Wt:2*Wt] = base_rgb
    else:
        canvas = base_rgb.copy()

    # PRED full-res prob map on (Hh,Ww) -> visualize at canvas scale
    with torch.no_grad():
        pred_prob_full = torch.sigmoid(logits_BSC[..., heat_class_idx]).reshape(Hh, Ww).clamp(0,1)
    pred_uint8 = (pred_prob_full.cpu().numpy() * 255).astype(np.uint8)
    pred_color = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)

    target_hw = (Ht*3, Wt*3) if EAD else (Ht, Wt)
    pred_for_canvas = cv2.resize(pred_uint8, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    pred_color_canvas = cv2.applyColorMap(pred_for_canvas, cv2.COLORMAP_JET)

    # GT full-res (binary 0/1) on (Hh,Ww)
    gt_full = hm_BCHW[0, heat_class_idx].clamp(0,1)
    gt_uint8 = (gt_full.cpu().numpy() * 255).astype(np.uint8)
    gt_color = cv2.applyColorMap(gt_uint8, cv2.COLORMAP_JET)

    gt_for_canvas = cv2.resize(gt_uint8, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    gt_color_canvas = cv2.applyColorMap(gt_for_canvas, cv2.COLORMAP_JET)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    overlay_pred = cv2.addWeighted(canvas_bgr, 1.0, pred_color_canvas, overlay_alpha, 0.0)
    overlay_gt   = cv2.addWeighted(canvas_bgr, 1.0, gt_color_canvas,   overlay_alpha, 0.0)

    # draw boxes on the overlay (center-crop coords; shift if EAD)
    if pred_cc['boxes'].numel() > 0:
        shift = np.array([Wt, Ht, Wt, Ht], dtype=np.float32) if EAD else np.array([0,0,0,0], dtype=np.float32)
        for i in range(pred_cc['boxes'].size(0)):
            x1, y1, x2, y2 = (pred_cc['boxes'][i].cpu().numpy() + shift).astype(int).tolist()
            s = float(pred_cc['scores'][i].item())
            cv2.rectangle(overlay_pred, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(overlay_pred, f"{s:.2f}", (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # --- save artifacts ---
    os.makedirs('./viz_single', exist_ok=True)
    stamp = f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'
    out_overlay_pred = f'./viz_single/overlay_single_pred_idx{index:06d}_id{int(img_id):012d}_{stamp}.png'
    out_overlay_gt   = f'./viz_single/overlay_single_gt_idx{index:06d}_id{int(img_id):012d}_{stamp}.png'
    pred_full_png    = f'./viz_single/pred_full_idx{index:06d}_id{int(img_id):012d}_{stamp}.png'
    pred_full_color  = f'./viz_single/pred_full_color_idx{index:06d}_id{int(img_id):012d}_{stamp}.png'
    gt_full_png      = f'./viz_single/gt_full_idx{index:06d}_id{int(img_id):012d}_{stamp}.png'
    gt_full_color    = f'./viz_single/gt_full_color_idx{index:06d}_id{int(img_id):012d}_{stamp}.png'
    pred_npz         = f'./viz_single/pred_full_idx{index:06d}_id{int(img_id):012d}_{stamp}.npz'
    gt_npz           = f'./viz_single/gt_full_idx{index:06d}_id{int(img_id):012d}_{stamp}.npz'

    cv2.imwrite(out_overlay_pred, overlay_pred)
    cv2.imwrite(out_overlay_gt,   overlay_gt)
    cv2.imwrite(pred_full_png,    pred_uint8)
    cv2.imwrite(pred_full_color,  pred_color)
    cv2.imwrite(gt_full_png,      gt_uint8)
    cv2.imwrite(gt_full_color,    gt_color)

    # raw arrays for perfect reproducibility (pred logits already on GT grid)
    np.savez(pred_npz, logits_BSC=logits_BSC.detach().cpu().numpy(), Hh=Hh, Ww=Ww, class_idx=heat_class_idx)
    np.savez(gt_npz,   hm_BCHW=hm_BCHW.detach().cpu().numpy(),       Hh=Hh, Ww=Ww, class_idx=heat_class_idx)

    # pretty print
    print(f"[single-overlay] idx={index} id={int(img_id)} | "
          f"AP50_95={ap50_95:.4f} AP50={ap50:.4f} | "
          f"B: P={b_metrics['Precision']:.3f} R={b_metrics['Recall']:.3f} pixelAP={b_metrics['pixelAP']:.3f} "
          f"IoU@TP={b_metrics['meanIoU_TP']:.3f} | "
          f"H: mIoU={mres['mIoU']:.3f} mIoU_int={mres['mIoU_int']:.3f} pixelAP={mres['pixelAP']:.3f} | "
          f"pred_overlay → {out_overlay_pred}")

    return {
        'index': index,
        'image_id': int(img_id),
        'overlay_pred_png': out_overlay_pred,
        'overlay_gt_png': out_overlay_gt,
        'pred_full_png': pred_full_png,
        'pred_full_color_png': pred_full_color,
        'pred_npz': pred_npz,
        'gt_full_png': gt_full_png,
        'gt_full_color_png': gt_full_color,
        'gt_npz': gt_npz,
        'COCO': {'AP50_95': ap50_95, 'AP50': ap50},
        'B': b_metrics,
        'H': {'CE': ce, 'SE': se, **mres},
        'conf_thresh': conf_thr, 'nms_thresh': nms_thr, 'topk': (None if topk is None else int(topk))
    }

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

def build_eval_dataloader(cfg, evaluator, num_classes=2, mopts: MetricOptions = MetricOptions(), radius_down_ratio=0.5):
    dataset = evaluator.dataset
    dataset.transform = evaluator.transform
    collate = HeatmapCollector(num_classes, down_ratio=1, radius_down_ratio=radius_down_ratio, EAD_pad=cfg.get('EAD', True))
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
    gt_binarize: str = "relative"
    gt_rel_thresh: float = 0.25
    gt_percentile: float = 90.0
    beta: float = 1.0
    class_idx: int = 0
    ignore_center: bool = True

@torch.inference_mode()
def run_metrics(model, loader, device, mopts: MetricOptions, 
                miou_cfg: MiouCLI, skip_se_ce: bool = False, 
                skip_miou: bool = False) -> Dict[str, float]:

    model.eval().to(device)
    sums = {'ce_pred_gt':0.0, 'se_pred':0.0, 'mIoU':0.0, 'mIoU_int':0.0, 'pixelAP':0.0}
    n_batches = 0

    grid = ThresholdGrid(values=tuple(miou_cfg.thresholds))
    miou_opts = MiouOptions(
        ignore_center=miou_cfg.ignore_center,
        spatial_hw=None,   # set per-batch
        gt_binarize=miou_cfg.gt_binarize,
        gt_rel_thresh=miou_cfg.gt_rel_thresh,
        gt_percentile=miou_cfg.gt_percentile,
        beta=miou_cfg.beta,
        class_idx=miou_cfg.class_idx
    )

    nan_num = 0
    totoal_num = 0
    for img, target, mask, heatmaps in tqdm(loader, desc="Metric pass", leave=False):
        img = img.to(device, non_blocking=False)                  # (B,3,Ht,Wt)
        heatmaps = heatmaps.to(device, non_blocking=False)        # (B,C,3Ht,3Wt)
        B, C, H, W = heatmaps.shape
        gt_BSC = rearrange(heatmaps, 'b c h w -> b (h w) c')      # (B,S,C)
        gt_BSC[gt_BSC > 0] = 1.0                                  # binarize GT heatmaps for CE

        pred_logits_raw = model.heatmap(img)                      # (B,S_raw,C) logits
        pred_logits_BSC, (_, _), (Hc, Wc) = _unify_pred_to_gt_grid_simple(pred_logits_raw, heatmaps)
        
        # print(f'pred_logits_BSC.shape: {pred_logits_BSC.shape}, gt_BSC.shape: {gt_BSC.shape}')

        # CE(model, GT) and SE(model) with center-ignore
        if not skip_se_ce:
            ce_pred_gt = heatmap_2dce(
                pred_logits_BSC, gt_BSC,
                input_type=mopts.pred_input_type,
                reduction='none',
                ignore_center=mopts.ignore_center,
                spatial_hw=(H, W)
            )
            se_pred = heatmap_self_entropy_2d(
                pred_logits_BSC,
                input_type='pred_logits',
                reduction='none',
                class_idx=mopts.se_class_idx,
                ignore_center=mopts.ignore_center,
                spatial_hw=(H, W)
            )
            if torch.isnan(ce_pred_gt[:,0]).any():
                nan_num += torch.isnan(ce_pred_gt[:,0]).sum().item()
            totoal_num += B
            # print(f'ce_pred_gt: {ce_pred_gt[:,0]}, se_pred: {se_pred}, nan_num: {nan_num}, totoal_num: {totoal_num}')
            sums['ce_pred_gt'] += torch.nansum(ce_pred_gt[:,0]).item()
            sums['se_pred']    += torch.nanmean(se_pred).item()
        else:
            sums['ce_pred_gt'] += 0.0
            sums['se_pred']    += 0.0

        # mIoU suite (vectorized) for selected class
        if not skip_miou:
            miou_opts.spatial_hw = (H, W)
            miou_opts.class_idx = miou_cfg.class_idx
            mres = miou_over_thresholds_batch(pred_logits_BSC, heatmaps, grid, miou_opts)
            sums['mIoU']     += mres['mIoU']
            sums['mIoU_int'] += mres['mIoU_int']
            sums['pixelAP']      += mres['pixelAP']
        else:
            sums['mIoU_int'] += 0.0
            sums['pixelAP']      += 0.0

        n_batches += 1

    # free per-iter tensors early
    del img, heatmaps, gt_BSC, pred_logits_BSC
    if not skip_se_ce:
        del ce_pred_gt, se_pred
    if not skip_miou:
        del mres
    torch.cuda.empty_cache()
    
    result_dict = {k: (v / max(n_batches,1)) for k, v in sums.items()}
    result_dict['ce_pred_gt'] = sums['ce_pred_gt'] / max(totoal_num - nan_num,1)

    return result_dict

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

def get_evaluator_and_loader(cfg, device, args, radius_down_ratio):
    evaluator = get_evaluator(cfg, device, args.val_img_folder, args.val_ann_file, args.cocoeval_iouthr)
    mopts = MetricOptions(
        pred_input_type=args.pred_input_type,
        se_class_idx=args.se_class_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ignore_center=(not args.no_entropy_ignore_center)  # default ON
    )
    loader = build_eval_dataloader(cfg, evaluator, num_classes=2, mopts=mopts, radius_down_ratio=radius_down_ratio)
    return evaluator, loader, mopts

@torch.inference_mode()
def run_sanity_ce_se(loader,
                     device,
                     pred_input_type: str = 'pred_logits',
                     ignore_center: bool = True,
                     spatial_hw_from_batch: bool = True,
                     cls_idx: int = 0,
                     sanity_limit: Optional[int] = None,
                     miou_cfg: Optional[MiouCLI] = None,
                     recall_iou_thr: float = 0.5):
    """
    Sanity check using only GT heatmaps/boxes from the loader:
      Heatmap metrics (H-mode, for class cls_idx):
        - CE/SE (uniform & GT-as-pred)
        - mIoU / mIoU_int / pixelAP (uniform & GT-as-pred)
      Box metrics (B-mode):
        - Recall / Precision / F1 (uniform=empty preds, GT-as-pred=use GT boxes)

    No model is loaded or used.
    """
    # accumulators
    sums = dict(
        # CE/SE
        ce_uni_sum=0.0, ce_uni_cnt=0, ce_uni_nan=0,
        se_uni_sum=0.0, se_uni_cnt=0, se_uni_nan=0,
        ce_gt_sum=0.0,  ce_gt_cnt=0,  ce_gt_nan=0,
        se_gt_sum=0.0,  se_gt_cnt=0,  se_gt_nan=0,
        # H-mode (mIoU suite)
        H_mIoU_uni=0.0, H_mIoU_int_uni=0.0, H_pixelAP_uni=0.0,
        H_mIoU_gt=0.0,  H_mIoU_int_gt=0.0,  H_pixelAP_gt=0.0,
        H_batches=0,
        # B-mode (per-image avg)
        B_rec_uni=0.0, B_prec_uni=0.0, B_f1_uni=0.0,
        B_rec_gt=0.0,  B_prec_gt=0.0,  B_f1_gt=0.0,
        B_images=0,
        n_images=0
    )

    # Prepare mIoU/pixelAP config if requested
    grid = ThresholdGrid(values=tuple(miou_cfg.thresholds)) if miou_cfg else None
    miou_opts = None
    if miou_cfg:
        miou_opts = MiouOptions(
            ignore_center=miou_cfg.ignore_center,
            spatial_hw=None,  # not used inside miou_over_thresholds_batch
            gt_binarize=miou_cfg.gt_binarize,
            gt_rel_thresh=miou_cfg.gt_rel_thresh,
            gt_percentile=miou_cfg.gt_percentile,
            beta=miou_cfg.beta,
            class_idx=miou_cfg.class_idx
        )

    BIG = 12.0  # logits magnitude for GT-as-pred (sigmoid(±BIG) ~ 0/1)

    for img, targets, mask, heatmaps in tqdm(loader, desc="Sanity CE/SE + mIoU/pixelAP + Recall", leave=False):
        heatmaps = heatmaps.to(device, non_blocking=False)  # (B,C,H,W) with H=3*Ht, W=3*Wt
        B, C, H, W = heatmaps.shape
        S = H * W
        spatial_hw = (H, W) if spatial_hw_from_batch else None

        # ---- GT as (B,S,C), binarized for CE/SE ----
        gt_BSC = rearrange(heatmaps, 'b c h w -> b (h w) c').contiguous()
        gt_BSC = gt_BSC.clone()
        gt_BSC[gt_BSC > 0] = 1.0

        # ---- Uniform predictions (space-uniform) ----
        if pred_input_type == 'pred_logits':
            pred_uni_BSC = torch.zeros_like(gt_BSC, device=device)  # logits -> softmax uniform
        elif pred_input_type == 'pred_probs':
            pred_uni_BSC = torch.ones_like(gt_BSC, device=device)   # scores -> normalized uniform
        else:
            raise ValueError("pred_input_type must be 'pred_logits' or 'pred_probs'")

        # CE/SE (uniform)
        ce_uni_BC = heatmap_2dce(
            pred_uni_BSC, gt_BSC,
            input_type=pred_input_type,
            reduction='none',
            ignore_center=ignore_center,
            spatial_hw=spatial_hw
        )  # (B,C)
        se_uni_BC = heatmap_self_entropy_2d(
            pred_uni_BSC,
            input_type=pred_input_type,
            reduction='none',
            class_idx=cls_idx,
            ignore_center=ignore_center,
            spatial_hw=spatial_hw
        )  # (B,1) or (B,C)

        ce_uni_c = ce_uni_BC[:, cls_idx]
        se_uni_c = se_uni_BC.squeeze(-1) if se_uni_BC.dim() == 2 and se_uni_BC.size(1) == 1 else se_uni_BC[:, 0]
        ce_uni_fin = torch.isfinite(ce_uni_c)
        se_uni_fin = torch.isfinite(se_uni_c)

        sums['ce_uni_sum'] += float(torch.nansum(ce_uni_c).item())
        sums['ce_uni_cnt'] += int(ce_uni_fin.sum().item())
        sums['ce_uni_nan'] += int((~ce_uni_fin).sum().item())
        sums['se_uni_sum'] += float(torch.nansum(se_uni_c).item())
        sums['se_uni_cnt'] += int(se_uni_fin.sum().item())
        sums['se_uni_nan'] += int((~se_uni_fin).sum().item())

        # CE/SE (GT-as-pred)
        ce_gt_BC = heatmap_2dce(
            gt_BSC, gt_BSC,
            input_type='gt',  # CE(G,G)=H(G)
            reduction='none',
            ignore_center=ignore_center,
            spatial_hw=spatial_hw
        )
        se_gt_BC = heatmap_self_entropy_2d(
            gt_BSC, input_type='gt',
            reduction='none',
            class_idx=cls_idx,
            ignore_center=ignore_center,
            spatial_hw=spatial_hw
        )
        ce_gt_c = ce_gt_BC[:, cls_idx]
        se_gt_c = se_gt_BC.squeeze(-1) if se_gt_BC.dim() == 2 and se_gt_BC.size(1) == 1 else se_gt_BC[:, 0]
        ce_gt_fin = torch.isfinite(ce_gt_c)
        se_gt_fin = torch.isfinite(se_gt_c)

        sums['ce_gt_sum']  += float(torch.nansum(ce_gt_c).item())
        sums['ce_gt_cnt']  += int(ce_gt_fin.sum().item())
        sums['ce_gt_nan']  += int((~ce_gt_fin).sum().item())
        sums['se_gt_sum']  += float(torch.nansum(se_gt_c).item())
        sums['se_gt_cnt']  += int(se_gt_fin.sum().item())
        sums['se_gt_nan']  += int((~se_gt_fin).sum().item())

        # ---- H-mode: mIoU / mIoU_int / pixelAP on uniform & GT-as-pred ----
        if miou_cfg is not None:
            # logits for uniform (zeros -> sigmoid 0.5 everywhere)
            pred_logits_uni_BSC = torch.zeros((B, S, C), dtype=heatmaps.dtype, device=device)

            # logits for GT-as-pred: +BIG on GT pixels, -BIG elsewhere (per class)
            logits_gt_BCHW = torch.where(heatmaps > 0,
                                         torch.tensor(BIG, dtype=heatmaps.dtype, device=device),
                                         torch.tensor(-BIG, dtype=heatmaps.dtype, device=device))
            pred_logits_gt_BSC = rearrange(logits_gt_BCHW, 'b c h w -> b (h w) c')

            mres_uni = miou_over_thresholds_batch(pred_logits_uni_BSC, heatmaps, grid, miou_opts)
            mres_gt  = miou_over_thresholds_batch(pred_logits_gt_BSC,  heatmaps, grid, miou_opts)

            sums['H_mIoU_uni']      += mres_uni['mIoU']
            sums['H_mIoU_int_uni']  += mres_uni['mIoU_int']
            sums['H_pixelAP_uni']   += mres_uni['pixelAP']
            sums['H_mIoU_gt']       += mres_gt['mIoU']
            sums['H_mIoU_int_gt']   += mres_gt['mIoU_int']
            sums['H_pixelAP_gt']    += mres_gt['pixelAP']
            sums['H_batches']       += 1

        # ---- B-mode: Recall / Precision / F1 (per image) ----
        # uniform: predict nothing; gt-as-pred: predict GT boxes as detections
        for b in range(B):
            G = targets[b]['boxes'].to(torch.float32)  # (Ni, 4) in transformed coords
            L = targets[b]['labels'].to(torch.int64) if 'labels' in targets[b] else torch.zeros((G.size(0),), dtype=torch.int64)

            # empty preds
            pred_empty = {
                'boxes': torch.empty((0,4), dtype=torch.float32),
                'scores': torch.empty((0,), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.int64),
            }
            m_uni = _bmode_single_image_metrics(pred_empty, G, iou_thr=recall_iou_thr)

            # GT-as-pred
            pred_gt = {
                'boxes': G.clone(),
                'scores': torch.ones((G.size(0),), dtype=torch.float32),
                'labels': L.clone(),
            }
            m_gt = _bmode_single_image_metrics(pred_gt, G, iou_thr=recall_iou_thr)

            sums['B_rec_uni']  += m_uni['Recall'];    sums['B_prec_uni'] += m_uni['Precision']; sums['B_f1_uni'] += m_uni['F1']
            sums['B_rec_gt']   += m_gt['Recall'];     sums['B_prec_gt']  += m_gt['Precision'];  sums['B_f1_gt']  += m_gt['F1']
            sums['B_images']   += 1

        sums['n_images'] += B
        if sanity_limit is not None and sums['n_images'] >= sanity_limit:
            break

    def avg(x, n): return x / max(n, 1)

    out = {
        'class_idx': cls_idx,
        'ignore_center': bool(ignore_center),
        'pred_input_type': pred_input_type,
        'n_images_seen': sums['n_images'],

        # CE/SE
        'ce_uniform_mean':  avg(sums['ce_uni_sum'], sums['ce_uni_cnt']),
        'se_uniform_mean':  avg(sums['se_uni_sum'], sums['se_uni_cnt']),
        'ce_gt_mean':       avg(sums['ce_gt_sum'],  sums['ce_gt_cnt']),
        'se_gt_mean':       avg(sums['se_gt_sum'],  sums['se_gt_cnt']),
        'nan_counts': {
            'ce_uniform': sums['ce_uni_nan'],
            'se_uniform': sums['se_uni_nan'],
            'ce_gt':      sums['ce_gt_nan'],
            'se_gt':      sums['se_gt_nan'],
        },
    }

    if miou_cfg is not None:
        out.update({
            'heatmap_metrics': {
                'uniform': {
                    'mIoU':     avg(sums['H_mIoU_uni'],     sums['H_batches']),
                    'mIoU_int': avg(sums['H_mIoU_int_uni'], sums['H_batches']),
                    'pixelAP':  avg(sums['H_pixelAP_uni'],  sums['H_batches']),
                },
                'gt_as_pred': {
                    'mIoU':     avg(sums['H_mIoU_gt'],      sums['H_batches']),
                    'mIoU_int': avg(sums['H_mIoU_int_gt'],  sums['H_batches']),
                    'pixelAP':  avg(sums['H_pixelAP_gt'],   sums['H_batches']),
                },
                'thresholds': list(miou_cfg.thresholds),
            }
        })

    out.update({
        'box_metrics_per_image_mean': {
            'uniform_preds': {
                'Recall':    avg(sums['B_rec_uni'],  sums['B_images']),
                'Precision': avg(sums['B_prec_uni'], sums['B_images']),
                'F1':        avg(sums['B_f1_uni'],   sums['B_images']),
            },
            'gt_as_pred': {
                'Recall':    avg(sums['B_rec_gt'],   sums['B_images']),
                'Precision': avg(sums['B_prec_gt'],  sums['B_images']),
                'F1':        avg(sums['B_f1_gt'],    sums['B_images']),
            },
            'iou_thr': recall_iou_thr,
        }
    })

    return out


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
    # ---------- SANITY CE/SE ONLY ----------
    if args.sanity_ce_se:
        cfg = yoloh_config[config_names[0]]
        if args.img_size is not None:
            cfg['input_size'] = (args.img_size, args.img_size)
        else:
            cfg['input_size'] = (320, 320)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluator, loader, mopts = get_evaluator_and_loader(cfg, device, args, radius_down_ratio)

        res = run_sanity_ce_se(
            loader=loader,
            device=device,
            pred_input_type=args.pred_input_type,
            ignore_center=(not args.no_entropy_ignore_center),
            cls_idx=args.miou_class_idx,
            sanity_limit=args.sanity_limit,
            miou_cfg=miou_cfg,
            recall_iou_thr=args.single_iou_thr
        )
        print("\n=== Sanity Check (CE/SE + mIoU/mIoU_int/pixelAP + Box Recall) ===")
        print(json.dumps(res, indent=2))
        print("\n[Notes]")
        print(" - CE(uniform) ≈ 1.0 ; SE(uniform) ≈ 1.0 (log-normalized over masked spatial support)")
        print(" - CE(GT) == SE(GT) = normalized entropy of GT; can be NaN if the mask removes all GT")
        print(" - For mIoU/pixelAP: uniform uses P=0.5; GT-as-pred uses hard logits (±BIG)")
        print(" - Box sanity: uniform makes no detections → Recall≈0 (unless image has no GT); GT-as-pred → Recall=1, F1=1")
        return

    for conf_thresh in conf_threshs:
        for nms_thresh in nms_threshs:
            for topk in topks:
                for config_name, weight_file in zip(config_names, weight_files):

                    cfg = yoloh_config[config_name]
                    if args.img_size is not None:
                        cfg['input_size'] = (args.img_size, args.img_size)
                    else:
                        cfg['input_size'] = (320, 320)

                    model = build_model_noargs(
                        cfg=cfg, device=device, num_classes=2, trainable=False,
                        coco_pretrained=weight_file, fig_size=cfg['input_size'],
                        conf_thresh=conf_thresh, nms_thresh=nms_thresh, topk=topk,
                    )
                    model.eval().to(device)

                    evaluator, loader, mopts = get_evaluator_and_loader(cfg, device, args, radius_down_ratio)
                    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Evaluating {config_name} | conf={conf_thresh} nms={nms_thresh} topk={topk}")

                    # ---------- SINGLE-SAMPLE PATH ----------
                    if args.single_index is not None:
                        res = visualize_single_index_overlay(
                            cfg=cfg, model=model, device=device, evaluator=evaluator,
                            index=args.single_index,
                            conf_thr=conf_thresh, nms_thr=nms_thresh, topk=topk,
                            iou_thr=args.single_iou_thr,
                            miou_cfg=miou_cfg,
                            radius_down_ratio=radius_down_ratio,
                            heat_class_idx=args.miou_class_idx,
                            overlay_alpha=0.8,
                            verbose=bool(args.verbose)
                        )
                        # save JSON summary
                        os.makedirs('./viz_single', exist_ok=True)
                        safe_date = f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'
                        out_json = f'./viz_single/single_overlay_{config_name.replace("/", "_")}_idx{args.single_index}_conf{conf_thresh}_nms{nms_thresh}_topk{topk}_{safe_date}.json'
                        with open(out_json, 'w') as f:
                            json.dump(res, f, indent=2)
                        # tidy and move to next
                        del model, evaluator, loader
                        torch.cuda.empty_cache()
                        continue

                    # ---------- ORIGINAL DATASET-WIDE PATH ----------
                    if not args.skip_ap_mse:
                        start = datetime.datetime.now()
                        eval_result_dict = evaluator.evaluate(model)
                        print(f"COCO eval time: {datetime.datetime.now() - start}")
                    else:
                        print("Skipping COCO eval")
                        eval_result_dict = {}

                    start = datetime.datetime.now()
                    mdict = run_metrics(model, loader, device, mopts, miou_cfg,
                                        skip_miou=args.skip_miou, skip_se_ce=args.skip_se_ce)
                    print(f"Metric pass time: {datetime.datetime.now() - start}")

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
                    csv_out = './result/normalized_density_metrics.csv'
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
    ap.add_argument('--verbose', action='store_true',
                    help='Print sizes (H0/W0,Ht/Wt,Hc/Wc,Hh/Ww), B-mode TP/FP/FN, and save GT/pred heatmaps.')
    
    # ----- sanity mode: CE/SE only, no model -----
    ap.add_argument('--sanity_ce_se', action='store_true',
                    help='Run CE/SE sanity check with uniform prediction and GT-as-pred; skips model/AP/mIoU.')
    ap.add_argument('--sanity_limit', type=int, default=None,
                    help='Optional: limit the number of images evaluated in sanity mode.')

    args = ap.parse_args()
    main(args)