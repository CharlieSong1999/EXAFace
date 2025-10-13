import os, math, json
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Iterable, List
import datetime

import cv2
import numpy as np
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
from deepface import DeepFace

# --- your project imports (same paths you used in eval_runner.py) ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'yoloh')))
from data.transforms import ValTransforms
from config.yoloh_config_expand import yoloh_config
from models.yoloh import build_model_noargs

# =============== Helpers ===================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def overlay_heatmap_full(DA_bgr: np.ndarray, probs_HW: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay probs (H,W) [0..1] on the whole DA canvas."""
    probs_clamped = np.clip(probs_HW, 0.0, 1.0)
    return overlay_heatmap(DA_bgr.copy(), probs_clamped, alpha=alpha)  # you already have overlay_heatmap()

def rebuild_probs_from_logits(heat_logits, cls_idx: int, W_DA: int, H_DA: int):
    """heat_logits: (1,S,C) numpy or torch; returns resized probs (H_DA, W_DA)."""
    if isinstance(heat_logits, np.ndarray):
        hl = torch.from_numpy(heat_logits)
    else:
        hl = heat_logits
    hl = hl.float()
    B, S, C = hl.shape
    Hh = int(round(math.sqrt(S))); Wh = S // Hh
    assert Hh * Wh == S, "Non-square head? Provide Hh,Wh explicitly."
    probs = torch.sigmoid(hl[0, :, cls_idx]).reshape(Hh, Wh).detach().cpu().numpy()
    probs_up = cv2.resize(probs, (W_DA, H_DA), interpolation=cv2.INTER_CUBIC)
    return probs_up

# --- compatibility: two-click rectangle selection (works on older Gradio) ---
def _draw_temp_rect(rgb_img: np.ndarray, p1: tuple[int,int] | None, p2: tuple[int,int] | None):
    """Return an overlay image with a green rectangle if both points exist."""
    vis = rgb_img.copy()
    if p1 is not None:
        cv2.circle(vis, p1, 4, (0,255,0), -1)
    if p1 is not None and p2 is not None:
        x1, y1 = p1; x2, y2 = p2
        x, y = min(x1,x2), min(y1,y2)
        w, h = abs(x2-x1), abs(y2-y1)
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
    return vis

def select_two_click(evt: gr.SelectData,
                     image: np.ndarray,
                     p1_state: tuple[int,int] | None):
    """
    On first click: set p1 and draw a dot.
    On second click: finalize p2, draw rect, emit JSON and reset p1=None.
    Returns: preview_image, rect_json, new_p1_state
    """
    if image is None:
        return None, json.dumps({}), None

    idx = evt.index  # (x, y)
    if not isinstance(idx, (list, tuple)) or len(idx) != 2:
        return _draw_temp_rect(image, None, None), json.dumps({}), None

    x, y = int(idx[0]), int(idx[1])

    # First click -> set p1
    if p1_state is None:
        vis = _draw_temp_rect(image, (x,y), None)
        return vis, json.dumps({}), (x,y)

    # Second click -> finalize rectangle
    x1, y1 = p1_state
    x0, y0 = min(x1,x), min(y1,y)
    w, h = abs(x - x1), abs(y - y1)
    # Clamp tiny boxes
    w = max(8, int(w)); h = max(8, int(h))
    rect = {"x": int(x0), "y": int(y0), "w": int(w), "h": int(h)}
    vis = _draw_temp_rect(image, (x0,y0), (x0+w, y0+h))
    return vis, json.dumps(rect), None  # reset p1

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: (4,) [x1,y1,x2,y2]
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
    inter = w * h
    ua = max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / ua

def nms_per_class(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thresh: float) -> np.ndarray:
    """
    Run greedy NMS independently for each class label.
    Returns indices (into original arrays) to keep.
    """
    keep_all = []
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    classes = np.unique(labels)
    for c in classes:
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue
        b = boxes[idx_c]
        s = scores[idx_c]
        order = s.argsort()[::-1]
        kept_local = []
        while order.size > 0:
            i = order[0]
            kept_local.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = np.array([iou_xyxy(b[i], b[j]) for j in rest], dtype=np.float32)
            order = rest[ious <= iou_thresh]
        # map local indices back to global
        keep_all.extend(idx_c[np.array(kept_local, dtype=np.int64)])
    return np.array(keep_all, dtype=np.int64)

def apply_conf_nms_class_aware(
    boxes_raw: np.ndarray, scores_raw: np.ndarray, labels_raw: np.ndarray,
    conf_thresh: float, nms_thresh: float, show_bodies: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter by confidence, optionally drop bodies (label==1), then run class-aware NMS.
    Returns filtered (boxes, scores, labels).
    """
    if boxes_raw is None or boxes_raw.size == 0:
        return (np.zeros((0,4), np.float32),
                np.zeros((0,),  np.float32),
                np.zeros((0,),  np.int64))
    # confidence mask
    m_conf = scores_raw >= float(conf_thresh)
    boxes = boxes_raw[m_conf]
    scores = scores_raw[m_conf]
    labels = labels_raw[m_conf]
    # show bodies?
    if not show_bodies:
        m_face = labels == 0
        boxes, scores, labels = boxes[m_face], scores[m_face], labels[m_face]
    if boxes.size == 0:
        return (boxes, scores, labels)
    # class-aware NMS
    keep = nms_per_class(boxes, scores, labels, float(nms_thresh))
    return boxes[keep], scores[keep], labels[keep]

def render_da_and_gp_classwise(
    DA_bgr: np.ndarray,
    gt_in_DA: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    face_color=(0,255,0), body_color=(255,128,0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw faces (label=0) and bodies (label=1) with different colors.
    Returns (DA+Pred, DA+GT&Pred), both in BGR.
    """
    img_da_pred = DA_bgr.copy()
    if pred_boxes.size > 0:
        m_face = (pred_labels == 0)
        m_body = (pred_labels == 1)
        if m_face.any():
            img_da_pred = overlay_boxes(img_da_pred, pred_boxes[m_face], color=face_color, thickness=2, scores=pred_scores[m_face])
        if m_body.any():
            img_da_pred = overlay_boxes(img_da_pred, pred_boxes[m_body], color=body_color, thickness=2, scores=pred_scores[m_body])

    img_da_gp = overlay_boxes(DA_bgr.copy(), gt_in_DA, color=(255,0,0), thickness=2)
    if pred_boxes.size > 0:
        m_face = (pred_labels == 0)
        m_body = (pred_labels == 1)
        if m_face.any():
            img_da_gp = overlay_boxes(img_da_gp, pred_boxes[m_face], color=face_color, thickness=2, scores=pred_scores[m_face])
        if m_body.any():
            img_da_gp = overlay_boxes(img_da_gp, pred_boxes[m_body], color=body_color, thickness=2, scores=pred_scores[m_body])
    return img_da_pred, img_da_gp


# ============================================================
# DA builder (3h x 3w) with zero padding and ring darkening
# ============================================================
def build_DA_with_padding(img_bgr: np.ndarray, px:int, py:int, h:int, w:int, dark_ratio:float=0.5):
    H, W = img_bgr.shape[:2]
    need_left  = max(0, w - px)
    need_top   = max(0, h - py)
    need_right = max(0, (px + 2*w) - W)
    need_bot   = max(0, (py + 2*h) - H)

    if any([need_left, need_top, need_right, need_bot]):
        img_exp = cv2.copyMakeBorder(
            img_bgr, need_top, need_bot, need_left, need_right,
            borderType=cv2.BORDER_CONSTANT, value=(0,0,0),
        )
        px_e, py_e = px + need_left, py + need_top
    else:
        img_exp = img_bgr
        px_e, py_e = px, py

    x0, y0 = px_e - w, py_e - h
    x1, y1 = px_e + 2*w, py_e + 2*h
    DA = img_exp[max(y0,0):y1, max(x0,0):x1].copy()
    DA = cv2.resize(DA, (3*w, 3*h), interpolation=cv2.INTER_NEAREST)

    ring_mask = np.ones((3*h, 3*w, 1), dtype=np.float32)
    ring_mask[h:2*h, w:2*w, :] = 0.0
    DA_f = DA.astype(np.float32)
    DA_f = DA_f * (1 - dark_ratio * ring_mask)
    DA_bgr = np.clip(DA_f, 0, 255).astype(np.uint8)

    Is_bgr = img_exp[py_e:py_e+h, px_e:px_e+w].copy()
    return DA_bgr, Is_bgr, (x0, y0, img_exp.shape[0], img_exp.shape[1])

# ============================================================
# Heatmap GT from boxes (your same recipe as eval_runner)
# ============================================================
def gaussian2D(shape, sigma=1.0):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(hm: np.ndarray, center_xy: Tuple[int,int], radius:int, k:float=1.0):
    diameter = 2*radius + 1
    g = gaussian2D((diameter, diameter), sigma=diameter/6)
    x, y = int(center_xy[0]), int(center_xy[1])
    H, W = hm.shape[:2]
    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)
    masked = hm[y - top:y + bottom, x - left:x + right]
    mg = g[radius - top:radius + bottom, radius - left:radius + right]
    if masked.size and mg.size:
        np.maximum(masked, mg * k, out=masked)

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1, b1 = 1, (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1**2 - 4*a1*c1)) / 2
    a2, b2 = 4, 2*(height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2**2 - 4*a2*c2)) / 2
    a3, b3 = 4*min_overlap, -2*min_overlap*(height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3**2 - 4*a3*c3)) / (2*a3)
    return int(min(r1, r2, r3))

def boxes_to_heatmap(Is_hw: Tuple[int,int], boxes_abs: List[Tuple[int,int,int,int]],
                     out_hw: Tuple[int,int], num_classes:int, cls_idx:int=0, radius_scale:float=1.0):
    H, W = Is_hw
    h, w = out_hw
    hm = np.zeros((num_classes, h, w), dtype=np.float32)
    sx, sy = w / W, h / H
    rad = max(0, int(gaussian_radius((h,w), 0.7) * radius_scale))
    for (x1,y1,x2,y2) in boxes_abs:
        cx = (x1 + x2) / 2 * sx
        cy = (y1 + y2) / 2 * sy
        draw_umich_gaussian(hm[cls_idx], (int(cx), int(cy)), rad)
    hm_t = torch.from_numpy(hm).permute(1,2,0).reshape(1, h*w, num_classes)
    return hm_t

# ============================================================
# CE / SE (normalized with optional center-ignore)
# ============================================================
def _build_center_mask(B, S, spatial_hw, device, dtype):
    H, W = spatial_hw
    assert (H*W)==S
    m = torch.ones((H,W), dtype=dtype, device=device)
    h1,h2 = H//3, 2*(H//3); w1,w2 = W//3, 2*(W//3)
    m[h1:h2, w1:w2] = 0
    return m.reshape(1,S).expand(B,S)

def _gt_norm_over_space(gt_BSC, eps=1e-12, valid_mask_BS=None):
    if valid_mask_BS is not None:
        gt_BSC = gt_BSC * valid_mask_BS.unsqueeze(-1)
    Z = gt_BSC.clamp_min(0).sum(dim=1, keepdim=True)
    present = (Z.squeeze(1) > eps)
    G = (gt_BSC / Z.clamp_min(eps)).clamp_min(eps)
    return G, present

def _pred_spatial_probs_from_logits(pred_BSC, eps=1e-12, valid_mask_BS=None):
    B,S,C = pred_BSC.shape
    pred_BCS = pred_BSC.permute(0,2,1)
    if valid_mask_BS is not None:
        m = valid_mask_BS.unsqueeze(1)
        pred_BCS = pred_BCS.masked_fill(m==0, float('-inf'))
    P = F.softmax(pred_BCS, dim=-1)
    if valid_mask_BS is not None:
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        Z = (P * m).sum(dim=-1, keepdim=True).clamp_min(eps)
        P = P / Z
    return P.permute(0,2,1).clamp_min(eps)

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

def heatmap_2dce(pred_BSC, gt_BSC, input_type='pred_logits', reduction='mean',
                 ignore_center=False, spatial_hw: Optional[Tuple[int,int]]=None):
    assert pred_BSC.shape == gt_BSC.shape
    B,S,C = pred_BSC.shape
    valid_mask_BS = None
    if ignore_center:
        assert spatial_hw is not None
        valid_mask_BS = _build_center_mask(B,S,spatial_hw, pred_BSC.device, pred_BSC.dtype)
    G, present = _gt_norm_over_space(gt_BSC, valid_mask_BS=valid_mask_BS)
    if input_type == 'pred_logits':
        D = _pred_spatial_probs_from_logits(pred_BSC, valid_mask_BS=valid_mask_BS)
    elif input_type == 'pred_probs':
        D = _pred_spatial_probs_from_scores(pred_BSC, valid_mask_BS=valid_mask_BS)
    elif input_type == 'gt':
        D, _ = _gt_norm_over_space(pred_BSC, valid_mask_BS=valid_mask_BS)
    else:
        raise ValueError
    logX = _log_cardinality_from_mask(S, valid_mask_BS, pred_BSC.dtype, pred_BSC.device)
    CE_BC = -(G * D.log()).sum(dim=1) / logX
    CE_BC = CE_BC.masked_fill(~present, float('nan'))
    return torch.nanmean(CE_BC) if reduction=='mean' else CE_BC

def heatmap_self_entropy_2d(X_BSC, input_type='pred_logits', reduction='mean',
                            class_idx: Optional[int]=None,
                            ignore_center=False, spatial_hw: Optional[Tuple[int,int]]=None):
    assert X_BSC.dim()==3
    if class_idx is not None:
        X_BSC = X_BSC[..., [class_idx]]
    B,S,C = X_BSC.shape
    valid_mask_BS = None
    if ignore_center:
        assert spatial_hw is not None
        valid_mask_BS = _build_center_mask(B,S,spatial_hw, X_BSC.device, X_BSC.dtype)
    if input_type == 'gt':
        P, present = _gt_norm_over_space(X_BSC, valid_mask_BS=valid_mask_BS)
    elif input_type == 'pred_logits':
        P = _pred_spatial_probs_from_logits(X_BSC, valid_mask_BS=valid_mask_BS)
        present = torch.ones((B,C), dtype=torch.bool, device=X_BSC.device)
    elif input_type == 'pred_probs':
        P = _pred_spatial_probs_from_scores(X_BSC, valid_mask_BS=valid_mask_BS)
        present = torch.ones((B,C), dtype=torch.bool, device=X_BSC.device)
    else:
        raise ValueError
    logX = _log_cardinality_from_mask(S, valid_mask_BS, X_BSC.dtype, X_BSC.device)
    H_BC = -(P * P.log()).sum(dim=1) / logX
    H_BC = H_BC.masked_fill(~present, float('nan'))
    return torch.nanmean(H_BC) if reduction=='mean' else H_BC

# ============================================================
# mIoU / IoU_int / F1 (single-image curves)
# ============================================================
@dataclass
class MiouOptions:
    ignore_center: bool = False
    spatial_hw: Optional[Tuple[int,int]] = None
    gt_binarize: str = "relative"
    gt_rel_thresh: float = 0.25
    gt_percentile: float = 90.0
    beta: float = 1.0
    class_idx: int = 0

def _build_center_mask_hw(H, W, device=None, dtype=None):
    m = torch.ones((H,W), dtype=dtype or torch.float32, device=device)
    h1,h2 = H//3, 2*(H//3); w1,w2 = W//3, 2*(W//3)
    m[h1:h2, w1:w2] = 0
    return m

def _binarize_gt_spatial(gt_BHW: torch.Tensor, opts: MiouOptions) -> torch.Tensor:
    if opts.gt_binarize == "gt>0":
        return (gt_BHW > 0).to(gt_BHW.dtype)
    elif opts.gt_binarize == "relative":
        B,H,W = gt_BHW.shape
        gmax = gt_BHW.view(B,-1).amax(dim=1).clamp_min(1e-8)
        thr = (gmax * opts.gt_rel_thresh).view(B,1,1)
        return (gt_BHW >= thr).to(gt_BHW.dtype)
    elif opts.gt_binarize == "percentile":
        B,H,W = gt_BHW.shape
        flat = gt_BHW.view(B,-1)
        kth = torch.tensor([np.percentile(flat[b].cpu().numpy(), opts.gt_percentile)
                            for b in range(B)], device=gt_BHW.device, dtype=gt_BHW.dtype).view(B,1,1)
        return (gt_BHW >= kth).to(gt_BHW.dtype)
    else:
        raise ValueError

def miou_over_thresholds_single(
    pred_logits_BSC: torch.Tensor,  # (1,S,C)
    gt_heat_BCHW: torch.Tensor,     # (1,C,H,W)
    thresholds: Iterable[float],
    opts: MiouOptions
) -> Dict[str, Any]:
    assert pred_logits_BSC.shape[0]==1 and gt_heat_BCHW.shape[0]==1
    _, S, C = pred_logits_BSC.shape
    _, Cg, H, W = gt_heat_BCHW.shape
    assert C==Cg and S==H*W
    cls = opts.class_idx
    logits_HW = pred_logits_BSC[0,:,cls].reshape(H,W)
    P = torch.sigmoid(logits_HW).unsqueeze(0)  # (1,H,W)
    Y = _binarize_gt_spatial(gt_heat_BCHW[:, cls, ...], opts)    # (1,H,W)
    if opts.ignore_center:
        m = _build_center_mask_hw(H,W, device=P.device, dtype=P.dtype)
        P = P * m; Y = Y * m
    P = P.view(1,-1); Y = Y.view(1,-1)
    thr_t = torch.tensor(list(thresholds), dtype=P.dtype, device=P.device)
    pred_m = (P.unsqueeze(-1) >= thr_t.view(1,1,-1)).to(P.dtype)  # (1,S,T)
    gt_sum = Y.sum(dim=1, keepdim=True)      # (1,1)
    pred_sum = pred_m.sum(dim=1)             # (1,T)
    tp = (pred_m * Y.unsqueeze(-1)).sum(dim=1)
    fp = pred_sum - tp
    fn = gt_sum - tp
    gt_empty = (gt_sum == 0)
    if bool(gt_empty):
        pred_empty = (pred_sum == 0)
        iou_std = pred_empty.to(P.dtype)
        iou_int = pred_empty.to(P.dtype)
        prec = pred_empty.to(P.dtype)
        rec  = torch.ones_like(pred_sum, dtype=P.dtype)
    else:
        iou_std = tp / (tp + fp + fn).clamp_min(1e-12)
        iou_int = tp / (tp + fn).clamp_min(1e-12)
        prec = tp / (tp + fp).clamp_min(1e-12)
        rec  = tp / (tp + fn).clamp_min(1e-12)
    b2 = opts.beta * opts.beta
    f1 = (1+b2) * prec * rec / (b2*prec + rec).clamp_min(1e-12)
    return {
        "thresholds": [float(t) for t in thr_t],
        "iou_std": [float(v) for v in iou_std[0]],
        "iou_int": [float(v) for v in iou_int[0]],
        "f_beta":  [float(v) for v in f1[0]],
        "mIoU": float(iou_std.mean()),
        "mIoU_int": float(iou_int.mean()),
        "mF1": float(f1.mean()),
    }

# ============================================================
# mAP & MAE (centers)
# ============================================================
def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    N,M = a.shape[0], b.shape[0]
    if N==0 or M==0:
        return np.zeros((N,M), dtype=np.float32)
    xx1 = np.maximum(a[:,None,0], b[None,:,0])
    yy1 = np.maximum(a[:,None,1], b[None,:,1])
    xx2 = np.minimum(a[:,None,2], b[None,:,2])
    yy2 = np.minimum(a[:,None,3], b[None,:,3])
    inter = np.clip(xx2-xx1,0,None) * np.clip(yy2-yy1,0,None)
    area_a = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
    area_b = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    union = area_a[:,None] + area_b[None,:] - inter
    return np.where(union>0, inter/union, 0.0)

def compute_ap_at_iou(pred_boxes_xyxy: np.ndarray, pred_scores: np.ndarray,
                      gt_boxes_xyxy: np.ndarray, iou_thr: float=0.5) -> float:
    if pred_boxes_xyxy.size==0 and gt_boxes_xyxy.size==0:
        return 1.0
    if pred_boxes_xyxy.size==0:
        return 0.0
    if gt_boxes_xyxy.size==0:
        return 0.0
    order = np.argsort(-pred_scores)
    preds = pred_boxes_xyxy[order]
    scores = pred_scores[order]
    Np = preds.shape[0]
    matched_gt = np.zeros(gt_boxes_xyxy.shape[0], dtype=bool)
    tp = np.zeros(Np, dtype=np.float32)
    fp = np.zeros(Np, dtype=np.float32)
    ious = box_iou_xyxy(preds, gt_boxes_xyxy)
    for i in range(Np):
        j = np.argmax(ious[i])
        if ious[i, j] >= iou_thr and not matched_gt[j]:
            tp[i] = 1.0; matched_gt[j] = True
        else:
            fp[i] = 1.0
    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(1, gt_boxes_xyxy.shape[0])
    precisions = cum_tp / np.maximum(1, (cum_tp + cum_fp))
    ap = 0.0
    for t in np.linspace(0,1,11):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
        ap += p
    return ap / 11.0

def mean_abs_center_error(pred_boxes_xyxy: np.ndarray, gt_boxes_xyxy: np.ndarray, H: int, W: int) -> float:
    if gt_boxes_xyxy.size == 0:
        return 0.0
    pred_centers = (np.stack([(pred_boxes_xyxy[:,0]+pred_boxes_xyxy[:,2])/2,
                              (pred_boxes_xyxy[:,1]+pred_boxes_xyxy[:,3])/2], axis=1)
                    if pred_boxes_xyxy.size else np.zeros((0,2)))
    gt_centers = np.stack([(gt_boxes_xyxy[:,0]+gt_boxes_xyxy[:,2])/2,
                           (gt_boxes_xyxy[:,1]+gt_boxes_xyxy[:,3])/2], axis=1)
    if pred_centers.shape[0] == 0:
        half_diag = 0.5 * math.sqrt(H*H + W*W)
        return float(np.mean(np.ones(len(gt_centers)) * (half_diag / half_diag)))
    dmat = np.linalg.norm(gt_centers[:,None,:] - pred_centers[None,:,:], axis=2)
    mins = dmat.min(axis=1)
    half_diag = 0.5 * math.sqrt(H*H + W*W)
    return float(np.clip(mins / half_diag, 0.0, 1.0).mean())

# ============================================================
# Viz
# ============================================================
def overlay_boxes(image_bgr: np.ndarray, boxes_xyxy: np.ndarray, color=(0,255,0), thickness=2, scores: Optional[np.ndarray]=None):
    img = image_bgr.copy()
    for i, b in enumerate(boxes_xyxy):
        x1,y1,x2,y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
        if scores is not None:
            cv2.putText(img, f"{scores[i]:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

def overlay_heatmap(image_bgr: np.ndarray, heat_HW: np.ndarray, alpha=0.5):
    heat_uint8 = np.clip(heat_HW*255, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0, heat_color, alpha, 0)

def plot_curves(thr: List[float], iou: List[float], iou_int: List[float], f1: List[float]) -> np.ndarray:
    plt.figure(figsize=(4,3))
    plt.plot(thr, iou, label='IoU'); plt.plot(thr, iou_int, label='IoU_int'); plt.plot(thr, f1, label='F1')
    plt.xlabel('Prob threshold'); plt.ylabel('Score'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "curves_tmp.png"; plt.savefig(path, dpi=140); plt.close()
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# ============================================================
# RetinaFace GT
# ============================================================
def detect_faces_bgr_deepface(img_bgr: np.ndarray, backend: str = "opencv", thresh: float = 0.8) -> list[tuple[int,int,int,int]]:
    """
    DeepFace detector. Returns list of (x1, y1, x2, y2) absolute pixels.
    backends: 'opencv' (fast, CPU), 'retinaface', 'mediapipe', 'mtcnn', 'yolov8' (if installed)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = DeepFace.extract_faces(img_path=img_rgb, detector_backend=backend, enforce_detection=False)
    boxes = []
    for f in faces:
        # DeepFace returns either 'facial_area' or 'region'
        region = f.get("facial_area", None)
        if region is None:
            region = f.get("region", None)
        if isinstance(region, dict):
            x, y, w, h = int(region.get("x", 0)), int(region.get("y", 0)), int(region.get("w", 0)), int(region.get("h", 0))
            if w > 0 and h > 0:
                boxes.append((x, y, x + w, y + h))
    # No scores from most backends; we’ll treat scores as 1.0 later
    return boxes

# ============================================================
# Model & transform management
# ============================================================
@dataclass
class ModelCfg:
    config_name: str
    weight_file: str
    conf_thresh: float = 0.0
    nms_thresh: float = 1.0
    topk: int = 5000
    num_classes: int = 2
    fig_size: Tuple[int,int] = (320,320)

MODEL: Optional[nn.Module] = None
TRANSFORM: Optional[ValTransforms] = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(cfg: ModelCfg):
    global MODEL, TRANSFORM
    cfg_dict = yoloh_config[cfg.config_name]
    MODEL = build_model_noargs(
        cfg=cfg_dict, device=DEVICE, num_classes=cfg.num_classes, trainable=False,
        coco_pretrained=cfg.weight_file, fig_size=cfg.fig_size,
        conf_thresh=cfg.conf_thresh, nms_thresh=cfg.nms_thresh, topk=cfg.topk
    ).to(DEVICE).eval()
    # Build val_transform same as get_evaluator in your eval_runner
    TRANSFORM = ValTransforms(
        min_size=cfg.fig_size[0], max_size=cfg.fig_size[1],
        pixel_mean=cfg_dict['pixel_mean'], pixel_std=cfg_dict['pixel_std'],
        trans_config=cfg_dict.get('val_transform', None),
        format=cfg_dict['format']
    )
    return f"Loaded {cfg.config_name} with weights: {os.path.basename(cfg.weight_file)}"

def apply_transform(Is_bgr: np.ndarray) -> torch.Tensor:
    # Your ValTransforms returns (tensor, _) — same as evaluator.transform(img)[0]
    t = TRANSFORM(Is_bgr)[0]  # (C,h,w) torch
    return t.unsqueeze(0).to(DEVICE)

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)

def parse_model_outputs(out):
    """
    Supports:
      - dict with keys: 'bboxes' (Nx4, normalized [0,1]), 'scores' (N,), 'labels' (N,)
      - tuple/list: (bboxes, scores, labels)
    Returns: (bboxes_np, scores_np, labels_np); each is np.ndarray (may be empty with shape (0,4)/(0,))
    """
    bboxes = scores = labels = None
    if isinstance(out, dict):
        bboxes = _to_numpy(out.get('bboxes'))
        scores = _to_numpy(out.get('scores'))
        labels = _to_numpy(out.get('labels'))
    elif isinstance(out, (tuple, list)) and len(out) >= 2:
        bboxes = _to_numpy(out[0])
        scores = _to_numpy(out[1])
        labels = _to_numpy(out[2]) if len(out) > 2 else None
    else:
        # unexpected; try best-effort
        bboxes = _to_numpy(out)

    # default shapes
    if bboxes is None:
        bboxes = np.zeros((0,4), dtype=np.float32)
    if bboxes.ndim == 1 and bboxes.size == 4:
        bboxes = bboxes.reshape(1,4)
    if scores is None:
        scores = np.ones((bboxes.shape[0],), dtype=np.float32)
    if labels is None:
        labels = np.zeros((bboxes.shape[0],), dtype=np.int64)
    return bboxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int64)

# ============================================================
# Gradio callbacks
# ============================================================
def on_select(evt: gr.SelectData):
    # evt.index is (x, y) for click OR (x, y, w, h) for drag-select
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        if len(idx) == 4:
            x, y, w, h = map(int, idx)
            # Clamp tiny/zero rectangles
            w = max(8, w)
            h = max(8, h)
            return json.dumps({"x": x, "y": y, "w": w, "h": h})
        elif len(idx) == 2:
            x, y = map(int, idx)
            # Default small box if just clicked (optional)
            return json.dumps({"x": max(0, x-32), "y": max(0, y-32), "w": 64, "h": 64})
    # Fallback: ask user to drag
    return json.dumps({})

def do_load_model(config_name, weight_file, conf_thresh, nms_thresh, topk):
    try:
        msg = load_model(ModelCfg(
            config_name=config_name, weight_file=weight_file,
            conf_thresh=float(conf_thresh), nms_thresh=float(nms_thresh), topk=int(topk)
        ))
        return gr.update(value=msg)
    except Exception as e:
        return gr.update(value=f"Load failed: {e}")

def process(image: np.ndarray,
            rect_json: str,
            dark_ratio: float,
            cls_idx: int,
            ignore_center_entropy: bool,
            miou_ignore_center: bool,
            miou_rel: float,
            miou_thr_min: float, miou_thr_max: float, miou_thr_num: int,
            ap_iou_thr: float,
            heat_alpha: float,
            face_thresh: float):

    if image is None:
        return None, None, None, None, "Please upload an image."

    if MODEL is None or TRANSFORM is None:
        return None, None, None, None, "Please load the model (left panel)."
    
    if not rect_json:
        # Default to center 1/3 if no selection
        H, W = img_bgr.shape[:2]
        w = max(8, W // 3); h = max(8, H // 3)
        px = (W - w) // 2; py = (H - h) // 2
    else:
        rect = json.loads(rect_json)
        px, py, w, h = rect["x"], rect["y"], rect["w"], rect["h"]

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # Selection rectangle
    try:
        rect = json.loads(rect_json) if rect_json else {}
        px, py, w, h = int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"])
        px = max(0, min(W-1, px)); py = max(0, min(H-1, py))
        w = max(8, min(W, w)); h = max(8, min(H, h))
    except Exception:
        return None, None, None, None, "Please drag to select a rectangle on the image."

    # Build DA & Is
    DA_bgr, Is_bgr, (x0, y0, Hexp, Wexp) = build_DA_with_padding(img_bgr, px, py, h, w, dark_ratio)

    # DeepFace GT on original image
    face_boxes = detect_faces_bgr_deepface(img_bgr, backend="retinaface", thresh=float(face_thresh))
    # project into Is
    gt_in_Is = []
    for (x1,y1,x2,y2) in face_boxes:
        xx1 = max(px, x1); yy1 = max(py, y1)
        xx2 = min(px+w, x2); yy2 = min(py+h, y2)
        if xx2>xx1 and yy2>yy1:
            gt_in_Is.append((xx1-px, yy1-py, xx2-px, yy2-py))
    gt_in_Is = np.array(gt_in_Is, dtype=np.float32).reshape(-1,4)

    with torch.inference_mode():
        x = apply_transform(Is_bgr)             # (1,C,h,w)
        out = MODEL(x)                          # dict OR tuple; torch OR numpy
        heat_logits = MODEL.heatmap(x)          # (1, S, C) torch
        
    if not isinstance(heat_logits, torch.Tensor):
        heat_logits = torch.from_numpy(heat_logits)
    heat_logits = heat_logits.to(DEVICE)  # if your code expects GPU; otherwise .cpu()

    # Parse predictions robustly
    pred_b, pred_s, pred_l = parse_model_outputs(out)  # all numpy

    # Rescale from [0,1] to Is pixel coords
    if pred_b.size > 0:
        pred_abs = pred_b.copy()
        pred_abs[:, [0,2]] *= w
        pred_abs[:, [1,3]] *= h
    else:
        pred_abs = np.zeros((0,4), dtype=np.float32)
        pred_s = np.zeros((0,), dtype=np.float32)
        
    if pred_b.size > 0:
        pred_abs = pred_b.copy()
        pred_abs[:, [0,2]] *= w
        pred_abs[:, [1,3]] *= h
    else:
        pred_abs = np.zeros((0,4), dtype=np.float32)

    # CE / SE on Is
    B,S,C = heat_logits.shape
    Hh = int(math.sqrt(S)); Wh = Hh
    assert Hh*Wh == S, "Heatmap not square; adapt if needed."
    gt_heat_BSC = boxes_to_heatmap(Is_hw=(h,w), boxes_abs=gt_in_Is.tolist(),
                                   out_hw=(Hh,Wh), num_classes=C, cls_idx=cls_idx)
    ce_val = heatmap_2dce(heat_logits, gt_heat_BSC, input_type='pred_logits',
                          reduction='mean', ignore_center=ignore_center_entropy,
                          spatial_hw=(Hh, Wh))
    se_val = heatmap_self_entropy_2d(heat_logits, input_type='pred_logits',
                                     reduction='mean', class_idx=cls_idx,
                                     ignore_center=ignore_center_entropy,
                                     spatial_hw=(Hh, Wh))

    # mIoU curves + aggregates
    thresholds = np.linspace(miou_thr_min, miou_thr_max, miou_thr_num).tolist()
    miou_opts = MiouOptions(ignore_center=miou_ignore_center, spatial_hw=(Hh,Wh),
                            gt_binarize="relative", gt_rel_thresh=miou_rel, class_idx=cls_idx)
    gt_heat_HW = gt_heat_BSC[0,:,cls_idx].reshape(Hh,Wh).unsqueeze(0).unsqueeze(0)
    miou_res = miou_over_thresholds_single(heat_logits, gt_heat_HW.repeat(1,C,1,1), thresholds, miou_opts)
    curve_img = plot_curves(miou_res["thresholds"], miou_res["iou_std"], miou_res["iou_int"], miou_res["f_beta"])

    # mAP & MAE on Is
    ap = compute_ap_at_iou(pred_abs, pred_s, gt_in_Is, iou_thr=float(ap_iou_thr))
    mae = mean_abs_center_error(pred_abs, gt_in_Is, H=h, W=w)

    # Visuals on DA
    da_pred_abs = pred_abs.copy()
    da_pred_abs[:, [0,2]] += w
    da_pred_abs[:, [1,3]] += h
    da_boxes_img = overlay_boxes(DA_bgr, da_pred_abs, color=(0,255,0), thickness=2, scores=pred_s)

    probs = torch.sigmoid(heat_logits[0,:,cls_idx]).reshape(Hh,Wh).detach().cpu().numpy()
    probs_up = cv2.resize(probs, (w,h), interpolation=cv2.INTER_CUBIC)
    da_heat_img = DA_bgr.copy()
    da_heat_img[h:2*h, w:2*w] = overlay_heatmap(DA_bgr[h:2*h, w:2*w], probs_up, alpha=float(heat_alpha))

    # Is with GT+Pred
    is_pred_img = overlay_boxes(Is_bgr, pred_abs, color=(0,255,0), thickness=2, scores=pred_s)
    is_gt_img = overlay_boxes(is_pred_img, gt_in_Is, color=(255,0,0), thickness=2, scores=None)

    msg = (f"mAP@{ap_iou_thr:.2f}: {ap:.3f} | MAE(center): {mae:.3f} | "
           f"CE: {float(ce_val):.3f} | SE: {float(se_val):.3f} | "
           f"mIoU: {miou_res['mIoU']:.3f} | mIoU_int: {miou_res['mIoU_int']:.3f} | mF1: {miou_res['mF1']:.3f}")

    return (
        cv2.cvtColor(da_boxes_img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(da_heat_img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(is_gt_img, cv2.COLOR_BGR2RGB),
        curve_img,
        msg
    )

def on_image_uploaded(image: np.ndarray, det_backend: str, det_thresh: float):
    """
    1) Convert to BGR, run DeepFace, overlay pseudo-GT on the uploaded image.
    2) Cache original BGR + GT boxes in state.
    """
    if image is None:
        return None, None, None

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    boxes = detect_faces_bgr_deepface(img_bgr, backend=det_backend, thresh=float(det_thresh))
    # visualize GT in RED
    vis = overlay_boxes(img_bgr, np.array(boxes, dtype=np.float32), color=(0,0,255), thickness=2)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis_rgb, img_bgr, np.array(boxes, dtype=np.float32)  # preview, state_image_bgr, state_gt_boxes_xyxy

def _draw_temp_rect(rgb_img, p1, p2):
    vis = rgb_img.copy()
    if p1 is not None:
        cv2.circle(vis, p1, 4, (0,255,0), -1)
    if p1 is not None and p2 is not None:
        x1,y1 = p1; x2,y2 = p2
        x, y = min(x1,x2), min(y1,y2)
        w, h = abs(x2-x1), abs(y2-y1)
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
    return vis

def select_two_click(evt: gr.SelectData, image: np.ndarray, p1_state):
    if image is None:
        return None, json.dumps({}), None
    idx = evt.index
    if not isinstance(idx, (list, tuple)) or len(idx)!=2:
        return image, json.dumps({}), None
    x,y = int(idx[0]), int(idx[1])
    if p1_state is None:
        # first click
        return _draw_temp_rect(image, (x,y), None), json.dumps({}), (x,y)
    # second click: finalize
    x1,y1 = p1_state
    x0,y0 = min(x1,x), min(y1,y)
    w,h = max(8, abs(x-x1)), max(8, abs(y-y1))
    rect = {"x": int(x0), "y": int(y0), "w": int(w), "h": int(h)}
    vis = _draw_temp_rect(image, (x0,y0), (x0+w,y0+h))
    return vis, json.dumps(rect), None

def on_generate_DA(
    rect_json_str: str,
    dark_ratio: float,
    image_bgr: np.ndarray,
    gt_boxes_xyxy: Optional[np.ndarray]
):
    if image_bgr is None:
        return None, None, None, None, None, "Upload an image first.", None, None

    try:
        r = json.loads(rect_json_str) if rect_json_str else {}
        px, py, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
    except Exception:
        return None, None, None, None, None, "Select a rectangle (two clicks) before generating DA.", None, None
    
    H, W = image_bgr.shape[:2]

    # Build DA (3h x 3w) with padding + darkened ring; Is is the center h x w
    DA_bgr, Is_bgr, _ = build_DA_with_padding(image_bgr, px, py, h, w, dark_ratio)

    # -------- GT projected into Is (center tile) --------
    gt_in_Is = []
    if gt_boxes_xyxy is not None and gt_boxes_xyxy.size > 0:
        for (x1, y1, x2, y2) in gt_boxes_xyxy.tolist():
            xx1 = max(px, x1); yy1 = max(py, y1)
            xx2 = min(px + w, x2); yy2 = min(py + h, y2)
            if xx2 > xx1 and yy2 > yy1:
                gt_in_Is.append((xx1 - px, yy1 - py, xx2 - px, yy2 - py))
    gt_in_Is = np.array(gt_in_Is, dtype=np.float32).reshape(-1, 4)

    # -------- GT projected into DA (full 3h x 3w) --------
    # Compute DA rect analytically (no need for da_offsets)
    x0 = px - w
    y0 = py - h
    W_DA = 3 * w
    H_DA = 3 * h
    DA_rect = (x0, y0, x0 + W_DA, y0 + H_DA)

    gt_in_DA = []
    if gt_boxes_xyxy is not None and gt_boxes_xyxy.size > 0:
        for (x1, y1, x2, y2) in gt_boxes_xyxy.tolist():
            xx1 = max(DA_rect[0], x1); yy1 = max(DA_rect[1], y1)
            xx2 = min(DA_rect[2], x2); yy2 = min(DA_rect[3], y2)
            if xx2 > xx1 and yy2 > yy1:
                # shift to DA-local coords
                gt_in_DA.append((xx1 - x0, yy1 - y0, xx2 - x0, yy2 - y0))
    gt_in_DA = np.array(gt_in_DA, dtype=np.float32).reshape(-1, 4)

    da_rgb = cv2.cvtColor(DA_bgr, cv2.COLOR_BGR2RGB)
    # Return exactly the same outputs you wired previously (+ keep DA_bgr and gt_in_DA in state)
    return (
        da_rgb, Is_bgr, (h, w), (x0, y0, H_DA, W_DA),  # or your existing offsets tuple
        gt_in_Is,
        "DA generated.",
        DA_bgr,
        gt_in_DA,
        (px, py, w, h),         # <<< state_pxpywh
        (H, W)                  # <<< state_image_shape
    )

def on_run_model(
    cls_idx: int,
    ignore_center_entropy: bool,
    miou_ignore_center: bool,
    miou_rel: float,
    miou_thr_min: float, miou_thr_max: float, miou_thr_num: int,
    ap_iou_thr: float,
    heat_alpha: float,
    Is_bgr: np.ndarray,              # feed Is to the model (unchanged)
    gt_in_DA: Optional[np.ndarray],  # <<< GT already projected into DA coords
    Is_hw: Optional[tuple],          # (h, w) so DA = (3h, 3w)
    DA_bgr: Optional[np.ndarray],    # <<< DA canvas to draw on
    conf_thr: float,
    nms_thr: float,
    show_bodies: bool,
):
    if MODEL is None or TRANSFORM is None:
        return None, None, None, None, "Load the model first."
    if Is_bgr is None or Is_hw is None:
        return None, None, None, None, "Generate DA first."

    h, w = Is_hw
    H_DA, W_DA = 3*h, 3*w
    gt_in_DA = gt_in_DA if gt_in_DA is not None else np.zeros((0,4), dtype=np.float32)

    # Transform + model
    with torch.inference_mode():
        x = apply_transform(Is_bgr)             # (1,C,h,w)
        out = MODEL(x)                          # dict OR tuple; torch OR numpy
        heat_logits = MODEL.heatmap(x)          # (1, S, C) torch

    if not isinstance(heat_logits, torch.Tensor):
        heat_logits = torch.from_numpy(heat_logits)
    heat_logits = heat_logits.to(DEVICE)  # if your code expects GPU; otherwise .cpu()

    # Parse predictions robustly
    pred_b, pred_s, pred_l = parse_model_outputs(out)  # all numpy

    # ---------- scale boxes by DA size (NOT Is size) ----------
    if pred_b.size > 0:
        pred_abs = pred_b.copy()
        pred_abs[:, [0, 2]] *= W_DA
        pred_abs[:, [1, 3]] *= H_DA
    else:
        pred_abs = np.zeros((0,4), dtype=np.float32)
        pred_s = np.zeros((0,), dtype=np.float32)

    # infer DA heatmap grid Hh x Wh from S
    B, S, C = heat_logits.shape
    Hh = int(round(math.sqrt(S)))
    Wh = S // Hh
    assert Hh * Wh == S, "Adapt Hh/Wh if your head is not square."

    # ---------- GT heatmap on DA grid ----------
    gt_heat_BSC = boxes_to_heatmap(
        Is_hw=(H_DA, W_DA),               # build over DA
        boxes_abs=gt_in_DA.tolist(),
        out_hw=(Hh, Wh),
        num_classes=C,
        cls_idx=cls_idx
    ).to(device=heat_logits.device, dtype=heat_logits.dtype)

    # ---------- CE / SE over DA ----------
    ce_val = heatmap_2dce(
        heat_logits, gt_heat_BSC, input_type='pred_logits',
        reduction='mean', ignore_center=ignore_center_entropy,
        spatial_hw=(Hh, Wh)
    )
    se_val = heatmap_self_entropy_2d(
        heat_logits, input_type='pred_logits', reduction='mean',
        class_idx=cls_idx, ignore_center=ignore_center_entropy,
        spatial_hw=(Hh, Wh)
    )

    # ---------- mIoU / mIoU_int / mF1 over DA ----------
    thresholds = np.linspace(miou_thr_min, miou_thr_max, miou_thr_num).tolist()
    miou_opts = MiouOptions(ignore_center=miou_ignore_center, spatial_hw=(Hh,Wh),
                            gt_binarize="relative", gt_rel_thresh=miou_rel, class_idx=cls_idx)
    gt_heat_HW = gt_heat_BSC[0, :, cls_idx].reshape(Hh,Wh).unsqueeze(0).unsqueeze(0)
    gt_heat_HW = gt_heat_HW.to(device=heat_logits.device, dtype=heat_logits.dtype)
    miou_res = miou_over_thresholds_single(heat_logits, gt_heat_HW.repeat(1,C,1,1), thresholds, miou_opts)
    curve_img = plot_curves(miou_res["thresholds"], miou_res["iou_std"], miou_res["iou_int"], miou_res["f_beta"])

    # ---------- mAP & MAE in DA coords ----------
    if pred_s.size == 0 and pred_abs.shape[0] > 0:
        pred_s = np.ones((pred_abs.shape[0],), dtype=np.float32)
    ap  = compute_ap_at_iou(pred_abs, pred_s, gt_in_DA, iou_thr=float(ap_iou_thr))
    mae = mean_abs_center_error(pred_abs, gt_in_DA, H=H_DA, W=W_DA)


    # ---------- Visualizations on FULL DA ---------

    # 2) DA + Heatmap (full canvas)
    probs = torch.sigmoid(heat_logits[0, :, cls_idx]).reshape(Hh, Wh).detach().cpu().numpy()
    probs_up = cv2.resize(probs, (W_DA, H_DA), interpolation=cv2.INTER_CUBIC)
    da_heat_img = overlay_heatmap(DA_bgr.copy(), probs_up, alpha=float(heat_alpha))

    # ---------- STORE RAW PREDICTIONS IN STATE ----------
    # (we'll also return these via outputs to update the gr.State objects)
    pred_abs_raw = pred_abs.copy()
    pred_scores_raw = pred_s.copy()
    pred_labels_raw = pred_l.copy()

    # ---------- initial filter & render with current UI values ----------
    def _filter_and_render_now(DA_bgr, gt_in_DA, b_raw, s_raw, l_raw, conf_thr, nms_thr, show_bodies):
        b, s, l = apply_conf_nms_class_aware(b_raw, s_raw, l_raw, float(conf_thr), float(nms_thr), bool(show_bodies))
        img_da_pred, img_da_gp = render_da_and_gp_classwise(DA_bgr, gt_in_DA, b, s, l)
        return (cv2.cvtColor(img_da_pred, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(img_da_gp,  cv2.COLOR_BGR2RGB),
                b, s, l)

    da_pred_rgb, da_gp_rgb, pred_abs_filt, pred_scores_filt, pred_labels_filt = _filter_and_render_now(
        DA_bgr, gt_in_DA, pred_abs_raw, pred_scores_raw, pred_labels_raw,
        conf_thr, nms_thr, show_bodies
    )

    # ---------- Compose message (unchanged) ----------
    msg = (f"mAP@{ap_iou_thr:.2f}: {ap:.3f} | MAE(center): {mae:.3f} | "
        f"CE: {float(ce_val):.3f} | SE: {float(se_val):.3f} | "
        f"mIoU: {miou_res['mIoU']:.3f} | mIoU_int: {miou_res['mIoU_int']:.3f} | mF1: {miou_res['mF1']:.3f}")

    # keep heat logits & last-used class for saving
    heat_np_or_t = heat_logits.detach().cpu() if isinstance(heat_logits, torch.Tensor) else heat_logits

    return (
        da_pred_rgb,                                        # DA + Pred (filtered)
        cv2.cvtColor(da_heat_img, cv2.COLOR_BGR2RGB),       # DA + Heat
        da_gp_rgb,                                          # DA + GT & Pred
        pred_abs_raw, pred_scores_raw, pred_labels_raw,
        heat_np_or_t.numpy() if isinstance(heat_np_or_t, torch.Tensor) else heat_np_or_t,
        int(cls_idx)
    )
    
def on_filter_change_class_aware(
    DA_bgr, gt_in_DA,
    pred_abs_raw, pred_scores_raw, pred_labels_raw,
    conf_thr, nms_thr, show_bodies
):
    if DA_bgr is None or pred_abs_raw is None or pred_scores_raw is None or pred_labels_raw is None:
        return gr.update(), gr.update()
    b, s, l = apply_conf_nms_class_aware(pred_abs_raw, pred_scores_raw, pred_labels_raw,
                                         float(conf_thr), float(nms_thr), bool(show_bodies))
    img_da_pred, img_da_gp = render_da_and_gp_classwise(DA_bgr, gt_in_DA, b, s, l)
    return cv2.cvtColor(img_da_pred, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_da_gp, cv2.COLOR_BGR2RGB)

def on_save(
    base_name: str,
    pxpywh,                 # state_pxpywh  -> (px, py, w, h) in original I
    image_hw,               # state_image_shape -> (H, W)
    DA_bgr,                 # state_DA_bgr
    heat_logits,            # state_heat_logits (1,S,C) torch/np
    cls_idx_cached,         # state_cls_idx_cache (int)
    pred_abs_raw, pred_scores_raw, pred_labels_raw,
    conf_thr, nms_thr, show_bodies, heat_alpha,
    gt_boxes_image_xyxy,    # <<< NEW: state_gt_boxes_xyxy  (M,4) in I coords
    gt_boxes_DA_xyxy        # <<< NEW: state_gt_in_DA       (M',4) in DA coords (after cropping)
):
    # sanity
    if DA_bgr is None or pxpywh is None or image_hw is None:
        return None, None, "Please generate DA first."
    if base_name is None or str(base_name).strip() == "":
        base_name = datetime.datetime.now().strftime("save_%Y%m%d_%H%M%S")

    save_dir = "./saved"
    ensure_dir(save_dir)

    # unpack sizes
    px, py, w, h = map(int, pxpywh)
    H, W = map(int, image_hw)
    x0, y0 = px - w, py - h
    W_DA, H_DA = 3 * w, 3 * h

    # tidy GT lists
    gt_img = [] if gt_boxes_image_xyxy is None or len(np.atleast_2d(gt_boxes_image_xyxy)) == 0 \
             else np.atleast_2d(gt_boxes_image_xyxy).astype(float).tolist()
    gt_da  = [] if gt_boxes_DA_xyxy is None or len(np.atleast_2d(gt_boxes_DA_xyxy)) == 0 \
             else np.atleast_2d(gt_boxes_DA_xyxy).astype(float).tolist()

    # 1) JSON: Is + DA + GT
    is_json = {
        "image_size": {"H": H, "W": W},
        "selection_xywh": {"x": px, "y": py, "w": w, "h": h},
        "DA_xywh": {"x": x0, "y": y0, "w": W_DA, "h": H_DA},
        "gt_boxes_image_xyxy": gt_img,  # detector GT in original image coords
        "gt_boxes_DA_xyxy": gt_da,      # detector GT projected onto DA coords
        "timestamp": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    }
    json_path = os.path.join(save_dir, f"{base_name}_Is.json")
    with open(json_path, "w") as f:
        json.dump(is_json, f, indent=2)

    # 2) Combined DA image = heatmap over DA + filtered predicted boxes
    try:
        probs_up = rebuild_probs_from_logits(heat_logits, int(cls_idx_cached), W_DA, H_DA)
    except Exception:
        probs_up = np.zeros((H_DA, W_DA), dtype=np.float32)

    da_overlay = overlay_heatmap_full(DA_bgr, probs_up, float(heat_alpha))

    if pred_abs_raw is None or pred_scores_raw is None or pred_labels_raw is None:
        boxes_f, scores_f, labels_f = (
            np.zeros((0,4), np.float32),
            np.zeros((0,),  np.float32),
            np.zeros((0,),  np.int64),
        )
    else:
        boxes_f, scores_f, labels_f = apply_conf_nms_class_aware(
            pred_abs_raw, pred_scores_raw, pred_labels_raw,
            float(conf_thr), float(nms_thr), bool(show_bodies)
        )

    da_comb_bgr = render_da_and_gp_classwise(
        da_overlay, np.zeros((0,4), dtype=np.float32),  # no GT drawn on combined save
        boxes_f, scores_f, labels_f
    )[0]

    png_path = os.path.join(save_dir, f"{base_name}_DA_combined.png")
    cv2.imwrite(png_path, da_comb_bgr)

    return json_path, png_path, f"Saved:\n- {json_path}\n- {png_path}"

# ============================================================
# UI
# ============================================================
with gr.Blocks(title="DeepFace + Heatmap Metrics Demo") as demo:
    gr.Markdown("# DeepFace-GT Heatmap & Detection Metrics Demo")
    # persistent states across steps
    state_image_bgr      = gr.State(value=None)   # original BGR
    state_gt_boxes_xyxy  = gr.State(value=None)   # GT boxes on original image
    state_Is_bgr         = gr.State(value=None)   # cropped Is (center h×w)
    state_DA_bgr         = gr.State(value=None)   # 3h×3w ring view
    state_Is_hw          = gr.State(value=None)   # (h, w)
    state_DA_offsets     = gr.State(value=None)   # (x0, y0, Hexp, Wexp) from DA build
    state_gt_in_Is       = gr.State(value=None)   # GT boxes projected into Is
    state_heat_hw        = gr.State(value=None)   # (Hh, Wh) heatmap spatial size
    # add a new state alongside your others
    state_gt_in_DA = gr.State(value=None)  # GT boxes projected into DA coords
    
    state_pred_abs_raw     = gr.State(value=None)  # (N,4) DA coords
    state_pred_scores_raw  = gr.State(value=None)  # (N,)
    state_pred_labels_raw  = gr.State(value=None)  # (N,) int64 (0: face, 1: body)
    
    state_pxpywh        = gr.State(value=None)   # (px, py, w, h) in original image I coords
    state_image_shape   = gr.State(value=None)   # (H, W) of original I
    
    # add this so we can rebuild the heatmap overlay on save:
    state_heat_logits   = gr.State(value=None)   # (1, S, C) torch or numpy
    state_cls_idx_cache = gr.State(value=0)      # last-used cls_idx for heatmap

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Config")
            config_name = gr.Dropdown(choices=sorted(list(yoloh_config.keys())), 
                                      value="yoloh_expand-50-DC5-640-expand-attn-2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2", 
                                      label="config_name",
                                      interactive=False  # This freezes the dropdown
                                      )
            weight_file = gr.Textbox(label="Weight file path (.pth)", value="/home/u7707452/Projects/eaft/checkpoints/2scale2/basic/yoloh_expand-50-DC5-640-expand-attn-2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2_epoch_14_Thu18Sep2025-014729.pth")
            # conf_thresh = gr.Number(value=0.0, label="conf_thresh")
            # nms_thresh = gr.Number(value=1.0, label="nms_thresh")
            # topk = gr.Number(value=3000, label="topk", precision=0)
            conf_thresh = gr.State(value=0.0)
            nms_thresh = gr.State(value=1.0)
            topk = gr.State(value=1000)
            load_btn = gr.Button("Load / Reload Model")
            load_msg = gr.Markdown("")

            gr.Markdown("### Inputs")
            # Replace your current image component (remove tool=... which errors)
            # Image input and GT overlay preview
            in_img = gr.Image(label="Upload image", type="numpy", image_mode="RGB", sources=["upload"], interactive=True)
            gt_preview = gr.Image(label="Pseudo-GT overlay", type="numpy")
            

            preview_img = gr.Image(label="Selection preview", type="numpy")
            rect_json = gr.Textbox(label="Selection JSON", interactive=False)
            p1_state = gr.State(value=None)

            # Bind the select event to two-click handler
            in_img.select(
                fn=select_two_click,
                inputs=[in_img, p1_state],
                outputs=[preview_img, rect_json, p1_state],
            )
            
            reset_btn = gr.Button("Reset selection")

            dark_ratio = gr.State(value=0.0)  # default dark_ratio state
            cls_idx = gr.State(value=0)
            ignore_center_entropy = gr.State(value=False)
            miou_ignore_center = gr.State(value=False)
            miou_rel = gr.State(value=0.25)
            miou_thr_min = gr.State(value=0.05)
            miou_thr_max = gr.State(value=0.95)
            miou_thr_num = gr.State(value=19)
            ap_iou_thr = gr.State(value=0.5)
            heat_alpha = gr.State(value=0.5)
            # dark_ratio = gr.Slider(0.0, 0.9, value=0.0, step=0.05, label="Darken ratio (DA ring)")
            # # entropy / miou controls (keep your existing widgets or use these)
            # cls_idx = gr.Slider(0, 10, value=0, step=1, label="Class index")
            # ignore_center_entropy = gr.Checkbox(value=False, label="Ignore center (CE/SE)")
            # miou_ignore_center    = gr.Checkbox(value=False, label="Ignore center (mIoU curves)")
            # miou_rel = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="GT binarize (relative to max)")
            # miou_thr_min = gr.Slider(0.0, 0.9, value=0.05, step=0.05, label="mIoU threshold min (prob)")
            # miou_thr_max = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="mIoU threshold max (prob)")
            # miou_thr_num = gr.Slider(5, 50, value=19, step=1, label="mIoU number of thresholds")
            # ap_iou_thr   = gr.Slider(0.3, 0.9, value=0.5, step=0.05, label="mAP IoU threshold")
            # heat_alpha   = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Heatmap overlay alpha")
            # Detector backend and threshold controls (left panel)
            with gr.Accordion("Detector Settings", open=False):
                det_backend = gr.Dropdown(choices=["yolov8", "yolov11m"],
                                        value="yolov11m", label="DeepFace detector backend")
                face_thresh = gr.Slider(0.3, 0.99, value=0.8, step=0.01, label="Detector confidence threshold")
            with gr.Accordion("Display settings", open=False):
                conf_thr_ui = gr.Slider(0.0, 1.0, value=0.18, step=0.01, label="Pred box confidence threshold")
                nms_thr_ui  = gr.Slider(0.0, 1.0, value=0.1,  step=0.01, label="NMS IoU threshold")
                show_bodies_ui = gr.Checkbox(value=True, label="Show body boxes (class 1)")

        with gr.Column(scale=1):
            

            da_view = gr.Image(label="DA (3h×3w)", type="numpy")
            
            with gr.Row():
                gen_da_btn = gr.Button("1) Generate DA")
                run_btn = gr.Button("2) Run Model on Is")

            out_da_boxes = gr.Image(label="DA with predicted boxes")
            out_da_heat  = gr.Image(label="DA with heatmap overlay")
            out_is_view  = gr.Image(label="Is with GT (red) & Pred (green)")
            
            # save_base_ui = gr.Textbox(value="sample_0001", label="Save name (base filename)")
            # save_btn     = gr.Button("Save selection & combined DA")
            # file_Is_json = gr.File(label="Saved Is JSON")
            # file_da_png  = gr.File(label="Saved DA combined image")
            # out_curves   = gr.Image(label="IoU / IoU_int / F1 vs threshold")
            # msg          = gr.Markdown("")
    
    
    # bind selection on the GT overlay image so user sees faces while selecting
    gt_preview.select(fn=select_two_click, inputs=[gt_preview, p1_state], outputs=[preview_img, rect_json, p1_state])

    def reset_sel(image):
        return image, json.dumps({}), None
    reset_btn.click(fn=reset_sel, inputs=[gt_preview], outputs=[preview_img, rect_json, p1_state])
    
    gen_da_btn.click(
        fn=on_generate_DA,
        inputs=[rect_json, dark_ratio, state_image_bgr, state_gt_boxes_xyxy],
        outputs=[da_view, state_Is_bgr, state_Is_hw, state_DA_offsets, state_gt_in_Is,
                gr.Textbox(label="Status", interactive=False),
                state_DA_bgr, state_gt_in_DA, state_pxpywh, state_image_shape]  # <<< add here
    )
    

    # when image changes, run detector and overlay GT
    in_img.change(
        fn=on_image_uploaded,
        inputs=[in_img, det_backend, face_thresh],
        outputs=[gt_preview, state_image_bgr, state_gt_boxes_xyxy],
    )

    load_btn.click(
        fn=do_load_model,
        inputs=[config_name, weight_file, conf_thresh, nms_thresh, topk],
        outputs=[load_msg],
    )

    run_btn.click(
        fn=on_run_model,
        inputs=[
            cls_idx, ignore_center_entropy, miou_ignore_center, miou_rel,
            miou_thr_min, miou_thr_max, miou_thr_num, ap_iou_thr, heat_alpha,
            state_Is_bgr, state_gt_in_DA, state_Is_hw, state_DA_bgr,
            conf_thr_ui, nms_thr_ui, show_bodies_ui
        ],
        outputs=[
            out_da_boxes, out_da_heat, out_is_view,
            state_pred_abs_raw, state_pred_scores_raw, state_pred_labels_raw,
            state_heat_logits, state_cls_idx_cache
        ],
    )

    # Wire both sliders to update the two panels
    for ctl in (conf_thr_ui, nms_thr_ui, show_bodies_ui):
        ctl.change(
            fn=on_filter_change_class_aware,
            inputs=[state_DA_bgr, state_gt_in_DA,
                    state_pred_abs_raw, state_pred_scores_raw, state_pred_labels_raw,
                    conf_thr_ui, nms_thr_ui, show_bodies_ui],
            outputs=[out_da_boxes, out_is_view],
        )
        
    # save_btn.click(
    #     fn=on_save,
    #     inputs=[
    #         save_base_ui,
    #         state_pxpywh, state_image_shape,
    #         state_DA_bgr,
    #         state_heat_logits, state_cls_idx_cache,
    #         state_pred_abs_raw, state_pred_scores_raw, state_pred_labels_raw,
    #         conf_thr_ui, nms_thr_ui, show_bodies_ui, heat_alpha,
    #         state_gt_boxes_xyxy,   # <<< detector GT in image coords
    #         state_gt_in_DA         # <<< detector GT projected in DA coords
    #     ],
    #     outputs=[file_Is_json, file_da_png, msg]
    # )


if __name__ == "__main__":
    demo.launch()