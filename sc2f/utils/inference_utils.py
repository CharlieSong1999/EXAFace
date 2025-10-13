import os
import cv2
import math
import numpy as np
import torch


def build_DA_with_padding(img_bgr: np.ndarray, px: int, py: int, h: int, w: int, dark_ratio: float = 0.5):
    """Build 3×3 padded image with dark ring."""
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


def overlay_heatmap(image_bgr: np.ndarray, heat_HW: np.ndarray, alpha=0.5):
    heat_uint8 = np.clip(heat_HW * 255, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0, heat_color, alpha, 0)


def overlay_boxes(image_bgr: np.ndarray, boxes_xyxy: np.ndarray, color=(0,255,0), thickness=2, scores=None):
    img = image_bgr.copy()
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if scores is not None:
            cv2.putText(img, f"{scores[i]:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def parse_model_outputs(out, face_only=True):
    """
    Supports:
      - dict with keys: 'bboxes', 'scores', 'labels'
      - tuple or list: (bboxes, scores, labels)
    Returns: (bboxes, scores, labels) as np.ndarray
    """
    def _to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

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
        bboxes = _to_numpy(out)

    if bboxes is None:
        bboxes = np.zeros((0, 4), dtype=np.float32)
    if bboxes.ndim == 1 and bboxes.size == 4:
        bboxes = bboxes.reshape(1, 4)
    if scores is None:
        scores = np.ones((bboxes.shape[0],), dtype=np.float32)
    if labels is None:
        labels = np.zeros((bboxes.shape[0],), dtype=np.int64)
        
    if face_only:
        m_face = labels == 0
        bboxes, scores, labels = bboxes[m_face], scores[m_face], labels[m_face]

    return bboxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int64)


def rebuild_probs_from_logits(heat_logits, cls_idx, W_DA, H_DA):
    if isinstance(heat_logits, np.ndarray):
        hl = torch.from_numpy(heat_logits)
    else:
        hl = heat_logits
    hl = hl.float()
    B, S, C = hl.shape
    Hh = int(round(math.sqrt(S)))
    Wh = S // Hh
    probs = torch.sigmoid(hl[0, :, cls_idx]).reshape(Hh, Wh).cpu().numpy()
    probs_up = cv2.resize(probs, (W_DA, H_DA), interpolation=cv2.INTER_CUBIC)
    return probs_up


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    ua = max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / ua


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thresh: float) -> np.ndarray:
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
        keep_all.extend(idx_c[np.array(kept_local, dtype=np.int64)])
    return np.array(keep_all, dtype=np.int64)


def apply_conf_nms_class_aware(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    conf_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    show_bodies: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_conf = scores >= float(conf_thresh)
    boxes = boxes[m_conf]
    scores = scores[m_conf]
    labels = labels[m_conf]
    if not show_bodies:
        m_face = labels == 0
        boxes, scores, labels = boxes[m_face], scores[m_face], labels[m_face]
    if boxes.size == 0:
        return boxes, scores, labels
    keep = nms_per_class(boxes, scores, labels, float(nms_thresh))
    return boxes[keep], scores[keep], labels[keep]