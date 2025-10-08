#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extend-to-3x canvas + Pix2Gestalt completion pipeline WITH multi-ann merging
and optional **multi-GPU** parallelism (one process per GPU) in *work* mode.

VRAM-friendly version:
  - Checkpoint is loaded on CPU first, then model is moved to GPU (prevents CUDA
    deserialization OOM).
  - Precision selectable (bf16/fp16/fp32) + channels_last + allocator tuning.
  - Inference wrapped with torch.inference_mode() and AMP autocast.
  - SAM2 removed entirely.
  - Optional mask extractor "threshold" (no extra model) instead of CarveKit.

Modes:
  - verbose: save every extension step for selected images (single GPU)
  - sanity : save only finals for selected images (single GPU)
  - work   : process the entire dataset (supports multi-GPU via --num-gpus / --gpu-ids)

Skip rules:
  - Skip a **face** if difficulty ∈ {normal, hard}
  - Skip a **body** if its paired face difficulty ∈ {normal, hard}

Requirements:
  pycocotools, OpenCV, torch, tqdm, PIL, matplotlib
  Your Pix2Gestalt codebase (inference.py, ldm.util.create_carvekit_interface)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import logging, traceback, multiprocessing as mp
import torch

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("pix2gestalt.extend")

# hotfix for taming import
sys.path.append(os.path.join(os.path.dirname(__file__), "taming-transformers"))

# -----------------------------------------------------------------------------
# COCO + util
# -----------------------------------------------------------------------------
try:
    import pycocotools.mask as mask_utils
except Exception as e:
    print("[ERROR] pycocotools is required. pip install pycocotools", file=sys.stderr)
    raise

# Pix2Gestalt
from omegaconf import OmegaConf
from inference import run_inference, load_model_from_config
from ldm.util import create_carvekit_interface

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_coco(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    for k in ["images", "annotations", "categories"]:
        if k not in data:
            raise ValueError(f"COCO file missing key '{k}': {path}")
    return data


def anns_by_image(coco: Dict) -> Dict[int, List[Dict]]:
    from collections import defaultdict
    m = defaultdict(list)
    for a in coco["annotations"]:
        m[a["image_id"]].append(a)
    return m


def category_id_to_name(coco: Dict) -> Dict[int, str]:
    return {c["id"]: c.get("name", str(c["id"])) for c in coco["categories"]}

# -----------------------------------------------------------------------------
# Mask utilities
# -----------------------------------------------------------------------------

def fit_mask_to(mask: np.ndarray, H: int, W: int, center_crop_if_larger: bool = True) -> np.ndarray:
    """Conform arbitrary mask to target (H,W). Returns 0/255 uint8."""
    if mask is None:
        return np.zeros((H, W), dtype=np.uint8)
    m = mask[..., 0] if (mask.ndim == 3) else mask
    h, w = m.shape[:2]
    if (h, w) == (H, W):
        return ((m > 0).astype(np.uint8) * 255)
    if center_crop_if_larger and h >= H and w >= W:
        y0 = (h - H) // 2
        x0 = (w - W) // 2
        m2 = m[y0:y0 + H, x0:x0 + W]
        return ((m2 > 0).astype(np.uint8) * 255)
    m2 = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return ((m2 > 0).astype(np.uint8) * 255)


def rle_to_mask(rle: Dict, height: int, width: int) -> np.ndarray:
    """Robust RLE -> binary mask decoder (supports compressed and uncompressed)."""
    counts = rle.get("counts")
    if isinstance(counts, list):
        rle_obj = mask_utils.frPyObjects(rle, height, width)
        m = mask_utils.decode(rle_obj)
    else:
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        rle_c = {"counts": counts, "size": rle.get("size", [height, width])}
        m = mask_utils.decode(rle_c)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8)


def polys_to_mask(polys: List[List[float]], H: int, W: int) -> np.ndarray:
    if polys is None or len(polys) == 0:
        raise ValueError("empty polygon list")
    rles = mask_utils.frPyObjects(polys, H, W)
    m = mask_utils.decode(rles)
    if m.ndim == 3:
        m = np.any(m > 0, axis=2)
    else:
        m = (m > 0)
    return m.astype(np.uint8)


def ann_to_binary_mask(ann: Dict, H: int, W: int) -> Optional[np.ndarray]:
    seg = ann.get("segmentation", None)
    if seg is None:
        return None
    try:
        if isinstance(seg, dict) and "counts" in seg:
            logger.debug("[ann %s] decoding RLE mask (size=%s)", ann.get("id"), seg.get("size"))
            return rle_to_mask(seg, H, W)
        if isinstance(seg, list):
            logger.debug("[ann %s] decoding polygon mask (len=%d)", ann.get("id"), len(seg))
            return polys_to_mask(seg, H, W)
    except Exception as e:
        logger.warning("[ann %s] invalid segmentation; fallback to bbox: %s", ann.get("id"), e)
        return None
    return None

# -----------------------------------------------------------------------------
# Category & difficulty utilities
# -----------------------------------------------------------------------------

def is_face(ann: Dict, cats: Dict[int, str]) -> bool:
    if "label" in ann:
        return int(ann["label"]) == 0
    cname = cats.get(ann.get("category_id"), "").lower()
    return "face" in cname


def is_body(ann: Dict, cats: Dict[int, str]) -> bool:
    if "label" in ann:
        return int(ann["label"]) == 1
    cname = cats.get(ann.get("category_id"), "").lower()
    return "body" in cname


def get_difficulty(ann: Dict, default: str = "normal") -> str:
    if "difficulty" in ann:
        return str(ann["difficulty"]).lower()
    attrs = ann.get("attributes", {})
    if isinstance(attrs, dict) and "difficulty" in attrs:
        return str(attrs["difficulty"]).lower()
    return default

# -----------------------------------------------------------------------------
# Pairing face <-> body
# -----------------------------------------------------------------------------

def bbox_xywh_to_xyxy(b):
    x, y, w, h = b
    return (int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h)))


def bbox_contains_point(b, px, py):
    x1, y1, x2, y2 = bbox_xywh_to_xyxy(b)
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    denom = aw * ah + bw * bh - inter + 1e-6
    return inter / denom


def get_pair_key(ann: Dict, link_key: Optional[str]) -> Optional[int]:
    if not link_key:
        return None
    v = ann.get(link_key, None)
    if v is None:
        v = ann.get("attributes", {}).get(link_key, None)
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def find_body_for_face(face_ann: Dict, anns: List[Dict], cats: Dict[int, str], link_key: Optional[str]) -> Optional[Dict]:
    fk = get_pair_key(face_ann, link_key)
    if fk is not None:
        for a in anns:
            if is_body(a, cats) and get_pair_key(a, link_key) == fk:
                return a
    fx, fy, fw, fh = face_ann["bbox"]
    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    cand, best_iou = None, -1.0
    for a in anns:
        if not is_body(a, cats):
            continue
        if bbox_contains_point(a["bbox"], cx, cy):
            return a
        iou = iou_xywh(face_ann["bbox"], a["bbox"])
        if iou > best_iou:
            best_iou, cand = iou, a
    return cand


def find_face_for_body(body_ann: Dict, anns: List[Dict], cats: Dict[int, str], link_key: Optional[str]) -> Optional[Dict]:
    bk = get_pair_key(body_ann, link_key)
    if bk is not None:
        for a in anns:
            if is_face(a, cats) and get_pair_key(a, link_key) == bk:
                return a
    cand, best_iou = None, -1.0
    for a in anns:
        if not is_face(a, cats):
            continue
        fx, fy, fw, fh = a["bbox"]
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        if bbox_contains_point(body_ann["bbox"], cx, cy):
            return a
        iou = iou_xywh(a["bbox"], body_ann["bbox"])
        if iou > best_iou:
            best_iou, cand = iou, a
    return cand

# -----------------------------------------------------------------------------
# Pix2Gestalt helpers
# -----------------------------------------------------------------------------

def preprocess_256(img_rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pre_img = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    pre_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    return pre_img, pre_mask


def get_mask_from_pred(pred_image_rgb: np.ndarray, interface, thresholding=False) -> np.ndarray:
    """Extract amodal mask from Pix2Gestalt prediction using CarveKit.
    Falls back to thresholding if CarveKit fails or when thresholding=True."""
    arr = np.asarray(pred_image_rgb)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if thresholding or interface is None:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, pred_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        return pred_mask

    try:
        pil = Image.fromarray(arr, mode="RGB")
        amodal_rgba = np.asarray(interface([pil])[0])
        alpha = amodal_rgba[:, :, 3]
        return ((alpha > 0).astype(np.uint8) * 255)
    except Exception as e:
        logger.warning("CarveKit matting failed (%s); falling back to thresholding.", e)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, pred_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        return pred_mask


def run_pix2gestalt(preprocessed_image_256: np.ndarray,
                    target_size_image: np.ndarray,
                    visible_mask_256: np.ndarray,
                    model,
                    guidance_scale: float,
                    n_samples: int,
                    ddim_steps: int,
                    interface,
                    use_threshold: bool):
    # AMP + inference_mode for lower VRAM
    preds = None
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    with torch.inference_mode():
        if dev.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dt):
                preds = run_inference(preprocessed_image_256, visible_mask_256, model,
                                      guidance_scale, n_samples, ddim_steps)
        else:
            preds = run_inference(preprocessed_image_256, visible_mask_256, model,
                                  guidance_scale, n_samples, ddim_steps)
    H, W = target_size_image.shape[:2]
    pred0 = preds[0]
    pred_resized = cv2.resize(pred0, (W, H), interpolation=cv2.INTER_LANCZOS4)
    pred_mask = get_mask_from_pred(pred_resized, interface=interface, thresholding=use_threshold)
    return pred_resized, pred_mask

# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def overlay_mask_rgb(img_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.35) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    base = Image.fromarray(img_rgb).convert("RGBA")
    m = fit_mask_to(mask, h, w)
    m_img = Image.fromarray(m, mode="L")
    color_overlay = Image.new("RGBA", (w, h), color + (int(alpha * 255),))
    composed = Image.composite(color_overlay, base, m_img)
    return np.asarray(composed.convert("RGB"))


def save_side_by_side(left_rgb: np.ndarray, right_rgb: np.ndarray, title_left: str, title_right: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1); ax1.imshow(left_rgb); ax1.set_title(title_left, fontsize=10); ax1.axis("off")
    ax2 = plt.subplot(1, 2, 2); ax2.imshow(right_rgb); ax2.set_title(title_right, fontsize=10); ax2.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)


def composite_with_mask(canvas: np.ndarray, pred_img: np.ndarray, mask: np.ndarray,
                        mode: str = "overwrite", alpha: float = 1.0) -> np.ndarray:
    """Paste pred_img into canvas where mask > 0. If mode==alpha, blend."""
    H, W = canvas.shape[:2]
    pred = np.asarray(pred_img)
    if pred.ndim == 2:
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    if pred.shape[:2] != (H, W):
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LANCZOS4)
    if pred.dtype != canvas.dtype:
        pred = pred.astype(canvas.dtype)

    m = fit_mask_to(mask, H, W)
    mb = (m > 0)
    if not np.any(mb):
        return canvas

    if mode == "alpha":
        a = float(max(0.0, min(1.0, alpha)))
        mb3 = np.repeat(mb[..., None], 3, axis=2)
        canvas[mb3] = (a * pred[mb3] + (1 - a) * canvas[mb3]).astype(canvas.dtype)
    else:
        canvas[mb] = pred[mb]
    return canvas

# -----------------------------------------------------------------------------
# Extension logic
# -----------------------------------------------------------------------------

@dataclass
class ExtRecord:
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0


def is_mask_touch_canvas_boundary(mask: np.ndarray) -> Tuple[bool, bool, bool, bool]:
    H, W = mask.shape[:2]
    m = (mask > 0)
    left = m[:, 0].any()
    right = m[:, W - 1].any()
    top = m[0, :].any()
    bottom = m[H - 1].any()
    return bool(left), bool(right), bool(top), bool(bottom)


def reach_limit_for_touched_dirs(touched: Tuple[bool, bool, bool, bool], rec: ExtRecord, H0: int, W0: int) -> bool:
    tL, tR, tT, tB = touched
    no_left = (rec.left >= W0) if tL else False
    no_right = (rec.right >= W0) if tR else False
    no_top = (rec.top >= H0) if tT else False
    no_bot = (rec.bottom >= H0) if tB else False
    touched_any = any(touched)
    return touched_any and all([not tL or no_left, not tR or no_right, not tT or no_top, not tB or no_bot])


def extend_it(extended_img: np.ndarray,
              extended_mask: np.ndarray,
              rec: ExtRecord,
              H0: int, W0: int,
              extension_rate: float,
              pad_val_img: int,
              pad_val_mask: int) -> Tuple[np.ndarray, np.ndarray, ExtRecord, Tuple[bool, bool, bool, bool]]:
    tL, tR, tT, tB = is_mask_touch_canvas_boundary(extended_mask)
    H, W = extended_img.shape[:2]

    add_left = min(int(round(W * extension_rate)), max(0, W0 - rec.left)) if tL else 0
    add_right = min(int(round(W * extension_rate)), max(0, W0 - rec.right)) if tR else 0
    add_top = min(int(round(H * extension_rate)), max(0, H0 - rec.top)) if tT else 0
    add_bottom = min(int(round(H * extension_rate)), max(0, H0 - rec.bottom)) if tB else 0

    if add_left == add_right == add_top == add_bottom == 0:
        return extended_img, extended_mask, rec, (tL, tR, tT, tB)

    img = np.pad(
        extended_img,
        pad_width=((add_top, add_bottom), (add_left, add_right), (0, 0)),
        mode="constant",
        constant_values=pad_val_img,
    )
    msk = np.pad(
        extended_mask,
        pad_width=((add_top, add_bottom), (add_left, add_right)),
        mode="constant",
        constant_values=pad_val_mask,
    )

    rec.left += add_left
    rec.right += add_right
    rec.top += add_top
    rec.bottom += add_bottom
    return img, msk, rec, (tL, tR, tT, tB)


def extend_all_direction(extended_img: np.ndarray,
                         extended_mask: np.ndarray,
                         rec: ExtRecord,
                         H0: int, W0: int,
                         pad_val_img: int,
                         pad_val_mask: int) -> Tuple[np.ndarray, np.ndarray]:
    add_left = max(0, W0 - rec.left)
    add_right = max(0, W0 - rec.right)
    add_top = max(0, H0 - rec.top)
    add_bottom = max(0, H0 - rec.bottom)
    out_img = np.pad(
        extended_img,
        pad_width=((add_top, add_bottom), (add_left, add_right), (0, 0)),
        mode="constant",
        constant_values=pad_val_img,
    )
    out_mask = np.pad(
        extended_mask,
        pad_width=((add_top, add_bottom), (add_left, add_right)),
        mode="constant",
        constant_values=pad_val_mask,
    )
    return out_img, out_mask

# -----------------------------------------------------------------------------
# Per-annotation routine (composites predicted pixels back!)
# -----------------------------------------------------------------------------

def run_one_ann(
    img_rgb: np.ndarray,
    ann: Dict,
    anns_in_image: List[Dict],
    cats: Dict[int, str],
    H0: int, W0: int,
    *,
    model,
    link_key: Optional[str],
    extension_rate: float,
    pad_val_img: int,
    pad_val_mask: int,
    guidance_scale: float,
    n_samples: int,
    ddim_steps: int,
    out_dir: Optional[Path],
    verbose: bool,
    step_prefix: str,
    carvekit_interface,
    always_run: bool,
    save_trace: bool,
    trace_dir: Optional[Path],
    use_threshold_mask: bool,
) -> Optional[Dict]:
    ann_id = ann.get("id", None)
    selected_kind = "face" if is_face(ann, cats) else "body"
    logger.info("[ann %s] start kind=%s", ann_id, selected_kind)

    # --- selection gates ---
    if is_face(ann, cats):
        diff = get_difficulty(ann)
        logger.debug("[ann %s] face difficulty=%s", ann_id, diff)
        if diff in ["normal", "hard"]:
            logger.info("[ann %s] SKIP: face difficulty in {normal,hard}", ann_id)
            return None
        body_ann = find_body_for_face(ann, anns_in_image, cats, link_key)
        if body_ann is not None:
            logger.debug("[ann %s] paired body ann id=%s", ann_id, body_ann.get("id"))
            use_ann = body_ann
            used_pair_ids = [body_ann.get("id", -1)]
        else:
            logger.debug("[ann %s] no paired body, using face ann itself", ann_id)
            use_ann = ann
            used_pair_ids = []
    else:
        face_ann = find_face_for_body(ann, anns_in_image, cats, link_key)
        if face_ann is None:
            logger.info("[ann %s] SKIP: body ann has no paired face", ann_id)
            return None
        face_diff = get_difficulty(face_ann)
        logger.debug("[ann %s] paired face difficulty=%s", ann_id, face_diff)
        if face_diff in ["normal", "hard"]:
            logger.info("[ann %s] SKIP: paired face difficulty in {normal,hard}", ann_id)
            return None
        use_ann = ann
        used_pair_ids = [face_ann.get("id", -1)]

    # --- initial mask ---
    H, W = img_rgb.shape[:2]
    mask = ann_to_binary_mask(use_ann, H, W)
    if mask is None:
        x1, y1, x2, y2 = bbox_xywh_to_xyxy(use_ann["bbox"])
        m = np.zeros((H, W), dtype=np.uint8)
        m[y1:y2, x1:x2] = 255
        mask = m
    mask = fit_mask_to(mask, H, W)

    # --- setup state ---
    ext_img = img_rgb.copy()
    ext_mask = mask.copy()
    rec = ExtRecord(0, 0, 0, 0)

    # per-ann verbose dir
    vdir = None
    if (verbose or save_trace) and trace_dir is not None:
        vdir = trace_dir / f"{step_prefix}_ann{ann_id if ann_id is not None else 'x'}"
        vdir.mkdir(parents=True, exist_ok=True)

    if vdir is not None:
        init_over = overlay_mask_rgb(ext_img, ext_mask, color=(255, 0, 0), alpha=0.35)
        Image.fromarray(init_over).save(vdir / "initial.png")

    trace = {"ann_id": ann_id, "kind": selected_kind, "used_pair_ids": used_pair_ids,
             "steps": [], "pix2gestalt_calls": 0, "forced_run": False}

    # --- extend loop ---
    step_idx = 0
    touched = is_mask_touch_canvas_boundary(ext_mask)
    logger.debug("[ann %s] initial touch flags L,R,T,B=%s", ann_id, touched)

    last_pred_img = None
    last_pred_mask = None

    while True:
        if not any(touched):
            logger.info("[ann %s] mask not touching; extension loop stops", ann_id)
            break
        if reach_limit_for_touched_dirs(touched, rec, H0, W0):
            logger.info("[ann %s] reach_limit_all_direction true; stop extending", ann_id)
            break

        prev_shape = ext_img.shape
        ext_img, ext_mask, rec, touched_after = extend_it(
            ext_img, ext_mask, rec, H0, W0,
            extension_rate=extension_rate,
            pad_val_img=pad_val_img,
            pad_val_mask=pad_val_mask,
        )
        logger.debug("[ann %s] extend from %s to %s; rec=%s", ann_id, prev_shape, ext_img.shape, rec.__dict__)

        try:
            pre_img, pre_mask = preprocess_256(ext_img, ext_mask)
            pred_img, pred_amodal_mask = run_pix2gestalt(
                pre_img, ext_img, pre_mask,
                model=model,
                guidance_scale=guidance_scale,
                n_samples=n_samples,
                ddim_steps=ddim_steps,
                interface=carvekit_interface,
                use_threshold=use_threshold_mask,
            )
            trace["pix2gestalt_calls"] += 1
        except Exception as e:
            logger.error("[ann %s] Pix2Gestalt failed: %s\n%s", ann_id, e, traceback.format_exc())
            if vdir is not None:
                with open(vdir / "error.txt", "w") as f:
                    f.write(str(e) + "\n" + traceback.format_exc())
            break

        # >>> paste predicted pixels into the extended canvas <<<
        ext_img = composite_with_mask(ext_img, pred_img, pred_amodal_mask, mode="overwrite", alpha=1.0)

        # update mask for next iteration
        ext_mask = pred_amodal_mask
        last_pred_img = pred_img
        last_pred_mask = pred_amodal_mask

        if vdir is not None:
            # Save raw prediction and the canvas-after-composite (no overlays)
            try:
                Image.fromarray(np.asarray(pred_img)).save(vdir / f"step_{step_idx:02d}_pred.png")
                Image.fromarray(np.asarray(ext_img)).save(vdir / f"step_{step_idx:02d}_canvas.png")
            except Exception as e:
                logger.warning("[ann %s] saving step preds failed: %s", ann_id, e)
            # Side-by-side with overlays for quick QA
            over_ext = overlay_mask_rgb(ext_img, ext_mask, color=(255, 0, 0), alpha=0.35)
            over_pred = overlay_mask_rgb(pred_img, pred_amodal_mask, color=(0, 128, 255), alpha=0.35)
            save_side_by_side(over_ext, over_pred,
                              title_left=f"Step {step_idx}: Canvas after composite",
                              title_right=f"Step {step_idx}: Pix2Gestalt pred",
                              out_path=vdir / f"step_{step_idx:02d}.png")

        trace["steps"].append({
            "idx": step_idx,
            "shape": list(ext_img.shape),
            "rec": rec.__dict__.copy(),
            "touched_after": tuple(map(bool, touched_after)),
            "mask_area": int((ext_mask > 0).sum()),
        })

        step_idx += 1
        touched = is_mask_touch_canvas_boundary(ext_mask)
        logger.debug("[ann %s] post-step touch flags=%s", ann_id, touched)

    # Optionally force one pass if nothing ran
    if step_idx == 0 and always_run:
        logger.info("[ann %s] always_run=True -> forcing one Pix2Gestalt pass", ann_id)
        try:
            pre_img, pre_mask = preprocess_256(ext_img, ext_mask)
            pred_img, pred_amodal_mask = run_pix2gestalt(
                pre_img, ext_img, pre_mask,
                model=model,
                guidance_scale=guidance_scale,
                n_samples=n_samples,
                ddim_steps=ddim_steps,
                interface=carvekit_interface,
                use_threshold=use_threshold_mask,
            )
            ext_img = composite_with_mask(ext_img, pred_img, pred_amodal_mask, mode="overwrite", alpha=1.0)
            ext_mask = pred_amodal_mask
            last_pred_img = pred_img
            last_pred_mask = pred_amodal_mask
            trace["pix2gestalt_calls"] += 1
            trace["forced_run"] = True
            if vdir is not None:
                Image.fromarray(np.asarray(pred_img)).save(vdir / "step_00_forced_pred.png")
                Image.fromarray(np.asarray(ext_img)).save(vdir / "step_00_forced_canvas.png")
        except Exception as e:
            logger.error("[ann %s] forced Pix2Gestalt failed: %s\n%s", ann_id, e, traceback.format_exc())

    # Pad to exact 3H×3W (centered original)
    final_img, final_mask = extend_all_direction(ext_img, ext_mask, rec, H0, W0,
                                                pad_val_img=pad_val_img, pad_val_mask=pad_val_mask)

    if vdir is not None:
        over_final = overlay_mask_rgb(final_img, final_mask, color=(255, 0, 0), alpha=0.35)
        Image.fromarray(over_final).save(vdir / "final.png")
        # Also save the last Pix2Gestalt raw prediction for your inspection
        if last_pred_img is not None:
            Image.fromarray(np.asarray(last_pred_img)).save(vdir / "final_pred.png")
        if save_trace:
            import json as _json
            with open(vdir / "trace.json", "w") as f:
                _json.dump(trace, f, indent=2)

    logger.info("[ann %s] done: steps=%d, pix2gestalt_calls=%d, final_shape=%s",
                ann_id, step_idx, trace["pix2gestalt_calls"], final_img.shape)

    return {
        "ann_id": ann_id,
        "used_ids": [x for x in used_pair_ids if x is not None],
        "rec": {"left": rec.left, "right": rec.right, "top": rec.top, "bottom": rec.bottom},
        "final_img": final_img,
        "final_mask": final_mask,
    }

# -----------------------------------------------------------------------------
# Merging
# -----------------------------------------------------------------------------

def pad_original_to_3x(image_rgb: np.ndarray, pad_val_img: int) -> np.ndarray:
    H0, W0 = image_rgb.shape[:2]
    return np.pad(image_rgb, ((H0, H0), (W0, W0), (0, 0)), mode="constant", constant_values=pad_val_img)


def merge_images_with_masks(extended_images: List[np.ndarray],
                            extended_masks: List[np.ndarray],
                            base_image_3x: np.ndarray,
                            composite: str = "overwrite",
                            alpha: float = 1.0) -> np.ndarray:
    """Merge a list of extended images into base_image_3x using their masks."""
    out = base_image_3x.copy()
    for img, m in zip(extended_images, extended_masks):
        assert img.shape[:2] == out.shape[:2], "All extended images must share the same H×W"
        mask = m[..., 0] if (m.ndim == 3) else m
        mb = (mask > 0)
        if not np.any(mb):
            continue
        if composite == "overwrite":
            out[mb] = img[mb]
        elif composite == "alpha":
            a = float(max(0.0, min(1.0, alpha)))
            mb3 = np.repeat(mb[..., None], 3, axis=2)
            out[mb3] = (a * img[mb3] + (1 - a) * out[mb3]).astype(out.dtype)
        else:
            raise ValueError(f"Unknown composite mode: {composite}")
    return out

# -----------------------------------------------------------------------------
# CLI utils
# -----------------------------------------------------------------------------

def parse_indices(spec: Optional[str]) -> List[int]:
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1); a = int(a); b = int(b)
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    seen, res = set(), []
    for i in out:
        if i not in seen:
            res.append(i); seen.add(i)
    return res


def parse_gpu_ids(spec: Optional[str]) -> List[int]:
    if not spec:
        return []
    return parse_indices(spec)


def shard_round_robin(n_items: int, n_buckets: int) -> List[List[int]]:
    buckets = [[] for _ in range(max(1, n_buckets))]
    for i in range(n_items):
        buckets[i % max(1, n_buckets)].append(i)
    return buckets

# -----------------------------------------------------------------------------
# Model prep (low VRAM)
# -----------------------------------------------------------------------------

def _dtype_from_flag(flag: str):
    if flag == "fp16":  return torch.float16
    if flag == "bf16":  return torch.bfloat16
    return torch.float32

def prepare_model(cfg, ckpt, device_str: str, precision: str, channels_last: bool):
    # Load checkpoint/model on CPU first to avoid CUDA deserialization OOM.
    model = load_model_from_config(cfg, ckpt, "cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    dtype = _dtype_from_flag(precision)
    model = model.to(device=device_str, dtype=dtype)
    return model

# -----------------------------------------------------------------------------
# Worker (per-GPU) entry
# -----------------------------------------------------------------------------

def worker_main(worker_rank: int, gpu_local: int, index_list: List[int], args):
    # isolate a single GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local)

    # allocator hint (helps fragmentation)
    if args.max_split_size_mb and args.max_split_size_mb > 0:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.max_split_size_mb}"

    logger.info("[worker %d] start on GPU local=%s, indices=%s",
                worker_rank, gpu_local,
                index_list[:5] + (["..."] if len(index_list) > 5 else []))

    device = "cuda:0"
    cfg = OmegaConf.load(args.config)
    model = prepare_model(cfg, args.ckpt, device, args.precision, args.channels_last)

    carvekit_interface = None
    if args.mask_extractor == "carvekit":
        try:
            carvekit_interface = create_carvekit_interface()
        except Exception as e:
            logger.warning("create_carvekit_interface failed (%s); falling back to thresholding.", e)
            carvekit_interface = None

    coco = load_coco(args.coco)
    cats = category_id_to_name(coco)
    ann_map = anns_by_image(coco)
    images = coco["images"]
    out_root = Path(args.out_dir)

    for gi in index_list:
        # boundary checks
        if gi < 0 or gi >= len(images):
            continue
        im = images[gi]
        img_id = im["id"]
        fn = im["file_name"]
        path = Path(args.images_dir) / fn
        bgr = cv2.imread(str(path))
        if bgr is None:
            logger.warning("[worker %d] failed to read image: %s", worker_rank, path)
            continue
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]

        anns = ann_map.get(img_id, [])
        used_ann_ids: set = set()

        sub = out_root / f"img_{gi:05d}_id{img_id}" / "work"
        sub.mkdir(parents=True, exist_ok=True)

        final_imgs: List[np.ndarray] = []
        final_masks: List[np.ndarray] = []

        for k, ann in enumerate(anns):
            a_id = ann.get("id", None)
            if a_id is not None and a_id in used_ann_ids:
                continue

            # gates
            if is_face(ann, cats):
                diff = get_difficulty(ann)
                if diff in ["normal", "hard"]:
                    logger.info("[img %s ann %s] skip face: difficulty=%s", img_id, a_id, diff)
                    continue
            else:
                face_ann = find_face_for_body(ann, anns, cats, args.link_key)
                if face_ann is None:
                    logger.info("[img %s ann %s] skip body: no paired face", img_id, a_id)
                    continue
                face_diff = get_difficulty(face_ann)
                if face_diff in ["normal", "hard"]:
                    logger.info("[img %s ann %s] skip body: paired face difficulty=%s", img_id, a_id, face_diff)
                    continue

            res = run_one_ann(
                img_rgb=img_rgb,
                ann=ann,
                anns_in_image=anns,
                cats=cats,
                H0=H0, W0=W0,
                model=model,
                link_key=args.link_key,
                extension_rate=args.extension_rate,
                pad_val_img=args.pad_val_img,
                pad_val_mask=args.pad_val_mask,
                guidance_scale=args.guidance_scale,
                n_samples=args.n_samples,
                ddim_steps=args.ddim_steps,
                out_dir=sub,
                verbose=False,  # work mode in worker
                step_prefix=f"img{gi:05d}_k{k:03d}",
                carvekit_interface=carvekit_interface,
                always_run=args.always_run,
                save_trace=False,
                trace_dir=sub,
                use_threshold_mask=(args.mask_extractor == "threshold"),
            )

            if res is not None:
                if a_id is not None:
                    used_ann_ids.add(a_id)
                for uid in res.get("used_ids", []):
                    if uid is not None:
                        used_ann_ids.add(uid)
                final_imgs.append(res["final_img"])  # H3×W3×3
                final_masks.append(res["final_mask"])  # H3×W3 (0/255)

        base_3x = pad_original_to_3x(img_rgb, pad_val_img=args.pad_val_img)

        ok = True
        for fi, fm in zip(final_imgs, final_masks):
            if fi.shape[:2] != base_3x.shape[:2] or fm.shape[:2] != base_3x.shape[:2]:
                ok = False; break
        if not ok:
            logger.warning("[worker %d] Skip merge for image idx=%d, shapes mismatch.", worker_rank, gi)
            continue

        merged = merge_images_with_masks(final_imgs, final_masks, base_3x,
                                         composite=args.composite, alpha=args.alpha)

        Image.fromarray(base_3x).save(sub / "base_3x.png")
        Image.fromarray(merged).save(sub / "merged_final.png")

        if final_masks:
            union = np.zeros(base_3x.shape[:2], dtype=np.uint8)
            for m in final_masks:
                union |= (m > 0).astype(np.uint8) * 255
            cv2.imwrite(str(sub / "merged_union_mask.png"), union)

        # help allocator fragmentation between images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("[worker %d] done", worker_rank)

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extend-to-3x + Pix2Gestalt with multi-ann merge (multi-GPU work mode, low-VRAM)")
    # Modes
    ap.add_argument("--mode", choices=["verbose", "sanity", "work"], default="sanity")

    # COCO + images
    ap.add_argument("--coco", required=True, help="Path to COCO JSON")
    ap.add_argument("--images-dir", required=True, help="Directory containing images")
    ap.add_argument("--link-key", default=None, help="Pairing key (e.g., person_id)")

    # Pix2Gestalt
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device-idx", type=int, default=0, help="Single-GPU device index for sanity/verbose")
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--ddim-steps", type=int, default=250)

    # Extension config
    ap.add_argument("--extension-rate", type=float, default=0.25)
    ap.add_argument("--pad-val-img", type=int, default=0)
    ap.add_argument("--pad-val-mask", type=int, default=0)

    # Verbosity / Debug
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    ap.add_argument("--always-run", action="store_true", help="Force one Pix2Gestalt pass if loop never triggers")
    ap.add_argument("--save-trace", action="store_true", help="Save per-ann trace.json (verbose/sanity)")

    # Merge config
    ap.add_argument("--composite", choices=["overwrite", "alpha"], default="overwrite")
    ap.add_argument("--alpha", type=float, default=1.0, help="alpha for composite=alpha")

    # Output
    ap.add_argument("--out-dir", default="./pix2gestalt_ext")

    # Selection for non-work modes
    ap.add_argument("--image-ids", default="", help="Comma/ranges over COCO images array, e.g., '0,2,5-8'")

    # Multi-GPU controls (work mode only)
    ap.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use in work mode (one proc/GPU)")
    ap.add_argument("--gpu-ids", default="", help="Override visible GPU ids (e.g., '0,1,2,3'); if empty uses 0..num_gpus-1")

    # Low-VRAM knobs
    ap.add_argument("--precision", choices=["fp32","fp16","bf16"], default="fp16",
                    help="Inference dtype. A100 often works best with bf16.")
    ap.add_argument("--channels-last", action="store_true",
                    help="Use channels_last memory format (Ampere/Ada).")
    ap.add_argument("--max-split-size-mb", type=int, default=64,
                    help="Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:<N> to reduce fragmentation.")
    ap.add_argument("--mask-extractor", choices=["carvekit","threshold"], default="carvekit",
                    help="Use CarveKit matting or a simple threshold (no extra model, lowest VRAM).")

    args = ap.parse_args()

    # logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    logger.info("Args: %s", vars(args))

    # allocator hint (helps fragmentation)
    if args.max_split_size_mb and args.max_split_size_mb > 0:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.max_split_size_mb}"

    if args.mode in ["verbose", "sanity"] or args.num_gpus <= 1:
        # -------------------- single-GPU path (original behavior) --------------------
        device = f"cuda:{args.device_idx}"
        cfg = OmegaConf.load(args.config)
        model = prepare_model(cfg, args.ckpt, device, args.precision, args.channels_last)

        carvekit_interface = None
        if args.mask_extractor == "carvekit":
            try:
                carvekit_interface = create_carvekit_interface()
            except Exception as e:
                logger.warning("create_carvekit_interface failed (%s); falling back to thresholding.", e)
                carvekit_interface = None

        coco = load_coco(args.coco)
        cats = category_id_to_name(coco)
        ann_map = anns_by_image(coco)
        images = coco["images"]

        out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

        if args.mode in ["verbose", "sanity"]:
            idxs = parse_indices(args.image_ids)
            if not idxs:
                logger.warning("No --image-ids provided; defaulting to index 0")
                idxs = [0]
            selected = [(j, images[j]) for j in idxs if 0 <= j < len(images)]
        else:
            # work mode but num_gpus<=1 → sequential over all images
            selected = list(enumerate(images))

        iterator = tqdm(selected, desc="Processing images") if args.mode == "work" else selected

        for enum_idx, im in iterator:
            img_id = im["id"]
            fn = im["file_name"]
            path = Path(args.images_dir) / fn
            bgr = cv2.imread(str(path))
            if bgr is None:
                logger.warning("Failed to read image: %s", path)
                continue
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H0, W0 = img_rgb.shape[:2]
            anns = ann_map.get(img_id, [])
            used_ann_ids: set = set()

            sub = out_root / f"img_{enum_idx:05d}_id{img_id}" / ("verbose" if args.mode=="verbose" else ("sanity" if args.mode=="sanity" else "work"))
            sub.mkdir(parents=True, exist_ok=True)

            final_imgs: List[np.ndarray] = []
            final_masks: List[np.ndarray] = []

            for k, ann in enumerate(anns):
                a_id = ann.get("id", None)
                if a_id is not None and a_id in used_ann_ids:
                    continue

                # gates
                if is_face(ann, cats):
                    diff = get_difficulty(ann)
                    if diff in ["normal", "hard"]:
                        logger.info("[img %s ann %s] skip face: difficulty=%s", img_id, a_id, diff)
                        continue
                else:
                    face_ann = find_face_for_body(ann, anns, cats, args.link_key)
                    if face_ann is None:
                        logger.info("[img %s ann %s] skip body: no paired face", img_id, a_id)
                        continue
                    face_diff = get_difficulty(face_ann)
                    if face_diff in ["normal", "hard"]:
                        logger.info("[img %s ann %s] skip body: paired face difficulty=%s", img_id, a_id, face_diff)
                        continue

                res = run_one_ann(
                    img_rgb=img_rgb,
                    ann=ann,
                    anns_in_image=anns,
                    cats=cats,
                    H0=H0, W0=W0,
                    model=model,
                    link_key=args.link_key,
                    extension_rate=args.extension_rate,
                    pad_val_img=args.pad_val_img,
                    pad_val_mask=args.pad_val_mask,
                    guidance_scale=args.guidance_scale,
                    n_samples=args.n_samples,
                    ddim_steps=args.ddim_steps,
                    out_dir=sub,
                    verbose=(args.mode=="verbose"),
                    step_prefix=f"img{enum_idx:05d}_k{k:03d}",
                    carvekit_interface=carvekit_interface,
                    always_run=args.always_run,
                    save_trace=args.save_trace and (args.mode!="work"),
                    trace_dir=sub,
                    use_threshold_mask=(args.mask_extractor == "threshold"),
                )

                if res is not None:
                    if a_id is not None:
                        used_ann_ids.add(a_id)
                    for uid in res.get("used_ids", []):
                        if uid is not None:
                            used_ann_ids.add(uid)
                    final_imgs.append(res["final_img"])  # H3×W3×3
                    final_masks.append(res["final_mask"])  # H3×W3 (0/255)

            base_3x = pad_original_to_3x(img_rgb, pad_val_img=args.pad_val_img)

            ok = True
            for fi, fm in zip(final_imgs, final_masks):
                if fi.shape[:2] != base_3x.shape[:2] or fm.shape[:2] != base_3x.shape[:2]:
                    ok = False; break
            if not ok:
                logger.warning("Skip merge for image idx=%d, shapes mismatch.", enum_idx)
                continue

            merged = merge_images_with_masks(final_imgs, final_masks, base_3x,
                                             composite=args.composite, alpha=args.alpha)

            Image.fromarray(base_3x).save(sub / "base_3x.png")
            Image.fromarray(merged).save(sub / "merged_final.png")

            if final_masks:
                union = np.zeros(base_3x.shape[:2], dtype=np.uint8)
                for m in final_masks:
                    union |= (m > 0).astype(np.uint8) * 255
                cv2.imwrite(str(sub / "merged_union_mask.png"), union)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    else:
        # -------------------- multi-GPU path (work mode) --------------------
        coco = load_coco(args.coco)
        images = coco["images"]
        n_imgs = len(images)
        if n_imgs == 0:
            logger.info("No images to process.")
            return

        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if not gpu_ids:
            gpu_ids = list(range(max(1, args.num_gpus)))
        num_workers = min(len(gpu_ids), n_imgs)
        if num_workers <= 1:
            logger.info("Falling back to single-GPU sequential processing.")
            args.num_gpus = 1
            all_indices = list(range(n_imgs))
            worker_main(0, gpu_ids[0] if gpu_ids else 0, all_indices, args)
            logger.info("[INFO] Done. Results saved under: %s", Path(args.out_dir).resolve())
            return

        shards = shard_round_robin(n_imgs, num_workers)
        logger.info("Launching %d workers over GPUs %s", num_workers, gpu_ids[:num_workers])

        procs: List[mp.Process] = []
        for rank in range(num_workers):
            p = mp.Process(target=worker_main, args=(rank, gpu_ids[rank], shards[rank], args), daemon=False)
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    logger.info("[INFO] Done. Results saved under: %s", Path(args.out_dir).resolve())


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()