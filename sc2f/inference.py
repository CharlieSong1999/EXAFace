import argparse, os, math, cv2, torch
import numpy as np
from models.yoloh import build_model_noargs
from data.transforms import ValTransforms
from config.yoloh_config_expand import yoloh_config
from utils.inference_utils import (
    build_DA_with_padding, apply_conf_nms_class_aware,
    overlay_heatmap, overlay_boxes, parse_model_outputs, rebuild_probs_from_logits
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(config_name, weight_path):
    cfg = yoloh_config[config_name]
    model = build_model_noargs(
        cfg=cfg, device=DEVICE, num_classes=2, trainable=False,
        coco_pretrained=weight_path, fig_size=(320,320),
        conf_thresh=0.18, nms_thresh=0.1, topk=1000
    ).to(DEVICE).eval()

    transform = ValTransforms(
        min_size=320, max_size=320,
        pixel_mean=cfg["pixel_mean"],
        pixel_std=cfg["pixel_std"],
        trans_config=cfg.get("val_transform", None),
        format=cfg["format"]
    )
    return model, transform


def main(args):
    # 1. Load and preprocess image
    img_bgr = cv2.imread(args.img_path)
    H, W = img_bgr.shape[:2]
    px, py, w, h = (W - 320) // 2, (H - 320) // 2, 320, 320

    DA_bgr, Is_bgr, _ = build_DA_with_padding(img_bgr, px, py, h, w, dark_ratio=0.0)

    # 2. Load model and transform
    model, transform = load_model(config_name="yoloh_expand-50-DC5-640-expand-attn-2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2", weight_path=args.weight_path)

    # 3. Apply transform to Is
    x = transform(Is_bgr)[0].unsqueeze(0).to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        out = model(x)
        heat_logits = model.heatmap(x)

    bboxes, scores, labels = parse_model_outputs(out)
    if bboxes.shape[0] > 0:
        bboxes[:, [0,2]] *= 3*w
        bboxes[:, [1,3]] *= 3*h
    else:
        bboxes = np.zeros((0,4), dtype=np.float32)

    # 5. Map to DA coords
    # bboxes[:, [0,2]] += w
    # bboxes[:, [1,3]] += h

    # 6. Heatmap
    probs_up = rebuild_probs_from_logits(heat_logits, cls_idx=0, W_DA=3*w, H_DA=3*h)
    da_overlay = overlay_heatmap(DA_bgr, probs_up, alpha=0.5)

    # 7. Final overlay with boxes
    da_final = overlay_boxes(da_overlay, bboxes, scores=scores, color=(0,255,0), thickness=2)

    # 8. Save output
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    cv2.imwrite(args.out_path, da_final)
    print(f"[✓] Saved: {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single-image inference")
    parser.add_argument("--weight_path", required=True, help="Path to .pth model weights")
    parser.add_argument("--img_path", required=True, help="Path to input RGB image")
    parser.add_argument("--out_path", required=True, help="Path to save output (PNG)")

    args = parser.parse_args()
    main(args)