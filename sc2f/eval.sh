 python eval.py \
  --config_name "yoloh_expand-50-DC5-640-expand-attn-2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2" \
  --weight_path /path/to/weight.pth \
  --thresholds_json ./config/eval.json \
  --val_img_folder /path/to/eval_images \
  --val_ann_file /path/to/val_annotations.json \
  --img_size 320 \
  --cocoeval_iouthr 0.25