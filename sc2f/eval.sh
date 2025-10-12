 python eval.py \
  --model_list_json /home/u7707452/Projects/eaft/eval/config/casual.json \
  --thresholds_json /home/u7707452/Projects/eaft/eval/config/nms_threshold3.json \
  --val_img_folder /home/u7707452/Projects/eaft/data/amodal_val_2024_fb_v2 \
  --val_ann_file /home/u7707452/Projects/eaft/data/annotation/coco_amodal_fb_val_v2_classified_rb.json \
  --img_size 320 \
  --cocoeval_iouthr 0.25