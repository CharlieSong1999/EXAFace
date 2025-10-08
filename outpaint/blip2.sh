 python blip2_coco_outpainting_captions.py \
  --ann /data/anu_unseen/data/annotations/coco_amodal_fb_val_v2_classified_with_source_mask.json \
  --img-root /data/anu_unseen/data/amodal_val_2024_fb_v2 \
  --source-img-root /data/anu_unseen/data/val2017 \
  --source-pattern "{id:012d}.jpg" \
  --mode work \
  --seed 0 \
  --out /data/anu_unseen/data/annotations/amodal_val_2024_fb_v2_captions10.jsonl \
  --gpus 0,1,2,3 --num-workers 4