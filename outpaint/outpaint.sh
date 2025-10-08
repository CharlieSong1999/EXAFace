  python sdxl_outpaint_batch.py \
  --val_img_folder /data/anu_unseen/data/amodal_val_2024_fb_v2 \
  --captions /data/anu_unseen/data/annotations/amodal_val_2024_fb_v2_captions10.jsonl \
  --out_dir /data/anu_unseen/data/outpaint10_amodal_val_2024_fb_v2 \
  --device cuda \
  --seed 0 \
  --gpus 0,1,2,3 \
  --work_indices "5601-7593" \
  --steps 35 --guidance 5.0 --mask_blur 0 \
  --work_long_side 1024 --paste_original_center