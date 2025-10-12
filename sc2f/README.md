# Extreme amodal face detection

Extreme amodal detection is the task of inferring the 2D location of objects that are not fully visible in the input image but are visible within an expanded field-of-view. This differs from amodal detection, where the object is partially visible within the input image, but is occluded. In this paper, we consider the sub-problem of face detection, since this class provides motivating applications involving safety and privacy, but do not tailor our method specifically to this class. Existing approaches rely on image sequences so that missing detections may be interpolated from surrounding frames or make use of generative models to sample possible completions. In contrast, we consider the single-image task and propose a more efficient, sample-free approach that makes use of the contextual cues from the image to infer the presence of unseen faces. We design a heatmap-based extreme amodal object detector that addresses the problem of efficiently predicting a lot (the out-of-frame region) from a little (the image) with a selective coarse-to-fine decoder. Our method establishes strong results for this new task, even outperforming less efficient generative approaches.

# Train

Environment setup:
```bash
conda create envName python==3.11
conda activate envName
pip install -r requirements.txt
```

If you use slurm, change the paths and settings accordingly and run
```bash
sbatch train.slurm
```

or 

```bash
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py \
            --cuda -dist --num_gpu 2 \
            -d coco_fb_diff \
            --train_img_folder /path/to/train_folder \
            --train_ann_file /path/to/train_ann \
            --val_img_folder /path/to/val_folder \
            --val_ann_file /path/to/val_ann \
            -v yoloh_expand-50-DC5-640-expand-attn-2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2 \
            -lr 0.024 -lr_bk 0.004 \
            --half_precision --reduce_steps 100 --eval_epoch 8 \
            --batch_size 32 --subset_ratio 1.0 \
            --train_min_size 320 --train_max_size 320 \
            --val_min_size 320 --val_max_size 320 \
            --skip_attention_map --manual_max_epoch 8 \
            --schedule 1x --grad_clip_norm 4.0 \
            --save_folder /path/to/save_folder \
            --wandb_token your_wandb_token_or_delete_it \
            --exp_name delete_it_if_wandb_not_use \
            -p /path/to/yoloh.pth
```

# Eval

Change the path in `eval.sh` and run
```bash
sh eval.sh
```

# Inference

Coming soon...

# Comparsion methods

Coming soon...


