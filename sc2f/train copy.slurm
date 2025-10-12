#!/bin/bash 
#SBATCH --job-name=yoloh-attn-fine-tune-2enc-2dec-all-dp-320-tconvdecoder-lq0-C2F-noposvalue-rope-residual-tconv-2scale2-1x # Job name 
#SBATCH --mail-type=END,FAIL # Mail events 
#SBATCH --mail-user=changlin.song@anu.edu.au # Where to send mail 
#SBATCH --ntasks-per-node=2 # Run on 2 CPU 
#SBATCH --mem=48gb # Job memory request 
#SBATCH --time=120:00:00 # Time limit hrs:min:sec 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3090:2
#SBATCH --output=./slog/yoloh-attn-fine-tune-2enc-2dec-all-dp-320-tconvdecoder-lq0-C2F-noposvalue-rope-residual-tconv-2scale2-1x-%j.log # Standard output and error log

pwd; hostname; date 
nvidia-smi
echo "--- Pytorch Training ---" 
srun --account=optimalvi+ singularity exec --nv --bind /home/projects /opt/apps/containers/conda/conda-nvidia-22.04-latest.sif python train.py \
--cuda \
-dist \
--num_gpu 2 \
-d coco_fb_diff \
--train_img_folder /home/projects/u7707452/data/coco/amodal_train_2024_fb_v2_withSource \
--train_ann_file /home/projects/u7707452/data/coco/annotations/coco_amodal_fb_train_v2_withSource.json \
--val_img_folder /home/projects/u7707452/data/coco/amodal_val_2024_fb_v2 \
--val_ann_file /home/projects/u7707452/data/coco/annotations/coco_amodal_fb_val_v2_classified_rb.json \
-v yoloh_expand-50-DC5-640-expand-attn-2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2 \
-lr 0.024 \
-lr_bk 0.004 \
--half_precision \
--reduce_steps 100 \
--eval_epoch 2 \
--batch_size 64 \
--train_min_size 320 \
--train_max_size 320 \
--val_min_size 320 \
--val_max_size 320 \
--schedule 1x \
--manual_max_epoch 8 \
--grad_clip_norm 4.0 \
--save_folder /home/projects/u7707452/models/yoloh \
--wandb_token 469e207a63db0a4817bff81ecbcd9187be656e40 \
--exp_name 2enc-2dec-all-dp-320-tconvdecoder-lq0-C2F-noposvalue-rope-residual-tconv-2scale2-1x \
-p /home/projects/u7707452/models/yoloh/coco_fb/yoloh50-DC5-640/yoloh50-DC5-640_epoch_14_51.35.pth
echo "--- done ---" 
date