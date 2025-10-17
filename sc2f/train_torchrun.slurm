#!/bin/bash 
#SBATCH --job-name=yoloh-attn-fine-tune-2enc-2dec-all-dp-320-tconvdecoder-lq0-C2F-noposvalue-rope-residual-tconv-2scale2-1x # Job name 
#SBATCH --mail-type=END,FAIL # Mail events 
#SBATCH --mail-user=your_email@example.com # Where to send mail 
#SBATCH --ntasks-per-node=1 # Run on 1 CPU 
#SBATCH --mem=48gb # Job memory request 
#SBATCH --time=120:00:00 # Time limit hrs:min:sec 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --output=path/to/output.log # Standard output and error log

pwd; hostname; date 
nvidia-smi
echo "--- Pytorch Training ---" 
export MASTER_ADDR="$(hostname -s)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
srun --account=optimalvi+ singularity exec --nv --bind /home/projects /opt/apps/containers/conda/conda-nvidia-22.04-latest.sif \
     bash -lc '
        source /opt/miniconda-latest/etc/profile.d/conda.sh
        cd /path/to/project_dir
        conda activate your_conda_env
        echo "Rank env: SLURM_PROCID=$SLURM_PROCID SLURM_LOCALID=$SLURM_LOCALID CWD=$(pwd)"
        torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py \
            --cuda -dist --num_gpu 4 \
            -d coco_fb_diff \
            --train_img_folder /path/to/train_img_folder \
            --train_ann_file /path/to/train_ann_file \
            --val_img_folder /path/to/val_img_folder \
            --val_ann_file /path/to/val_ann_file \
            -v yoloh_expand-50-DC5-640-expand-attn-2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2 \
            -lr 0.024 -lr_bk 0.004 \
            --half_precision --reduce_steps 100 --eval_epoch 8 \
            --batch_size 64 --subset_ratio 1.0 \
            --train_min_size 320 --train_max_size 320 \
            --val_min_size 320 --val_max_size 320 \
            --skip_attention_map --manual_max_epoch 8 \
            --schedule 1x --grad_clip_norm 4.0 \
            --save_folder /path/to/save_folder \
            --wandb_token your_wandb_token \
            --exp_name your_exp_name \
            -p /path/to/pretrained_model.pth
     '