#!/bin/bash 
#SBATCH --job-name=yoloh-attn-fine-tune-2enc-2dec-all-dp-320-tconvdecoder-lq0-C2F-noposvalue-rope-residual-tconv-2scale2-1x # Job name 
#SBATCH --mail-type=END,FAIL # Mail events 
#SBATCH --mail-user=changlin.song@anu.edu.au # Where to send mail 
#SBATCH --ntasks-per-node=2 # Run on 2 CPU 
#SBATCH --mem=48gb # Job memory request 
#SBATCH --time=120:00:00 # Time limit hrs:min:sec 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3090:2
#SBATCH --output=/home/projects/u7707452/logs/yoloh-attn-fine-tune-2enc-2dec-all-dp-320-tconvdecoder-lq0-C2F-noposvalue-rope-residual-tconv-2scale2-1x-%j.log # Standard output and error log

set -eo pipefail
mkdir -p /home/projects/u7707452/logs

echo "PWD: $PWD"
echo "Host: $(hostname)"
date
nvidia-smi || true

export PS1="${PS1:-sbatch}"
module add miniconda/4.8.3 || true
. /d/sw/miniconda3/4.8.3/etc/profile.d/conda.sh
conda activate eaft

export WORK_DIR=/data/anu_unseen
export CONDA_ENVS_PATH="/data/anu_unseen/conda/envs"
export HF_HOME=$WORK_DIR/.hf_cache
export DIFFUSERS_CACHE=$WORK_DIR/.hf_cache
export TRANSFORMERS_CACHE=/data/anu_unseen/.hf_cache
export https_proxy="http://proxy.per.dug.com:3128"

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export MASTER_ADDR="$(hostname -s)"
export MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"
export WORLD_SIZE="${SLURM_NTASKS}"

# Launch 2 ranks; set CWD for each task; no cpu-bind flags
srun --ntasks="${WORLD_SIZE}" --mpi=none --export=ALL \
     --chdir=/data/anu_unseen/yoloh \
     bash -lc '
        export RANK=$SLURM_PROCID
        export WORLD_SIZE=$SLURM_NTASKS
        export LOCAL_RANK=$SLURM_LOCALID
        export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # <- each rank sees exactly one GPU (index 0)
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        echo "rank=$RANK local_rank=$LOCAL_RANK CVD=$CUDA_VISIBLE_DEVICES"
        cd /data/anu_unseen/yoloh
        conda activate eaft
        echo "Rank env: SLURM_PROCID=$SLURM_PROCID SLURM_LOCALID=$SLURM_LOCALID CWD=$(pwd)"
        python train.py \
            --cuda -dist --num_gpu 4 \
            -d coco_fb_diff \
            --root /home/projects/compVisionDatasets \
            --source_img_folder /home/projects/compVisionDatasets/coco/train2017 \
            --train_img_folder /data/anu_unseen/data/amodal_train_2024_fb_v2_withSource \
            --train_ann_file /data/anu_unseen/data/annotations/coco_amodal_fb_train_v2_withSource.json \
            --val_img_folder /data/anu_unseen/data/amodal_val_2024_fb_v2 \
            --val_ann_file /data/anu_unseen/data/annotations/coco_amodal_fb_val_v2_classified_rb.json \
            -v yoloh_expand-50-DC5-640-expand-attn-2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2 \
            -lr 0.024 -lr_bk 0.004 \
            --half_precision --reduce_steps 100 --eval_epoch 4 \
            --batch_size 64 --subset_ratio 1.0 \
            --train_min_size 320 --train_max_size 320 \
            --val_min_size 320 --val_max_size 320 \
            --skip_attention_map --manual_max_epoch 8 \
            --schedule 1x --grad_clip_norm 4.0 \
            --save_folder /data/anu_unseen/models/yoloh \
            --wandb_token 469e207a63db0a4817bff81ecbcd9187be656e40 \
            --exp_name 2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2-1x \
            -p /data/anu_unseen/models/yoloh50-DC5-640_epoch_14_51.35.pth
     '