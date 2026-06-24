#!/bin/bash
#SBATCH --job-name=ff_v7
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

mkdir -p logs/slurm

source ~/venvs/faceforensics/bin/activate

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "=========================================="
echo "V7: EfficientNet-B4 380x380 + LR uniforme 3e-05"
echo "     + class weights + margin 0.15 + freeze_bn"
echo "     + freeze 4 epochs + warmup 3 + patience 10"
echo "=========================================="

python3 -m src.train \
    --data_root data \
    --faces_dir data/faces_v6 \
    --compression c23 \
    --model efficientnet_b4 \
    --dropout 0.4 \
    --batch_size 32 \
    --lr 0.00003 \
    --weight_decay 1e-4 \
    --freeze_epochs 4 \
    --warmup_epochs 3 \
    --freeze_bn \
    --epochs 50 \
    --patience 10 \
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir logs/tensorboard

echo "Termine: $(date)"
