#!/bin/bash
#SBATCH --job-name=ff_train_v3
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
echo "V3: EfficientNet + Face Extraction + Mixup + Freeze/Unfreeze"
echo "=========================================="

python3 -m src.train \
    --data_root ~/projects/FaceForensics/data \
    --compression c23 \
    --model efficientnet \
    --dropout 0.5 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.00003 \
    --weight_decay 5e-4 \
    --patience 20 \
    --frames_per_video 5 \
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir logs/tensorboard

echo "Terminé: $(date)"
