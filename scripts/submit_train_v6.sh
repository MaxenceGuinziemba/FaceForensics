#!/bin/bash
#SBATCH --job-name=ff_v6
#SBATCH --output=logs/slurm/train_%j.out
#SBATCH --error=logs/slurm/train_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00

mkdir -p logs/slurm

source ~/venvs/faceforensics/bin/activate

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "=========================================="
echo "V6: EfficientNet-B4 380x380 + differential LR"
echo "     + class weights + margin 0.15 + freeze_bn"
echo "=========================================="

# Etape 1 : re-extraction des visages (margin=0.15, 50 faces/video)
# Resume automatique si extraction partielle
echo "=== Extraction des visages (margin=0.15, 50 faces/video) ==="
python3 -m scripts.extract_faces \
    --data_root data \
    --output_dir data/faces_v6 \
    --compression c23 \
    --frames_per_video 50
echo "=== Extraction terminee ==="

# Etape 2 : entrainement V6-Aggressive
python3 -m src.train \
    --data_root data \
    --faces_dir data/faces_v6 \
    --compression c23 \
    --model efficientnet_b4 \
    --dropout 0.4 \
    --batch_size 32 \
    --lr 0.0002 \
    --weight_decay 1e-4 \
    --freeze_epochs 5 \
    --warmup_epochs 3 \
    --freeze_bn \
    --differential_lr \
    --backbone_lr_factor 0.1 \
    --epochs 50 \
    --patience 15 \
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir logs/tensorboard

echo "Termine: $(date)"
