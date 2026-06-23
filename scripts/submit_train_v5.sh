#!/bin/bash
#SBATCH --job-name=ff_train_v5
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
echo "V5: EfficientNet-B4 + faces pré-extraites"
echo "=========================================="

# Étape 1 : extraire les faces (resume automatique si extraction partielle)
echo "=== Extraction des visages ==="
python3 -m scripts.extract_faces \
    --data_root data \
    --output_dir data/faces \
    --compression c23 \
    --frames_per_video 30
echo "=== Extraction terminée ==="

# Étape 2 : entraînement
python3 -m src.train \
    --data_root data \
    --faces_dir data/faces \
    --compression c23 \
    --model efficientnet_b4 \
    --dropout 0.5 \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.00001 \
    --weight_decay 5e-4 \
    --patience 20 \
    --freeze_epochs 5 \
    --warmup_epochs 3 \
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir logs/tensorboard

echo "Terminé: $(date)"
