#!/bin/bash
# ==============================================================================
# Job SLURM - Entraînement XceptionNet sur FaceForensics++
# Cluster Télécom Paris
#
# Usage: sbatch scripts/submit_train.sh
# ==============================================================================

#SBATCH --job-name=ff_train                 # Nom du job (visible dans squeue)
#SBATCH --output=logs/slurm/train_%j.out    # Stdout (%j = job ID)
#SBATCH --error=logs/slurm/train_%j.err     # Stderr
#SBATCH --partition=3090                    # GPU RTX 3090 (24GB VRAM) ou P100 (16GB)
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --cpus-per-task=8                   # 8 CPUs pour le DataLoader
#SBATCH --mem=32G                           # 32GB RAM
#SBATCH --time=24:00:00                     # 24h max (limite cluster: 36h)

# --- Création du dossier de logs ---
mkdir -p logs/slurm

# --- Environnement ---
source ~/venvs/faceforensics/bin/activate

# --- Afficher les infos GPU ---
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "---"

# --- Entraînement ---
python3 -m src.train \
    --data_root data \
    --compression c23 \
    --model xception \
    --dropout 0.5 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0002 \
    --weight_decay 1e-4 \
    --patience 10 \
    --frames_per_video 10 \
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir logs/tensorboard

echo "---"
echo "Terminé: $(date)"
