#!/bin/bash
# ==============================================================================
# Job SLURM - Entraînement OPTIMISÉ sur FaceForensics++
# Cluster Télécom Paris
#
# AMÉLIORATIONS APPLIQUÉES (Phase 1 - Quick Wins) :
# - Label Smoothing (0.1)
# - LR Warmup + Cosine Annealing
# - Augmentations avancées (JPEG, GaussianNoise, CutOut)
# - LR réduit (0.00005 au lieu de 0.0002)
# - Dropout augmenté (0.6 au lieu de 0.5)
# - Frames per video réduit (5 au lieu de 10) → epochs 2× plus rapides
# - Patience augmentée (15 au lieu de 10)
#
# GAINS ATTENDUS : Val acc 66% → 72-75% (+6-9%)
#
# Usage: sbatch scripts/submit_train_optimized.sh
# ==============================================================================

#SBATCH --job-name=ff_train_opt             # Nom du job
#SBATCH --output=logs/slurm/train_%j.out    # Stdout
#SBATCH --error=logs/slurm/train_%j.err     # Stderr
#SBATCH --partition=3090                    # GPU RTX 3090 (24GB VRAM)
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --cpus-per-task=8                   # 8 CPUs pour le DataLoader
#SBATCH --mem=32G                           # 32GB RAM
#SBATCH --time=24:00:00                     # 24h max

# --- Création du dossier de logs ---
mkdir -p logs/slurm

# --- Environnement ---
source ~/venvs/faceforensics/bin/activate

# --- Afficher les infos GPU ---
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "CONFIGURATION OPTIMISÉE (Phase 1):"
echo "  - LR: 0.00005 (au lieu de 0.0002)"
echo "  - Dropout: 0.6 (au lieu de 0.5)"
echo "  - Frames per video: 5 (au lieu de 10)"
echo "  - Patience: 15 (au lieu de 10)"
echo "  - Label Smoothing: 0.1"
echo "  - Augmentations: JPEG + GaussianNoise + CutOut"
echo ""
echo "GAINS ATTENDUS: Val acc 66% → 72-75%"
echo "=========================================="
echo ""

# --- Entraînement ---
python3 -m src.train \
    --data_root data \
    --compression c23 \
    --model resnet18 \
    --dropout 0.6 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.00005 \
    --weight_decay 1e-4 \
    --patience 15 \
    --frames_per_video 5 \
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir logs/tensorboard

echo ""
echo "=========================================="
echo "Terminé: $(date)"
echo "=========================================="
