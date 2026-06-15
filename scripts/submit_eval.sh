#!/bin/bash
# ==============================================================================
# Job SLURM - Évaluation du modèle entraîné
# À lancer APRÈS que submit_train.sh soit terminé
#
# Usage: sbatch scripts/submit_eval.sh
# ==============================================================================

#SBATCH --job-name=ff_eval                  # Nom du job
#SBATCH --output=logs/slurm/eval_%j.out     # Stdout
#SBATCH --error=logs/slurm/eval_%j.err      # Stderr
#SBATCH --partition=3090                    # GPU
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --cpus-per-task=4                   # 4 CPUs suffisent pour l'évaluation
#SBATCH --mem=16G                           # 16GB RAM
#SBATCH --time=02:00:00                     # 2h max (l'évaluation est rapide)

# --- Création du dossier de logs ---
mkdir -p logs/slurm

# --- Environnement ---
source ~/venvs/faceforensics/bin/activate

# --- Infos ---
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "---"

# --- Évaluation ---
python3 -m src.evaluate \
    --checkpoint checkpoints/best_model.pth \
    --data_root data \
    --compression c23 \
    --batch_size 32 \
    --frames_per_video 10 \
    --output_dir checkpoints/evaluation

echo "---"
echo "Terminé: $(date)"
