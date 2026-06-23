#!/bin/bash
#SBATCH --job-name=ff_eval_v5
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

mkdir -p logs/slurm results/v5

source ~/venvs/faceforensics/bin/activate

echo "=========================================="
echo "Evaluation V5 - EfficientNet-B4 (90.93%)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "=========================================="

# Evaluation
python3 -m src.evaluate \
    --checkpoint checkpoints/best_model.pth \
    --data_root data \
    --faces_dir data/faces \
    --compression c23 \
    --batch_size 32 \
    --output_dir checkpoints/evaluation

# Sauvegarder tous les resultats V5
cp checkpoints/best_model.pth results/v5/best_model_v5.pth
cp checkpoints/training_curves.png results/v5/
cp checkpoints/evaluation/* results/v5/ 2>/dev/null
cp logs/slurm/train_861281.out results/v5/train_log.out
cp logs/slurm/train_861281.err results/v5/train_log.err 2>/dev/null

echo "=========================================="
echo "Resultats sauvegardes dans results/v5/"
ls -lh results/v5/
echo "=========================================="
echo "Termine: $(date)"
