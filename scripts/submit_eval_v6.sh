#!/bin/bash
#SBATCH --job-name=ff_eval_v6
#SBATCH --output=logs/slurm/eval_%j.out
#SBATCH --error=logs/slurm/eval_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

mkdir -p logs/slurm

source ~/venvs/faceforensics/bin/activate

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "=========================================="
echo "Evaluation V6: EfficientNet-B4 380x380"
echo "=========================================="

python3 -m src.evaluate \
    --checkpoint checkpoints/best_model.pth \
    --data_root data \
    --faces_dir data/faces_v6 \
    --compression c23 \
    --batch_size 32 \
    --frames_per_video 50 \
    --output_dir checkpoints/evaluation_v6

echo "Termine: $(date)"
