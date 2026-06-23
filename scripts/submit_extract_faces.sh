#!/bin/bash
#SBATCH --job-name=ff_extract
#SBATCH --output=logs/slurm/extract_%j.out
#SBATCH --error=logs/slurm/extract_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

mkdir -p logs/slurm
source ~/venvs/faceforensics/bin/activate

echo "=== Extraction des visages (GPU) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"

python3 -m scripts.extract_faces \
    --data_root data \
    --output_dir data/faces \
    --compression c23 \
    --frames_per_video 50

echo "Terminé: $(date)"
