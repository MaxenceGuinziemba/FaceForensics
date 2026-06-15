#!/bin/bash
# ==============================================================================
# Setup initial sur le cluster Télécom Paris
# À exécuter UNE SEULE FOIS après la première connexion SSH
#
# Usage: bash scripts/setup_cluster.sh
# ==============================================================================

echo "=== Setup du projet FaceForensics++ sur le cluster ==="

# 1. Environnement virtuel Python
echo "[1/4] Création de l'environnement virtuel..."
python3 -m venv ~/venvs/faceforensics
source ~/venvs/faceforensics/bin/activate
pip install --upgrade pip

# 2. Dépendances
echo "[2/4] Installation des dépendances..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Vérification CUDA
echo "[3/4] Vérification GPU/CUDA..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 4. Téléchargement du dataset
echo "[4/4] Téléchargement du dataset (c23, ~10GB)..."
echo "  Cela peut prendre 30-60 minutes selon la connexion."
mkdir -p data
echo "" | python3 scripts/download_dataset.py data -d all -c c23 -t videos --server EU2

echo ""
echo "=== Setup terminé ==="
echo "Pour lancer l'entraînement: sbatch scripts/submit_train.sh"
