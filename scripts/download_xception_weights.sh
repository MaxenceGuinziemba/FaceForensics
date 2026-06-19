#!/bin/bash
# ==============================================================================
# Téléchargement manuel des poids XceptionNet
# (contournement du certificat SSL expiré)
# ==============================================================================

set -e

WEIGHTS_DIR="$HOME/.cache/torch/hub/checkpoints"
WEIGHTS_FILE="xception-b5690688.pth"
WEIGHTS_URL="http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth"

echo "📥 Téléchargement des poids XceptionNet..."
echo "Destination: $WEIGHTS_DIR/$WEIGHTS_FILE"

# Créer le dossier si nécessaire
mkdir -p "$WEIGHTS_DIR"

# Télécharger avec wget (ignore les erreurs SSL)
wget --no-check-certificate \
     --continue \
     --progress=bar:force \
     -O "$WEIGHTS_DIR/$WEIGHTS_FILE" \
     "$WEIGHTS_URL"

# Vérifier que le fichier existe et a une taille raisonnable
if [ -f "$WEIGHTS_DIR/$WEIGHTS_FILE" ]; then
    SIZE=$(du -h "$WEIGHTS_DIR/$WEIGHTS_FILE" | cut -f1)
    echo "✅ Téléchargement terminé : $SIZE"
    echo "MD5: $(md5sum $WEIGHTS_DIR/$WEIGHTS_FILE | cut -d' ' -f1)"
else
    echo "❌ Échec du téléchargement"
    exit 1
fi
