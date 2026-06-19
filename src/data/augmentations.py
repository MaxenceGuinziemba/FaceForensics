"""
Augmentations de données avancées pour la détection de deepfakes.

Ces augmentations simulent les artifacts présents dans les vidéos manipulées :
- Compression JPEG (artifacts de blocs)
- Bruit gaussien (faible luminosité)
- CutOut (masquage aléatoire pour forcer la généralisation)
"""
import random
import cv2
import numpy as np
import torch
from PIL import Image


class JPEGCompression:
    """
    Simule la compression JPEG.

    Crée des artifacts de blocs typiques des deepfakes compressés.
    Les vidéos du dataset sont en c23/c40 (compression H.264), ce qui
    introduit des artifacts similaires.
    """

    def __init__(self, quality_range=(50, 95), p=0.3):
        """
        Args:
            quality_range: plage de qualité JPEG (0-100, plus bas = plus d'artifacts)
            p: probabilité d'appliquer la compression
        """
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        """
        Args:
            img: PIL Image

        Returns:
            PIL Image (compressée avec probabilité p)
        """
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)

            # Convertir PIL → numpy
            img_np = np.array(img)

            # Compresser en JPEG en mémoire
            _, encoded = cv2.imencode(
                '.jpg',
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )

            # Décompresser
            img_compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            img_compressed = cv2.cvtColor(img_compressed, cv2.COLOR_BGR2RGB)

            return Image.fromarray(img_compressed)

        return img


class GaussianNoise:
    """
    Ajoute du bruit gaussien.

    Simule les artifacts de faible luminosité ou de capteurs de caméra.
    Appliqué APRÈS ToTensor (travaille sur des tensors).
    """

    def __init__(self, std_range=(0.01, 0.05), p=0.2):
        """
        Args:
            std_range: plage d'écart-type du bruit (0-1)
            p: probabilité d'appliquer le bruit
        """
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor):
        """
        Args:
            tensor: torch.Tensor [C, H, W] normalisé entre -1 et 1

        Returns:
            torch.Tensor avec bruit ajouté
        """
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(tensor) * std
            return torch.clamp(tensor + noise, -1, 1)

        return tensor


class CutOut:
    """
    Masque aléatoire de zones rectangulaires (remplies de 0).

    Force le modèle à regarder toutes les parties du visage au lieu de se
    concentrer sur une seule région (ex: uniquement la bouche).

    Référence : "Improved Regularization of Convolutional Neural Networks with Cutout"
    https://arxiv.org/abs/1708.04552
    """

    def __init__(self, n_holes=1, length=50, p=0.15):
        """
        Args:
            n_holes: nombre de rectangles à masquer
            length: taille des rectangles (en pixels)
            p: probabilité d'appliquer CutOut
        """
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, tensor):
        """
        Args:
            tensor: torch.Tensor [C, H, W]

        Returns:
            torch.Tensor avec zones masquées
        """
        if random.random() < self.p:
            h = tensor.shape[1]
            w = tensor.shape[2]

            # Créer un masque binaire (1 = visible, 0 = masqué)
            mask = torch.ones((h, w), dtype=torch.float32)

            for _ in range(self.n_holes):
                # Centre du rectangle aléatoire
                y = random.randint(0, h)
                x = random.randint(0, w)

                # Coordonnées du rectangle
                y1 = max(0, y - self.length // 2)
                y2 = min(h, y + self.length // 2)
                x1 = max(0, x - self.length // 2)
                x2 = min(w, x + self.length // 2)

                # Masquer
                mask[y1:y2, x1:x2] = 0.0

            # Appliquer le masque à tous les canaux
            tensor = tensor * mask.unsqueeze(0)

        return tensor


class RandomErase:
    """
    Variant de CutOut qui remplit avec du bruit au lieu de 0.

    Peut être plus naturel pour les visages (moins de zones noires artificielles).
    """

    def __init__(self, n_holes=1, length=50, p=0.15):
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            h = tensor.shape[1]
            w = tensor.shape[2]

            for _ in range(self.n_holes):
                y = random.randint(0, h)
                x = random.randint(0, w)

                y1 = max(0, y - self.length // 2)
                y2 = min(h, y + self.length // 2)
                x1 = max(0, x - self.length // 2)
                x2 = min(w, x + self.length // 2)

                # Remplir avec du bruit aléatoire
                noise = torch.randn((tensor.shape[0], y2 - y1, x2 - x1))
                tensor[:, y1:y2, x1:x2] = noise

        return tensor
