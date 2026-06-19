"""
Mixup : augmentation au niveau des batch.

Mixup mélange deux exemples du batch pour créer des exemples synthétiques.
Réduit drastiquement l'overfitting et améliore la généralisation.

Référence : "mixup: Beyond Empirical Risk Minimization"
https://arxiv.org/abs/1710.09412
"""
import torch
import numpy as np


def mixup_data(x, y, alpha=0.4, device='cuda'):
    """
    Mixup : mélange deux exemples avec un ratio aléatoire.

    Args:
        x: batch d'images [B, C, H, W]
        y: batch de labels [B]
        alpha: paramètre de la distribution Beta (plus grand = mixup plus fort)
        device: device (cuda ou cpu)

    Returns:
        mixed_x: batch mixé [B, C, H, W]
        y_a: labels originaux [B]
        y_b: labels mélangés [B]
        lam: ratio de mélange (scalaire)

    Exemple:
        >>> mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=0.4)
        >>> outputs = model(mixed_x)
        >>> loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)

    # Permutation aléatoire des indices
    index = torch.randperm(batch_size).to(device)

    # Mixer : mixed_x = lam * x + (1 - lam) * x[index]
    mixed_x = lam * x + (1 - lam) * x[index]

    # Labels originaux et mélangés
    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Loss combinée pour Mixup.

    Args:
        criterion: fonction de loss (ex: CrossEntropyLoss)
        pred: prédictions du modèle [B, num_classes]
        y_a: labels originaux [B]
        y_b: labels mélangés [B]
        lam: ratio de mélange (scalaire)

    Returns:
        loss: scalaire

    Exemple:
        >>> loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, y_a, y_b, lam)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    CutMix : coupe une région d'une image et la remplace par une région d'une autre.

    Alternative à Mixup qui préserve mieux les structures locales.

    Référence : "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    https://arxiv.org/abs/1905.04899

    Args:
        x: batch d'images [B, C, H, W]
        y: batch de labels [B]
        alpha: paramètre de la distribution Beta
        device: device (cuda ou cpu)

    Returns:
        mixed_x: batch mixé [B, C, H, W]
        y_a: labels originaux [B]
        y_b: labels de la région coupée [B]
        lam: ratio de surface coupée (scalaire)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    # Taille du patch à couper
    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # Coordonnées du patch (centre aléatoire)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Copier le patch
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Ratio réel de surface coupée
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam
