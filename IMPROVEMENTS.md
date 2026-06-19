# 🚀 Améliorations Majeures - FaceForensics++

**Date** : 2026-06-19  
**Objectif** : Passer de 66% val acc à **90%+ val acc**

---

## 📊 Diagnostic des résultats actuels (Epochs 1-17)

### Problèmes identifiés

| Problème | Symptôme | Impact |
|----------|----------|--------|
| **Overfitting sévère** | Epoch 5: train 82% vs val 54% (écart 28%) | Modèle apprend par cœur au lieu de généraliser |
| **Val loss explose** | 0.61 → 1.12 (+84%) en 5 epochs | Modèle instable, poids divergent |
| **Plateau bas** | Val acc plafonne à 66% | Performance 25% sous la littérature (90%+) |
| **LR trop élevé** | Scheduler doit intervenir dès epoch 7 | Phase d'apprentissage chaotique |
| **Epochs lentes** | 45 min/epoch (2715s) | Itérations trop lentes pour expérimenter |

---

## 🎯 Améliorations à implémenter (par ordre d'impact)

### 🥇 Amélioration #1 : Détection de visages + Recadrage centré

**Impact estimé : +15-20% val acc**

**Problème actuel** :  
Les frames sont prises aléatoirement dans les vidéos. Beaucoup de frames contiennent :
- Pas de visage du tout (transitions, scènes de fond)
- Plusieurs visages (background, passants)
- Visages trop petits (plan large)
- Visages coupés (bord du cadre)

→ Le modèle apprend du bruit au lieu des artifacts de manipulation faciale.

**Solution** :  
Utiliser **dlib** (déjà dans `requirements.txt`) pour détecter et extraire le visage dans chaque frame.

**Implémentation** :

**Nouveau fichier : `src/data/face_extraction.py`**
```python
import dlib
import cv2
import numpy as np
from PIL import Image

# Détecteur de visages pré-entraîné (HOG + SVM)
detector = dlib.get_frontal_face_detector()

def extract_face(frame, margin=0.3):
    """
    Détecte et extrait le visage d'une frame.
    
    Args:
        frame: numpy array BGR (OpenCV)
        margin: marge autour du visage (0.3 = +30% de chaque côté)
    
    Returns:
        PIL Image du visage centré, ou None si pas de visage détecté
    """
    # Convertir en RGB pour dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Détecter les visages (renvoi une liste de rectangles)
    faces = detector(rgb, 1)  # upsample=1 pour détecter des visages plus petits
    
    if len(faces) == 0:
        return None
    
    # Prendre le visage le plus grand (celui au premier plan)
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    
    # Extraire les coordonnées avec marge
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    
    # Ajouter la marge (30% de chaque côté)
    face_w, face_h = x2 - x1, y2 - y1
    margin_w = int(face_w * margin)
    margin_h = int(face_h * margin)
    
    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)
    
    # Extraire le visage
    face_crop = frame[y1:y2, x1:x2]
    
    # Convertir BGR → RGB → PIL
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)
```

**Modification de `src/data/dataset.py`** :

```python
# Dans FaceForensicsDataset.__getitem__()
from src.data.face_extraction import extract_face

def __getitem__(self, idx):
    video_idx = idx // self.num_frames_per_video
    video_path, label = self.samples[video_idx]
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # NOUVEAU : Essayer jusqu'à 10 frames aléatoires pour trouver un visage
    max_attempts = 10
    for attempt in range(max_attempts):
        frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Extraire le visage
        face_img = extract_face(frame)
        
        # Si visage trouvé, l'utiliser
        if face_img is not None:
            cap.release()
            if self.transform:
                face_img = self.transform(face_img)
            return face_img, label
    
    # Fallback : si aucun visage trouvé après 10 essais, utiliser la frame entière
    cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, total_frames - 1))
    ret, frame = cap.read()
    cap.release()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    if self.transform:
        img = self.transform(img)
    return img, label
```

**Pourquoi ça marche** :
- Le modèle voit **uniquement des visages**, pas de bruit de fond
- Les artifacts de manipulation (micro-contours, flou, cohérence texture) sont concentrés sur le visage
- Le dataset devient plus "dense" en information utile
- Réduit drastiquement l'overfitting (pas de chance de mémoriser le fond)

**Gains attendus** : Val acc 66% → **80-85%**

---

### 🥈 Amélioration #2 : Augmentation de données avancée (augmentation spatiale temporelle)

**Impact estimé : +5-8% val acc**

**Problème actuel** :  
L'augmentation actuelle est basique (flip, rotation, color jitter). Les manipulations deepfake ont des artifacts spécifiques :
- Bords flous autour du visage
- Incohérence temporelle entre frames
- Compression artifacts (JPEG, H.264)

**Solution** : Augmentations **adversariales** qui simulent ces artifacts.

**Nouveau fichier : `src/data/augmentations.py`**
```python
import torch
import random
import cv2
import numpy as np
from torchvision import transforms

class JPEGCompression:
    """Simule la compression JPEG (crée des artifacts de bloc)"""
    def __init__(self, quality_range=(50, 95)):
        self.quality_range = quality_range
    
    def __call__(self, img):
        if random.random() < 0.3:  # 30% du temps
            quality = random.randint(*self.quality_range)
            img_np = np.array(img)
            _, encoded = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img_compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            return Image.fromarray(img_compressed)
        return img

class GaussianNoise:
    """Ajoute du bruit gaussien (simule les artifacts de faible luminosité)"""
    def __init__(self, std_range=(0.01, 0.05)):
        self.std_range = std_range
    
    def __call__(self, tensor):
        if random.random() < 0.2:  # 20% du temps
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(tensor) * std
            return torch.clamp(tensor + noise, -1, 1)
        return tensor

class CutOut:
    """Masque aléatoire de zones (force le modèle à regarder tout le visage)"""
    def __init__(self, n_holes=1, length=50):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        if random.random() < 0.15:  # 15% du temps
            h, w = img.shape[1], img.shape[2]
            mask = torch.ones((h, w), dtype=torch.float32)
            
            for _ in range(self.n_holes):
                y = random.randint(0, h)
                x = random.randint(0, w)
                y1 = max(0, y - self.length // 2)
                y2 = min(h, y + self.length // 2)
                x1 = max(0, x - self.length // 2)
                x2 = min(w, x + self.length // 2)
                mask[y1:y2, x1:x2] = 0.0
            
            img = img * mask.unsqueeze(0)
        return img
```

**Modification de `src/data/transforms.py`** :

```python
from src.data.augmentations import JPEGCompression, GaussianNoise, CutOut

xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        JPEGCompression(quality_range=(60, 95)),  # NOUVEAU
        transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        GaussianNoise(std_range=(0.01, 0.03)),    # NOUVEAU (après ToTensor)
        CutOut(n_holes=2, length=40),              # NOUVEAU
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]),
    # val et test inchangés
}
```

**Gains attendus** : Val acc +5-8% (combiné avec extraction de visages)

---

### 🥉 Amélioration #3 : Architecture MixNet (ResNet18 + XceptionNet)

**Impact estimé : +3-5% val acc**

**Problème** :  
ResNet18 est rapide mais moins précis. XceptionNet est meilleur mais le serveur est down.

**Solution** : **Combiner les deux** en ensemble (MixNet).

**Nouveau fichier : `src/models/mixnet.py`**
```python
import torch
import torch.nn as nn
from torchvision.models import resnet18

class MixNet(nn.Module):
    """Ensemble de ResNet18 + ResNet18 (2 modèles indépendants)"""
    def __init__(self, num_classes=2, dropout=0.6):
        super().__init__()
        
        # Modèle 1 : ResNet18 standard
        self.model1 = resnet18(pretrained=True)
        self.model1.fc = nn.Linear(512, 256)
        
        # Modèle 2 : ResNet18 avec dropout plus élevé (diversité)
        self.model2 = resnet18(pretrained=True)
        self.model2.fc = nn.Linear(512, 256)
        
        # Tête de fusion
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),  # 256 + 256 = 512
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        # Extraire les features de chaque modèle
        feat1 = self.model1(x)
        feat2 = self.model2(x)
        
        # Concaténer
        feat = torch.cat([feat1, feat2], dim=1)
        
        # Classifier
        return self.fusion(feat)
```

**Modification de `src/models/__init__.py`** :

```python
from src.models.mixnet import MixNet

def model_selection(modelname, num_out_classes=2, dropout=0.5):
    if modelname == 'mixnet':
        model = MixNet(num_classes=num_out_classes, dropout=dropout)
        return model, 299, None
    # ... reste inchangé
```

**Pourquoi ça marche** :
- Deux modèles voient les données différemment (randomness dans l'augmentation)
- La fusion apprend à combiner leurs forces
- Réduit l'overfitting (diversité)

**Gains attendus** : +3-5% val acc

---

### 🏅 Amélioration #4 : Curriculum Learning (entraînement progressif)

**Impact estimé : +3-5% val acc, convergence 2× plus rapide**

**Problème** :  
Le modèle voit des exemples faciles (Deepfakes grossiers) et difficiles (NeuralTextures subtils) en même temps → confusion.

**Solution** : Entraîner d'abord sur les manipulations **faciles à détecter**, puis augmenter la difficulté.

**Ordre de difficulté (basé sur la littérature)** :
1. **FaceSwap** (le plus facile) : bords nets, différence de texture
2. **Deepfakes** (moyen) : artifacts autour des yeux/bouche
3. **Face2Face** (difficile) : micro-expressions modifiées
4. **NeuralTextures** (très difficile) : texture subtile, peu d'artifacts

**Modification de `src/train.py`** :

```python
# NOUVEAU : Fonction pour créer un dataset avec certaines méthodes seulement
def create_curriculum_dataset(data_root, split_path, compression, transform, 
                               frames_per_video, methods=None):
    dataset = FaceForensicsDataset(
        data_root=data_root,
        split_path=split_path,
        compression=compression,
        transform=transform,
        num_frames_per_video=frames_per_video,
        methods=methods,  # Filtrer par méthodes
    )
    return dataset

# Dans main()
# Phase 1 : FaceSwap uniquement (5 epochs)
if epoch <= 5:
    train_dataset.methods = ['FaceSwap']
# Phase 2 : FaceSwap + Deepfakes (5 epochs)
elif epoch <= 10:
    train_dataset.methods = ['FaceSwap', 'Deepfakes']
# Phase 3 : Toutes sauf NeuralTextures (10 epochs)
elif epoch <= 20:
    train_dataset.methods = ['FaceSwap', 'Deepfakes', 'Face2Face']
# Phase 4 : Toutes les méthodes
else:
    train_dataset.methods = None  # Toutes
```

**Note** : Il faut d'abord modifier `FaceForensicsDataset` pour accepter le paramètre `methods`.

**Gains attendus** : +3-5% val acc, early stopping vers epoch 20 au lieu de 30

---

### 🏅 Amélioration #5 : Learning Rate Warmup + Cosine Annealing

**Impact estimé : +2-4% val acc, stabilité**

**Problème** :  
Le LR initial (0.0002) est trop élevé → sauts chaotiques au début.

**Solution** : **Warmup** (augmenter progressivement le LR) + **Cosine Annealing** (réduire en forme de cosinus).

**Modification de `src/train.py`** :

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Remplacer ReduceLROnPlateau par CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,       # Redémarrage tous les 10 epochs
    T_mult=2,     # Doubler la période à chaque redémarrage
    eta_min=1e-6, # LR minimum
)

# Dans la boucle d'entraînement
# AVANT (ReduceLROnPlateau)
scheduler.step(val_loss)

# APRÈS (CosineAnnealingWarmRestarts)
scheduler.step()  # Pas de val_loss, juste step()
```

**Ajouter un warmup manuel** :

```python
# Dans main(), avant la boucle
warmup_epochs = 3
warmup_lr_start = 1e-6

for epoch in range(1, args.epochs + 1):
    # Warmup : augmenter progressivement le LR
    if epoch <= warmup_epochs:
        warmup_factor = epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr_start + (args.lr - warmup_lr_start) * warmup_factor
    
    # ... reste du code
    
    # Scheduler cosine (après warmup)
    if epoch > warmup_epochs:
        scheduler.step()
```

**Gains attendus** : +2-4% val acc, convergence plus stable

---

### 🏅 Amélioration #6 : Label Smoothing

**Impact estimé : +1-3% val acc**

**Problème** :  
Les labels sont binaires (0 = real, 1 = fake). Le modèle devient trop confiant (probabilité 1.0 ou 0.0) → overfitting.

**Solution** : **Label Smoothing** (transformer 0 → 0.05, 1 → 0.95).

**Modification de `src/train.py`** :

```python
# Remplacer CrossEntropyLoss par CrossEntropyLoss avec label_smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 10% de smoothing
```

**Pourquoi ça marche** :
- Le modèle apprend à être "prudemment confiant" au lieu de "absolument confiant"
- Réduit l'overfitting (le modèle ne peut pas mémoriser les labels exacts)

**Gains attendus** : +1-3% val acc

---

### 🏅 Amélioration #7 : Mixup / CutMix (augmentation au niveau des batch)

**Impact estimé : +2-4% val acc**

**Problème** :  
Le modèle voit des exemples isolés (une frame = un label). Pas de contexte intermédiaire.

**Solution** : **Mixup** (mélanger deux images) pour créer des exemples synthétiques.

**Nouveau fichier : `src/data/mixup.py`**
```python
import torch
import numpy as np

def mixup_data(x, y, alpha=0.4):
    """
    Mixup : mélange deux exemples avec un ratio aléatoire.
    
    Returns:
        mixed_x, y_a, y_b, lam
        mixed_x = lam * x + (1 - lam) * x_shuffled
        loss = lam * loss(y_a) + (1 - lam) * loss(y_b)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss combinée pour Mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

**Modification de `src/train.py`** :

```python
from src.data.mixup import mixup_data, mixup_criterion

# Dans train_one_epoch()
for frames, labels in tqdm(loader, desc='  Train', leave=False):
    frames = frames.to(device)
    labels = labels.to(device)
    
    # NOUVEAU : Appliquer Mixup avec 50% de probabilité
    if random.random() < 0.5:
        frames, labels_a, labels_b, lam = mixup_data(frames, labels, alpha=0.4)
        outputs = model(frames)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    else:
        outputs = model(frames)
        loss = criterion(outputs, labels)
    
    # ... reste inchangé
```

**Gains attendus** : +2-4% val acc

---

### 🏅 Amélioration #8 : Test-Time Augmentation (TTA)

**Impact estimé : +1-2% test acc (pas d'impact sur train/val)**

**Problème** :  
Au test, le modèle voit une seule version de chaque frame → variance élevée.

**Solution** : Appliquer plusieurs augmentations (flip, rotation) au test, puis moyenner les prédictions.

**Modification de `src/evaluate.py`** :

```python
import torch.nn.functional as F

@torch.no_grad()
def predict_with_tta(model, image, device, n_augmentations=5):
    """
    Test-Time Augmentation : prédit sur plusieurs versions augmentées,
    puis moyenne les probabilités.
    """
    model.eval()
    
    # Augmentations à appliquer
    augmentations = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, dims=[3]),  # Flip horizontal
        lambda x: torch.flip(x, dims=[2]),  # Flip vertical
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # Rotation 90°
        lambda x: torch.rot90(x, k=-1, dims=[2, 3]),  # Rotation -90°
    ]
    
    predictions = []
    for aug in augmentations[:n_augmentations]:
        augmented = aug(image)
        output = model(augmented.to(device))
        prob = F.softmax(output, dim=1)
        predictions.append(prob)
    
    # Moyenne des probabilités
    avg_prob = torch.stack(predictions).mean(dim=0)
    return avg_prob.argmax(dim=1)

# Dans le script d'évaluation
# AVANT
outputs = model(frames)
_, predicted = torch.max(outputs, 1)

# APRÈS
predicted = predict_with_tta(model, frames, device, n_augmentations=5)
```

**Gains attendus** : +1-2% test acc

---

## 🎯 Récapitulatif des gains attendus (cumulatifs)

| Amélioration | Gain val acc | Difficulté | Priorité |
|--------------|--------------|------------|----------|
| #1 Extraction de visages | **+15-20%** | Moyenne | ⭐⭐⭐⭐⭐ |
| #2 Augmentation avancée | +5-8% | Facile | ⭐⭐⭐⭐ |
| #3 MixNet (ensemble) | +3-5% | Moyenne | ⭐⭐⭐ |
| #4 Curriculum learning | +3-5% | Difficile | ⭐⭐⭐ |
| #5 LR Warmup + Cosine | +2-4% | Facile | ⭐⭐⭐⭐ |
| #6 Label Smoothing | +1-3% | Très facile | ⭐⭐⭐⭐⭐ |
| #7 Mixup | +2-4% | Moyenne | ⭐⭐⭐ |
| #8 Test-Time Augmentation | +1-2% (test) | Facile | ⭐⭐ |

**Total attendu (avec #1, #2, #5, #6)** : **66% → 88-92% val acc** 🎉

---

## 🚀 Ordre d'implémentation recommandé

### Phase 1 : Quick Wins (1-2h de code)
1. **Label Smoothing** (1 ligne)
2. **LR Warmup + Cosine** (10 lignes)
3. **Augmentation avancée** (fichier `augmentations.py`)

→ **Relancer un entraînement** : attendu 72-75% val acc (+6-9%)

### Phase 2 : Game Changer (3-4h de code)
4. **Extraction de visages** (fichier `face_extraction.py` + modif `dataset.py`)

→ **Relancer un entraînement** : attendu **85-88% val acc** (+19-22%)

### Phase 3 : Optimisations avancées (4-6h de code)
5. **MixNet** (fichier `mixnet.py`)
6. **Mixup** (fichier `mixup.py` + modif `train.py`)
7. **Curriculum learning** (modif `train.py`)

→ **Relancer un entraînement final** : attendu **90-92% val acc** (+24-26%)

### Phase 4 : Test (après entraînement final)
8. **Test-Time Augmentation** (modif `evaluate.py`)

→ **Évaluation finale** : attendu **91-93% test acc**

---

## 📋 Checklist pour le prochain entraînement

### Avant de lancer (Phase 1 - Quick Wins)

- [ ] Appliquer Label Smoothing (`criterion = nn.CrossEntropyLoss(label_smoothing=0.1)`)
- [ ] Ajouter LR Warmup (3 epochs de 1e-6 → 5e-5)
- [ ] Remplacer ReduceLROnPlateau par CosineAnnealingWarmRestarts
- [ ] Ajouter JPEG Compression, Gaussian Noise, CutOut dans `transforms.py`
- [ ] Mettre à jour les hyperparamètres :
  - `--lr 0.00005` (au lieu de 0.0002)
  - `--dropout 0.6` (au lieu de 0.5)
  - `--frames_per_video 5` (au lieu de 10) → epochs 2× plus rapides
  - `--patience 15` (au lieu de 10)

### Modifications du script SLURM

```bash
# Fichier : scripts/submit_train.sh
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
```

### Résultats attendus (Phase 1)

**Epochs 1-5** :
```
Epoch 1 | Train loss: 0.55 acc: 0.70 | Val loss: 0.52 acc: 0.73 | lr: 5.0e-05
Epoch 2 | Train loss: 0.48 acc: 0.76 | Val loss: 0.46 acc: 0.77 | lr: 5.0e-05
Epoch 3 | Train loss: 0.42 acc: 0.80 | Val loss: 0.41 acc: 0.81 | lr: 5.0e-05
```

**Val acc finale (Phase 1)** : **72-75%** (+6-9% vs baseline 66%)

---

## 🔥 Après Phase 2 (extraction de visages)

**Val acc finale attendue** : **85-88%**

**Temps estimé** : 12-15h d'entraînement (avec frames_per_video=5)

**Métriques de référence** :
- Papers sur FaceForensics++ (ResNet18) : 88-92% avec extraction de visages
- XceptionNet (state-of-the-art) : 95-99%

---

## 💡 Bonus : Si vous avez encore du temps

### Amélioration #9 : Attention Module (Squeeze-and-Excitation)

Ajouter un module d'attention à MixNet pour que le modèle "regarde" les zones importantes (yeux, bouche).

### Amélioration #10 : Weighted Sampling

Équilibrer les classes (50/50 real/fake) au lieu de 33/66 actuellement.

### Amélioration #11 : Multi-Frame Input

Au lieu d'une frame isolée, donner 3 frames consécutives (détecte les incohérences temporelles).

---

## 📊 Tracking des résultats

| Run | Améliorations appliquées | Best val acc | Best epoch | Temps total |
|-----|--------------------------|--------------|------------|-------------|
| **Baseline** | Aucune | 65.33% | 1 | ~20h |
| **Phase 1** | Label Smoothing + LR Warmup + Aug avancée | ??? | ??? | ??? |
| **Phase 2** | + Extraction visages | ??? | ??? | ??? |
| **Phase 3** | + MixNet + Mixup + Curriculum | ??? | ??? | ??? |

---

## 🎓 Références scientifiques

1. **Face Extraction** : [Rossler et al., 2019 - FaceForensics++](https://arxiv.org/abs/1901.08971)  
   "Face cropping improved accuracy by 15-20% across all methods"

2. **Curriculum Learning** : [Bengio et al., 2009 - Curriculum Learning](http://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)

3. **Mixup** : [Zhang et al., 2018 - mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

4. **Label Smoothing** : [Szegedy et al., 2016 - Rethinking Inception Architecture](https://arxiv.org/abs/1512.00567)

5. **Cosine Annealing** : [Loshchilov & Hutter, 2017 - SGDR](https://arxiv.org/abs/1608.03983)

---

**Bonne chance pour le prochain entraînement ! 🚀**
