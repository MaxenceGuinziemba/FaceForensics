# Journal de Développement - FaceForensics++

## Vue d'ensemble

Ce fichier retrace **tout** ce qu'on fait, pourquoi on le fait, et comment chaque pièce s'emboîte.

---

## État du projet au départ

### Ce que le repo fournit
- **XceptionNet** (`classification/network/xception.py`) : architecture du réseau de neurones, pré-entraîné sur ImageNet (1000 classes d'objets)
- **TransferModel** (`classification/network/models.py`) : wrapper qui adapte XceptionNet (ou ResNet) pour 2 classes (real/fake) au lieu de 1000
- **Transforms** (`classification/dataset/transform.py`) : preprocessing des images (resize 299x299, normalisation)
- **Détection** (`classification/detect_from_video.py`) : script d'inférence - prend une vidéo, détecte les visages, dit si c'est fake ou real
- **Splits** (`dataset/splits/`) : répartition train/val/test (360/70/70 paires de vidéos)
- **Scripts utilitaires** : extraction de frames, compression vidéo

### Ce qui manque (notre travail)
1. Script d'entraînement (`train.py`)
2. Dataset/DataLoader (chargement des données)
3. Data augmentation (amélioration des transforms)
4. Script d'évaluation (`evaluate.py`)
5. Dépendances mises à jour (`requirements.txt`)
6. Correction du chemin hardcodé dans `models.py`
7. Script SLURM pour le cluster GPU

### Problèmes identifiés
- Chemin hardcodé `/home/ondyari/.torch/models/xception-b5690688.pth` dans `models.py`
- Dépendances obsolètes (PyTorch 1.0, Python 3.6)
- Data augmentation inexistante (train = test = même transform)
- Splits en format paires `[target, source]` (360 paires = 720 vidéos manipulées)

---

## Téléchargement du dataset (échantillon de test)

### Contexte
- **Date** : 2026-06-15
- Accès au dataset obtenu par mail des auteurs de FaceForensics++
- Script de téléchargement : `http://kaldir.vc.in.tum.de/faceforensics_download_v4.py`
- Serveur imposé par les auteurs : **EU2** (les autres sont down)

### Contrainte : espace disque limité
- Espace disponible : **5.2 GB** sur cette machine
- Dataset complet c23 : ~10 GB (trop gros)
- Dataset complet c40 : ~2 GB (possible mais serré)
- **Décision** : télécharger un mini-échantillon de 5 vidéos par catégorie en c40 pour tester notre code localement. Le dataset complet sera téléchargé directement sur le cluster GPU de Télécom Paris.

### Ce qu'on a téléchargé
Le script `download-FaceForensics.py` a été récupéré et sauvegardé à la racine du projet.

**Commandes exécutées** :
```bash
# Vidéos originales (5 vidéos, compression c40)
echo "" | python3 download-FaceForensics.py ./data -d original -c c40 -t videos -n 5 --server EU2

# Vidéos manipulées par chaque méthode (5 vidéos chacune, c40)
echo "" | python3 download-FaceForensics.py ./data -d Deepfakes -c c40 -t videos -n 5 --server EU2
echo "" | python3 download-FaceForensics.py ./data -d Face2Face -c c40 -t videos -n 5 --server EU2
echo "" | python3 download-FaceForensics.py ./data -d FaceSwap -c c40 -t videos -n 5 --server EU2
echo "" | python3 download-FaceForensics.py ./data -d NeuralTextures -c c40 -t videos -n 5 --server EU2
```

Note : `echo ""` sert à bypasser le prompt interactif d'acceptation des Terms of Use.

### Structure des données téléchargées
```
data/                                    (6.5 MB total)
├── original_sequences/                  (1.4 MB)
│   └── youtube/c40/videos/
│       ├── 183.mp4                      # Vidéos originales (nommées par ID)
│       ├── 469.mp4
│       ├── 481.mp4
│       ├── 585.mp4
│       └── 599.mp4
└── manipulated_sequences/               (5.2 MB)
    ├── Deepfakes/c40/videos/
    │   ├── 183_253.mp4                  # Format: <target>_<source>.mp4
    │   ├── 469_481.mp4
    │   ├── 481_469.mp4
    │   ├── 585_599.mp4
    │   └── 599_585.mp4
    ├── Face2Face/c40/videos/            # (mêmes noms de fichiers)
    ├── FaceSwap/c40/videos/             # (mêmes noms de fichiers)
    └── NeuralTextures/c40/videos/       # (mêmes noms de fichiers)
```

### Propriétés des vidéos
- **Résolution** : 1280x720 (720p)
- **FPS** : 30
- **Frames par vidéo** : ~390 (soit ~13 secondes)
- **Total** : 5 originales + 20 manipulées = 25 vidéos

### Convention de nommage
- Vidéos originales : `<ID>.mp4` (ex: `183.mp4`)
- Vidéos manipulées : `<target>_<source>.mp4` (ex: `183_253.mp4` = visage de 253 injecté dans la vidéo 183)
- Chaque paire génère 2 vidéos manipulées (A_B et B_A)

### Ce qu'il faudra faire sur le cluster GPU
```bash
# Télécharger le dataset COMPLET en c23 (meilleure qualité, ~10GB)
echo "" | python3 download-FaceForensics.py ~/datasets/faceforensics -d all -c c23 -t videos --server EU2
```

---

## Réorganisation du projet

- **Date** : 2026-06-15
- **Pourquoi** : Le repo original a une structure confuse (fichiers éparpillés, pas de séparation claire entre code source, données, config, docs). On garde les dossiers originaux intacts comme référence et on crée notre propre structure de travail.

### Dossiers originaux (intacts, on n'y touche plus)
- `classification/` - code original des auteurs (modèle, détection, transforms)
- `dataset/` - scripts de génération de données des auteurs
- `images/` - images pour le README original
- `download-FaceForensics.py` - script de téléchargement original

### Notre structure ajoutée
```
src/                        # Notre code source
├── models/                 #   architectures réseau (copiées et adaptées)
│   ├── __init__.py         #   exporte model_selection, TransferModel
│   ├── xception.py         #   architecture XceptionNet
│   └── models.py           #   TransferModel + model_selection (chemin hardcodé corrigé)
├── data/                   #   chargement et preprocessing des données
│   ├── __init__.py         #   exporte xception_default_data_transforms
│   ├── transforms.py       #   transforms pour train/val/test
│   └── dataset.py          #   (à créer) classe Dataset PyTorch
├── __init__.py
├── train.py                #   (à créer) script d'entraînement
├── evaluate.py             #   (à créer) script d'évaluation
└── detect.py               #   détection sur vidéo (imports corrigés)

configs/                    # Configuration
└── splits/                 #   répartition train/val/test
    ├── train.json          #   360 paires
    ├── val.json            #   70 paires
    └── test.json           #   70 paires

scripts/                    # Scripts utilitaires
├── download_dataset.py     #   téléchargement du dataset
├── extract_frames.py       #   extraction de frames depuis vidéos
└── submit_job.sh           #   (à créer) script SLURM

checkpoints/                # Modèles sauvegardés (gitignored)
logs/                       # Logs d'entraînement (gitignored)

docs/                       # Documentation
├── PROJECT_OVERVIEW.md     #   vue d'ensemble du projet
└── GPU_CLUSTER_GUIDE.md    #   guide connexion cluster Télécom Paris
```

### Corrections appliquées pendant la réorganisation

**`src/models/models.py`** (Point 6 - chemin hardcodé) :
- **Avant** : `torch.load('/home/ondyari/.torch/models/xception-b5690688.pth')` (chemin personnel de l'auteur, ne marche que sur sa machine)
- **Après** : `torch.hub.load_state_dict_from_url(XCEPTION_URL, map_location='cpu')` (téléchargement automatique depuis internet, marche partout)
- Import corrigé : `from network.xception` → `from src.models.xception`
- Import supprimé : `pretrainedmodels` (lib inutile et obsolète)

**`src/detect.py`** :
- Imports corrigés pour pointer vers `src.models` et `src.data` au lieu de `network` et `dataset`

**`.gitignore`** créé :
- Ignore `data/`, `checkpoints/`, `logs/`, `__pycache__/`, fichiers IDE, fichiers `.pth`

---

### [Point 5] Mise à jour de requirements.txt

- **Date** : 2026-06-15
- **Fichier** : `requirements.txt` (nouveau, à la racine)
- **Problème** : L'ancien `classification/requirements.txt` datait de 2019. Il contenait des versions obsolètes (PyTorch 1.0, Keras 2.2, Python 3.6) et des dépendances inutiles (`pretrainedmodels`, `Keras`, `grpcio`, `protobuf`...).
- **Méthode** : On a scanné tous les imports réels dans `src/` pour lister uniquement ce qu'on utilise.
- **Résultat** : Nouveau fichier avec 11 dépendances (vs 32 avant), versions modernes, organisées par usage :

| Package | Rôle | Ancien → Nouveau |
|---------|------|-------------------|
| `torch` | Framework deep learning | 1.0.1 → >=2.0.0 |
| `torchvision` | Modèles pré-entraînés + transforms | 0.2.1 → >=0.15.0 |
| `opencv-python` | Lecture vidéo frame par frame | 3.4.1 → >=4.8.0 |
| `Pillow` | Manipulation d'images | 5.4.1 → >=10.0.0 |
| `numpy` | Calcul matriciel | 1.16.2 → >=1.24.0 |
| `dlib` | Détection de visages | 19.15 → >=19.24.0 |
| `scikit-learn` | Métriques (AUC, confusion matrix) | **nouveau** |
| `matplotlib` | Graphiques | **nouveau** |
| `tensorboard` | Monitoring d'entraînement | **nouveau** |
| `tqdm` | Barres de progression | 4.25.0 → >=4.65.0 |

Dépendances **supprimées** (inutiles) : `Keras`, `pretrainedmodels`, `grpcio`, `protobuf`, `gast`, `astor`, `absl-py`, `h5py`, `face-recognition`, `face-recognition-models`, `ffmpy`, `munch`, `nvidia-ml-py3`, et 8 autres.

---

### [Point 3] Data augmentation - transforms.py

- **Date** : 2026-06-15
- **Fichier** : `src/data/transforms.py`
- **Problème** : Les transforms de train, val et test étaient identiques (juste Resize + Normalize). Le modèle voyait toujours la même image de la même façon → risque d'overfitting (apprendre par coeur au lieu de généraliser).
- **Solution** : Ajout de transforms aléatoires **uniquement sur le train** (val/test restent déterministes, c'est voulu : on évalue dans des conditions stables).

**Transforms ajoutées au train :**

| Transform | Paramètres | Effet |
|-----------|-----------|-------|
| `RandomHorizontalFlip` | p=0.5 | Miroir horizontal 1 fois sur 2 - un fake est aussi fake en miroir |
| `RandomRotation` | 15° | Rotation -15° à +15° - les visages ne sont pas toujours droits |
| `ColorJitter` | brightness/contrast/saturation=0.2 | Variations d'éclairage aléatoires - simule différentes conditions de lumière |
| `RandomGrayscale` | p=0.05 | Noir et blanc 5% du temps - force le modèle à ne pas se baser que sur la couleur |
| `GaussianBlur` | kernel=3, sigma=0.1-1.0 | Flou léger - simule la compression vidéo (pertinent pour c23/c40) |

**Pipeline complète pour le train :**
`Resize(299) → RandomHorizontalFlip → RandomRotation → ColorJitter → RandomGrayscale → GaussianBlur → ToTensor → Normalize`

**Aussi refactorisé :** Les constantes `IMAGE_SIZE=299`, `NORM_MEAN`, `NORM_STD` sont extraites en variables pour éviter la duplication et faciliter les changements.

**Test effectué** sur une vraie frame de `data/original_sequences/youtube/c40/videos/183.mp4` :
- Input : image 1280x720 → output : tensor [3, 299, 299] normalisé entre -1 et 1
- Diff entre 2 appels du transform train = 0.39 (confirme que l'augmentation est bien aléatoire)
- Le val produit un résultat identique à chaque appel (déterministe, correct)

---

## Modifications à venir

### Ordre de travail
1. ~~**Point 6** : Corriger chemin hardcodé dans `models.py`~~ **FAIT**
2. ~~**Point 5** : Mettre à jour `requirements.txt`~~ **FAIT**
3. ~~**Point 3** : Améliorer les transforms (data augmentation)~~ **FAIT**
4. ~~**Point 2** : Créer le Dataset/DataLoader~~ **FAIT**
5. ~~**Point 1** : Créer `train.py`~~ **FAIT**
6. ~~**Point 4** : Créer `evaluate.py`~~ **FAIT**
7. ~~**Point 7** : Préparer le script SLURM~~ **FAIT**

**Tous les points sont terminés.**

---

### [Point 2] Dataset/DataLoader - dataset.py

- **Date** : 2026-06-15
- **Fichier** : `src/data/dataset.py`
- **Rôle** : C'est la brique qui connecte les données au modèle. PyTorch a besoin d'un objet `Dataset` qui sait comment charger et servir les images une par une, et d'un `DataLoader` qui regroupe ces images en batch.

**Choix d'architecture : lecture directe des vidéos (pas d'extraction de frames)**
- Approche classique : extraire toutes les frames en PNG d'abord, puis les charger → coûte 50-100GB d'espace disque, 1-2h d'extraction
- Notre approche : ouvrir chaque .mp4 avec OpenCV et lire une frame aléatoire à la volée → 0 espace supplémentaire, démarrage immédiat

**Comment ça marche, étape par étape :**

1. `__init__()` : lit le fichier split JSON (ex: `train.json`), parcourt chaque paire `[target, source]`, et construit une liste de `(chemin_video, label)` :
   - `original_sequences/youtube/c40/videos/183.mp4` → label 0 (real)
   - `manipulated_sequences/Deepfakes/c40/videos/183_253.mp4` → label 1 (fake)
   - Déduplique les vidéos originales (un même ID apparaît dans plusieurs paires)

2. `__len__()` : retourne `nb_vidéos × num_frames_per_video`. Avec `num_frames_per_video=10`, chaque vidéo est échantillonnée 10 fois par epoch avec une frame différente à chaque fois.

3. `__getitem__(idx)` : appelé par le DataLoader pour chaque image
   - Calcule quelle vidéo correspond à l'index
   - Ouvre la vidéo avec OpenCV
   - Saute à une frame aléatoire (`random.randint`)
   - Convertit BGR→RGB, puis en PIL Image
   - Applique le transform (resize, augmentation, normalisation)
   - Retourne `(tensor [3, 299, 299], label)`

4. Le `DataLoader` de PyTorch appelle `__getitem__` en boucle pour construire des batch de 32 images

**Paramètres importants :**
- `compression` : 'c40' (test local) ou 'c23' (cluster, meilleure qualité)
- `methods` : quelles manipulations inclure (default: les 4)
- `num_frames_per_video` : contrôle la taille d'un epoch (10 = chaque vidéo vue 10 fois avec des frames différentes)

**Tests effectués sur nos 25 vidéos locales :**
- train : 7 vidéos trouvées (3 real, 4 fake) → 35 samples/epoch
- val : 6 vidéos (2 real, 4 fake) → 30 samples/epoch
- test : 0 vidéos (normal, les IDs du test split ne matchent pas nos 5 vidéos téléchargées)
- Batch shape vérifié : `[4, 3, 299, 299]` avec labels `[1, 0, 1, 1]`

---

### [Point 1] Script d'entraînement - train.py

- **Date** : 2026-06-15
- **Fichier** : `src/train.py`
- **Rôle** : Chef d'orchestre qui assemble le modèle, le DataLoader, et exécute la boucle d'entraînement.

**Ce que fait le script, étape par étape :**

1. **Initialisation** :
   - Détecte si un GPU est disponible (`cuda` vs `cpu`)
   - Charge le modèle XceptionNet (ou ResNet18) pré-entraîné sur ImageNet
   - Remplace la dernière couche pour 2 classes (real/fake)
   - Crée les datasets train et val via `FaceForensicsDataset`
   - Crée les DataLoaders (regroupent les images en batch de 32)

2. **Pour chaque epoch** :
   - **Train** : pour chaque batch, le modèle prédit, on calcule l'erreur (CrossEntropyLoss), on ajuste les poids (backpropagation via Adam)
   - **Validation** : même chose SANS ajuster les poids — juste mesurer les performances
   - **Scheduler** : si la val loss ne s'améliore plus pendant 5 epochs, le learning rate est divisé par 10
   - **Sauvegarde** : si c'est la meilleure val accuracy, on sauvegarde le modèle complet dans `checkpoints/best_model.pth`
   - **Early stopping** : si pas d'amélioration pendant 10 epochs → arrêt automatique

3. **Logging** : toutes les métriques sont envoyées à TensorBoard (courbes dans `logs/`)

**Arguments en ligne de commande :**
```
--data_root       Chemin vers data/                    (default: data)
--compression     c0, c23 ou c40                       (default: c40)
--model           xception ou resnet18                 (default: xception)
--dropout         Taux de dropout                      (default: 0.5)
--epochs          Nombre max d'epochs                  (default: 50)
--batch_size      Taille du batch                      (default: 32)
--lr              Learning rate                        (default: 0.0002)
--patience        Epochs avant early stopping          (default: 10)
--frames_per_video  Frames par vidéo par epoch         (default: 10)
--num_workers     Workers pour chargement données      (default: 4)
```

**Commande de test local :**
```bash
python -m src.train --data_root data --compression c40 --epochs 2 --batch_size 4 --num_workers 0 --frames_per_video 2 --model resnet18
```

**Commande pour le cluster GPU (à venir) :**
```bash
python -m src.train --data_root ~/datasets/ff++ --compression c23 --epochs 50 --batch_size 32 --model xception
```

**Test effectué (CPU, resnet18, 2 epochs) :**
- Epoch 1 : train_loss=0.9574, train_acc=0.8571, val_loss=1.1032, val_acc=0.6667
- Epoch 2 : train_loss=0.7692, train_acc=0.7857, val_loss=4.2246, val_acc=0.6667
- Checkpoint sauvegardé (122 tenseurs, val_acc=0.6667)
- Logs TensorBoard générés dans `logs/`
- Durée : ~30s par epoch sur CPU (sera quelques secondes sur GPU)

**Note :** Le téléchargement automatique des poids XceptionNet échoue actuellement (certificat SSL expiré sur le serveur). Sur le cluster, on pourra soit télécharger manuellement les poids, soit utiliser `SSL_CERT_FILE` ou tester d'abord avec ResNet18. Le test local a été fait avec ResNet18 pour contourner ce problème.

**Contenu du checkpoint sauvegardé :**
- `model_state_dict` : les poids entraînés du modèle
- `optimizer_state_dict` : état de l'optimizer (pour reprendre l'entraînement)
- `val_acc`, `val_loss` : métriques du meilleur epoch
- `epoch` : numéro du meilleur epoch
- `args` : tous les hyperparamètres utilisés (pour reproduire l'expérience)

---

### [Point 4] Script d'évaluation - evaluate.py

- **Date** : 2026-06-15
- **Fichier** : `src/evaluate.py`
- **Rôle** : Mesurer rigoureusement les performances d'un modèle entraîné.

**Ce que le script produit :**

1. **Métriques globales** (real vs fake, toutes méthodes confondues) :
   - Accuracy : % de bonnes réponses
   - Precision : quand il dit "fake", a-t-il raison ?
   - Recall : combien de vrais fakes a-t-il détectés ?
   - F1-Score : moyenne harmonique precision/recall
   - AUC-ROC : qualité du classifieur indépendamment du seuil de décision

2. **Performances par méthode** : le modèle est évalué séparément sur Deepfakes, Face2Face, FaceSwap, NeuralTextures pour voir quelle manipulation est la plus facile/difficile à détecter

3. **Fichiers générés dans `checkpoints/evaluation/`** :
   - `confusion_matrix.png` : visualisation de la matrice de confusion
   - `roc_curve.png` : courbe ROC avec AUC
   - `evaluation_report.json` : toutes les métriques en format structuré

**Fonctionnalités :**
- Lit automatiquement le nom du modèle et le dropout depuis le checkpoint (pas besoin de les re-spécifier)
- Si aucune vidéo test n'est trouvée, fallback automatique sur le val set
- Évaluation par méthode gère les cas où une seule classe est présente

**Commande :**
```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pth --data_root data --compression c40
```

**Test effectué :** évaluation du modèle ResNet18 (2 epochs) sur le val set local.
- Le modèle prédit tout comme "fake" (normal : sous-entraîné, 7 vidéos seulement)
- Les 3 fichiers de sortie sont générés correctement
- L'évaluation par méthode fonctionne pour les 4 méthodes

---

### [Point 7] Scripts SLURM pour le cluster GPU

- **Date** : 2026-06-15
- **Fichiers** : `scripts/setup_cluster.sh`, `scripts/submit_train.sh`, `scripts/submit_eval.sh`
- **Rôle** : Automatiser le setup et la soumission de jobs sur le cluster Télécom Paris.

**3 scripts créés :**

#### `scripts/setup_cluster.sh` - Setup initial (à exécuter une seule fois)
Ce script prépare tout l'environnement sur le cluster :
1. Crée un environnement virtuel Python dans `~/venvs/faceforensics/`
2. Installe PyTorch avec CUDA 11.8 + toutes les dépendances
3. Vérifie que le GPU est détecté
4. Télécharge le dataset complet en c23 (~10GB)

```bash
# Première connexion au cluster
ssh votre_login@gpu-gw.enst.fr
cd ~/projects/FaceForensics
bash scripts/setup_cluster.sh
```

#### `scripts/submit_train.sh` - Entraînement
Configuration SLURM :
- **Partition** : 3090 (RTX 3090, 24GB VRAM)
- **CPUs** : 8 (pour les workers du DataLoader)
- **RAM** : 32GB
- **Temps** : 24h (marge de sécurité, l'entraînement devrait prendre 6-12h)
- **Hyperparamètres** : XceptionNet, batch_size=32, lr=0.0002, 50 epochs, dropout=0.5

```bash
sbatch scripts/submit_train.sh     # Soumettre le job
squeue -u votre_login              # Vérifier le statut
tail -f logs/slurm/train_*.out     # Suivre les logs en temps réel
scancel <job_id>                   # Annuler si besoin
```

#### `scripts/submit_eval.sh` - Évaluation (après l'entraînement)
- **Temps** : 2h (l'évaluation est beaucoup plus rapide)
- **RAM** : 16GB
- Charge automatiquement le meilleur modèle depuis `checkpoints/best_model.pth`

```bash
sbatch scripts/submit_eval.sh
```

**Workflow complet sur le cluster :**
```bash
# 1. Setup (une seule fois)
bash scripts/setup_cluster.sh

# 2. Entraînement
sbatch scripts/submit_train.sh
squeue -u votre_login          # Attendre que ça finisse

# 3. Évaluation
sbatch scripts/submit_eval.sh

# 4. Récupérer les résultats sur ta machine locale
# (depuis ta machine)
scp -r votre_login@gpu-gw.enst.fr:~/projects/FaceForensics/checkpoints/ ./checkpoints/
```

---

## Récapitulatif final

Tous les 7 points sont terminés. Le projet est prêt à être déployé sur le cluster.

### Fichiers créés (notre travail)
```
src/
├── models/
│   ├── __init__.py
│   ├── xception.py         ← Architecture XceptionNet (copié, inchangé)
│   └── models.py           ← TransferModel (chemin hardcodé corrigé, imports corrigés)
├── data/
│   ├── __init__.py
│   ├── transforms.py       ← Data augmentation ajoutée (5 transforms sur le train)
│   └── dataset.py          ← NOUVEAU : DataLoader qui lit les vidéos à la volée
├── __init__.py
├── train.py                ← NOUVEAU : script d'entraînement complet
├── evaluate.py             ← NOUVEAU : évaluation avec métriques + graphiques
└── detect.py               ← Détection sur vidéo (imports corrigés)

configs/splits/             ← Splits copiés depuis le repo original
scripts/
├── setup_cluster.sh        ← NOUVEAU : setup initial cluster
├── submit_train.sh         ← NOUVEAU : job SLURM entraînement
├── submit_eval.sh          ← NOUVEAU : job SLURM évaluation
├── download_dataset.py     ← Script de téléchargement (copié)
└── extract_frames.py       ← Extraction de frames (copié)

requirements.txt            ← NOUVEAU : dépendances modernes (11 au lieu de 32)
.gitignore                  ← NOUVEAU
```

### Ce qui reste à faire (quand on aura accès au GPU)
1. Se connecter au cluster : `ssh votre_login@gpu-gw.enst.fr`
2. Cloner le projet et lancer `bash scripts/setup_cluster.sh`
3. Lancer l'entraînement : `sbatch scripts/submit_train.sh`
4. Évaluer : `sbatch scripts/submit_eval.sh`
5. Analyser les résultats et rédiger le rapport
