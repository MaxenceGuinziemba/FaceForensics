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

---

## Premier entraînement sur le cluster GPU

### Setup et problèmes rencontrés

- **Date** : 2026-06-17 puis 2026-06-19
- **Cluster** : gpu-gw.enst.fr, node40, RTX 3090 24GB
- **Dataset** : 2160 vidéos train (720 real, 1440 fake), 420 vidéos val (140 real, 280 fake)

#### Problème 1 : Certificat SSL expiré pour XceptionNet

**Erreur** :
```
ssl.SSLCertVerificationError: certificate has expired
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```

Le serveur `data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth` a un certificat SSL expiré ET retourne une erreur 503 Service Unavailable.

**Solutions tentées** :
1. ❌ Téléchargement manuel avec `wget --no-check-certificate` : fichier vide (0 bytes)
2. ❌ Serveur down (503 Service Unavailable)

**Solution retenue** : Utiliser **ResNet18** à la place de XceptionNet
- ResNet18 est inclus dans torchvision (pas de téléchargement externe)
- Performances attendues : 95-97% au lieu de 98-99% (acceptable pour le projet)
- Plus rapide à entraîner (11M params vs 23M)

Modification du script SLURM :
```bash
sed -i 's/--model xception/--model resnet18/' scripts/submit_train.sh
```

### Résultats du premier entraînement (Job 856564)

**Hyperparamètres initiaux** :
- Model : ResNet18
- Learning rate : 0.0002
- Dropout : 0.5
- Batch size : 32
- Frames per video : 10
- Epochs : 50 max
- Early stopping patience : 10

**Résultats par epoch** :

| Epoch | Train loss | Train acc | Val loss | Val acc | LR | Temps | Observation |
|-------|------------|-----------|----------|---------|-----|-------|-------------|
| 1 | 0.6354 | 66.25% | **0.6118** | **65.33%** ✅ | 2.0e-04 | 2745s (46 min) | **Meilleur modèle** - performances équilibrées |
| 2 | 0.5279 | 74.30% | 0.7038 | 62.93% | 2.0e-04 | 2698s | Début d'overfitting (val acc baisse) |
| 3 | 0.4603 | 79.18% | 0.6797 | 63.14% | 2.0e-04 | 2708s | Overfitting se confirme (écart +16%) |
| 4 | 0.4257 | 80.96% | 1.0487 | 53.24% ⚠️ | 2.0e-04 | 2715s | Effondrement : val acc -12% |
| 5 | 0.4004 | 82.07% | 1.1247 | 54.31% ⚠️ | 2.0e-04 | 2725s | Pire epoch : val loss explose |
| 6 | 0.3850 | 82.95% | 0.8490 | 60.74% | 2.0e-04 | 2720s | Légère remontée (+6%) |
| 7 | 0.3662 | 83.76% | 0.8950 | 65.24% ✅ | **2.0e-05** ⭐ | 2715s | **Scheduler intervient** - retour à 65% |

**État actuel** : Job en cours, epoch 7 terminée (19/06/2026 ~11h)

### Analyse des problèmes d'overfitting (epochs 1-7)

#### Diagnostic

**Overfitting sévère observé (epochs 2-6)** :
- Train acc augmente : 66% → 83% (+17%)
- Val acc chute : 65% → 54% (-11%) puis remonte à 61%
- Écart train/val max : 28% (epoch 5)
- Val loss explose : 0.61 → 1.12 (+84%)

**Causes identifiées** :
1. **Learning rate trop élevé** (0.0002) : le modèle apprend trop vite et "mémorise" au lieu de généraliser
2. **Dropout faible** (0.5) : régularisation insuffisante
3. **Dataset relativement petit** (2160 vidéos) : risque d'overfitting accru
4. **Epochs longues** (45 min) : détection tardive des problèmes

#### Intervention du scheduler ReduceLROnPlateau

**Epoch 7 : Le scheduler a sauvé l'entraînement**
- Configuration : `patience=5, factor=0.1`
- Déclenchement : après 5 epochs sans amélioration de val_loss
- Action : LR divisé par 10 (2.0e-04 → 2.0e-05)
- Résultat immédiat : val_acc remonte de 60.74% → 65.24%
- Écart train/val réduit : 28% → 18.5%

**Pourquoi ça fonctionne** :
- LR plus faible = pas à pas plus petits dans l'espace des poids
- Le modèle "affine" au lieu de "sauter brutalement"
- Moins de risque de surajustement

### Recommandations pour le prochain entraînement

#### 🎯 Hyperparamètres optimisés

Si vous devez **relancer un entraînement depuis le début**, utilisez ces paramètres :

**Fichier : `scripts/submit_train.sh`**

```bash
python3 -m src.train \
    --data_root data \
    --compression c23 \
    --model resnet18 \
    --dropout 0.6 \              # ↑ de 0.5 : régularisation renforcée
    --epochs 50 \
    --batch_size 32 \
    --lr 0.00005 \               # ↓ de 0.0002 : apprentissage stable dès le départ
    --weight_decay 1e-4 \
    --patience 15 \              # ↑ de 10 : plus de tolérance avec LR faible
    --frames_per_video 5 \       # ↓ de 10 : epochs 2× plus courtes (22 min)
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir logs/tensorboard
```

#### 📊 Gains attendus avec les paramètres optimisés

| Métrique | Config actuelle | Config optimisée | Amélioration |
|----------|----------------|------------------|--------------|
| **Epoch 1 val acc** | 65.33% | 68-72% | +3-7% |
| **Overfitting epochs 2-6** | Oui (chute à 54%) | Non (montée stable) | ✅ Évité |
| **Temps par epoch** | 45 min | 22 min | **÷2** |
| **Val acc finale estimée** | 85-90% | 90-92% | +2-5% |
| **Temps total estimé** | ~20h | **10-12h** | **÷2** |
| **Early stopping** | Epoch ~25 | Epoch ~25-30 | Similaire |

#### 🔧 Script complet pour relancer

```bash
# 1. Annuler le job actuel (si nécessaire)
scancel <job_id>

# 2. Sauvegarder les résultats actuels
cd ~/projects/FaceForensics
mkdir -p logs/archive
cp logs/slurm/train_*.{out,err} logs/archive/
cp checkpoints/best_model.pth checkpoints/best_model_run1.pth 2>/dev/null || true

# 3. Appliquer les modifications
sed -i 's/--lr 0.0002/--lr 0.00005/' scripts/submit_train.sh
sed -i 's/--dropout 0.5/--dropout 0.6/' scripts/submit_train.sh
sed -i 's/--frames_per_video 10/--frames_per_video 5/' scripts/submit_train.sh
sed -i 's/--patience 10/--patience 15/' scripts/submit_train.sh

# 4. Vérifier les modifications
grep -E "(lr|dropout|frames_per_video|patience)" scripts/submit_train.sh

# 5. Relancer
sbatch scripts/submit_train.sh

# 6. Suivre la progression
squeue -u $USER
tail -f logs/slurm/train_*.out
```

#### 📈 Résultats attendus avec la config optimisée

**Epochs 1-5 (premières 2 heures)** :
```
Epoch 1 | Train loss: 0.58 acc: 0.68 | Val loss: 0.55 acc: 0.70 | lr: 5.0e-05 | 1300s
Epoch 2 | Train loss: 0.50 acc: 0.74 | Val loss: 0.48 acc: 0.75 | lr: 5.0e-05 | 1280s
Epoch 3 | Train loss: 0.44 acc: 0.78 | Val loss: 0.43 acc: 0.78 | lr: 5.0e-05 | 1290s
Epoch 4 | Train loss: 0.40 acc: 0.81 | Val loss: 0.39 acc: 0.81 | lr: 5.0e-05 | 1285s
Epoch 5 | Train loss: 0.37 acc: 0.83 | Val loss: 0.36 acc: 0.83 | lr: 5.0e-05 | 1295s
```

**Caractéristiques attendues** :
- ✅ Train acc et val acc augmentent ensemble (pas d'overfitting)
- ✅ Écart train/val < 5% (excellent)
- ✅ Amélioration régulière et stable

**Epochs 15-20 (plateau)** :
```
Epoch 18 | Train loss: 0.22 acc: 0.92 | Val loss: 0.24 acc: 0.91 | lr: 5.0e-06 | 1280s
  → Nouveau meilleur modèle sauvegardé (val_acc=0.9100)
```

**Early stopping** : vers epoch 25-30 avec **val_acc finale : 90-92%**

#### 🧪 Pourquoi ces changements fonctionnent

1. **LR 0.00005 (au lieu de 0.0002)** :
   - Le scheduler a prouvé qu'un LR réduit stabilise l'apprentissage (epoch 7)
   - Autant commencer directement avec un LR optimal
   - Évite la phase chaotique des epochs 2-6
   - **Why:** Learning rate = taille des pas dans l'espace des poids. Trop grand → le modèle "saute" et rate les minima. Plus petit → convergence stable.

2. **Dropout 0.6 (au lieu de 0.5)** :
   - Plus de régularisation = moins d'overfitting
   - Force le modèle à apprendre des features robustes
   - **Why:** Dropout désactive aléatoirement des neurones pendant l'entraînement. Plus de dropout = le modèle ne peut pas se reposer sur des neurones spécifiques → généralisation forcée.

3. **Frames 5 (au lieu de 10)** :
   - Epochs 2× plus courtes (22 min vs 45 min)
   - Itérations plus rapides = détection plus rapide des problèmes
   - Moins de risque de surajustement par epoch
   - **Why:** Avec frames=10, chaque vidéo est vue 10 fois par epoch. Avec frames=5, seulement 5 fois. Le modèle voit moins de répétitions par epoch, mais fait plus d'epochs dans le même temps → meilleure exploration.

4. **Patience 15 (au lieu de 10)** :
   - Avec un LR plus faible, l'amélioration est plus progressive
   - Il faut plus de patience avant de conclure à un plateau
   - **Why:** LR faible = petits pas = amélioration lente mais stable. Il faut plus d'epochs pour atteindre un plateau réel.

#### 💡 Leçons apprises

**Ce qui a bien fonctionné** :
- ✅ Dataset bien structuré (2160 train, 420 val)
- ✅ Data augmentation efficace (transforms.py)
- ✅ Scheduler ReduceLROnPlateau sauve l'entraînement (intervention epoch 7)
- ✅ Early stopping évite le gaspillage de compute

**Ce qui nécessite amélioration** :
- ⚠️ LR initial trop élevé → commencer avec 0.00005
- ⚠️ Dropout trop faible → passer à 0.6
- ⚠️ Epochs trop longues (45 min) → réduire à 22 min avec frames=5
- ⚠️ Certificat SSL XceptionNet expiré → utiliser ResNet18 ou télécharger manuellement

**Métriques de référence (ResNet18 sur FaceForensics++)** :
- Epoch 1 baseline : **65-70% val acc** ✅
- Val acc finale attendue : **90-92%** (avec hyperparamètres optimisés)
- Temps total estimé : **10-12 heures** (50 epochs × 22 min, early stopping ~epoch 25)

#### 🎯 Checklist avant de lancer un nouvel entraînement

- [ ] Vérifier que le dataset est bien présent (`ls ~/projects/FaceForensics/data/`)
- [ ] Utiliser les hyperparamètres optimisés (lr=0.00005, dropout=0.6, frames=5)
- [ ] Vérifier l'environnement virtuel (`source ~/venvs/faceforensics/bin/activate`)
- [ ] Créer les dossiers de logs (`mkdir -p logs/slurm`)
- [ ] Soumettre le job (`sbatch scripts/submit_train.sh`)
- [ ] Vérifier que le job démarre (`squeue -u $USER`)
- [ ] Suivre l'epoch 1 pour valider que val_acc > 68%
- [ ] Vérifier epochs 3-5 : pas d'overfitting (train_acc ≈ val_acc)
- [ ] Laisser tourner jusqu'à early stopping (10-12h)

### Commandes utiles pour surveiller l'entraînement

```bash
# Voir l'état du job
squeue -u mguinzie-24

# Voir toutes les epochs terminées
grep "Epoch" ~/projects/FaceForensics/logs/slurm/train_<job_id>.out

# Suivre la progression en temps réel (barre tqdm dans .err)
tail -f ~/projects/FaceForensics/logs/slurm/train_<job_id>.err

# Vérifier le checkpoint sauvegardé
ls -lh ~/projects/FaceForensics/checkpoints/

# Voir les 5 dernières epochs
grep "Epoch" ~/projects/FaceForensics/logs/slurm/train_<job_id>.out | tail -5

# Annuler un job
scancel <job_id>
```

### Déconnexion sans arrêter l'entraînement

**Important** : Le job SLURM tourne sur le cluster, pas sur votre machine locale.

Vous pouvez **fermer le terminal, éteindre votre ordinateur, partir** sans affecter l'entraînement.

```bash
# Quitter proprement
exit

# Revenir plus tard
ssh mguinzie-24@gpu-gw.enst.fr
source ~/venvs/faceforensics/bin/activate
grep "Epoch" ~/projects/FaceForensics/logs/slurm/train_*.out
```

Le job continuera jusqu'à :
- Completion normale (early stopping ou 50 epochs)
- Annulation manuelle (`scancel <job_id>`)
- Limite de temps SLURM (24h)

---

## Prochaines étapes (quand l'entraînement actuel sera terminé)

### 1. Évaluation sur le test set

```bash
cd ~/projects/FaceForensics
sbatch scripts/submit_eval.sh
```

Cela générera :
- `checkpoints/evaluation/evaluation_report.json` : métriques détaillées
- `checkpoints/evaluation/confusion_matrix.png` : matrice de confusion
- `checkpoints/evaluation/roc_curve.png` : courbe ROC avec AUC

### 2. Récupération des résultats en local

```bash
# Depuis votre machine locale
scp -r mguinzie-24@gpu-gw.enst.fr:~/projects/FaceForensics/checkpoints/ ./
scp -r mguinzie-24@gpu-gw.enst.fr:~/projects/FaceForensics/logs/ ./
```

### 3. Visualisation avec TensorBoard

```bash
# En local
tensorboard --logdir=logs/tensorboard
# Ouvrir http://localhost:6006
```

### 4. Analyse des résultats

- Comparer les performances par méthode (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
- Identifier quelle manipulation est la plus facile/difficile à détecter
- Analyser la courbe ROC et l'AUC
- Examiner la matrice de confusion pour voir les faux positifs/négatifs

### 5. Rédaction du rapport

Points à inclure :
- Dataset utilisé (2160 train, 420 val, compression c23)
- Architecture : ResNet18 (11M params) avec dropout 0.6
- Hyperparamètres optimaux trouvés
- Problème d'overfitting rencontré et solution (scheduler)
- Métriques finales (accuracy, precision, recall, F1, AUC)
- Performances par méthode de manipulation
- Temps d'entraînement total

---

## Entraînement V3 : EfficientNet + Face Extraction + Mixup + Freeze/Unfreeze

### Configuration

- **Date** : 2026-06-20 → 2026-06-21
- **Job** : submit_train_optimized.sh
- **Modèle** : EfficientNet-B0 (5.3M params)
- **Hyperparamètres** : lr=0.00003, dropout=0.5, weight_decay=5e-4, batch_size=32, frames_per_video=5
- **Scheduler** : CosineAnnealingLR (T_max=50, eta_min=1e-6)
- **Améliorations appliquées** :
  - Extraction de visages (OpenCV DNN SSD) dans `src/data/face_extraction.py`
  - Mixup (alpha=0.4, probabilité 50%) dans `src/data/mixup.py`
  - Augmentations avancées (JPEGCompression, GaussianNoise, CutOut) dans `src/data/augmentations.py`
  - Label smoothing (0.1)
  - LR warmup (3 epochs, 1e-6 → 3e-5)
  - Freeze/unfreeze (classifier seul pendant 5 epochs, puis unfreeze tout)

### Résultats (50 epochs, pas d'early stopping)

| Epoch | Train loss | Train acc | Val loss | Val acc | LR |
|-------|------------|-----------|----------|---------|-----|
| 1 | - | - | - | - | ~1e-5 (warmup) |
| 45 | 0.6214 | 68.62% | 0.5973 | 72.33% | 2.8e-06 |
| 46 | 0.6249 | 67.86% | 0.5973 | 73.29% | 2.4e-06 |
| 47 | 0.6242 | 68.20% | 0.5952 | 73.29% | 2.0e-06 |
| 48 | 0.6242 | 68.23% | 0.6013 | 73.05% | 1.7e-06 |
| 49 | 0.6251 | 67.98% | 0.6048 | 72.95% | 1.5e-06 |
| 50 | 0.6241 | 68.28% | 0.5992 | 72.57% | 1.3e-06 |

**Meilleure val accuracy : 73.81%**
**Durée** : ~20h (50 × ~1430s/epoch)

### Analyse : pourquoi ça plafonne à 73%

**Symptôme principal** : train acc (68%) < val acc (73%). C'est un pattern d'underfitting — le modèle ne peut pas apprendre correctement sur le train set.

**Bug critique identifié** : dans `src/models/models.py`, méthode `set_trainable_up_to()`, le `return` à la ligne 88 était **à l'intérieur du for loop** au lieu d'être après :

```python
# BUGUÉ (V3) :
if layername is None:
    for i, param in self.model.named_parameters():
        param.requires_grad = True
        return  # ← retourne après le 1er paramètre !

# CORRIGÉ (V4) :
if layername is None:
    for i, param in self.model.named_parameters():
        param.requires_grad = True
    return  # ← retourne après avoir unfreeezé TOUS les paramètres
```

**Conséquence** : à l'epoch 6, quand `train.py` appelle `model.set_trainable_up_to(False, layername=None)` pour unfreezer le backbone, seul le **premier paramètre** était dégelé. Le backbone EfficientNet restait gelé pendant les 45 epochs suivantes. Le modèle n'entraînait que son classifier head — d'où le plafond à 68% train / 73% val.

**Problèmes secondaires** :
- CosineAnnealingLR poussait le LR à ~1e-6 dès epoch 40, le modèle arrêtait d'apprendre trop tôt
- Mixup à 50% des batches ajoutait trop de bruit sur un modèle qui ne pouvait déjà pas apprendre

---

## V4 : Corrections appliquées

- **Date** : 2026-06-22

### Modifications (3 changements ciblés)

1. **Fix unfreeze bug** (`src/models/models.py:88`) : dédentation du `return` pour qu'il soit après la boucle for. Tout le backbone s'unfreeze maintenant correctement à l'epoch 6.

2. **Scheduler** (`src/train.py:142-144`) : remplacement de `CosineAnnealingLR` par `ReduceLROnPlateau(mode='min', factor=0.5, patience=5, min_lr=1e-7)`. Le LR ne baisse que quand la val loss stagne réellement.

3. **Mixup** (`src/train.py:56`) : probabilité réduite de 0.5 à 0.3. Plus d'exemples réels pour que le backbone puisse apprendre.

### Résultats V4 (Job 860139 - en cours au 2026-06-23)

**Cluster** : gpu-gw.enst.fr, node40, RTX 3090 24GB
**Config** : EfficientNet-B0, lr=3e-05, dropout=0.5, weight_decay=5e-4, batch_size=32, frames_per_video=5, patience=20 (early stopping), ReduceLROnPlateau(factor=0.5, patience=5)

| Epoch | Train loss | Train acc | Val loss | Val acc | LR | Observation |
|-------|------------|-----------|----------|---------|-----|-------------|
| 1 | 0.6684 | 62.05% | 0.6583 | 65.81% | 1.1e-05 | Warmup (1/3) |
| 2 | 0.6573 | 65.86% | 0.6501 | 66.81% | 2.0e-05 | Warmup (2/3) |
| 3 | 0.6518 | 65.97% | 0.6423 | 67.43% | 3.0e-05 | Warmup terminé |
| 4 | 0.6474 | 66.35% | 0.6354 | 67.43% | 3.0e-05 | Freeze (classifier seul) |
| 5 | 0.6466 | 66.39% | 0.6353 | 67.29% | 3.0e-05 | Dernière epoch freeze |
| **6** | **0.6184** | **68.53%** | **0.5358** | **77.62%** | 3.0e-05 | **Unfreeze → saut +10% val acc** ✅ |
| 7 | 0.5669 | 73.48% | 0.4701 | 83.67% | 3.0e-05 | Backbone apprend rapidement |
| 8 | 0.5329 | 76.80% | 0.4297 | 86.05% | 3.0e-05 | |
| 9 | 0.5017 | 79.50% | 0.4083 | 87.57% | 3.0e-05 | |
| 10 | 0.4723 | 82.04% | 0.3920 | 88.19% | 3.0e-05 | Meilleur modèle |
| 11-15 | ↘ 0.41 | ↗ 87% | ~0.40 | ~88% | 3.0e-05 | Plateau, oscillations |
| **16** | 0.4123 | 86.74% | **0.3777** | **89.05%** | 3.0e-05 | **Nouveau best** |
| 17-21 | ↘ 0.37 | ↗ 90% | ~0.40 | ~88% | 3.0e-05 | Val loss stagne 5 epochs |
| **22** | 0.3758 | 89.31% | 0.3993 | 87.67% | **1.5e-05** | **Scheduler ÷2** |
| **23** | 0.3589 | 90.51% | 0.3801 | **89.71%** | 1.5e-05 | **Nouveau best** ✅ |
| 24 | 0.3585 | 90.55% | 0.3783 | 89.00% | 1.5e-05 | |
| 25 | 0.3539 | 91.02% | 0.3695 | 89.24% | 1.5e-05 | |
| 26 | 0.3564 | 90.72% | 0.3832 | 88.90% | 1.5e-05 | |
| 27 | 0.3546 | 90.71% | 0.3781 | 89.67% | 1.5e-05 | |
| 28 | 0.3681 | 89.63% | 0.4152 | 86.76% | 1.5e-05 | Spike négatif |
| 29 | 0.3576 | 90.46% | 0.3984 | 88.33% | 1.5e-05 | |
| 30 | 0.3553 | 90.82% | 0.3833 | 89.48% | 1.5e-05 | |
| **31** | 0.3515 | 90.86% | 0.3890 | 88.71% | **7.5e-06** | **Scheduler ÷2** (2e intervention) |
| 32 | 0.3467 | 91.31% | 0.3844 | 88.19% | 7.5e-06 | |
| 33 | 0.3515 | 90.85% | 0.3732 | 89.14% | 7.5e-06 | |
| 34 | - | - | - | - | 7.5e-06 | **Arrêt forcé** |

**Meilleur modèle sauvegardé** : epoch 23, val_acc = **89.71%**, val_loss = 0.3801
**Arrêt** : epoch 34 (forcé manuellement, early stopping aurait déclenché ~epoch 43)

### Analyse V4 : le fix du bug a tout changé

#### Preuve que le fix fonctionne

Le moment clé est l'**epoch 6** (unfreeze du backbone) :

| | V3 (bugué) | V4 (fixé) |
|---|---|---|
| Val acc epoch 5 (freeze) | ~67% | 67.29% |
| Val acc epoch 6 (unfreeze) | ~68% (+1%) | **77.62% (+10%)** |
| Val acc epoch 10 | ~70% | **88.19%** |
| Val acc finale | 73.81% (epoch 50) | **89.71%** (epoch 23) |
| Train acc finale | 68% | 91% |
| Pattern | **Underfitting** (train < val) | **Sain** (train ≈ val) |

En V3, `set_trainable_up_to(False, layername=None)` ne dégelait qu'**un seul paramètre** (le `return` était à l'intérieur du `for`). Le backbone EfficientNet restait gelé — le modèle ne pouvait apprendre qu'avec son classifier head de 2 couches. D'où le plafond à 68% train / 73% val.

En V4, tous les paramètres du backbone sont dégelés. Le modèle peut maintenant ajuster ses features de bas niveau (textures, artefacts de compression) → +16% de val accuracy.

#### Comportement du scheduler ReduceLROnPlateau

Le scheduler intervient **deux fois** pendant l'entraînement V4 :

1. **Epoch 22** : val_loss stagne pendant 6 epochs après le record de 0.3777 (epoch 16) → LR divisé par 2 (3e-05 → 1.5e-05)
2. Résultat immédiat : epoch 23 atteint un **nouveau record** (89.71%), confirmant que le scheduler a fait le bon choix
3. **Epoch 31** : val_loss stagne à nouveau pendant 6 epochs → LR divisé par 2 (1.5e-05 → 7.5e-06)
4. Résultat : epochs 31-33 n'améliorent pas le record → le modèle a atteint son plateau

Avec le `ReduceLROnPlateau` adaptatif (au lieu du `CosineAnnealingLR` de V3 qui poussait le LR à 1e-6 trop tôt), le modèle garde un LR utile tant qu'il progresse. L'arrêt forcé à epoch 34 était justifié : le modèle n'aurait plus progressé (early stopping aurait déclenché vers epoch 43).

#### Écart train/val

L'écart train acc - val acc reste faible : **~2%** (91% vs 89%). C'est un signe de bonne généralisation — le modèle ne surapprend pas. Les augmentations (mixup 30%, ColorJitter, GaussianBlur, label smoothing 0.1) font leur travail de régularisation.

Comparaison avec V1 (ResNet18, lr=0.0002) qui avait un écart de **28%** à l'epoch 5 :

| | V1 (ResNet18) | V4 (EfficientNet-B0) |
|---|---|---|
| Écart max train/val | 28% (epoch 5) | 4% (epoch 28) |
| Cause de l'écart | LR trop élevé, pas d'augmentation | Régularisation efficace |
| Val acc finale | 65% | 89.71% |

### Leçons apprises de V4

#### 1. Un bug d'une ligne peut ruiner 20h de GPU

Le bug V3 (`return` mal indenté dans `set_trainable_up_to()`) a coûté **20h de calcul GPU** et plafonné les performances à 73% au lieu de 90%.

**Comment on aurait pu l'éviter** : un test unitaire de 3 lignes après l'unfreeze :
```python
model.set_trainable_up_to(False, layername=None)
trainable = sum(p.requires_grad for p in model.parameters())
assert trainable == total_params, f"Seulement {trainable}/{total_params} params dégelés"
```

**Leçon** : sur un cluster GPU (ressource rare et partagée), un test rapide avant de lancer un job de 20h est toujours rentable. Le coût d'un test = 2 secondes. Le coût du bug = 20 heures + diagnostic + relance.

#### 2. Le diagnostic par les métriques

Le pattern `train_acc < val_acc` est un signal d'alarme spécifique :
- **train > val** (classique) = overfitting → ajouter de la régularisation
- **train < val** (inhabituel) = underfitting → le modèle ne peut pas apprendre → chercher un bug dans le pipeline d'entraînement (données corrompues, couches gelées, gradient qui ne passe pas)

Ce pattern a été la clé pour identifier que le backbone ne s'entraînait pas.

#### 3. ReduceLROnPlateau vs CosineAnnealingLR

| | CosineAnnealingLR (V3) | ReduceLROnPlateau (V4) |
|---|---|---|
| Comportement | LR suit un cosinus prédéfini | LR baisse quand la val loss stagne |
| Problème | Pousse le LR à 1e-6 dès epoch 40, même si le modèle apprenait encore | - |
| Avantage | Prévisible, pas de paramètre à tuner | S'adapte au rythme réel d'apprentissage |
| Résultat | LR trop faible trop tôt → stagnation prématurée | LR utile tant que le modèle progresse |

**Recommandation** : pour un premier entraînement (quand on ne connaît pas la dynamique du modèle), `ReduceLROnPlateau` est plus sûr. `CosineAnnealingLR` est utile quand on connaît déjà le nombre optimal d'epochs.

#### 4. L'importance du freeze/unfreeze progressif

La stratégie en 2 phases (5 epochs classifier gelé → unfreeze tout) a bien fonctionné :
- **Epochs 1-5** : le classifier apprend à utiliser les features ImageNet existantes (65% → 67%)
- **Epoch 6+** : le backbone s'ajuste pour détecter les artefacts de deepfake (67% → 88% en 4 epochs)

Sans freeze initial, le classifier aléatoire envoie des gradients incohérents dans le backbone → les features pré-entraînées sont détruites avant d'être utiles. Le freeze protège ces features pendant que le classifier s'initialise.

---

## V5 : EfficientNet-B4 + faces pré-extraites

- **Date** : 2026-06-22 (préparation) → 2026-06-23 (corrections)

### Motivations

1. **EfficientNet-B4** (19M params) au lieu de B0 (5.3M params) — le benchmark FaceForensics++ montre B4 nettement au-dessus de B0. Plus de capacité = meilleures features pour distinguer les artefacts subtils.

2. **Pré-extraction des visages sur disque** — au lieu d'exécuter le détecteur de visages à chaque frame à chaque epoch (~1400s/epoch), on extrait 30 faces par vidéo une seule fois (~2h). Les epochs passent à ~200-300s, ce qui permet d'itérer 5-7x plus vite.

3. **Plus de données** — 30 faces/vidéo × 2160 vidéos = 64,800 images train (vs 10,800 avec frames_per_video=5).

### Fichiers modifiés/créés

- `src/models/models.py` — ajout EfficientNet-B4 comme choix de modèle (input 299×299)
- `src/data/dataset.py` — paramètre `faces_dir` pour charger les visages pré-extraits (JPEG)
- `src/train.py` — argument `--faces_dir` et `--model efficientnet_b4`
- `scripts/extract_faces.py` — script de pré-extraction (30 faces/vidéo, frames équidistantes, logique de resume intégrée)
- `scripts/submit_train_v5.sh` — script SLURM (extraction + entraînement)
- `scripts/submit_extract_faces.sh` — script SLURM extraction seule

### Problème rencontré : extraction coupée par le time limit

**Job 860234** (extraction des faces) a été tué à **56%** (1219/2160 vidéos) après 3h :

```
Extraction:  56%|█████▋    | 1219/2160 [2:59:57<1:44:33,  6.67s/it]
slurmstepd: error: *** JOB 860234 ON nodemm01 CANCELLED AT 2026-06-23T04:07:05 DUE TO TIME LIMIT ***
```

**Cause** : `submit_extract_faces.sh` avait `--time=03:00:00` (3h), mais l'extraction de 2160 vidéos à 6.67s/vidéo nécessite ~4h.

**Calcul qu'on aurait dû faire avant de lancer** : 2160 vidéos × 6.67s/vidéo = 14,407s ≈ **4h**. Avec une marge de sécurité → 6h minimum.

**État après le crash** : extraction partielle sur le disque :
- `data/faces/c23/original/` : 408/720 vidéos extraites
- `data/faces/c23/Deepfakes/` : 203/~360 vidéos extraites
- Les autres méthodes : partiellement extraites

### Corrections appliquées (2026-06-23)

**2 changements ciblés :**

1. **`scripts/submit_extract_faces.sh`** : time limit augmenté de 3h à **6h** (`--time=06:00:00`)

2. **`scripts/submit_train_v5.sh`** : supprimé le `if [ ! -d "data/faces/c23/original" ]` qui skipait l'extraction si le dossier existait. Problème : le crash avait créé le dossier avec des données incomplètes, donc V5 aurait entraîné sur 56% des faces sans s'en rendre compte.

   L'extraction tourne maintenant **systématiquement** avant l'entraînement. Grâce à la logique de resume dans `extract_faces.py` (qui skip les vidéos déjà extraites), relancer est rapide : seules les ~941 vidéos restantes seront traitées (~1.5h au lieu de 4h).

**Leçon** : toujours calculer le temps nécessaire avant de configurer le `--time` SLURM. Et ne jamais se fier à l'existence d'un dossier pour décider si une étape est terminée — vérifier le contenu ou utiliser un flag de completion.

### Commandes de lancement

```bash
# Sur le cluster, après que V4 ait fini
git pull origin dev/training-pipeline
cp checkpoints/best_model.pth checkpoints/best_model_v4.pth
sbatch scripts/submit_train_v5.sh
```

### Optimisations V5 après audit (2026-06-23)

Avant de lancer V5, un audit approfondi du code et une recherche dans la littérature ont révélé **un bug critique** et plusieurs optimisations manquantes.

#### Référence : DeepfakeBench (NeurIPS 2023)

**Notre cible à battre** : le benchmark [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) ([paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf)) évalue EfficientNet-B4 sur FF++ c23 et obtient :

| Détecteur | AUC sur FF++ c23 |
|-----------|-----------------|
| UCF (Xception) | 0.9705 |
| Xception | 0.9637 |
| **EfficientNet-B4** | **0.9567** |

**Config DeepfakeBench** : Adam, lr=0.0002, batch=32, 32 frames/vidéo, résolution 256×256, A100 80GB.

Notre objectif : **dépasser 0.9567 AUC** avec notre pipeline qui a des avantages que DeepfakeBench n'utilise pas (freeze/unfreeze progressif, mixup, label smoothing, augmentations avancées, face extraction avec marge généreuse).

Sources :
- [DeepfakeBench GitHub](https://github.com/SCLBD/DeepfakeBench)
- [DeepfakeBench Paper (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf)
- [DeepfakeBench ar5iv (tables complètes)](https://ar5iv.labs.arxiv.org/html/2307.01426)
- [Keras EfficientNet fine-tuning guide](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)
- [PyTorch Forums - BatchNorm freeze](https://discuss.pytorch.org/t/finetuning-pretrained-networks-with-batchnorm/4121)
- [EfficientNet-B4 deepfake detection (ResearchGate)](https://www.researchgate.net/publication/378157870_Using_the_CNN_architecture_based_on_the_EfficientNetB4_model_to_efficiently_detect_Deepfake_images)

#### Bug critique : BatchNorm pas vraiment gelé

**Problème** : `model.train()` est appelé à chaque epoch (`train.py:47`), ce qui met **toutes** les couches BatchNorm en mode training. Même avec `requires_grad=False`, les buffers `running_mean` et `running_var` sont mis à jour.

**Conséquence** : pendant les epochs de freeze (classifier seul), les statistiques BatchNorm d'ImageNet — soigneusement apprises sur 14 millions d'images — sont **corrompues** par notre dataset de ~65k faces. Quand le backbone est ensuite unfreeezé, il part de stats BatchNorm dégradées au lieu des stats ImageNet originales.

**Fix** (`src/models/models.py`) : ajout d'une méthode `freeze_bn()` :
```python
def freeze_bn(self):
    for module in self.model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
```

Appelée dans `train_one_epoch()` juste après `model.train()`. Les couches BatchNorm restent en mode eval pendant tout l'entraînement (y compris après l'unfreeze), utilisant les statistiques ImageNet pré-calculées au lieu de stats de batch bruités.

**Pourquoi garder BN en eval même après l'unfreeze ?** Avec batch_size=16, les statistiques de batch sont bruitées (calculées sur seulement 16 images). Les running stats d'ImageNet (calculées sur des millions d'images) sont plus fiables. C'est la pratique recommandée pour le fine-tuning avec des petits batch sizes ([source](https://discuss.pytorch.org/t/finetuning-pretrained-networks-with-batchnorm/4121)).

#### Gradient clipping

**Problème** : quand le backbone se défreeze (epoch 3 en V5), les gradients venant du classifier déjà entraîné peuvent être énormes et déstabiliser les poids pré-entraînés.

**Fix** (`src/train.py`) : ajout de `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` après `loss.backward()`. Les gradients sont clippés à une norme maximale de 1.0, empêchant les mises à jour explosives.

#### Paramètres ajustés pour le dataset 6× plus gros

Avec les faces pré-extraites, chaque epoch contient ~64,800 samples (30 faces × 2160 vidéos) au lieu de 10,800 (5 frames × 2160 vidéos). Ça change la dynamique d'apprentissage :

| Paramètre | V4 | V5 | Raison |
|-----------|----|----|--------|
| **batch_size** | 32 | **16** | B4 (19M params) à 299×299 sur RTX 3090 (24GB) → OOM probable avec 32 |
| **lr** | 3e-05 | **2e-05** | B4 est plus sensible aux grands LR (plus de params) |
| **freeze_epochs** | 5 (hardcodé) | **2** (CLI) | 2 epochs × 4050 steps = 8100 steps de classifier (vs 10,125 en V4). Suffisant. |
| **warmup_epochs** | 3 (hardcodé) | **1** (CLI) | 1 epoch × 4050 steps = 4050 steps de warmup (vs 6075). Proportionnel. |
| **gradient clipping** | non | **max_norm=1.0** | Stabilité à l'unfreeze |
| **BN freeze** | non | **oui** | Préserve stats ImageNet |

`freeze_epochs` et `warmup_epochs` sont maintenant des arguments CLI (au lieu d'être hardcodés), ce qui permet de les ajuster sans modifier le code.

#### Config finale V5

```bash
python3 -m src.train \
    --data_root data \
    --faces_dir data/faces \
    --compression c23 \
    --model efficientnet_b4 \
    --dropout 0.5 \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.00002 \
    --weight_decay 5e-4 \
    --patience 20 \
    --freeze_epochs 2 \
    --warmup_epochs 1 \
    --num_workers 8
```

### Résultats attendus

- **Extraction** : ~1.5h (resume des ~941 vidéos restantes)
- **Entraînement** : ~3-5h (epochs de ~200-300s grâce aux faces pré-extraites)
- **Val acc finale** : 93-96% / AUC > 0.9567 (objectif : battre DeepfakeBench)
- **Durée totale** : ~5-7h

### Nos avantages par rapport à DeepfakeBench

| | DeepfakeBench | Notre pipeline |
|---|---|---|
| Freeze/unfreeze | Non | Oui (2 epochs freeze → unfreeze progressif) |
| BatchNorm freeze | Non mentionné | Oui (stats ImageNet préservées) |
| Mixup | Non | Oui (alpha=0.4, p=0.3) |
| Label smoothing | Non | Oui (0.1) |
| Gradient clipping | Non | Oui (max_norm=1.0) |
| Augmentations | Flip, rotation, JPEG, blur, brightness, FancyPCA | Flip, rotation, ColorJitter, JPEG, blur, GaussianNoise, CutOut, Grayscale |
| Résolution | 256×256 | 299×299 (plus de détails) |
| Face crop margin | 1.3× | 1.6× (plus de contexte autour du visage) |
| LR scheduler | Non mentionné | ReduceLROnPlateau (adaptatif) |
| Warmup | Non | Oui (1 epoch, 1e-6 → 2e-5) |

### Résultats V5 (Job 861281 - en cours au 2026-06-23)

**Cluster** : gpu-gw.enst.fr, RTX 3090 24GB
**Config** : EfficientNet-B4, lr=1e-05, dropout=0.5, batch_size=16, 30 faces/vidéo, freeze_epochs=5, warmup_epochs=3, BN freeze désactivé (commit ced81d4)

| Epoch | Train loss | Train acc | Val loss | Val acc | LR | Observation |
|-------|------------|-----------|----------|---------|-----|-------------|
| 16 | 0.3553 | 90.39% | 0.3532 | 90.54% | 1.0e-05 | **Nouveau best** |
| 17 | 0.3479 | 90.81% | 0.3797 | 89.58% | 1.0e-05 | |
| 18 | 0.3419 | 91.23% | 0.3700 | 89.93% | 1.0e-05 | |
| **19** | 0.3396 | 91.38% | 0.3567 | **90.93%** | 1.0e-05 | **Meilleur modèle** |
| 20 | 0.3397 | 91.29% | 0.3686 | 90.71% | 1.0e-05 | |
| 21 | 0.3409 | 91.21% | 0.3783 | 89.11% | 1.0e-05 | |
| 22 | 0.3356 | 91.58% | 0.4131 | 88.20% | **5.0e-06** | **Scheduler ÷2** |
| 23 | 0.3298 | 91.89% | 0.3607 | 90.69% | 5.0e-06 | |
| 24 | 0.3276 | 92.05% | 0.3979 | 89.09% | 5.0e-06 | |
| 25 | 0.3282 | 91.96% | 0.3765 | 89.90% | 5.0e-06 | |
| 26 | 0.3265 | 92.24% | 0.4079 | 88.68% | 5.0e-06 | Oscillations |

**Meilleur modèle** : epoch 19, val_acc = **90.93%**, val_loss = 0.3567

### Résultats d'évaluation V5 (test set, 12347 samples)

| Métrique | V4 (B0, 5.3M) | V5 (B4, 19M) | Écart |
|----------|---------------|--------------|-------|
| **Accuracy** | **89.79%** | 88.77% | **-1.02%** |
| **AUC** | **0.9671** | 0.9392 | **-0.028** |
| **Precision** | **96.22%** | 90.36% | **-5.86%** |
| **Recall** | 88.14% | **93.05%** | +4.91% |
| **F1** | **0.9200** | 0.9168 | **-0.003** |

**V5 est une régression par rapport à V4** malgré un modèle 3.6× plus gros.

#### AUC par méthode — V5 pire que V4 sur les 4

| Méthode | V4 | V5 | Écart |
|---------|-----|-----|-------|
| Deepfakes | **0.9797** | 0.9685 | -0.011 |
| Face2Face | **0.9712** | 0.9436 | -0.028 |
| FaceSwap | **0.9683** | 0.9294 | -0.039 |
| NeuralTextures | **0.9458** | 0.9156 | -0.030 |

#### Accuracy par méthode

| Méthode | V4 | V5 |
|---------|-----|-----|
| Deepfakes | **92.24%** | 86.40% |
| Face2Face | **92.62%** | 84.79% |
| FaceSwap | **91.29%** | 83.56% |
| NeuralTextures | **88.95%** | 83.36% |

#### Confusion matrix V5 : `[[3321, 815], [571, 7640]]`

- 815 faux positifs (20% des real classées fake) vs 97 en V4 (7%)
- 571 faux négatifs (7% des fakes manqués) vs 332 en V4 (12%)
- Le modèle a un **biais vers "fake"** : recall élevé (93%) mais precision faible (90%)
- Cause directe : pas de class weighting → le modèle apprend que prédire "fake" est plus sûr (classe 4× plus fréquente)

#### Analyse des courbes d'entraînement V5

Courbes dans `results/v5/training_curves.png` :
- **Epochs 1-5 (freeze)** : classifier seul, ~251s/epoch, progression lente (60% → 67%)
- **Epoch 6 (unfreeze)** : saut spectaculaire 67% → 77% (même pattern que V4)
- **Epochs 6-16 (montée)** : progression rapide 77% → 90.5%, train ≈ val
- **Epochs 17-39 (plateau + overfitting)** : train loss continue de baisser (0.35 → 0.32) mais **val loss remonte** (0.35 → 0.40). Divergence train/val visible : train acc 92% vs val acc 88%. Le scheduler intervient 3 fois (epochs 22, 28, 34) sans amélioration.

**Pattern clé** : la divergence train/val loss est le symptôme d'un overfitting que les augmentations (trop agressives) n'arrivent pas à compenser. Le BN freeze désactivé + batch_size=16 amplifient le bruit → oscillations de 3% sur val acc.

#### Analyse des courbes ROC V5

Courbes dans `results/v5/roc_curves_per_method.png` :
- Deepfakes reste la meilleure (AUC=0.969) mais en recul vs V4 (0.980)
- NeuralTextures la plus difficile (AUC=0.916), même tendance que V4
- Les 4 courbes sont moins proches du coin haut-gauche que V4 → classifieur globalement moins discriminant
- AUC global 0.9392 vs 0.9671 en V4 → **régression de 2.8 points**

### Analyse V5 : pourquoi un modèle 3.6× plus gros fait pire

Le passage de B0 (5.3M params) à B4 (19M params) a **dégradé** les performances (-1% accuracy, -2.8% AUC). Causes identifiées par audit approfondi :

1. **BN freeze désactivé** : le flag `--freeze_bn` n'était pas passé dans `submit_train_v5.sh` (commit ced81d4 l'a rendu optionnel avec `default=False`). Avec batch_size=16, les statistiques de batch sont bruitées → corruption des running stats ImageNet → oscillations de 3% entre epochs (88.2% → 90.9%).

2. **Résolution incorrecte** : `IMAGE_SIZE = 299` hardcodé dans `transforms.py` alors que B4 natif = 380×380. Le modèle recevait des images sous-dimensionnées → ses couches finales travaillaient sur des feature maps trop petites → perte de détails fins.

3. **Pas de class weighting** : ratio real:fake de 1:4 non compensé → biais vers "fake" → 815 faux positifs (vs 97 en V4). Le paper FaceForensics++ utilise explicitement des class weights.

4. **Augmentations trop agressives** : rotation 15°, GaussianBlur σ=1.0, 2 trous CutOut de 40px → destruction des artefacts de deepfake subtils que le modèle doit détecter. Un modèle plus gros (B4) est plus sensible à la qualité du signal d'entrée.

5. **Face margin trop large** : margin=0.3 (1.6× total) vs paper 1.3× → trop de background inutile, dilution des artefacts de frontière.

6. **LR uniforme** : backbone et classifier au même LR (1e-05) → features ImageNet potentiellement dégradées trop vite. Un modèle plus gros nécessite un differential LR pour protéger ses couches basses.

**Leçon principale** : un modèle plus gros ne garantit pas de meilleurs résultats si le pipeline d'entraînement n'est pas adapté. B4 a plus de capacité mais aussi plus de paramètres à corrompre — chaque erreur de configuration (BN, résolution, class weights) a un impact amplifié.

---

## V6 : Configuration Ultime — EfficientNet-B4 Optimisé

- **Date** : 2026-06-23

### Motivations

Audit approfondi du paper FaceForensics++ (ar5iv.labs.arxiv.org/html/1901.08971) et comparaison systématique avec notre pipeline. Objectif : combler l'écart entre V5 (90.93%) et le paper (95.91%) en corrigeant tous les problèmes identifiés.

### Référence Paper FaceForensics++ (Table 5, c23)

XceptionNet (23M params) entraîné sur toutes les méthodes :

| Méthode | Accuracy |
|---------|---------|
| Deepfakes | 97.49% |
| Face2Face | 97.69% |
| FaceSwap | 96.79% |
| NeuralTextures | 92.19% |
| Pristine | 95.41% |
| **Moyenne** | **95.91%** |

**Config paper** : Adam, lr=0.0002, batch=32, **270 frames/vidéo train**, 100 val/test, **class weights 1:4**, freeze 3 epochs FC → 15 epochs full, face crop **1.3×**

### Changements V6 (6 modifications)

#### 1. Résolution native B4 : 299×299 → 380×380

**Fichiers** : `src/data/transforms.py` (IMAGE_SIZE), `src/models/models.py` (model_selection)

EfficientNet-B4 est conçu pour des entrées 380×380. Le utiliser à 299×299 revient à sous-exploiter sa capacité — les couches finales reçoivent des feature maps plus petites et perdent des détails spatiaux. Le paper utilise XceptionNet à 299×299 (sa résolution native). Chaque architecture a sa résolution optimale.

#### 2. Class weighting dynamique

**Fichier** : `src/train.py`

```python
num_real = sum(1 for _, label in train_dataset.samples if label == 0)
num_fake = sum(1 for _, label in train_dataset.samples if label == 1)
weight_real = len(train_dataset.samples) / (2 * num_real)
weight_fake = len(train_dataset.samples) / (2 * num_fake)
class_weights = torch.tensor([weight_real, weight_fake], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

Le paper dit explicitement : "*we solve the imbalance between real and fake images by weighing the training images correspondingly*". Sans weighting, le modèle apprend un biais vers "fake" (classe majoritaire 4:1). V4 avait déjà ce problème : 96.22% precision mais seulement 88.14% recall.

#### 3. Differential Learning Rates

**Fichier** : `src/train.py`

Le backbone (features ImageNet) apprend à 1/10 du LR du classifier :
- **Backbone** : lr × 0.1 = 2e-05 (préserve les features pré-entraînées)
- **Classifier** : lr = 2e-04 (apprend rapidement la tâche real/fake)

Le paper utilise lr=0.0002 uniforme sur XceptionNet. Mais avec un modèle pré-entraîné ImageNet, les couches basses (détecteurs d'edges, textures) sont déjà optimales — les modifier trop vite détruit ces features avant que le classifier puisse les exploiter. C'est ce qui causait l'overfitting en V1 (lr=0.0002 uniforme).

Le differential LR résout ce dilemme : LR agressif sur le classifier (comme le paper) tout en protégeant le backbone (comme nos V4/V5 conservatrices).

Le warmup a été adapté pour respecter le ratio : chaque param group warmup vers son propre `initial_lr`, pas vers un LR global.

#### 4. Face margin réduite : 0.3 → 0.15

**Fichier** : `src/data/face_extraction.py`

Margin 0.3 = +30% de chaque côté = 1.6× la bbox du visage. Paper = 1.3×. Le surplus de contexte (background, cheveux, épaules) introduit de la variance non pertinente que le modèle doit apprendre à ignorer, gaspillant de la capacité. Les artefacts de deepfake sont concentrés à la frontière du visage — un crop plus serré maximise le ratio signal/bruit.

Nécessite une **re-extraction complète** des visages dans un nouveau dossier `data/faces_v6`.

#### 5. Augmentations allégées

**Fichier** : `src/data/transforms.py`

| Transform | V5 | V6 | Raison |
|-----------|-----|-----|--------|
| Rotation | 15° | **5°** | Les artefacts deepfake sont des features spatiales-fréquentielles sensibles à la rotation |
| GaussianBlur σ=(0.1, 1.0) | Oui | **Supprimé** | σ=1.0 détruit les artefacts de compression c23, exactement ce qu'on veut détecter |
| CutOut | 2×40, p=0.15 | **1×20, p=0.1** | 2 trous de 40px masquent 7% du visage, potentiellement les régions forensiques clés |
| GaussianNoise std | (0.01, 0.03) | **(0.005, 0.015)** | Réduit de moitié pour préserver le signal fin |
| Mixup | p=0.3, α=0.4 | **p=0.2, α=0.2** | Mixing plus léger, préserve la structure des artefacts |
| JPEG, ColorJitter, Flip, Grayscale | Garder | **Garder** | Pertinents pour la tâche |

Stratégie : le paper n'utilise **aucune augmentation avancée** et atteint 95.91%. Nos augmentations légères (JPEG, ColorJitter, flip) gardent l'avantage de robustesse sans détruire le signal forensique.

#### 6. Assertion d'unfreeze (leçon V3)

**Fichier** : `src/train.py`

```python
if epoch == freeze_epochs + 1:
    model.set_trainable_up_to(False, layername=None)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Unfreeze: {trainable:,}/{total:,} params trainable')
    assert trainable > 1_000_000, f"Bug: seulement {trainable} params dégelés!"
```

Coût : 0 secondes. Valeur : éviter 20h de GPU gaspillées (leçon V3).

### Config finale V6

```bash
python3 -m src.train \
    --data_root data \
    --faces_dir data/faces_v6 \
    --compression c23 \
    --model efficientnet_b4 \
    --dropout 0.4 \
    --batch_size 32 \
    --lr 0.0002 \
    --weight_decay 1e-4 \
    --freeze_epochs 5 \
    --warmup_epochs 3 \
    --freeze_bn \
    --differential_lr \
    --backbone_lr_factor 0.1 \
    --epochs 50 \
    --patience 15 \
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir logs/tensorboard
```

### Comparaison V5 → V6 (changements)

| Paramètre | V5 | V6 | Raison |
|-----------|-----|-----|--------|
| **IMAGE_SIZE** | 299 | **380** | B4 natif |
| **Face margin** | 0.3 (1.6×) | **0.15 (1.3×)** | Comme paper |
| **Class weights** | Non | **Oui (auto)** | Comme paper |
| **LR** | 1e-05 (uniforme) | **2e-04 / 2e-05** | Differential LR |
| **Batch size** | 16 | **32** | Comme paper |
| **Dropout** | 0.5 | **0.4** | Régularisation plus légère avec gros batch |
| **Weight decay** | 5e-4 | **1e-4** | Comme paper |
| **Rotation** | 15° | **5°** | Préserve artefacts spatiaux |
| **GaussianBlur** | σ=(0.1, 1.0) | **Supprimé** | Détruit signal forensique |
| **CutOut** | 2×40, p=0.15 | **1×20, p=0.1** | Moins agressif |
| **Mixup** | p=0.3, α=0.4 | **p=0.2, α=0.2** | Plus léger |
| **BN freeze** | Non (oublié) | **Oui** | Flag `--freeze_bn` |
| **Faces/vidéo** | 30 | **50** | Plus de diversité |
| **Assertion unfreeze** | Non | **Oui** | Leçon V3 |

### Résultats attendus

- **Extraction** : ~8h (re-extraction complète 2580 vidéos, margin=0.15, 50 faces/vidéo)
- **Entraînement** : ~12-16h (epochs plus longues avec batch=32 à 380×380)
- **Val acc finale** : **94-96%** (objectif : dépasser le paper à 95.91%)
- **Durée totale** : ~20-24h

### Commandes de lancement

```bash
# Sur le cluster
ssh mguinzie-24@gpu-gw.enst.fr
cd ~/projects/FaceForensics
git pull origin dev/training-pipeline
cp checkpoints/best_model.pth checkpoints/best_model_v5.pth
sbatch scripts/submit_train_v6.sh
```

---

### Récapitulatif de l'évolution du projet

#### Résultats d'entraînement

| Version | Modèle | Params | Val acc | Val loss | Meilleure epoch | Epochs total | Durée |
|---------|--------|--------|---------|----------|-----------------|-------------|-------|
| V1 | ResNet18 | 11M | 66.21% | 0.6118 | 8/18 | 18 (early stop) | ~13h |
| V3 | EfficientNet-B0 (bugué) | 5.3M | 73.81% | 0.5952 | 37/50 | 50 | 20h |
| V4 | EfficientNet-B0 (fixé) | 5.3M | **89.71%** | 0.3801 | 23/34 | 34 (arrêt forcé) | ~13h |
| V5 | EfficientNet-B4 | 19M | **90.93%** | 0.3567 | 19/? | En cours | ~10h |
| V6 | EfficientNet-B4 (diff LR) | 19M | **92.08%** | 0.3890 | 9/24 | 24 (early stop) | ~7h |
| V7 | EfficientNet-B4 (LR uniforme) | 19M | **?** | - | - | En cours | ~12-16h estimé |

Note : V2 n'a pas eu de run indépendant — les recommandations V2 ont été intégrées dans V3/V4.

#### Résultats d'évaluation (sur le test set)

| Métrique | V3 (bugué) | V4 (fixé) | V5 (B4) | V6 (diff LR) |
|----------|-----------|-----------|---------|-------------|
| **Samples** | 4200 | 4200 | 12347 | 20574 |
| **Accuracy** | 71.60% | **89.79%** | 88.77% | 89.37% |
| **AUC** | 0.7204 | **0.9671** | 0.9392 | 0.9587 |
| **Precision** | 71.51% | 96.22% | 90.36% | **95.11%** |
| **Recall** | 95.39% | 88.14% | **93.05%** | 88.56% |
| **F1-Score** | 0.8174 | **0.9200** | 0.9168 | 0.9172 |

Note : V5 évaluée sur 12347 samples (faces pré-extraites), V6 sur 20574 (50 faces/vidéo). V5 et V6 sont des régressions vs V4 malgré un modèle 3.6× plus gros — V5 à cause de problèmes de configuration, V6 à cause du differential LR qui limite l'entraînement utile à 4 epochs.

#### AUC par méthode de manipulation

| Méthode | V3 | V4 | V5 | V6 | DeepfakeBench (B4) |
|---------|-----|-----|-----|-----|-------------------|
| **Deepfakes** | 0.761 | **0.9797** | 0.9685 | 0.9725 | 0.9757 |
| **Face2Face** | 0.722 | **0.9712** | 0.9436 | 0.9703 | 0.9758 |
| **FaceSwap** | 0.685 | **0.9683** | 0.9294 | 0.9549 | 0.9797 |
| **NeuralTextures** | 0.730 | **0.9458** | 0.9156 | 0.9372 | 0.9308 |
| **Global** | 0.720 | **0.9671** | 0.9392 | 0.9587 | 0.9567 |

V4 reste le meilleur en AUC global. V6 rattrape V4 sur Deepfakes/Face2Face mais reste en retrait sur FaceSwap/NeuralTextures — le differential LR a coupé l'entraînement trop tôt. V7 (LR uniforme + corrections V6) devrait dépasser V4 sur toutes les méthodes.

#### Accuracy par méthode de manipulation

| Méthode | V3 | V4 | V5 | V6 |
|---------|-----|-----|-----|-----|
| **Deepfakes** | 48.57% | **92.24%** | 86.40% | 91.51% |
| **Face2Face** | 47.76% | **92.62%** | 84.79% | 91.30% |
| **FaceSwap** | 46.86% | **91.29%** | 83.56% | 90.13% |
| **NeuralTextures** | 46.81% | **88.95%** | 83.36% | 87.72% |

V6 récupère 5-7% par rapport à V5 sur chaque méthode grâce aux class weights et résolution native. L'écart avec V4 est réduit à ~1%. NeuralTextures reste la plus difficile dans toutes les versions.

#### Analyse des matrices de confusion

**V3 (bugué)** — confusion matrix : `[[336, 1064], [129, 2671]]`
- 1064 faux positifs (76% des vidéos real classées comme fake)
- Le modèle prédit presque tout comme "fake" → recall élevé (95%) mais precision faible (71%)
- Le classifier head seul (backbone gelé) a appris un biais vers "fake" car les fakes sont 2× plus nombreux dans le dataset

**V4 (fixé)** — confusion matrix : `[[1303, 97], [332, 2468]]`
- Seulement 97 faux positifs (7% des real classées comme fake) vs 1064 en V3
- 332 faux négatifs (12% des fakes passent inaperçus)
- Bonne balance precision/recall → le backbone entraîné détecte les vrais artefacts au lieu de se baser sur un biais statistique

**V5 (B4)** — confusion matrix : `[[3321, 815], [571, 7640]]`
- 815 faux positifs (20% des real classées comme fake) — **régression majeure** vs V4 (7%)
- 571 faux négatifs (7% des fakes manqués) — meilleur que V4 (12%)
- Le modèle a un biais vers "fake" : il préfère prédire fake (recall 93%) au détriment de la precision (90%)
- Cause : pas de class weighting (ratio 1:4 real:fake non compensé) + BN freeze désactivé → le modèle apprend un raccourci statistique au lieu des artefacts réels

**V6 (B4, differential LR)** — confusion matrix : `[[6268, 623], [1565, 12118]]`
- 623 faux positifs (9% des real classées fake) — entre V4 (7%) et V5 (20%)
- 1565 faux négatifs (11.4% des fakes manqués) — similaire à V4 (12%)
- Meilleur équilibre que V5 grâce aux class weights qui corrigent le biais vers "fake"
- Le modèle reste légèrement conservateur (precision 95% > recall 89%) — normal avec seulement 4 epochs d'entraînement utile

#### Analyse des courbes d'entraînement

**V1 (ResNet18)** — courbes dans `results/v1/training_curves.png`
- Overfitting classique : train acc monte à 90%, val acc stagne à 66%
- Val loss explose (0.6 → 1.1) pendant que train loss descend
- Le scheduler intervient 2 fois (LR ÷10 à epoch 7, ÷10 à epoch 13) mais ne sauve pas le run
- Early stopping à epoch 18

**V3 (EfficientNet-B0 bugué)** — courbes dans `results/v3/training_curves.png`
- Pattern d'underfitting visible : train acc (68%, bleu) SOUS val acc (73%, rouge)
- Les deux courbes progressent très lentement et en parallèle
- CosineAnnealingLR pousse le LR à 1.3e-06 → le modèle arrête d'apprendre
- 50 epochs pour seulement +8% d'accuracy (65% → 73%)

**V4 (EfficientNet-B0 fixé)** — courbes dans `results/v4/training_curves.png`
- Point de rupture spectaculaire à l'**epoch 6** (unfreeze du backbone)
- Val loss chute de 0.65 → 0.39 en 4 epochs (6-10)
- Val acc saute de 67% → 88% en 4 epochs
- Après epoch 16 : plateau sain avec train ≈ val (écart ~2%)
- Scheduler ÷2 à epoch 22 → petit boost à 89.71% (epoch 23)

#### Analyse des courbes ROC

**V3** — courbes dans `results/v3/evaluation/roc_curves_per_method.png`
- Toutes les courbes sont éloignées du coin haut-gauche (classifieur faible)
- FaceSwap le pire (AUC=0.685, proche du hasard)
- Les courbes se croisent → le modèle n'a pas appris de features discriminantes

**V4** — courbes dans `results/v4/roc_curves_per_method.png`
- Toutes les courbes épousent le coin haut-gauche (excellent classifieur)
- Deepfakes et Face2Face quasi parfaites (AUC > 0.97)
- NeuralTextures légèrement en retrait (AUC=0.946) mais reste très bon

#### Hyperparamètres par version

| Paramètre | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|-----------|----|----|----|----|-----|-----|-----|
| **Modèle** | ResNet18 | ResNet18 | EfficientNet-B0 | EfficientNet-B0 | EfficientNet-B4 | EfficientNet-B4 | EfficientNet-B4 |
| **IMAGE_SIZE** | 299 | 299 | 299 | 299 | 299 | 380 | **380** |
| **LR** | 2e-04 | 5e-05 | 3e-05 | 3e-05 | 1e-05 | 2e-04/2e-05 | **3e-05 uniforme** |
| **Differential LR** | Non | Non | Non | Non | Non | Oui (×0.1) | **Non** |
| **Batch size** | 32 | 32 | 32 | 32 | 16 | 32 | **32** |
| **Dropout** | 0.5 | 0.6 | 0.5 | 0.5 | 0.5 | 0.4 | **0.4** |
| **Weight decay** | 1e-4 | 1e-4 | 5e-4 | 5e-4 | 5e-4 | 1e-4 | **1e-4** |
| **Class weights** | Non | Non | Non | Non | Non | Oui (auto) | **Oui (auto)** |
| **Frames/vidéo** | 10 | 5 | 5 | 5 | 30 (pré-extraites) | 50 (pré-extraites) | **50 (pré-extraites)** |
| **Face margin** | — | — | 0.3 (1.6×) | 0.3 (1.6×) | 0.3 (1.6×) | 0.15 (1.3×) | **0.15 (1.3×)** |
| **Samples/epoch** | 21,600 | 10,800 | 10,800 | 10,800 | ~64,800 | ~108,000 | **~108,000** |
| **Scheduler** | ReduceLROnPlateau | ReduceLROnPlateau | CosineAnnealing | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau | **ReduceLROnPlateau** |
| **Freeze/unfreeze** | Non | Non | Oui (5 epochs) | Oui (5 epochs) | Oui (5 epochs) | Oui (5 epochs) | **Oui (4 epochs)** |
| **Warmup** | Non | Non | Oui (3 epochs) | Oui (3 epochs) | Oui (3 epochs) | Oui (3 epochs) | **Oui (3 epochs)** |
| **Mixup** | Non | Non | Oui (p=0.5) | Oui (p=0.3) | Oui (p=0.3) | Oui (p=0.2, α=0.2) | **Oui (p=0.2, α=0.2)** |
| **Label smoothing** | Non | Non | Oui (0.1) | Oui (0.1) | Oui (0.1) | Oui (0.1) | **Oui (0.1)** |
| **Face extraction** | Pas de crop | Pas de crop | On-the-fly (SSD) | On-the-fly (SSD) | Pré-extraite (JPEG) | Pré-extraite (margin=0.15) | **Pré-extraite (margin=0.15)** |
| **Augmentations** | Basiques | Basiques | Avancées | Avancées | Avancées | Allégées | **Allégées** |
| **BN freeze** | Non | Non | Non | Non | Non (oublié) | Oui | **Oui** |
| **Gradient clipping** | Non | Non | Non | Non | Oui (1.0) | Oui (1.0) | **Oui (1.0)** |
| **Assertion unfreeze** | Non | Non | Non | Non | Non | Oui | **Oui** |
| **Early stopping** | patience=10 | patience=15 | Non | patience=20 | patience=20 | patience=15 | **patience=10** |

#### Bugs et leçons par version

| Version | Bug / Problème | Diagnostic | Leçon |
|---------|---------------|------------|-------|
| V1 | LR=2e-04 trop élevé → overfitting (train 83%, val 54%, écart 28%) | Val acc chute pendant que train acc monte | Un LR trop grand fait "sauter" les minima → le scheduler a sauvé le run en divisant par 10 |
| V2 | Pas de bug, mais ResNet18 (11M params) atteint sa limite à ~89% | Val acc stagne malgré hyperparamètres optimaux | Pour aller plus loin, il faut un modèle avec plus de capacité |
| V3 | `return` mal indenté dans `set_trainable_up_to()` → seul 1 paramètre dégelé sur des milliers | train acc (68%) < val acc (73%) = pattern d'underfitting | **Toujours tester le freeze/unfreeze avec une assertion**. Coût du bug : 20h de GPU |
| V4 | Plateau à 90% → limite de capacité d'EfficientNet-B0 (5.3M params) | Écart train/val de 2% seulement = bonne généralisation, le modèle a donné tout ce qu'il pouvait | B0 est trop petit pour capturer les artefacts subtils de deepfake en c23 |
| V5 | BN freeze oublié (flag non passé dans script SLURM) + IMAGE_SIZE 299 au lieu de 380 + pas de class weights + augmentations trop agressives | Oscillations 3% val acc (88-91%) malgré B4 (19M params) | Toujours vérifier que **chaque flag** du code est passé dans le script. Auditer le paper original avant de choisir les hyperparamètres |
| V6 | Differential LR ×0.1 → désapprentissage catastrophique (train ET val chutent après epoch 9) | Pattern inédit : les deux métriques descendent ensemble après un pic (92.08%) | Le differential LR crée un désalignement backbone/classifier. Simplicité > sophistication : LR uniforme prouvé en V4 |

#### Ce que chaque version a apporté au pipeline

| Version | Ajout clé |
|---------|-----------|
| V1 | Pipeline de base fonctionnel (train.py, dataset.py, evaluate.py, SLURM) |
| V2 | Optimisation des hyperparamètres (LR, dropout, frames_per_video) |
| V3 | Face extraction, mixup, label smoothing, augmentations avancées, freeze/unfreeze, warmup |
| V4 | Fix bug unfreeze, ReduceLROnPlateau adaptatif, réduction mixup (0.5 → 0.3) |
| V5 | Modèle B4, faces pré-extraites, fix BatchNorm, gradient clipping, CLI configurable |
| V6 | Résolution native 380×380, class weights, differential LR, margin paper (1.3×), augmentations allégées, assertion unfreeze |
| V7 | Retour au LR uniforme 3e-05, freeze 4 epochs, patience 10 — combine la stabilité V4 avec les corrections V6 |

**Progression totale** : 65% → 91% → 95% (estimé V7), soit **+30 points d'accuracy** en 7 itérations.

**Temps GPU total consommé** : ~65h (dont 20h perdues sur le bug V3, ~7h sur V6). V7 estimé : +12-16h.

**Point clé pour le rapport** : chaque version a apporté une leçon — V1 sur l'overfitting et le learning rate, V2 sur l'optimisation des hyperparamètres, V3 sur l'importance des tests unitaires et du diagnostic par les métriques, V4 sur le choix du scheduler, V5 sur le fine-tuning des BatchNorm et l'audit de la littérature, V6 sur les dangers du differential LR et l'importance de la stabilité d'entraînement, V7 sur le retour aux fondamentaux (LR uniforme prouvé). L'amélioration n'est pas seulement venue du modèle plus gros, mais de la compréhension progressive du problème.

---

## Résultats d'entraînement V6 (Job 861929)

- **Date** : 2026-06-24
- **Cluster** : gpu-gw.enst.fr, RTX 3090 24GB
- **Config** : EfficientNet-B4 380×380, lr=2e-04/2e-05 (differential ×0.1), dropout=0.4, batch_size=32, weight_decay=1e-4, freeze_epochs=5, warmup_epochs=3, freeze_bn, class weights auto, faces pré-extraites (margin=0.15, 50/vidéo)

### Résultats d'entraînement

| Epoch | Train loss | Train acc | Val loss | Val acc | LR (backbone/classifier) | Observation |
|-------|------------|-----------|----------|---------|--------------------------|-------------|
| 1 | 0.6943 | 49.95% | 0.7076 | 48.14% | 7.3e-06/6.7e-05 | Warmup (1/3) |
| 2 | 0.6811 | 56.48% | 0.6901 | 60.43% | 1.4e-05/1.3e-04 | Warmup (2/3) |
| 3 | 0.6718 | 58.99% | 0.6840 | 62.11% | 2.0e-05/2.0e-04 | Warmup terminé |
| 4 | 0.6659 | 60.29% | 0.6795 | 63.12% | 2.0e-05/2.0e-04 | Freeze (classifier seul) |
| 5 | 0.6626 | 60.98% | 0.6815 | 61.41% | 2.0e-05/2.0e-04 | Dernière epoch freeze |
| **6** | **0.4697** | **82.33%** | **0.4281** | **88.06%** | 2.0e-05/2.0e-04 | **Unfreeze → saut +27% val acc** |
| 7 | 0.3736 | 90.56% | 0.4370 | 89.31% | 2.0e-05/2.0e-04 | Pic train acc |
| 8 | 0.3744 | 90.60% | 0.4002 | 91.74% | 2.0e-05/2.0e-04 | |
| **9** | **0.3888** | **89.51%** | **0.3890** | **92.08%** | 2.0e-05/2.0e-04 | **Meilleur modèle** ✅ |
| 10 | 0.4113 | 87.75% | 0.4180 | 89.76% | 2.0e-05/2.0e-04 | **Début dégradation** ⚠️ |
| 11 | 0.4401 | 85.70% | 0.4147 | 89.75% | 2.0e-05/2.0e-04 | Train acc continue de baisser |
| 12 | 0.4696 | 83.22% | 0.4417 | 88.26% | 2.0e-05/2.0e-04 | |
| 13 | 0.4995 | 80.46% | 0.4492 | 87.29% | 2.0e-05/2.0e-04 | |
| 14 | 0.5325 | 77.42% | 0.4893 | 84.02% | 2.0e-05/2.0e-04 | |
| 15 | 0.5635 | 74.16% | 0.5263 | 81.11% | **1.0e-05/1.0e-04** | **Scheduler ÷2** (1re intervention) |
| 16 | 0.5646 | 73.93% | 0.5266 | 80.85% | 1.0e-05/1.0e-04 | ÷2 ne stoppe pas la chute |
| 17 | 0.5779 | 72.36% | 0.5229 | 81.15% | 1.0e-05/1.0e-04 | |
| 18 | 0.5934 | 70.49% | 0.5531 | 78.34% | 1.0e-05/1.0e-04 | |
| 19 | 0.6069 | 69.23% | 0.5503 | 78.69% | 1.0e-05/1.0e-04 | |
| 20 | 0.6168 | 67.75% | 0.5645 | 77.67% | 1.0e-05/1.0e-04 | |
| 21 | 0.6306 | 65.98% | 0.6306 | 68.37% | **5.0e-06/5.0e-05** | **Scheduler ÷2** (2e intervention) |
| 22 | 0.6275 | 66.39% | 0.5968 | 73.18% | 5.0e-06/5.0e-05 | |
| 23 | 0.6332 | 65.47% | 0.6640 | 63.86% | 5.0e-06/5.0e-05 | |
| 24 | 0.6400 | 64.78% | 0.6945 | 59.98% | 5.0e-06/5.0e-05 | Early stopping (patience=15 atteinte) |

**Meilleur modèle sauvegardé** : epoch 9, val_acc = **92.08%**, val_loss = 0.3890
**Durée** : ~7h (24 epochs × ~600s/epoch en moyenne)

### Analyse V6 : le differential LR cause un désapprentissage catastrophique

#### Le pattern inédit : train ET val dégradent ensemble

C'est un pattern qu'on n'avait jamais vu dans les versions précédentes :
- **Overfitting** (V1) : train monte, val descend → régularisation insuffisante
- **Underfitting** (V3) : train < val, les deux progressent lentement → modèle ne peut pas apprendre
- **Désapprentissage** (V6) : **train ET val descendent après un pic** → le modèle oublie ce qu'il a appris

Le modèle atteint 92.08% val acc à epoch 9, puis régresse jusqu'à 59.98% à epoch 24 — pire que le hasard ajusté au ratio des classes. Le scheduler intervient 2 fois (epochs 15 et 21, ÷2 chaque fois) sans stopper la chute.

#### Cause : désalignement backbone/classifier dû au differential LR

Le differential LR à ratio 10× (backbone 2e-05, classifier 2e-04) crée un **désalignement progressif** :

1. **Epochs 6-9** : le classifier s'adapte rapidement aux features du backbone → résultats excellents (92%)
2. **Epoch 10+** : le classifier a "overshooté" — il s'est spécialisé sur les features actuelles du backbone, mais le backbone continue de bouger lentement (LR 10× plus faible). Les features changent, le classifier n'est plus aligné → les deux régressent
3. Le phénomène s'auto-renforce : les gradients du classifier devenu incohérent perturbent le backbone → les features dérivent encore plus → cercle vicieux

En V4 (LR uniforme 3e-05), backbone et classifier évoluaient au même rythme → pas de désalignement → pas de dégradation.

#### Facteur aggravant : BN freeze

Avec `--freeze_bn`, les couches BatchNorm utilisent les statistiques ImageNet fixes. Quand les poids du backbone dérivent (à cause du désalignement), les statistiques BN deviennent progressivement incorrectes. Sans BN freeze, les running stats auraient pu partiellement compenser la dérive — avec BN freeze, aucune correction n'est possible.

#### Analyse des courbes d'entraînement

Courbes dans `results/v6/training_curves.png` :

- **Loss** : les deux courbes (train et val) plongent à epoch 6 (unfreeze), atteignent un minimum à epoch 8-10, puis **remontent ensemble** jusqu'à ~0.65-0.70 — retour quasi au niveau pré-unfreeze
- **Accuracy** : pic à epoch 7-9 (~90% train, ~92% val), puis chute symétrique des deux courbes. À epoch 24, train (65%) et val (60%) sont revenus au niveau des epochs de freeze
- Le pattern est parfaitement symétrique : ce qui a été appris aux epochs 6-9 est systématiquement désappris aux epochs 10-24

### Résultats d'évaluation V6 (test set, 20574 samples)

| Métrique | V4 (B0) | V5 (B4) | V6 (B4) |
|----------|---------|---------|---------|
| **Samples** | 4200 | 12347 | 20574 |
| **Accuracy** | **89.79%** | 88.77% | 89.37% |
| **AUC** | **0.9671** | 0.9392 | 0.9587 |
| **Precision** | 96.22% | 90.36% | **95.11%** |
| **Recall** | 88.14% | **93.05%** | 88.56% |
| **F1** | **0.9200** | 0.9168 | 0.9172 |

V6 remonte par rapport à V5 (AUC +0.02, precision +5%) grâce aux corrections (résolution 380, class weights, margin 1.3×), mais ne bat pas V4 car le differential LR a limité l'entraînement utile à seulement 4 epochs (6-9).

#### AUC par méthode

| Méthode | V4 (B0) | V5 (B4) | V6 (B4) | DeepfakeBench (B4) |
|---------|---------|---------|---------|-------------------|
| Deepfakes | **0.9797** | 0.9685 | 0.9725 | 0.9757 |
| Face2Face | **0.9712** | 0.9436 | 0.9703 | 0.9758 |
| FaceSwap | **0.9683** | 0.9294 | 0.9549 | 0.9797 |
| NeuralTextures | **0.9458** | 0.9156 | 0.9372 | 0.9308 |
| **Global** | **0.9671** | 0.9392 | 0.9587 | 0.9567 |

V6 rattrape quasi V4 sur Deepfakes et Face2Face (écart <0.01 AUC). FaceSwap et NeuralTextures restent en retrait — le modèle n'a pas eu assez d'epochs d'entraînement stable pour apprendre ces artefacts plus subtils.

#### Accuracy par méthode

| Méthode | V4 | V5 | V6 |
|---------|-----|-----|-----|
| Deepfakes | **92.24%** | 86.40% | 91.51% |
| Face2Face | **92.62%** | 84.79% | 91.30% |
| FaceSwap | **91.29%** | 83.56% | 90.13% |
| NeuralTextures | **88.95%** | 83.36% | 87.72% |

V6 récupère 5-7% par rapport à V5 sur chaque méthode, grâce aux class weights et à la résolution native. L'écart avec V4 est réduit à ~1%.

#### Confusion matrix V6 : `[[6268, 623], [1565, 12118]]`

- 623 faux positifs (9% des real classées fake) — entre V4 (7%) et V5 (20%)
- 1565 faux négatifs (11.4% des fakes manqués) — similaire à V4 (12%)
- Meilleur équilibre precision/recall que V5 grâce aux class weights
- Les class weights ont corrigé le biais vers "fake" de V5 (815 FP → 623 FP)

#### Analyse des courbes ROC V6

Courbes dans `results/v6/roc_curves_per_method.png` :

- Deepfakes (AUC=0.973) et Face2Face (AUC=0.970) quasi superposées, proches du coin haut-gauche
- FaceSwap (AUC=0.955) légèrement en retrait
- NeuralTextures (AUC=0.937) nettement plus basse, avec une montée plus lente — les artefacts de NeuralTextures (modification de texture du visage) sont les plus subtils et les plus difficiles à détecter
- Globalement très similaire à V4 mais avec une courbe NeuralTextures un peu moins bonne

#### Analyse des exemples de prédictions

Grille dans `results/v6/example_predictions.png` :

- **Vrais Négatifs** (Real→Real) : confiance très élevée (0.98, 0.99) — le modèle est sûr de lui sur les vrais visages
- **Vrais Positifs** (Fake→Fake) : confiance bonne (0.90, 0.93) — détection fiable des fakes
- **Faux Négatifs** (Fake→Real) : un cas à confiance 0.93 (le modèle est très confiant que c'est real alors que c'est fake) et un à 0.50 (hésitation) — les faux négatifs confiants sont les plus dangereux
- **Faux Positifs** (Real→Fake) : confiances faibles (0.53, 0.57) — ce sont des cas borderline, le modèle n'est pas sûr

### Leçons V6

#### 1. Le differential LR est dangereux sans validation empirique

Le differential LR (backbone ×0.1) est une technique courante en fine-tuning, recommandée dans la littérature. Mais sur notre pipeline (avec freeze/unfreeze, BN freeze, class weights, mixup), il crée une instabilité qui détruit l'entraînement après seulement 4 epochs utiles.

**Diagnostic clé** : quand train acc ET val acc baissent ensemble, ce n'est ni de l'overfitting ni de l'underfitting — c'est de l'instabilité d'entraînement. Il faut chercher un problème dans la dynamique d'optimisation (LR, momentum, interactions entre composants).

#### 2. Les corrections V6 fonctionnent

Malgré le problème de LR, V6 a montré que les corrections (résolution 380, class weights, margin 1.3×) sont efficaces :
- Le pic à epoch 9 (92.08%) dépasse le meilleur de V4 (89.71%) de +2.4%
- L'AUC par méthode rattrape quasi V4 en seulement 4 epochs d'entraînement utile
- Les class weights corrigent le biais vers "fake" de V5

#### 3. Simplicité > sophistication

V4 (LR uniforme 3e-05) a battu V6 (differential LR 2e-04/2e-05) en stabilité. Ajouter de la complexité (differential LR) sans la tester isolément a introduit un problème que les autres améliorations n'ont pas pu compenser.

---

## V7 : LR Uniforme + Corrections V6

- **Date** : 2026-06-24

### Motivations

V6 a démontré que les corrections structurelles (résolution 380, class weights, margin 1.3×, BN freeze) fonctionnent — le pic à 92.08% le prouve. Mais le differential LR a causé un désapprentissage catastrophique. V7 garde tout de V6 et revient au LR uniforme 3e-05 (prouvé stable en V4).

### Changements V6 → V7 (4 modifications)

| Paramètre | V6 | V7 | Raison |
|-----------|-----|-----|--------|
| **LR** | 2e-04/2e-05 (differential ×0.1) | **3e-05 uniforme** | Le LR uniforme est prouvé stable en V4 (23 epochs sans dégradation) |
| **differential_lr** | Oui | **Non** | Supprimé — cause du désapprentissage V6 |
| **freeze_epochs** | 5 | **4** | 1 epoch de moins en freeze, le classifier a assez de temps en 4 epochs + 3 warmup |
| **patience** | 15 | **10** | Early stopping plus réactif pour capturer le pic |

Tout le reste de V6 est conservé : résolution 380×380, class weights, margin 0.15, freeze_bn, assertion unfreeze, augmentations allégées, gradient clipping.

### Config V7

```bash
python3 -m src.train \
    --data_root data \
    --faces_dir data/faces_v6 \
    --compression c23 \
    --model efficientnet_b4 \
    --dropout 0.4 \
    --batch_size 32 \
    --lr 0.00003 \
    --weight_decay 1e-4 \
    --freeze_epochs 4 \
    --warmup_epochs 3 \
    --freeze_bn \
    --epochs 50 \
    --patience 10 \
    --num_workers 8 \
    --checkpoint_dir checkpoints \
    --log_dir logs/tensorboard
```

**Fichier** : `scripts/submit_train_v7.sh`
**Faces** : réutilise `data/faces_v6` (margin=0.15, 50 faces/vidéo, déjà extraites)

### Timeline attendue

- Epochs 1-3 : warmup (1e-6 → 3e-05), classifier seul (~356s/epoch)
- Epoch 4 : plein LR, classifier seul
- Epoch 5 : unfreeze tout le backbone (~1020s/epoch)
- Epoch 6+ : entraînement complet, early stopping si stagnation 10 epochs
- Val acc finale estimée : **93-95%**

### Résultats V7 (Job 862284 - en cours au 2026-06-24)

**Cluster** : gpu-gw.enst.fr, RTX 3090 24GB

| Epoch | Train loss | Train acc | Val loss | Val acc | LR | Observation |
|-------|------------|-----------|----------|---------|-----|-------------|
| 1 | 0.7007 | 44.99% | 0.7112 | 42.76% | 1.1e-05 | Warmup (1/3) |
| 2 | 0.6964 | 48.38% | 0.7089 | 47.16% | 2.0e-05 | Warmup (2/3) |
| 3 | 0.6913 | 51.75% | 0.7007 | 57.38% | 3.0e-05 | Warmup terminé |
| 4 | 0.6863 | 54.93% | 0.7053 | 50.09% | 3.0e-05 | Dernière epoch freeze |
| **5** | **0.4797** | **81.22%** | **0.3861** | **91.39%** | 3.0e-05 | **Unfreeze → saut +41% val acc** ✅ |

**Assertion unfreeze confirmée** : 17,552,202/17,552,202 params trainable.

Le saut à l'unfreeze est encore plus fort qu'en V6 (+41% vs +27%). Le LR uniforme 3e-05 permet au backbone ET au classifier de s'ajuster ensemble dès la première epoch d'entraînement complet.

### Résultats complets V7 (Job 862284)

| Epoch | Train loss | Train acc | Val loss | Val acc | LR | Observation |
|-------|------------|-----------|----------|---------|-----|-------------|
| 1 | 0.7007 | 44.99% | 0.7112 | 42.76% | 1.1e-05 | Warmup (1/3) |
| 2 | 0.6964 | 48.38% | 0.7089 | 47.16% | 2.0e-05 | Warmup (2/3) |
| 3 | 0.6913 | 51.75% | 0.7007 | 57.38% | 3.0e-05 | Warmup terminé |
| 4 | 0.6863 | 54.93% | 0.7053 | 50.09% | 3.0e-05 | Dernière epoch freeze |
| **5** | **0.4797** | **81.22%** | **0.3861** | **91.39%** | 3.0e-05 | **Unfreeze → saut +41%** |
| 6 | 0.3836 | 89.89% | 0.4208 | 89.90% | 3.0e-05 | Ajustement |
| 7 | 0.3829 | 89.98% | 0.4022 | 90.86% | 3.0e-05 | |
| 8 | 0.3976 | 88.85% | 0.3917 | 91.54% | 3.0e-05 | Nouveau best |
| **9** | **0.4182** | **87.41%** | **0.3878** | **91.77%** | 3.0e-05 | **Meilleur modèle** ✅ |
| 10 | 0.4522 | 84.54% | 0.5090 | 83.22% | 3.0e-05 | **Début dégradation** ⚠️ |
| 11 | 0.4941 | 80.87% | 0.4628 | 86.24% | **1.5e-05** | Scheduler ÷2 |
| 12 | 0.5112 | 79.08% | 0.4663 | 84.91% | 1.5e-05 | |
| 13 | 0.5405 | 76.27% | 0.5315 | 81.05% | 1.5e-05 | |
| 14-16 | ↗ 0.62 | ↘ 67% | ~0.60 | ~71% | 1.5e-05 | Déclin continu |
| 17 | 0.6438 | 64.52% | 0.6236 | 69.00% | **7.5e-06** | Scheduler ÷2 |
| 18 | 0.6446 | 64.06% | 0.6023 | 74.33% | 7.5e-06 | |
| 19 | 0.6508 | 63.19% | 0.6190 | 73.28% | 7.5e-06 | Early stopping (patience=10) |

**Meilleur modèle sauvegardé** : epoch 9, val_acc = **91.77%**, val_loss = 0.3878
**Durée** : ~5.5h (19 epochs)

### Résultats d'évaluation V7 (test set, 20574 samples)

| Métrique | V4 (B0) | V6 (B4) | V7 (B4) |
|----------|---------|---------|---------|
| **Accuracy** | **89.79%** | 89.37% | 88.47% |
| **AUC** | **0.9671** | 0.9587 | 0.9508 |
| **Precision** | **96.22%** | 95.11% | 90.76% |
| **Recall** | 88.14% | 88.56% | **92.03%** |
| **F1** | **0.9200** | 0.9172 | 0.9139 |

#### AUC par méthode — V7

| Méthode | V4 | V6 | V7 |
|---------|-----|-----|-----|
| Deepfakes | **0.9797** | 0.9725 | 0.9611 |
| Face2Face | **0.9712** | 0.9703 | 0.9660 |
| FaceSwap | **0.9683** | 0.9549 | 0.9581 |
| NeuralTextures | **0.9458** | 0.9372 | 0.9179 |
| **Global** | **0.9671** | 0.9587 | 0.9508 |

#### Confusion matrix V7 : `[[5609, 1282], [1091, 12592]]`

- 1282 faux positifs (18.6% des real classées fake) — pire que V6 (9%) et V4 (7%)
- 1091 faux négatifs (8% des fakes manqués) — meilleur que V4 (12%)
- Le modèle a un biais vers "fake" plus marqué que V6

### Analyse V7 : le LR uniforme ne suffit pas

V7 reproduit le **même pattern de dégradation que V6** malgré le LR uniforme :
- Pic à epoch 9 (91.77%), puis train ET val déclinent ensemble
- Scheduler intervient 2 fois (epochs 11 et 17) sans stopper la chute
- Early stopping à epoch 19

**Le differential LR n'était pas la seule cause du problème.** Les différences entre V4 (stable) et V7 (instable) sont :

| | V4 (stable) | V7 (instable) |
|---|---|---|
| BN freeze | **Non** | Oui |
| Class weights | **Non** | Oui |
| Samples/epoch | **10,800** | 108,000 |
| Résolution | **299×299** | 380×380 |
| Modèle | **B0 (5.3M)** | B4 (19M) |

**Cause probable : le BN freeze.** En V4, les couches BatchNorm adaptaient leurs statistiques quand les poids du backbone changeaient. Avec BN freeze, les stats ImageNet deviennent progressivement incorrectes à mesure que le backbone évolue. Avec 108,000 samples/epoch (10× plus que V4), les poids bougent beaucoup plus par epoch, accélérant le décalage.

**Résultats test V7 inférieurs à V6** : malgré le LR uniforme, V7 (88.47% test accuracy, 0.9508 AUC) fait moins bien que V6 (89.37%, 0.9587). Le checkpoint V7 (epoch 9) a eu moins d'epochs utiles que V6 (epoch 9 aussi, mais V6 avait des gradients plus forts grâce au LR classifier à 2e-04).

### Conclusion des expériences

**V4 reste le meilleur modèle évalué** (89.79% test accuracy, 0.9671 AUC) malgré un modèle 3.6× plus petit que V5-V7. Le passage à EfficientNet-B4 a permis d'atteindre des pics de val acc plus élevés (92% vs 90%) mais l'instabilité d'entraînement (liée au BN freeze et/ou au volume de données) empêche de concrétiser ces gains sur le test set.

**Temps GPU total** : ~72h (V1: 13h, V3: 20h, V4: 13h, V5: 10h, V6: 7h, V7: 5.5h, extraction: 3.5h)

---

## Nettoyage du projet (2026-06-24)

### Fichiers supprimés
- `analysis_results.md`, `FIX_OVERFITTING.md`, `IMPROVEMENTS.md`, `QUICK_START_OPTIMIZED.md`, `RESUME_ANALYSE.txt` — notes temporaires obsolètes, tout est dans le journal
- `docs/` — `GPU_CLUSTER_GUIDE.md` et `PROJECT_OVERVIEW.md` redondants avec le journal
- `report.pdf` — fichier compilé, pas de source à versionner
- Scripts SLURM obsolètes : `submit_train.sh` (V1), `submit_train_v5.sh`, `submit_train_v7.sh`, `submit_eval.sh` (V1), `submit_eval_v5.sh`, `download_xception_weights.sh`

### Scripts conservés
- `scripts/submit_train_v4.sh` — config du meilleur modèle évalué (EfficientNet-B0)
- `scripts/submit_train_v6.sh` — config du meilleur pic val acc (EfficientNet-B4)
- `scripts/submit_eval_v6.sh` — évaluation
- `scripts/submit_extract_faces.sh` — extraction de visages
- `scripts/setup_cluster.sh`, `scripts/download_dataset.py`, `scripts/extract_faces.py`, `scripts/extract_frames.py` — utilitaires

### Structure finale du projet

```
FaceForensics/
├── src/                          # Notre code
│   ├── models/                   #   architectures (xception.py, models.py)
│   ├── data/                     #   dataset, transforms, augmentations, face extraction, mixup
│   ├── train.py                  #   script d'entraînement
│   ├── evaluate.py               #   script d'évaluation
│   └── detect.py                 #   détection sur vidéo
├── scripts/                      # Scripts utilitaires et SLURM
│   ├── submit_train_v4.sh        #   entraînement V4 (meilleur AUC test)
│   ├── submit_train_v6.sh        #   entraînement V6 (meilleur val acc)
│   ├── submit_eval_v6.sh         #   évaluation
│   ├── submit_extract_faces.sh   #   extraction de visages
│   ├── setup_cluster.sh          #   setup initial cluster
│   ├── download_dataset.py       #   téléchargement FF++
│   ├── extract_faces.py          #   extraction de visages
│   └── extract_frames.py         #   extraction de frames
├── configs/splits/               # Splits train/val/test
├── results/                      # Résultats par version (courbes, métriques, graphiques)
├── images/                       # Images du paper original (utilisées dans le rapport)
├── classification/               # Code original des auteurs (référence)
├── dataset/                      # Scripts de génération des auteurs (référence)
├── report.typ                    # Rapport Typst
├── JOURNAL.md                    # Journal de développement complet
├── README.md                     # README du repo original
├── requirements.txt              # Dépendances Python
├── .gitignore                    # Ignore data/, checkpoints/, logs/, *.pth, *.err, *.out
└── LICENSE
```
