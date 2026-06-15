# FaceForensics++ - Projet de Détection de Deepfakes

## 📋 Objectif du Projet

Développer et entraîner un modèle de détection de manipulations faciales (deepfakes) en utilisant le dataset FaceForensics++ et l'architecture XceptionNet.

## 🎯 Dataset FaceForensics++

### Composition
- **1000 vidéos originales** de YouTube
- **4 méthodes de manipulation** automatisées :
  - **DeepFakes** - Échange de visage via auto-encodeur
  - **Face2Face** - Transfert d'expressions faciales
  - **FaceSwap** - Échange de visage (méthode Kowalski)
  - **NeuralTextures** - Manipulation basée sur GANs
- **FaceShifter** - Méthode récente (CVPR 2020)
- **Deep Fake Detection Dataset** - Dataset Google/JigSaw (3000+ vidéos)

### Splits
- **Train**: 720 vidéos
- **Validation**: 140 vidéos  
- **Test**: 140 vidéos

## 🏗️ Architecture

**Modèle**: XceptionNet
- Architecture de classification binaire (Real vs Fake)
- Input: Images 299x299 pixels
- Pré-entraîné sur ImageNet
- Fine-tuning sur FaceForensics++

## 📅 Plan de Travail

### Phase 1: Setup & Exploration (1-2 jours)
**Objectif**: Comprendre le projet et préparer l'environnement

**Tâches**:
- [x] Clone du repository
- [x] Exploration de la structure du projet
- [ ] Lecture de la documentation
- [ ] Analyse du code existant (détection, preprocessing)
- [ ] Étude de l'architecture XceptionNet
- [ ] Installation des dépendances localement (test)

**Livrables**:
- Compréhension de l'architecture
- Documentation des scripts disponibles

---

### Phase 2: Accès & Préparation des Données (2-3 jours)
**Objectif**: Obtenir et préparer le dataset

**Tâches**:
- [ ] Remplir le [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform) pour accès dataset
- [ ] Télécharger le dataset (version **c23** recommandée = ~10GB)
  ```bash
  python download-FaceForensics.py <output_path> -d all -c c23 -t videos
  ```
- [ ] Transférer le dataset vers le cluster GPU Télécom Paris
- [ ] Extraire les frames des vidéos
  ```bash
  python extract_compressed_videos.py <output_path> -d all -c c23
  ```
- [ ] Vérifier l'intégrité des données
- [ ] Analyser les splits (train/val/test)

**Livrables**:
- Dataset téléchargé et vérifié
- Frames extraites et organisées
- Statistiques du dataset

**Espace requis**:
- Version c23 (recommandée): ~10-22 GB
- Version c0/raw (optionnel): ~500 GB
- Frames extraites: variable selon compression

---

### Phase 3: Développement du Script d'Entraînement (2-3 jours)
**Objectif**: Créer le pipeline d'entraînement

**Tâches**:
- [ ] Créer le script d'entraînement (non fourni dans le repo)
- [ ] Implémenter le data loader
  - Chargement des frames
  - Data augmentation
  - Normalisation (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
- [ ] Configurer le modèle XceptionNet
  - Chargement des poids pré-entraînés
  - Modification de la dernière couche (2 classes: real/fake)
- [ ] Implémenter la boucle d'entraînement
  - Loss function (CrossEntropy)
  - Optimizer (Adam/SGD)
  - Learning rate scheduler
- [ ] Ajouter logging et monitoring
  - TensorBoard ou Weights & Biases
  - Checkpointing
  - Early stopping
- [ ] Créer un script de validation

**Livrables**:
- `train.py` - Script d'entraînement complet
- `data_loader.py` - Gestion du dataset
- Configuration des hyperparamètres

**Hyperparamètres suggérés**:
```python
BATCH_SIZE = 32  # Ajuster selon VRAM disponible
LEARNING_RATE = 0.0001
EPOCHS = 50
IMAGE_SIZE = 299
NUM_CLASSES = 2
```

---

### Phase 4: Entraînement sur Cluster GPU (3-7 jours)
**Objectif**: Entraîner le modèle sur le cluster Télécom Paris

**Tâches**:
- [ ] Configurer l'environnement sur le cluster
  - Créer environnement virtuel
  - Installer PyTorch + CUDA
  - Installer dépendances
- [ ] Créer le script SLURM pour soumission de job
- [ ] Lancer entraînement sur partition P100 ou 3090
- [ ] Monitoring des métriques
  - Train loss/accuracy
  - Validation loss/accuracy
  - Temps par epoch
- [ ] Sauvegarder les checkpoints régulièrement
- [ ] Ajuster hyperparamètres si nécessaire

**Livrables**:
- Modèle entraîné (fichiers .pth)
- Logs d'entraînement
- Courbes de loss/accuracy

**Ressources cluster**:
- GPU: P100 ou 3090
- Temps max: 36h par job
- Jobs simultanés: 4 max

---

### Phase 5: Évaluation & Analyse (2-3 jours)
**Objectif**: Évaluer les performances du modèle

**Tâches**:
- [ ] Test sur le test set (140 vidéos)
- [ ] Calculer les métriques
  - Accuracy globale
  - Precision, Recall, F1-Score
  - AUC-ROC
  - Matrice de confusion
- [ ] Analyse par méthode de manipulation
  - Performance sur DeepFakes
  - Performance sur Face2Face
  - Performance sur FaceSwap
  - Performance sur NeuralTextures
- [ ] Test sur vidéos individuelles
  ```bash
  python detect_from_video.py -i <video_path> -m <model_path> -o <output_path> --cuda
  ```
- [ ] Analyse des erreurs
  - Faux positifs
  - Faux négatifs
  - Visualisation des cas difficiles

**Livrables**:
- Rapport d'évaluation
- Métriques détaillées
- Visualisations des résultats

---

### Phase 6: Rapport & Présentation (2-3 jours)
**Objectif**: Documenter et présenter les résultats

**Tâches**:
- [ ] Rédaction du rapport final
  - Introduction & contexte
  - Méthodologie
  - Résultats expérimentaux
  - Analyse et discussion
  - Conclusion et perspectives
- [ ] Création de visualisations
  - Courbes d'apprentissage
  - Matrice de confusion
  - Exemples de détections
- [ ] Préparation de la présentation
- [ ] Nettoyage et documentation du code

**Livrables**:
- Rapport final (PDF)
- Présentation (slides)
- Code documenté et organisé

---

## 🛠️ Structure du Projet

```
FaceForensics/
├── classification/           # Scripts de classification
│   ├── network/             # Architecture XceptionNet
│   ├── dataset/             # Data transforms
│   ├── detect_from_video.py # Détection sur vidéo
│   └── requirements.txt
├── dataset/                 # Scripts dataset
│   ├── download-FaceForensics.py  # Download script
│   ├── extract_compressed_videos.py
│   ├── compress.py
│   └── splits/              # Train/val/test splits
├── images/                  # Exemples visuels
└── README.md

À créer:
├── train.py                 # Script d'entraînement
├── evaluate.py              # Script d'évaluation
├── data_loader.py           # Dataset loader
├── config.py                # Configuration
└── scripts/
    └── submit_job.sh        # Script SLURM
```

## 📊 Métriques de Succès

- **Accuracy** > 85% sur le test set
- **AUC-ROC** > 0.90
- Temps d'entraînement < 24h
- Modèle capable de généraliser sur les 4 méthodes

## 🔗 Ressources

- **Paper**: [FaceForensics++ (ICCV 2019)](https://arxiv.org/abs/1901.08971)
- **Accès dataset**: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform)
- **Benchmark**: http://kaldir.vc.in.tum.de/faceforensics_benchmark/
- **Documentation cluster**: https://computing.telecom-paris.fr/
- **Video**: https://www.youtube.com/watch?v=x2g48Q2I2ZQ

## ⚠️ Points d'Attention

1. **Dataset nécessite autorisation** - Remplir le formulaire en priorité
2. **Pas de script d'entraînement fourni** - À développer
3. **Dépendances anciennes** - PyTorch 1.0, Python 3.6 (peut nécessiter update)
4. **Espace disque** - Prévoir minimum 50GB pour dataset + modèles
5. **Temps GPU** - Limité à 36h par job sur le cluster

## 📝 Notes

- Commencer avec version **c23** du dataset (10GB) plutôt que raw (2TB)
- Utiliser les splits fournis dans `dataset/splits/`
- Sauvegarder régulièrement les checkpoints
- Documenter tous les hyperparamètres utilisés
- Versionner le code avec git

## 📧 Contact

- **FaceForensics**: faceforensics@googlegroups.com
- **Cluster Télécom Paris**: Voir documentation computing.telecom-paris.fr
