# Guide de Connexion au Cluster GPU - Télécom Paris

## 📚 Documentation Officielle

**Lien principal**: https://computing.telecom-paris.fr/

⚠️ Accessible uniquement via :
- Réseau de l'école Télécom Paris
- VPN de l'école

La documentation couvre :
- Connexion au cluster
- Soumission de jobs
- Ressources disponibles
- Bonnes pratiques d'utilisation

---

## 🔐 Connexion au Cluster

### Première Connexion

**Commande**:
```bash
ssh votre_login@gpu-gw.enst.fr
```

**Exemple**:
```bash
ssh john.doe@gpu-gw.enst.fr
```

**Note importante**: 
- Votre compte cluster se créera **automatiquement** à votre première connexion
- Utilisez votre login Télécom Paris habituel

### Configuration SSH (Optionnel mais Recommandé)

Pour simplifier les connexions futures, créez un fichier `~/.ssh/config` :

```bash
# Sur votre machine locale
nano ~/.ssh/config
```

Ajoutez :
```
Host telecom-gpu
    HostName gpu-gw.enst.fr
    User votre_login
    ForwardX11 yes
    ServerAliveInterval 60
```

Puis connectez-vous simplement avec :
```bash
ssh telecom-gpu
```

### Génération de Clé SSH (Recommandé)

Pour éviter de taper votre mot de passe à chaque fois :

```bash
# Sur votre machine locale
ssh-keygen -t ed25519 -C "votre_email@example.com"

# Copier la clé publique vers le cluster
ssh-copy-id votre_login@gpu-gw.enst.fr
```

---

## 🖥️ Ressources Disponibles

### Partitions GPU

| Partition | Type GPU | Usage | VRAM |
|-----------|----------|-------|------|
| **P100** | NVIDIA Tesla P100 | Entraînement | 16 GB |
| **3090** | NVIDIA RTX 3090 | Entraînement | 24 GB |

### Partition CPU

- Pour tâches sans GPU
- Preprocessing, extraction de données, etc.

### Limites

⚠️ **Restrictions importantes** :
- **Jobs simultanés maximum** : 4
- **Temps maximum par job** : 36 heures
- Après 36h, le job sera automatiquement arrêté

---

## 🚀 Soumission de Jobs avec SLURM

### Structure d'un Script SLURM

Créez un fichier `submit_job.sh` :

```bash
#!/bin/bash
#SBATCH --job-name=faceforensics_train      # Nom du job
#SBATCH --output=logs/train_%j.out          # Fichier de sortie (%j = job ID)
#SBATCH --error=logs/train_%j.err           # Fichier d'erreur
#SBATCH --partition=3090                    # Partition (P100 ou 3090)
#SBATCH --gres=gpu:1                        # Nombre de GPUs (1 GPU)
#SBATCH --cpus-per-task=4                   # Nombre de CPUs
#SBATCH --mem=32G                           # Mémoire RAM
#SBATCH --time=24:00:00                     # Temps max (format HH:MM:SS)
#SBATCH --mail-type=END,FAIL                # Email à la fin ou en cas d'échec
#SBATCH --mail-user=votre_email@telecom-paris.fr

# Charger les modules nécessaires
module load cuda/11.8
module load python/3.9

# Activer l'environnement virtuel
source ~/venvs/faceforensics/bin/activate

# Aller dans le répertoire du projet
cd ~/FaceForensics

# Lancer l'entraînement
python train.py \
    --data_path /path/to/dataset \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.0001 \
    --output_dir checkpoints/

# Désactiver l'environnement
deactivate
```

### Soumettre le Job

```bash
sbatch submit_job.sh
```

### Commandes de Gestion des Jobs

```bash
# Vérifier l'état de vos jobs
squeue -u votre_login

# Voir tous les jobs en cours
squeue

# Annuler un job
scancel <job_id>

# Voir l'historique de vos jobs
sacct -u votre_login

# Voir les détails d'un job spécifique
scontrol show job <job_id>

# Voir les partitions disponibles
sinfo

# Voir l'utilisation des ressources
sstat <job_id>
```

### Job Interactif (Pour Tests)

Pour obtenir une session interactive sur un nœud GPU :

```bash
srun --partition=3090 --gres=gpu:1 --mem=16G --time=02:00:00 --pty bash
```

Cela vous donne un shell sur un nœud GPU pendant 2 heures.

---

## 📦 Configuration de l'Environnement

### 1. Première Connexion - Setup Initial

```bash
# Connexion au cluster
ssh votre_login@gpu-gw.enst.fr

# Créer un répertoire pour les projets
mkdir -p ~/projects
cd ~/projects

# Cloner le projet (si pas déjà fait)
git clone git@github.com:MaxenceGuinziemba/FaceForensics.git
cd FaceForensics
```

### 2. Créer un Environnement Virtuel Python

```bash
# Créer le répertoire pour les environnements
mkdir -p ~/venvs

# Créer l'environnement virtuel
python3 -m venv ~/venvs/faceforensics

# Activer l'environnement
source ~/venvs/faceforensics/bin/activate

# Mettre à jour pip
pip install --upgrade pip
```

### 3. Installer les Dépendances

**Option 1 : PyTorch avec CUDA** (recommandé)

```bash
# PyTorch avec support CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Autres dépendances
pip install -r classification/requirements.txt
```

**Option 2 : Mettre à jour les anciennes dépendances**

Le `requirements.txt` fourni utilise des versions anciennes (PyTorch 1.0). Vous pouvez créer un nouveau fichier :

```bash
# Créer requirements_updated.txt
cat > requirements_updated.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
pillow>=10.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tensorboard>=2.13.0
dlib>=19.24.0
face-recognition>=1.3.0
h5py>=3.9.0
EOF

pip install -r requirements_updated.txt
```

### 4. Vérifier l'Installation CUDA

```bash
# Dans un job interactif ou script SLURM
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

---

## 📁 Transfert de Fichiers

### Depuis Votre Machine Locale vers le Cluster

**Copier un fichier** :
```bash
scp /local/path/file.txt votre_login@gpu-gw.enst.fr:~/remote/path/
```

**Copier un répertoire** :
```bash
scp -r /local/path/folder votre_login@gpu-gw.enst.fr:~/remote/path/
```

**Avec rsync (recommandé pour gros fichiers)** :
```bash
rsync -avz --progress /local/path/dataset/ votre_login@gpu-gw.enst.fr:~/datasets/faceforensics/
```

### Depuis le Cluster vers Votre Machine

```bash
scp votre_login@gpu-gw.enst.fr:~/checkpoints/model.pth /local/path/
```

### Télécharger le Dataset Directement sur le Cluster

```bash
# Sur le cluster
cd ~/datasets
wget <url_from_faceforensics_form>
# ou
python download-FaceForensics.py ~/datasets/faceforensics -d all -c c23 -t videos
```

---

## 🔍 Monitoring et Debugging

### Surveiller l'Utilisation GPU en Temps Réel

```bash
# Dans une session interactive ou pendant un job
watch -n 1 nvidia-smi
```

### Voir les Logs en Temps Réel

```bash
# Suivre le fichier de sortie d'un job en cours
tail -f logs/train_<job_id>.out

# Suivre les erreurs
tail -f logs/train_<job_id>.err
```

### TensorBoard pour Monitoring

Dans votre script d'entraînement, ajoutez :
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')
```

Puis sur le cluster :
```bash
tensorboard --logdir=runs --port=6006 --bind_all
```

Créez un tunnel SSH depuis votre machine locale :
```bash
ssh -L 6006:localhost:6006 votre_login@gpu-gw.enst.fr
```

Accédez à http://localhost:6006 dans votre navigateur.

---

## ✅ Checklist de Setup Complet

- [ ] Connexion SSH réussie au cluster
- [ ] Compte cluster créé automatiquement
- [ ] Configuration SSH (config file + clé SSH)
- [ ] Projet cloné dans `~/projects/FaceForensics`
- [ ] Environnement virtuel créé dans `~/venvs/faceforensics`
- [ ] PyTorch + CUDA installés et vérifiés
- [ ] Dépendances installées
- [ ] Dataset téléchargé/transféré vers `~/datasets/`
- [ ] Script SLURM créé et testé
- [ ] Premier job de test soumis avec succès
- [ ] Capacité à monitorer les jobs (squeue, logs)

---

## 🎯 Workflow Typique

### 1. Développement Local
```bash
# Sur votre machine
cd FaceForensics
# Éditer le code, créer train.py, etc.
git add .
git commit -m "Add training script"
git push
```

### 2. Mise à Jour sur le Cluster
```bash
# Sur le cluster
cd ~/projects/FaceForensics
git pull
```

### 3. Test Interactif (Optionnel)
```bash
# Obtenir une session GPU
srun --partition=3090 --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash

# Activer l'environnement
source ~/venvs/faceforensics/bin/activate

# Tester le script
python train.py --epochs 1 --batch_size 8
```

### 4. Soumission du Job Complet
```bash
# Créer le répertoire de logs
mkdir -p logs

# Soumettre le job
sbatch submit_job.sh

# Vérifier le statut
squeue -u votre_login
```

### 5. Monitoring
```bash
# Suivre les logs
tail -f logs/train_<job_id>.out

# Vérifier l'utilisation GPU (si job interactif)
nvidia-smi
```

### 6. Récupération des Résultats
```bash
# Télécharger le modèle entraîné
scp votre_login@gpu-gw.enst.fr:~/projects/FaceForensics/checkpoints/* ./local_checkpoints/
```

---

## 🆘 Troubleshooting

### Problème : Job ne démarre pas
```bash
# Vérifier les partitions disponibles
sinfo

# Voir la file d'attente
squeue -p 3090
```

### Problème : Job tué prématurément
- Vérifier la mémoire demandée (peut-être insuffisante)
- Vérifier les logs d'erreur : `cat logs/train_<job_id>.err`
- Réduire le batch size si Out Of Memory (OOM)

### Problème : CUDA non disponible
```bash
# Vérifier que le job demande bien un GPU
#SBATCH --gres=gpu:1

# Charger le module CUDA
module load cuda/11.8
```

### Problème : Connexion SSH refuse
- Vérifier que vous êtes sur le réseau de l'école ou VPN
- Vérifier votre login Télécom Paris

---

## 📞 Support

**Pour problèmes techniques sur le cluster** :
- Consulter https://computing.telecom-paris.fr/
- Contacter l'équipe computing de Télécom Paris

**Pour questions sur FaceForensics** :
- Email : faceforensics@googlegroups.com

---

## 💡 Bonnes Pratiques

1. **Toujours tester avec un job court** avant de lancer un entraînement de 36h
2. **Sauvegarder régulièrement** les checkpoints (toutes les N epochs)
3. **Utiliser tensorboard** pour monitoring en temps réel
4. **Nommer clairement vos jobs** pour les retrouver facilement
5. **Nettoyer vos anciens fichiers** pour libérer l'espace disque
6. **Documenter vos expériences** (hyperparamètres, résultats)
7. **Ne pas laisser de jobs inutiles** en cours (scancel si besoin)
8. **Respecter les quotas** : max 4 jobs simultanés

---

## 🚀 Exemple Complet : Premier Entraînement

```bash
# 1. Connexion
ssh votre_login@gpu-gw.enst.fr

# 2. Préparation
cd ~/projects/FaceForensics
source ~/venvs/faceforensics/bin/activate
mkdir -p logs checkpoints

# 3. Création du script SLURM (voir section ci-dessus)
nano submit_train.sh
# (copier le contenu du template)

# 4. Soumission
sbatch submit_train.sh

# 5. Vérification
squeue -u votre_login

# 6. Monitoring
tail -f logs/train_*.out

# 7. En cas de problème
scancel <job_id>
```

Bon courage pour votre projet ! 🎓
