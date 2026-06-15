import os
import json
import random
from os.path import join

import cv2
from PIL import Image
from torch.utils.data import Dataset

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


class FaceForensicsDataset(Dataset):
    """
    Dataset qui lit les vidéos FaceForensics++ directement depuis les .mp4.

    À chaque appel de __getitem__, il ouvre une vidéo et extrait UNE frame
    aléatoire. Pas besoin d'extraire les frames en PNG sur le disque.

    Labels: 0 = real, 1 = fake
    """

    def __init__(self, data_root, split_path, compression='c40',
                 methods=None, transform=None, num_frames_per_video=10):
        """
        Args:
            data_root: chemin vers le dossier data/ (contient original_sequences/ et manipulated_sequences/)
            split_path: chemin vers le fichier split JSON (ex: configs/splits/train.json)
            compression: niveau de compression ('c0', 'c23', 'c40')
            methods: liste des méthodes de manipulation à inclure (default: toutes)
            transform: transforms torchvision à appliquer sur chaque frame
            num_frames_per_video: nombre de frames à échantillonner par vidéo
                                  (contrôle la taille effective du dataset par epoch)
        """
        self.data_root = data_root
        self.compression = compression
        self.methods = methods or METHODS
        self.transform = transform
        self.num_frames_per_video = num_frames_per_video

        with open(split_path, 'r') as f:
            pairs = json.load(f)

        # Construire la liste des (chemin_video, label, nombre_de_frames)
        self.samples = []
        self._build_sample_list(pairs)

    def _build_sample_list(self, pairs):
        """
        À partir des paires du split, on construit la liste de tous les
        échantillons (video, label).

        Chaque paire ["183", "253"] donne :
          - 1 vidéo originale : original_sequences/youtube/c40/videos/183.mp4 (real)
          - 1 vidéo originale : original_sequences/youtube/c40/videos/253.mp4 (real)
          - N vidéos manipulées par méthode : manipulated_sequences/<method>/c40/videos/183_253.mp4 (fake)
        """
        seen_originals = set()

        for target, source in pairs:
            # Vidéos originales (on évite les doublons car un même ID apparaît dans plusieurs paires)
            for vid_id in [target, source]:
                if vid_id not in seen_originals:
                    path = join(self.data_root, 'original_sequences', 'youtube',
                                self.compression, 'videos', f'{vid_id}.mp4')
                    if os.path.isfile(path):
                        self.samples.append((path, 0))
                    seen_originals.add(vid_id)

            # Vidéos manipulées (une par méthode)
            fake_name = f'{target}_{source}.mp4'
            for method in self.methods:
                path = join(self.data_root, 'manipulated_sequences', method,
                            self.compression, 'videos', fake_name)
                if os.path.isfile(path):
                    self.samples.append((path, 1))

    def __len__(self):
        # Chaque vidéo est échantillonnée num_frames_per_video fois par epoch
        return len(self.samples) * self.num_frames_per_video

    def __getitem__(self, idx):
        # Retrouver quelle vidéo correspond à cet index
        video_idx = idx // self.num_frames_per_video
        video_path, label = self.samples[video_idx]

        frame = self._read_random_frame(video_path)

        if self.transform:
            frame = self.transform(frame)

        return frame, label

    def _read_random_frame(self, video_path):
        """
        Ouvre une vidéo et extrait une frame à une position aléatoire.
        Retourne une PIL Image (RGB).
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            # Fallback : lire la première frame si la frame aléatoire échoue
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def get_label_counts(self):
        """Retourne le nombre de vidéos real vs fake (utile pour vérifier l'équilibre)."""
        real = sum(1 for _, label in self.samples if label == 0)
        fake = sum(1 for _, label in self.samples if label == 1)
        return {'real': real, 'fake': fake}
