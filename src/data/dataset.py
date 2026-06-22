import os
import json
import random
from os.path import join

import cv2
from PIL import Image
from torch.utils.data import Dataset

from src.data.face_extraction import extract_face

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


class FaceForensicsDataset(Dataset):

    def __init__(self, data_root, split_path, compression='c40',
                 methods=None, transform=None, num_frames_per_video=10,
                 faces_dir=None):
        self.data_root = data_root
        self.compression = compression
        self.methods = methods or METHODS
        self.transform = transform
        self.num_frames_per_video = num_frames_per_video
        self.faces_dir = faces_dir

        with open(split_path, 'r') as f:
            pairs = json.load(f)

        self.samples = []
        if faces_dir:
            self._build_faces_list(pairs)
        else:
            self._build_sample_list(pairs)

    def _build_sample_list(self, pairs):
        seen_originals = set()

        for target, source in pairs:
            for vid_id in [target, source]:
                if vid_id not in seen_originals:
                    path = join(self.data_root, 'original_sequences', 'youtube',
                                self.compression, 'videos', f'{vid_id}.mp4')
                    if os.path.isfile(path):
                        self.samples.append((path, 0))
                    seen_originals.add(vid_id)

            fake_name = f'{target}_{source}.mp4'
            for method in self.methods:
                path = join(self.data_root, 'manipulated_sequences', method,
                            self.compression, 'videos', fake_name)
                if os.path.isfile(path):
                    self.samples.append((path, 1))

    def _build_faces_list(self, pairs):
        seen_originals = set()

        for target, source in pairs:
            for vid_id in [target, source]:
                if vid_id not in seen_originals:
                    face_dir = join(self.faces_dir, self.compression,
                                    'original', str(vid_id))
                    if os.path.isdir(face_dir):
                        for fname in sorted(os.listdir(face_dir)):
                            if fname.endswith('.jpg'):
                                self.samples.append((join(face_dir, fname), 0))
                    seen_originals.add(vid_id)

            fake_name = f'{target}_{source}'
            for method in self.methods:
                face_dir = join(self.faces_dir, self.compression,
                                method, fake_name)
                if os.path.isdir(face_dir):
                    for fname in sorted(os.listdir(face_dir)):
                        if fname.endswith('.jpg'):
                            self.samples.append((join(face_dir, fname), 1))

    def __len__(self):
        if self.faces_dir:
            return len(self.samples)
        return len(self.samples) * self.num_frames_per_video

    def __getitem__(self, idx):
        if self.faces_dir:
            path, label = self.samples[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

        video_idx = idx // self.num_frames_per_video
        video_path, label = self.samples[video_idx]

        frame = self._read_random_frame(video_path)

        if self.transform:
            frame = self.transform(frame)

        return frame, label

    def _read_random_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in range(10):
            frame_idx = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            face = extract_face(frame)
            if face is not None:
                cap.release()
                return face

        # Fallback: frame entière si aucun visage trouvé
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cap.release()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_label_counts(self):
        real = sum(1 for _, label in self.samples if label == 0)
        fake = sum(1 for _, label in self.samples if label == 1)
        return {'real': real, 'fake': fake}
