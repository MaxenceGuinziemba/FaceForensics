"""
Pré-extraction des visages depuis les vidéos FaceForensics++.

Extrait N faces par vidéo et les sauvegarde en .jpg pour accélérer
l'entraînement (pas de face detection on-the-fly à chaque epoch).

Usage:
    python scripts/extract_faces.py --data_root data --output_dir data/faces --compression c23
"""
import argparse
import json
import os
from os.path import join

import cv2
from tqdm import tqdm

from src.data.face_extraction import extract_face

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def extract_from_video(video_path, output_dir, num_frames=30):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return 0

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    saved = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        face = extract_face(frame)
        if face is None:
            continue

        face_path = join(output_dir, f'{i:03d}.jpg')
        face.save(face_path, quality=95)
        saved += 1

    cap.release()
    return saved


def process_split(data_root, split_path, compression, output_dir, num_frames):
    with open(split_path) as f:
        pairs = json.load(f)

    seen_originals = set()
    videos = []

    for target, source in pairs:
        for vid_id in [target, source]:
            if vid_id not in seen_originals:
                path = join(data_root, 'original_sequences', 'youtube',
                            compression, 'videos', f'{vid_id}.mp4')
                if os.path.isfile(path):
                    out = join(output_dir, compression, 'original', str(vid_id))
                    videos.append((path, out, 'original'))
                seen_originals.add(vid_id)

        fake_name = f'{target}_{source}'
        for method in METHODS:
            path = join(data_root, 'manipulated_sequences', method,
                        compression, 'videos', f'{fake_name}.mp4')
            if os.path.isfile(path):
                out = join(output_dir, compression, method, fake_name)
                videos.append((path, out, method))

    total_faces = 0
    no_face_videos = 0

    for video_path, out_dir, category in tqdm(videos, desc='  Extraction'):
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            total_faces += len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
            continue

        saved = extract_from_video(video_path, out_dir, num_frames)
        total_faces += saved
        if saved == 0:
            no_face_videos += 1

    return len(videos), total_faces, no_face_videos


def main():
    parser = argparse.ArgumentParser(description='Pré-extraction des visages')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--compression', type=str, default='c23')
    parser.add_argument('--frames_per_video', type=int, default=30)
    parser.add_argument('--splits_dir', type=str, default='configs/splits')
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        split_path = join(args.splits_dir, f'{split}.json')
        if not os.path.isfile(split_path):
            continue

        print(f'\n=== {split.upper()} ===')
        n_videos, n_faces, n_failed = process_split(
            args.data_root, split_path, args.compression,
            args.output_dir, args.frames_per_video,
        )
        print(f'  {n_videos} vidéos → {n_faces} faces extraites')
        if n_failed:
            print(f'  ⚠ {n_failed} vidéos sans visage détecté')

    print('\nTerminé.')


if __name__ == '__main__':
    main()
