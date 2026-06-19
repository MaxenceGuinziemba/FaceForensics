"""
Extraction de visages depuis des frames vidéo.

Utilise dlib pour détecter et extraire les visages de manière centrée.
Résout le problème des frames sans visage ou avec plusieurs visages.
"""
import dlib
import cv2
import numpy as np
from PIL import Image


# Détecteur de visages pré-entraîné (HOG + SVM)
# Lazy loading : instancié à la première utilisation
_detector = None


def get_face_detector():
    """Retourne le détecteur de visages (singleton)."""
    global _detector
    if _detector is None:
        _detector = dlib.get_frontal_face_detector()
    return _detector


def extract_face(frame, margin=0.3, min_face_size=80):
    """
    Détecte et extrait le visage principal d'une frame.

    Args:
        frame: numpy array BGR (OpenCV format)
        margin: marge autour du visage (0.3 = +30% de chaque côté)
        min_face_size: taille minimale du visage en pixels

    Returns:
        PIL Image du visage centré, ou None si pas de visage détecté

    Exemple:
        >>> frame = cv2.imread('video_frame.jpg')
        >>> face = extract_face(frame)
        >>> if face is not None:
        >>>     face.save('extracted_face.jpg')
    """
    detector = get_face_detector()

    # Convertir BGR → RGB pour dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détecter les visages
    # upsample=1 permet de détecter des visages plus petits (mais plus lent)
    faces = detector(rgb, 1)

    if len(faces) == 0:
        return None

    # Filtrer les visages trop petits (probablement du bruit)
    faces = [f for f in faces if f.width() >= min_face_size and f.height() >= min_face_size]

    if len(faces) == 0:
        return None

    # Prendre le visage le plus grand (celui au premier plan)
    face = max(faces, key=lambda rect: rect.width() * rect.height())

    # Extraire les coordonnées avec marge
    h, w = frame.shape[:2]
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()

    # Ajouter la marge (30% de chaque côté par défaut)
    face_w = x2 - x1
    face_h = y2 - y1
    margin_w = int(face_w * margin)
    margin_h = int(face_h * margin)

    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)

    # Extraire le visage
    face_crop = frame[y1:y2, x1:x2]

    # Vérifier que le crop n'est pas vide
    if face_crop.size == 0:
        return None

    # Convertir BGR → RGB → PIL
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)


def extract_face_from_video(video_path, frame_idx=0, max_attempts=10):
    """
    Extrait un visage depuis une vidéo à un index de frame donné.

    Si la frame donnée ne contient pas de visage, essaie jusqu'à max_attempts
    frames aléatoires.

    Args:
        video_path: chemin vers la vidéo
        frame_idx: index de la frame à extraire (0-based)
        max_attempts: nombre max de frames à essayer

    Returns:
        (PIL Image du visage, frame_idx utilisée), ou (None, -1) si échec
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None, -1

    # Essayer la frame demandée d'abord
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx % total_frames)
    ret, frame = cap.read()

    if ret:
        face = extract_face(frame)
        if face is not None:
            cap.release()
            return face, frame_idx

    # Si échec, essayer des frames aléatoires
    import random
    for attempt in range(max_attempts - 1):
        random_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        face = extract_face(frame)
        if face is not None:
            cap.release()
            return face, random_idx

    # Aucun visage trouvé
    cap.release()
    return None, -1
