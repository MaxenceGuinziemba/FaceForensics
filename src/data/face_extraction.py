import os
import cv2
import numpy as np
from PIL import Image

_net = None

PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def _get_model_dir():
    d = os.path.join(os.path.dirname(__file__), ".face_models")
    os.makedirs(d, exist_ok=True)
    return d


def _download(url, dest):
    if os.path.isfile(dest):
        return
    import urllib.request
    print(f"Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)


def get_detector():
    global _net
    if _net is None:
        model_dir = _get_model_dir()
        proto = os.path.join(model_dir, "deploy.prototxt")
        model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        _download(PROTO_URL, proto)
        _download(MODEL_URL, model)
        _net = cv2.dnn.readNetFromCaffe(proto, model)
        try:
            _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass
    return _net


def extract_face(frame, margin=0.3, min_confidence=0.7):
    net = get_detector()
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    best_face = None
    best_area = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue

        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_face = (x1, y1, x2, y2)

    if best_face is None:
        return None

    x1, y1, x2, y2 = best_face
    face_w, face_h = x2 - x1, y2 - y1
    margin_w = int(face_w * margin)
    margin_h = int(face_h * margin)

    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
