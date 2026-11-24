"""Fetches the MediaPipe model files on first run and caches them locally."""

import urllib.request
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "shunshin-cam"  # per-user: an installed package dir may be read-only

MODELS = {
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}


def model_path(name: str) -> str:
    path = MODEL_DIR / name
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODELS[name], tmp)
        tmp.replace(path)  # rename only after a full download, so a crash never leaves a broken model
    return str(path)
