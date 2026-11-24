"""MediaPipe wrappers: who is in the frame, and is anyone showing the hand sign."""

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .models import model_path

RUNNING_MODE = vision.RunningMode.VIDEO


class Perception:
    def __init__(self, gesture: str = "Victory", min_score: float = 0.6):
        self.gesture = gesture
        self.min_score = min_score
        self._segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=model_path("selfie_segmenter.tflite")),
                running_mode=RUNNING_MODE,
                output_confidence_masks=True,
                output_category_mask=False,
            )
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=model_path("gesture_recognizer.task")),
                running_mode=RUNNING_MODE,
                num_hands=2,
            )
        )

    def process(self, rgb: np.ndarray, ts_ms: int):
        """Returns (person mask in 0..1 as float32 HxW, whether the sign is visible)."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        seg = self._segmenter.segment_for_video(image, ts_ms)
        mask = seg.confidence_masks[0].numpy_view().reshape(rgb.shape[:2]).astype(np.float32)
        gestures = self._recognizer.recognize_for_video(image, ts_ms).gestures
        sign = any(
            g[0].category_name == self.gesture and g[0].score >= self.min_score for g in gestures
        )
        return mask, sign

    def close(self):
        self._segmenter.close()
        self._recognizer.close()
