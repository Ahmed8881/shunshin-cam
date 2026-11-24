"""MediaPipe wrappers: who is in the frame, and is anyone showing the hand sign."""

import threading
import time

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .models import model_path

RUNNING_MODE = vision.RunningMode.VIDEO


def _image(rgb: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class Segmenter:
    """Per-frame person mask. Runs inline so the cut-out always lines up with the frame."""

    def __init__(self):
        self._seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=model_path("selfie_segmenter.tflite")),
                running_mode=RUNNING_MODE,
                output_confidence_masks=True,
                output_category_mask=False,
            )
        )

    def __call__(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray:
        """Person confidence in 0..1 as float32 HxW."""
        res = self._seg.segment_for_video(_image(rgb), ts_ms)
        return res.confidence_masks[0].numpy_view().reshape(rgb.shape[:2]).astype(np.float32)

    def close(self):
        self._seg.close()


class SignWatcher:
    """Hand-sign recognition on a background thread.

    The recognizer costs 2-4x the segmenter. The trigger already wants the sign held for
    a third of a second, so reading a result one or two frames old is invisible, and
    keeping it off the main loop keeps the video at full frame rate.
    """

    def __init__(self, gestures=("Victory",), min_score: float = 0.6, max_hz: float = 10):
        self.gestures = set(gestures)
        self.period = 1 / max_hz  # plenty for a 0.35s hold, and leaves CPU for the video
        self.min_score = min_score
        self.visible = False
        self._rec = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=model_path("gesture_recognizer.task")),
                running_mode=RUNNING_MODE,
                num_hands=2,
            )
        )
        self._frame = None
        self._ready = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, rgb: np.ndarray):
        with self._ready:
            self._frame = rgb
            self._ready.notify()

    def _loop(self):
        start = time.monotonic()
        last_ts = -1
        while True:
            with self._ready:
                self._ready.wait_for(lambda: self._frame is not None or not self._running)
                if not self._running:
                    return
                rgb, self._frame = self._frame, None
            began = time.monotonic()
            ts = max(last_ts + 1, int((began - start) * 1000))  # VIDEO mode needs rising ts
            last_ts = ts
            gestures = self._rec.recognize_for_video(_image(rgb), ts).gestures
            self.visible = any(
                g[0].category_name in self.gestures and g[0].score >= self.min_score for g in gestures
            )
            time.sleep(max(0.0, self.period - (time.monotonic() - began)))

    def close(self):
        with self._ready:
            self._running = False
            self._ready.notify()
        self._thread.join(timeout=2)
        self._rec.close()
