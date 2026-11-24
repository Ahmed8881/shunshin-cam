"""Swaps you for the clean plate, timed to happen while the smoke is at its thickest."""

import cv2
import numpy as np

# Seconds after the trigger. The poof needs ~50ms to cover you, so the swap waits for it.
VANISH_AT = (0.06, 0.22)
APPEAR_AT = (0.12, 0.30)


class Vanisher:
    def __init__(self, shape):
        h, w = shape[:2]
        self.gone = False
        self._t0 = float("-inf")
        self._prev = np.zeros((h, w), np.uint8)
        # Generous margins: the segmenter trails fast motion by a frame and is soft on hair,
        # so a tight cut leaves a ghostly outline of you behind. These scale with resolution.
        grow = max(3, round(w * 0.018))
        self._grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        self._feather = max(1.0, w * 0.008)

    def toggle(self, now: float):
        self.gone = not self.gone
        self._t0 = now

    def amount(self, now: float) -> float:
        """0 = fully you, 1 = fully gone."""
        a, b = VANISH_AT if self.gone else APPEAR_AT
        t = float(np.clip((now - self._t0 - a) / (b - a), 0, 1))
        t = t * t * (3 - 2 * t)
        return t if self.gone else 1 - t

    def compose(self, frame: np.ndarray, person: np.ndarray, plate, now: float) -> np.ndarray:
        cur = (person > 0.3).astype(np.uint8)
        # Union with last frame's mask to cover segmentation lag when you move.
        hard = cv2.dilate(cv2.max(cur, self._prev), self._grow)
        self._prev = cur
        k = self.amount(now)
        if k <= 0.001:
            return frame
        ys, xs = np.nonzero(hard)
        if not len(xs):
            return frame
        h, w = hard.shape
        pad = int(3 * self._feather) + 1
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
        alpha = cv2.GaussianBlur(hard[y0:y1, x0:x1].astype(np.float32), (0, 0), self._feather)
        alpha = (alpha * k)[..., None]
        out = frame.copy()
        roi = out[y0:y1, x0:x1].astype(np.float32)
        roi += alpha * (plate.render((y0, y1, x0, x1)) - roi)
        out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        return out
