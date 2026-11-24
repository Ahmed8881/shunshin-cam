"""Swaps you for the clean plate, timed to happen while the smoke is at its thickest."""

import cv2
import numpy as np

from .plate import MASK_SCALE

# Seconds after the trigger. The poof needs ~50ms to cover you, so the swap waits for it.
VANISH_AT = (0.06, 0.22)
APPEAR_AT = (0.12, 0.30)


class Vanisher:
    def __init__(self, shape):
        h, w = shape[:2]
        self.gone = False
        self._t0 = float("-inf")
        self._size = (w, h)
        self._small = (max(1, w // MASK_SCALE), max(1, h // MASK_SCALE))
        self._prev = np.zeros(self._small[::-1], np.uint8)
        # Generous margins: the segmenter trails fast motion by a frame and is soft on hair,
        # so a tight cut leaves a ghostly outline of you behind. These scale with resolution.
        grow = max(1, round(w * 0.018 / MASK_SCALE))
        self._grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        self._feather = max(0.6, w * 0.008 / MASK_SCALE)

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
        cur = (cv2.resize(person, self._small, interpolation=cv2.INTER_AREA) > 0.3).astype(np.uint8)
        # Union with last frame's mask to cover segmentation lag when you move.
        hard = cv2.dilate(cv2.max(cur, self._prev), self._grow)
        self._prev = cur
        k = self.amount(now)
        if k <= 0.001:
            return frame
        ys, xs = np.nonzero(hard)
        if not len(xs):
            return frame
        sh, sw = hard.shape
        pad = int(3 * self._feather) + 1
        sy0, sy1 = max(0, ys.min() - pad), min(sh, ys.max() + pad + 1)
        sx0, sx1 = max(0, xs.min() - pad), min(sw, xs.max() + pad + 1)
        alpha = cv2.GaussianBlur(hard[sy0:sy1, sx0:sx1].astype(np.float32), (0, 0), self._feather)
        alpha = np.minimum(alpha * 2, 1)  # solid over the whole cut, soft only outside it
        w, h = self._size
        fx, fy = w / sw, h / sh
        y0, y1 = int(sy0 * fy), min(h, int(sy1 * fy))
        x0, x1 = int(sx0 * fx), min(w, int(sx1 * fx))
        alpha = cv2.resize(alpha, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        alpha *= k
        fill = plate.render((y0, y1, x0, x1))
        if plate.noise > 0.5:
            # Fresh grain every frame, matched to the camera, so the empty spot "breathes" like video.
            grain = np.empty(fill.shape, np.int16)
            cv2.randn(grain, (0, 0, 0), (plate.noise,) * 3)  # per channel; a bare scalar hits only B
            fill = cv2.add(fill, grain, dtype=cv2.CV_8U)
        out = frame.copy()
        out[y0:y1, x0:x1] = cv2.blendLinear(frame[y0:y1, x0:x1], fill, 1 - alpha, alpha)
        return out


def replace_background(frame: np.ndarray, person: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Puts `background` behind you, using the 0..1 person mask at any resolution."""
    h, w = frame.shape[:2]
    alpha = cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip((alpha - 0.3) * 2.5, 0, 1)  # firm inside, soft 0.3-0.7 edge
    return cv2.blendLinear(frame, background, alpha, 1 - alpha)
