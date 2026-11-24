"""The clean plate: a live-updated picture of the room without you in it.

This is what makes the vanish believable. Every frame, every pixel that is clearly
*not* you is folded into the plate, so it:

* fills in the room behind you as you move, without asking you to step out of frame,
* follows slow lighting changes (sunset, a lamp turning on) instead of freezing them,
* corrects for webcam auto-exposure on the parts it has not seen in a while,
* notices when the camera gets bumped and relearns instead of showing a stale room,
* falls back to inpainting for any spot it has never seen.
"""

import cv2
import numpy as np

BORDER = 0.12  # outer frame strip used to measure exposure; usually wall, not person


def _luma(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR weights


class CleanPlate:
    def __init__(self, shape, learn_rate: float = 0.08, guard_px: int = 25,
                 moved_threshold: float = 28.0, moved_frames: int = 12):
        h, w = shape[:2]
        self.learn_rate = learn_rate
        self.moved_threshold = moved_threshold
        self.moved_frames = moved_frames
        self.plate = np.zeros((h, w, 3), np.float32)
        self.known = np.zeros((h, w), bool)
        self.stamp = np.ones((h, w), np.float32)  # exposure level when each pixel was last seen
        self.exposure = None
        self._guard = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard_px + 1,) * 2)
        self._border = np.ones((h, w), bool)
        bh, bw = int(h * BORDER), int(w * BORDER)
        self._border[bh:h - bh, bw:w - bw] = False
        self._mismatch_streak = 0
        self._fill_cache = None

    @property
    def coverage(self) -> float:
        return float(self.known.mean())

    def reset(self):
        self.known[:] = False
        self._fill_cache = None

    def update(self, frame: np.ndarray, person: np.ndarray) -> bool:
        """Learn from one frame. `person` is the 0..1 mask. Returns True if the camera moved."""
        f = frame.astype(np.float32)
        # Grow the person mask so hair fringes, hands and the shadow right next to you stay out.
        bg = cv2.dilate((person > 0.15).astype(np.uint8), self._guard) == 0

        moved = self._check_moved(f, bg)
        if moved:
            self.reset()
        elif self._mismatch_streak:
            return False  # possible bump: don't learn from it until it's confirmed or gone

        ref = bg & self._border
        if ref.sum() > 500:
            level = float(_luma(f[ref]).mean())
            if self.exposure is None:
                self.exposure = level
                self.stamp[:] = level
            self.exposure = 0.9 * self.exposure + 0.1 * level

        fresh = bg & ~self.known
        blend = bg & self.known
        self.plate[fresh] = f[fresh]
        self.plate[blend] += self.learn_rate * (f[blend] - self.plate[blend])
        self.known |= bg
        if self.exposure is not None:
            self.stamp[bg] = self.exposure
        if fresh.any():
            self._fill_cache = None
        return moved

    def _check_moved(self, f: np.ndarray, bg: np.ndarray) -> bool:
        # A bump shifts the whole scene at once; slow light changes don't trip this
        # because the learn rate keeps the plate within the threshold.
        overlap = bg & self.known
        if overlap.sum() < 0.2 * bg.size:
            self._mismatch_streak = 0
            return False
        diff = float(np.abs(f[overlap] - self.plate[overlap]).mean())
        self._mismatch_streak = self._mismatch_streak + 1 if diff > self.moved_threshold else 0
        if self._mismatch_streak >= self.moved_frames:
            self._mismatch_streak = 0
            return True
        return False

    def render(self, box=None) -> np.ndarray:
        """The plate as float32 BGR, exposure-matched to now, with never-seen holes inpainted.

        `box` = (y0, y1, x0, x1) renders just that region, which is all the compositor needs.
        """
        if box is None:
            box = (0, self.known.shape[0], 0, self.known.shape[1])
        sl = np.s_[box[0]:box[1], box[2]:box[3]]
        out = self.plate[sl]
        if self.exposure is not None:
            out = out * np.clip(self.exposure / np.maximum(self.stamp[sl], 1e-3), 0.6, 1.6)[..., None]
        known = self.known[sl]
        if not known.all():
            out = np.where(known[..., None], out, self._holes()[sl])
        return out

    def _holes(self) -> np.ndarray:
        # shortcut: inpaint at quarter res and cache until new pixels arrive; smeary on big
        # holes, which is why the HUD shows coverage and nudges you to lean aside once.
        if self._fill_cache is None:
            h, w = self.known.shape
            small = cv2.resize(np.clip(self.plate, 0, 255).astype(np.uint8), (w // 4, h // 4),
                               interpolation=cv2.INTER_AREA)
            holes = cv2.resize((~self.known).astype(np.uint8), (w // 4, h // 4),
                               interpolation=cv2.INTER_NEAREST)
            filled = cv2.inpaint(small, holes, 5, cv2.INPAINT_TELEA)
            filled = cv2.GaussianBlur(filled, (0, 0), 2)
            self._fill_cache = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        return self._fill_cache
