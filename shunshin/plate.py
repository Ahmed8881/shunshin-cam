"""The clean plate: a live-updated picture of the room without you in it.

This is what makes the vanish believable. Every frame, every pixel that is clearly
*not* you is folded into the plate, so it:

* fills in the room behind you as you move, without asking you to step out of frame,
* follows slow lighting changes (sunset, a lamp turning on) instead of freezing them,
* corrects for webcam auto-exposure on the parts it has not seen in a while,
* notices when the camera gets bumped and relearns instead of showing a stale room,
* falls back to a smooth fill for any spot it has never seen,
* measures your camera's grain so the patch where you were isn't suspiciously clean.
"""

import cv2
import numpy as np

BORDER = 0.12  # outer frame strip used to measure exposure; usually wall, not person
MASK_SCALE = 4  # masks come from a 256px model, so mask maths at 1/4 res loses nothing
REFILL_EVERY = 5  # updates between hole re-fills (~0.5s at 30fps with learning every 3rd frame)


def _luma(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.114, 0.587, 0.299], dtype=np.float32)  # BGR weights


class CleanPlate:
    def __init__(self, shape, learn_rate: float = 0.3, guard_px: int = 25,
                 moved_threshold: float = 28.0, moved_frames: int = 12):
        h, w = shape[:2]
        self.learn_rate = learn_rate
        self.moved_threshold = moved_threshold
        self.moved_frames = moved_frames
        self.plate = np.zeros((h, w, 3), np.float32)
        self.known = np.zeros((h, w), bool)
        self.view = np.zeros((h, w, 3), np.uint8)  # what render() hands out
        self._dirty = True  # view is rebuilt only when someone asks for it, i.e. while you're gone
        self.exposure = None
        self.noise = 0.0  # camera grain (std, 0-255); the averaged plate is cleaner than live video
        g = max(1, round(guard_px / MASK_SCALE))
        self._guard = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1,) * 2)
        self._small = (max(1, w // MASK_SCALE), max(1, h // MASK_SCALE))
        # Exposure level when each (1/4 res) pixel was last seen; light changes are smooth anyway.
        self._stamp = np.ones(self._small[::-1], np.float32)
        self._border = np.ones((h, w), bool)
        bh, bw = int(h * BORDER), int(w * BORDER)
        self._border[bh:h - bh, bw:w - bw] = False
        self._mismatch_streak = 0
        self._fill_cache = None
        self._fill_stale = True
        self._fill_age = 0

    @property
    def coverage(self) -> float:
        return float(self.known.mean())

    def reset(self):
        self.known[:] = False
        self._fill_cache = None
        self._fill_stale = True
        self._dirty = True

    def update(self, frame: np.ndarray, person: np.ndarray) -> bool:
        """Learn from one frame. `person` is the 0..1 mask. Returns True if the camera moved."""
        h, w = self.known.shape
        # Grow the person mask so hair fringes, hands and the shadow right next to you stay out.
        small = (cv2.resize(person, self._small, interpolation=cv2.INTER_AREA) > 0.15).astype(np.uint8)
        bg_small = 1 - cv2.dilate(small, self._guard)
        bg8 = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_NEAREST)
        bg = bg8.astype(bool)

        moved = self._check_moved(frame, bg)
        if moved:
            self.reset()
        elif self._mismatch_streak:
            return False  # possible bump: don't learn from it until it's confirmed or gone

        ref = (bg & self._border)[::MASK_SCALE, ::MASK_SCALE]
        if ref.sum() > 30:
            level = float(_luma(frame[::MASK_SCALE, ::MASK_SCALE][ref].astype(np.float32)).mean())
            if self.exposure is None:
                self.exposure = level
                self._stamp[:] = level
            self.exposure = 0.5 * self.exposure + 0.5 * level  # mean of many px: barely noisy

        fresh = bg & ~self.known
        cv2.accumulateWeighted(frame, self.plate, self.learn_rate, mask=bg8 & self.known.view(np.uint8))
        if fresh.any():
            np.copyto(self.plate, frame, where=fresh[..., None], casting="unsafe")
            self._fill_stale = True
        self.known |= bg
        if self.exposure is not None:
            self._stamp[bg_small.astype(bool)] = self.exposure
        self._dirty = True
        return moved

    def _check_moved(self, frame: np.ndarray, bg: np.ndarray) -> bool:
        # A bump shifts the whole scene at once; slow light changes don't trip this
        # because the learn rate keeps the plate within the threshold.
        s = np.s_[::MASK_SCALE, ::MASK_SCALE]  # a sparse grid is plenty to spot a whole-scene shift
        overlap = bg[s] & self.known[s]
        if overlap.sum() < 0.2 * overlap.size:
            self._mismatch_streak = 0
            return False
        residual = frame[s][overlap].astype(np.float32) - self.plate[s][overlap]
        diff = float(np.abs(residual).mean())
        if diff <= self.moved_threshold:
            # Robust std of the residual; 0.84 converts it to the grain the plate is missing
            # (with learn_rate 0.3 the plate keeps ~40% of the sensor noise itself).
            est = 0.84 * 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
            self.noise = est if self.noise == 0 else 0.8 * self.noise + 0.2 * est
        self._mismatch_streak = self._mismatch_streak + 1 if diff > self.moved_threshold else 0
        if self._mismatch_streak >= self.moved_frames:
            self._mismatch_streak = 0
            return True
        return False

    def _refresh(self):
        """Rebuilds the uint8 view: exposure-matched to now, never-seen holes inpainted."""
        h, w = self.known.shape
        gain = np.ones_like(self._stamp)
        if self.exposure is not None:
            gain = np.clip(self.exposure / np.maximum(self._stamp, 1e-3), 0.6, 1.6)
        if np.abs(gain - 1).max() > 0.01:
            # NEAREST keeps gain blocks aligned with the 4px blocks the plate was learned in.
            gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_NEAREST)
            self.view = cv2.convertScaleAbs(cv2.multiply(self.plate, cv2.merge([gain] * 3)))
        else:
            self.view = cv2.convertScaleAbs(self.plate)
        if not self.known.all():
            cv2.copyTo(self._holes(self.view), (~self.known).view(np.uint8), self.view)

    def render(self, box=None) -> np.ndarray:
        """The plate as uint8 BGR. `box` = (y0, y1, x0, x1) returns just that region."""
        if self._dirty:
            self._refresh()
            self._dirty = False
        if box is None:
            return self.view
        return self.view[box[0]:box[1], box[2]:box[3]]

    def _holes(self, view: np.ndarray) -> np.ndarray:
        # shortcut: a smooth push-pull fill from the surrounding wall, redone at most every
        # REFILL_EVERY updates; it can't invent texture, which is why the HUD shows coverage
        # and nudges you to lean aside once.
        self._fill_age += 1
        if self._fill_cache is None or (self._fill_stale and self._fill_age >= REFILL_EVERY):
            self._fill_stale, self._fill_age = False, 0
            h, w = self.known.shape
            self._fill_cache = cv2.resize(fill_holes(view, self.known), (w, h),
                                          interpolation=cv2.INTER_CUBIC)
        return self._fill_cache


def fill_holes(img: np.ndarray, known: np.ndarray, scale: int = 8) -> np.ndarray:
    """Fills unknown pixels with a smooth average of nearby known ones, at 1/`scale` res.

    Push-pull: shrink colour*weight and weight together until every cell has data, then
    grow back, keeping real data wherever it exists. Unknown pixels never leak in, which
    is what made plain inpainting smear dark streaks.
    """
    h, w = known.shape
    size = (max(1, w // scale), max(1, h // scale))
    wt = cv2.resize(known.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    col = cv2.resize(img.astype(np.float32) * known[..., None], size, interpolation=cv2.INTER_AREA)
    pyramid = [(col, wt)]
    while min(wt.shape) > 2:
        col, wt = cv2.pyrDown(col), cv2.pyrDown(wt)
        pyramid.append((col, wt))
    col, wt = pyramid.pop()
    out = col / np.maximum(wt, 1e-6)[..., None]
    for col, wt in reversed(pyramid):
        up = cv2.pyrUp(out, dstsize=wt.shape[::-1])
        a = np.minimum(wt * 4, 1)[..., None]  # trust local data once a cell is ~1/4 known
        out = a * col / np.maximum(wt, 1e-6)[..., None] + (1 - a) * up
    return cv2.convertScaleAbs(out)
