"""Procedural anime smoke: a dense "poof" that hides the swap, then lingers and drifts away."""

import math

import cv2
import numpy as np

SPRITE = 128
LAYER_W = 240  # smoke is soft, so it's drawn on a small layer and upscaled once, at any camera res


def _noise(rng, size, octaves=4):
    """Fractal value noise in 0..1: the billowy texture that makes a blob read as smoke."""
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = 4 * 2 ** o
        grid = rng.random((cells, cells)).astype(np.float32)
        out += amp * cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def make_sprites(count=8, seed=7):
    """Pre-renders puff sprites as (alpha, shade) pairs; shade darkens the underside for volume."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:SPRITE, 0:SPRITE].astype(np.float32)
    c = (SPRITE - 1) / 2
    sprites = []
    for _ in range(count):
        n = _noise(rng, SPRITE)
        # Wobble the radius with noise so puffs have lumpy, cauliflower edges, not circles.
        r = np.hypot(x - c, y - c) / c + 0.45 * (n - 0.5)
        # Solid body with a short soft edge: anime smoke is opaque, not a fog.
        edge = np.clip((0.92 - r) / 0.22, 0, 1)
        alpha = edge * edge * (3 - 2 * edge) * (0.88 + 0.12 * n)
        # Lit from the upper left, with darker lumps where the noise dips, for a cel-shaded look.
        light = 1 - ((x + y) / (2 * SPRITE)) ** 1.5
        shade = 0.68 + 0.22 * light + 0.18 * (n - 0.5)
        sprites.append((alpha.astype(np.float32), np.clip(shade, 0.5, 1.0).astype(np.float32)))
    return sprites


class SmokeSystem:
    def __init__(self, shape, color=(236, 236, 240), seed=None):
        h, w = shape[:2]
        self.size = (w, h)
        self.k = w / LAYER_W
        self.lh, self.lw = round(h / self.k), LAYER_W
        self.color = np.array(color, np.float32)
        self.rng = np.random.default_rng(seed)
        self.sprites = make_sprites()
        self._scaled = {}  # (sprite, flip, size) -> resized (alpha, premultiplied color)
        self.p = []  # each particle: dict(x, y, vx, vy, size, grow, life, age, sprite)

    @property
    def active(self) -> bool:
        return bool(self.p)

    def burst(self, person: np.ndarray, count: int = 40):
        """Spawn a poof covering whoever is in `person` (0..1 mask, any resolution)."""
        w, h = self.size
        k = w / person.shape[1]
        ys, xs = np.nonzero(person > 0.5)
        if len(xs) < 50 / k ** 2:  # nobody visible: puff in the middle of the frame
            xs, ys = np.array([w / 2]), np.array([h / 2])
            scale = 0.35 * min(w, h)
        else:
            scale = math.sqrt(len(xs)) * k
            xs, ys = xs * k, ys * k
        cx, cy = xs.mean(), ys.mean()
        r = self.rng
        for i in range(count):
            core = i < count // 4  # big puffs right on the body hide the swap
            k = r.integers(len(xs))
            x, y = xs[k], ys[k]
            if core:  # pull toward the body's middle line so head and torso are both covered
                x = 0.5 * (x + cx)
            x += r.normal(0, scale * 0.05)
            y += r.normal(0, scale * 0.05)
            ang = math.atan2(y - cy, x - cx) + r.normal(0, 0.5)
            speed = r.uniform(0.1, 0.5) * scale * (0.5 if core else 1.0)
            self.p.append(dict(
                x=x, y=y,
                vx=math.cos(ang) * speed,
                vy=math.sin(ang) * speed - r.uniform(0.05, 0.2) * scale,
                size=scale * (r.uniform(0.6, 0.9) if core else r.uniform(0.25, 0.5)),
                grow=r.uniform(0.6, 1.3),
                life=r.uniform(1.1, 1.6) if core else r.uniform(1.8, 3.4),
                age=0.0,
                sprite=int(r.integers(len(self.sprites))),
                flip=bool(r.integers(2)),
            ))

    def step(self, dt: float):
        for q in self.p:
            q["age"] += dt
            drag = math.exp(-2.6 * dt)
            q["vx"] *= drag
            q["vy"] = q["vy"] * drag - 18 * dt  # smoke is warm: it keeps rising after the burst
            # Lazy side-to-side curl so lingering wisps don't move in straight lines.
            q["vx"] += 12 * math.sin(q["age"] * 2.3 + q["sprite"]) * dt
            q["x"] += q["vx"] * dt
            q["y"] += q["vy"] * dt
        self.p = [q for q in self.p if q["age"] < q["life"]]

    def _sprite(self, idx, flip, size):
        key = (idx, flip, size)
        if key not in self._scaled:
            a, shade = self.sprites[idx]
            if flip:
                a, shade = a[:, ::-1], shade[:, ::-1]
            a = cv2.resize(a, (size, size))
            color = (a * cv2.resize(shade, (size, size)))[..., None] * self.color
            if len(self._scaled) > 4000:  # sizes drift over a session; don't grow forever
                self._scaled.clear()
            self._scaled[key] = (a, color)
        return self._scaled[key]

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Returns a uint8 BGR copy of `frame` with the smoke composited on top."""
        out = frame.copy()
        if not self.p:
            return out
        color = np.zeros((self.lh, self.lw, 3), np.float32)  # premultiplied
        alpha = np.zeros((self.lh, self.lw), np.float32)
        for q in self.p:
            t = q["age"] / q["life"]
            fade = min(1.0, max(0.0, (t - 0.3) / 0.7))  # hold dense while the swap happens, then thin out
            opacity = min(1.0, q["age"] / 0.05) * (1 - fade * fade * (3 - 2 * fade))
            if opacity < 0.01:
                continue
            ease = 1 - (1 - min(1.0, t * 3)) ** 3  # fast "pop" outward, then slow swell
            size = int(q["size"] * (0.45 + 0.55 * ease + q["grow"] * t) / self.k)
            if size < 4:
                continue
            a, s = self._sprite(q["sprite"], q["flip"], size)
            a = a * opacity
            x0 = int(q["x"] / self.k) - size // 2
            y0 = int(q["y"] / self.k) - size // 2
            # Clip the sprite to the layer bounds.
            sx0, sy0 = max(0, -x0), max(0, -y0)
            lx0, ly0 = max(0, x0), max(0, y0)
            lx1, ly1 = min(self.lw, x0 + size), min(self.lh, y0 + size)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            a = a[sy0:sy0 + ly1 - ly0, sx0:sx0 + lx1 - lx0]
            s = s[sy0:sy0 + ly1 - ly0, sx0:sx0 + lx1 - lx0] * opacity
            roi_c = color[ly0:ly1, lx0:lx1]
            roi_a = alpha[ly0:ly1, lx0:lx1]
            roi_c *= (1 - a)[..., None]
            roi_c += s
            roi_a *= 1 - a
            roi_a += a
        # Only touch the part of the full-res frame the smoke actually covers.
        ys, xs = np.nonzero(alpha > 0.003)
        if not len(xs):
            return out
        lx0, lx1, ly0, ly1 = max(xs.min() - 1, 0), xs.max() + 2, max(ys.min() - 1, 0), ys.max() + 2
        w, h = self.size
        x0, x1 = int(lx0 * self.k), min(w, int(lx1 * self.k))
        y0, y1 = int(ly0 * self.k), min(h, int(ly1 * self.k))
        big = (x1 - x0, y1 - y0)
        a = alpha[ly0:ly1, lx0:lx1]
        c = color[ly0:ly1, lx0:lx1] / np.maximum(a, 1e-4)[..., None]  # un-premultiply for blendLinear
        c = cv2.resize(cv2.convertScaleAbs(c), big, interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, big, interpolation=cv2.INTER_LINEAR)
        out[y0:y1, x0:x1] = cv2.blendLinear(out[y0:y1, x0:x1], c, 1 - a, a)
        return out
