import numpy as np

from shunshin.smoke import SmokeSystem

H, W = 240, 320


def body():
    m = np.zeros((H, W), np.float32)
    m[60:, 120:200] = 1.0
    m[20:60, 140:180] = 1.0  # head
    return m


def play(seconds, fps=30):
    frame = np.zeros((H, W, 3), np.uint8)
    smoke = SmokeSystem(frame.shape, seed=3)
    smoke.burst(body())
    frames = []
    for _ in range(int(seconds * fps)):
        frames.append(smoke.render(frame))
        smoke.step(1 / fps)
    return smoke, frames


def test_poof_hides_the_body_while_the_swap_happens():
    _, frames = play(0.4)
    covered = frames[6][..., 0][body() > 0]  # ~0.2s in, on a black frame: bright = smoke
    assert (covered > 120).mean() > 0.85


def test_smoke_lingers_then_clears_completely():
    smoke, frames = play(4)
    assert frames[45].max() > 30  # still visible 1.5s later: it's left behind, not instant
    assert not smoke.active
    assert frames[-1].max() < 1


def test_no_person_still_puffs_in_frame():
    frame = np.zeros((H, W, 3), np.uint8)
    smoke = SmokeSystem(frame.shape, seed=3)
    smoke.burst(np.zeros((H, W), np.float32))
    smoke.step(0.1)
    out = smoke.render(frame)
    assert out[H // 2, W // 2].mean() > 100
    assert np.isfinite(out).all()
