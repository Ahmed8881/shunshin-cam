import numpy as np

from shunshin.plate import CleanPlate
from shunshin.vanish import Vanisher

H, W = 120, 160
ROOM = np.full((H, W, 3), 90, np.uint8)


def scene():
    frame = ROOM.copy()
    person = np.zeros((H, W), np.float32)
    person[30:, 60:100] = 1.0
    frame[person > 0] = 230  # you, much brighter than the wall
    return frame, person


def setup():
    plate = CleanPlate((H, W), guard_px=3)
    plate.update(ROOM, np.zeros((H, W), np.float32))  # saw the empty room once
    return plate, Vanisher((H, W))


def test_untouched_until_triggered():
    plate, v = setup()
    frame, person = scene()
    assert np.array_equal(v.compose(frame, person, plate, now=5.0), frame)


def test_gone_after_vanish_shows_only_the_room():
    plate, v = setup()
    frame, person = scene()
    v.toggle(now=0.0)
    out = v.compose(frame, person, plate, now=1.0)
    assert np.abs(out.astype(int) - ROOM).max() <= 2


def test_swap_waits_for_the_smoke():
    plate, v = setup()
    frame, person = scene()
    v.toggle(now=0.0)
    assert np.array_equal(v.compose(frame, person, plate, now=0.02), frame)


def test_reappear_brings_you_back():
    plate, v = setup()
    frame, person = scene()
    v.toggle(now=0.0)
    v.toggle(now=2.0)
    assert np.array_equal(v.compose(frame, person, plate, now=3.0), frame)


def test_moving_while_gone_stays_hidden():
    plate, v = setup()
    v.toggle(now=0.0)
    for shift in range(0, 40, 5):  # walk sideways while invisible
        frame, person = scene()
        frame, person = np.roll(frame, shift, 1), np.roll(person, shift, 1)
        out = v.compose(frame, person, plate, now=1.0 + shift)
        assert np.abs(out.astype(int) - ROOM).max() <= 2
