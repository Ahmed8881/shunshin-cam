import numpy as np

from shunshin.plate import CleanPlate, fill_holes

H, W = 120, 160


def room(level=100):
    """A textured fake room so inpainting and exposure checks have something to work on."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([level + 20 * np.sin(x / 7), level + 20 * np.cos(y / 5), np.full((H, W), level)], -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def person_at(x0, x1):
    m = np.zeros((H, W), np.float32)
    m[30:, x0:x1] = 1.0
    return m


def test_learns_room_behind_you_as_you_move():
    plate = CleanPlate((H, W), guard_px=3)
    for x0 in (20, 60, 100):  # you shift around the frame
        for _ in range(3):
            plate.update(room(), person_at(x0, x0 + 40))
    assert plate.coverage > 0.99
    assert np.abs(plate.render() - room()).mean() < 1.0


def test_never_seen_spot_is_inpainted_not_black():
    plate = CleanPlate((H, W), guard_px=3)
    plate.update(room(), person_at(60, 100))
    assert plate.coverage < 0.9
    hole = plate.render()[60:, 70:90]
    assert hole.mean() > 60  # filled from the surroundings, not left at zero


def test_stale_pixels_follow_auto_exposure():
    plate = CleanPlate((H, W), guard_px=3)
    plate.update(room(100), person_at(0, 0))  # empty room at normal exposure
    for _ in range(60):  # you sit down and the camera brightens
        plate.update(room(140), person_at(60, 100))
    behind_you = plate.render()[60:, 70:90]
    assert abs(behind_you[..., 2].mean() - 140) < 12


def test_camera_bump_triggers_relearn():
    plate = CleanPlate((H, W), guard_px=3, moved_frames=12)
    rng = np.random.default_rng(0)
    scene = np.kron(rng.integers(0, 255, (H // 10, W // 10, 3)), np.ones((10, 10, 1))).astype(np.uint8)
    plate.update(scene, person_at(60, 100))
    bumped = np.roll(scene, 25, axis=1)
    moved = [plate.update(bumped, person_at(60, 100)) for _ in range(13)]
    assert any(moved)


def test_brief_glitch_does_not_wipe_the_plate():
    plate = CleanPlate((H, W), guard_px=3)
    plate.update(room(), person_at(60, 100))
    plate.update(255 - room(), person_at(60, 100))  # one corrupted frame
    plate.update(room(), person_at(60, 100))
    assert np.abs(plate.render()[:, :50] - room()[:, :50]).mean() < 1.0


def test_hole_fill_matches_the_wall_and_never_goes_dark():
    wall = np.full((H, W, 3), 200, np.uint8)
    known = np.ones((H, W), bool)
    known[20:110, 70:90] = False  # long narrow strip that was always behind you
    filled = fill_holes(wall * known[..., None], known)
    assert np.abs(filled.astype(int) - 200).max() <= 3
