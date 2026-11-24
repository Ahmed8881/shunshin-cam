from shunshin.trigger import GestureTrigger


def run(trigger, frames, fps=30):
    """Feeds a list of booleans at `fps` and returns the frame indexes that fired."""
    return [i for i, seen in enumerate(frames) if trigger.update(seen, i / fps)]


def test_brief_flicker_does_not_fire():
    assert run(GestureTrigger(), [True] * 5 + [False] * 30) == []


def test_holding_fires_exactly_once():
    assert len(run(GestureTrigger(), [True] * 120)) == 1


def test_must_release_before_firing_again():
    frames = [True] * 30 + [False] * 30 + [True] * 30
    assert len(run(GestureTrigger(cooldown_s=0), frames)) == 2


def test_short_dropout_does_not_rearm():
    frames = [True] * 30 + [False] * 3 + [True] * 30
    assert len(run(GestureTrigger(cooldown_s=0), frames)) == 1


def test_cooldown_blocks_rapid_retrigger():
    frames = [True] * 12 + [False] * 13 + [True] * 12
    assert len(run(GestureTrigger(cooldown_s=5), frames)) == 1
