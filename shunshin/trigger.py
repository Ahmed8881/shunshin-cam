"""Turns a noisy per-frame "is the hand sign visible?" signal into clean, single-fire triggers."""


class GestureTrigger:
    """Fires once when the sign is held for `hold_s` seconds.

    It then stays disarmed until the hand drops the sign for `release_s` seconds, and never
    fires twice within `cooldown_s`. Without this, holding the sign would toggle you
    in and out every frame, and one stray detection would make you vanish mid-sentence.
    """

    def __init__(self, hold_s: float = 0.35, release_s: float = 0.4, cooldown_s: float = 1.5):
        self.hold_s = hold_s
        self.release_s = release_s
        self.cooldown_s = cooldown_s
        self._seen_since = None
        self._gone_since = None
        self._armed = True
        self._last_fire = float("-inf")

    def update(self, sign_visible: bool, now: float) -> bool:
        if sign_visible:
            self._gone_since = None
            if self._seen_since is None:
                self._seen_since = now
            held = now - self._seen_since >= self.hold_s
            if self._armed and held and now - self._last_fire >= self.cooldown_s:
                self._armed = False
                self._last_fire = now
                return True
        else:
            self._seen_since = None
            if self._gone_since is None:
                self._gone_since = now
            if now - self._gone_since >= self.release_s:
                self._armed = True
        return False

    def progress(self, now: float) -> float:
        """0..1 charge for the on-screen hold indicator."""
        if not self._armed or self._seen_since is None:
            return 0.0
        return min(1.0, (now - self._seen_since) / self.hold_s)
