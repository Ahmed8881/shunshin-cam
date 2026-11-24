"""Webcam capture on its own thread, so decoding overlaps with the effect instead of stalling it."""

import sys
import threading

import cv2


class Camera:
    def __init__(self, index: int, width: int, height: int, fps: int):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        # Most webcams only reach 720p30 over MJPG; the default raw format drops to ~10 fps.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Could not read from camera {index}. Is another app (Zoom itself?) "
                               "using it, or is the privacy shutter closed?")
        self.first = frame
        self._frame, self._seq, self._taken = frame, 1, 0
        self._failed = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            with self._cond:
                if ok:
                    self._frame, self._seq, self._failed = frame, self._seq + 1, 0
                else:
                    self._failed += 1
                self._cond.notify()

    def read(self, timeout: float = 2.0):
        """Newest frame not handed out yet (older ones are dropped), or None if the feed died."""
        with self._cond:
            fresh = self._cond.wait_for(lambda: self._seq != self._taken or self._failed > 30, timeout)
            if not fresh or self._failed > 30:
                return None
            self._taken = self._seq
            return self._frame

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()
