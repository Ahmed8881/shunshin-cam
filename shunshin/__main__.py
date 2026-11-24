"""Shunshin Cam: flash a hand sign on your video call and vanish in a puff of smoke.

    python -m shunshin                 # webcam -> virtual camera + preview window
    python -m shunshin --gesture Thumb_Up --width 640 --height 480
"""

import argparse
import sys
import time

import cv2
import numpy as np

from .perception import Segmenter, SignWatcher
from .plate import CleanPlate
from .smoke import SmokeSystem
from .trigger import GestureTrigger
from .vanish import Vanisher

GESTURES = ["Victory", "Thumb_Up", "Thumb_Down", "Open_Palm", "Closed_Fist", "Pointing_Up", "ILoveYou"]
WINDOW = "Shunshin Cam (preview: only you see the overlay)"
PLATE_EVERY = 3  # the room changes slowly; learning every 3rd frame saves CPU for the video
MODEL_SCALE = 2  # both models shrink input to ~256px internally, so half res costs nothing


def parse_args():
    p = argparse.ArgumentParser(prog="shunshin", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--gesture", choices=GESTURES, default="Victory",
                   help="hand sign that toggles the vanish (default Victory = two fingers)")
    p.add_argument("--hold", type=float, default=0.35, help="seconds to hold the sign (default 0.35)")
    p.add_argument("--no-virtualcam", action="store_true", help="preview only, don't send to Zoom")
    p.add_argument("--no-preview", action="store_true", help="no window; stop with Ctrl+C")
    return p.parse_args()


def open_camera(index, width, height, fps):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    # Most webcams only reach 720p30 over MJPG; the default raw format drops to ~10 fps.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # if we fall behind, drop old frames instead of lagging
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"Could not read from camera {index}. Is another app (Zoom itself?) using it, "
                 "or is the privacy shutter closed?")
    if frame.mean() < 3:
        print("Warning: the camera image is almost black. Check the privacy shutter or lighting.")
    return cap, frame


def open_virtualcam(width, height, fps):
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR)
    except Exception as e:  # missing driver is the common case; preview still works
        print(f"Virtual camera unavailable ({e}).\n"
              "  Install OBS Studio (https://obsproject.com), open it once, click 'Start Virtual "
              "Camera' then 'Stop', close OBS, and run this again.\n"
              "  Continuing with the preview window only.")
        return None
    print(f"Virtual camera ready: pick '{cam.device}' as your camera in Zoom/Meet/Teams.")
    return cam


def draw_hud(img, vanisher, plate, trigger, now, fps, gesture):
    h, w = img.shape[:2]
    state = "GONE" if vanisher.gone else "VISIBLE"
    color = (80, 200, 255) if vanisher.gone else (120, 230, 120)
    cv2.rectangle(img, (0, 0), (w, 34), (20, 20, 20), -1)
    cv2.putText(img, f"{state}   sign: {gesture}   background learned: {plate.coverage:.0%}   "
                f"{fps:.0f} fps", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    charge = trigger.progress(now)
    if charge > 0:
        cv2.rectangle(img, (0, 34), (int(w * charge), 40), (80, 200, 255), -1)
    if plate.coverage < 0.92:
        msg = "Tip: lean left and right once so I can see the wall behind you"
        cv2.putText(img, msg, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, msg, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "[V] vanish  [B] relearn background  [Q] quit", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    cap, frame = open_camera(args.camera, args.width, args.height, args.fps)
    h, w = frame.shape[:2]
    print(f"Camera {args.camera}: {w}x{h}")

    segment = Segmenter()
    signs = SignWatcher(args.gesture)
    plate = CleanPlate(frame.shape)
    smoke = SmokeSystem(frame.shape)
    vanisher = Vanisher(frame.shape)
    trigger = GestureTrigger(hold_s=args.hold)
    cam = None if args.no_virtualcam else open_virtualcam(w, h, args.fps)

    start = last = time.monotonic()
    last_ts, fps, n, bad = -1, 0.0, 0, 0
    print(f"Running. Hold the {args.gesture} sign to vanish, hold it again to come back.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                bad += 1
                if bad > 30:
                    sys.exit("Lost the camera feed.")
                continue
            bad = 0
            now = time.monotonic()
            dt, last = now - last, now
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3)

            small = cv2.resize(frame, (w // MODEL_SCALE, h // MODEL_SCALE), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            last_ts = max(last_ts + 1, int((now - start) * 1000))
            person = segment(rgb, last_ts)
            signs.submit(rgb)
            if n % PLATE_EVERY == 0 or n < 10:
                if plate.update(frame, person) and now - start > 3:  # startup = auto-exposure settling
                    print("Camera moved: relearning the background.")
            n += 1

            key = -1 if args.no_preview else cv2.waitKey(1) & 0xFF
            if trigger.update(signs.visible, now) or key in (ord("v"), ord("V")):
                vanisher.toggle(now)
                smoke.burst(person)
            if key in (ord("b"), ord("B")):
                plate.reset()
            if key in (ord("q"), ord("Q"), 27):
                break

            out = vanisher.compose(frame, person, plate, now)
            smoke.step(dt)
            out = smoke.render(out)

            if cam is not None:
                cam.send(out)
            if not args.no_preview:
                view = cv2.flip(out, 1)  # mirrored like a mirror, the way Zoom shows your self-view
                draw_hud(view, vanisher, plate, trigger, now, fps, args.gesture)
                cv2.imshow(WINDOW, view)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        signs.close()
        segment.close()
        if cam is not None:
            cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
