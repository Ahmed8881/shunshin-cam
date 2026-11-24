"""Shunshin Cam: flash a hand sign on your video call and vanish in a puff of smoke.

    python -m shunshin                 # webcam -> virtual camera + preview window
    python -m shunshin --gesture Thumb_Up --width 640 --height 480
"""

import argparse
import shutil
import subprocess
import sys
import time

import cv2

from .camera import Camera
from .perception import Segmenter, SignWatcher
from .plate import CleanPlate
from .smoke import SmokeSystem
from .trigger import GestureTrigger
from .vanish import Vanisher

GESTURES = ["Victory", "Thumb_Up", "Thumb_Down", "Open_Palm", "Closed_Fist", "Pointing_Up", "ILoveYou"]
WINDOW = "Shunshin Cam (preview: only you see the overlay)"
PLATE_EVERY = 3  # the room changes slowly; learning every 3rd frame saves CPU for the video
# Both models shrink their input to ~256px internally, so feeding them less costs nothing.
# Hands get half res so small, distant hands still have enough pixels for landmarks.
SEG_SCALE, HAND_SCALE = 4, 2


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
    try:
        cam = Camera(index, width, height, fps)
    except RuntimeError as e:
        sys.exit(str(e))
    return cam, cam.first


CALL_SETUP = """
  Zoom:         Settings > Video > Camera > "{device}"
                Settings > Background & effects > None   (Zoom's backgrounds erase the smoke)
  Google Meet:  before joining: More options > Settings > Video > Camera > "{device}"
                Visual effects > No effect   (same reason)
                Not listed? Fully quit and reopen the browser; it only scans cameras at launch.
  Teams/Discord/OBS: pick "{device}" as the camera.
"""

DRIVER_HELP = {
    "win32": "Install OBS Studio (https://obsproject.com) and run this again.",
    "darwin": "Install OBS Studio 30+ (https://obsproject.com or `brew install --cask obs`), "
              "start its Virtual Camera once, then run this again.",
}
LINUX_HELP = ("Load the loopback driver: sudo apt install v4l2loopback-dkms && "
              "sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label=ShunshinCam")


def _create_virtualcam(width, height, fps):
    import pyvirtualcam
    return pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR)


def _offer_obs_install() -> bool:
    """On Windows, OBS ships the virtual camera driver; offer a one-key winget install."""
    if sys.platform != "win32" or not sys.stdin.isatty() or not shutil.which("winget"):
        return False
    try:
        answer = input("The virtual camera driver comes with OBS Studio. Install it now with winget? [Y/n] ")
    except EOFError:  # some terminals report a closed stdin as a tty
        return False
    if answer.strip().lower() not in ("", "y", "yes"):
        return False
    cmd = ["winget", "install", "--id", "OBSProject.OBSStudio", "-e",
           "--accept-package-agreements", "--accept-source-agreements"]
    return subprocess.run(cmd).returncode == 0


def open_virtualcam(width, height, fps):
    try:
        cam = _create_virtualcam(width, height, fps)
    except Exception as e:  # missing driver is the common case; preview still works
        print(f"Virtual camera unavailable: {e}")
        cam = None
        if _offer_obs_install():
            try:
                cam = _create_virtualcam(width, height, fps)
            except Exception as e2:
                print(f"Still unavailable after installing OBS ({e2}). Try restarting your computer.")
        if cam is None:
            print("  " + DRIVER_HELP.get(sys.platform, LINUX_HELP))
            print("  Continuing with the preview window only.")
            return None
    print("Virtual camera ready. To use it in a call:" + CALL_SETUP.format(device=cam.device))
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
    last_ts, fps, n = -1, 0.0, 0
    print(f"Running. Hold the {args.gesture} sign to vanish, hold it again to come back.")
    try:
        while True:
            frame = cap.read()
            if frame is None:
                sys.exit("Lost the camera feed.")
            now = time.monotonic()
            dt, last = now - last, now
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3)

            half = cv2.resize(frame, (w // HAND_SCALE, h // HAND_SCALE), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(half, cv2.COLOR_BGR2RGB)
            quarter = cv2.resize(rgb, (w // SEG_SCALE, h // SEG_SCALE), interpolation=cv2.INTER_AREA)
            last_ts = max(last_ts + 1, int((now - start) * 1000))
            person = segment(quarter, last_ts)
            signs.submit(rgb)
            if n % PLATE_EVERY == 0 or n < 10:
                if plate.update(frame, person) and now - start > 3:  # startup = auto-exposure settling
                    print("Camera moved: relearning the background.")
            n += 1
            if n == 60 and frame.mean() < 3:  # first frames are always dark while the sensor wakes up
                print("Your webcam is sending a black picture. Usually another app (Zoom, Meet, Teams) "
                      "has it open: switch that app to 'OBS Virtual Camera', then restart this.")

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
        cap.close()
        signs.close()
        segment.close()
        if cam is not None:
            cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
