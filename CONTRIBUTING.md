# Contributing to Shunshin Cam

Thanks for wanting to help. Bug reports, new hand signs, better smoke and performance
work are all welcome.

## Getting set up

```bash
git clone https://github.com/<your-username>/shunshin-cam.git
cd shunshin-cam
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Run the app with `python -m shunshin` (add `--no-virtualcam` if you don't have OBS installed).

## Before you open a pull request

```bash
ruff check .          # lint
python -m pytest      # tests, no webcam needed
```

CI runs both on Linux, Windows and macOS for Python 3.10 to 3.12, and a PR needs them green.

## Guidelines

- **Keep it real-time.** Every frame has a budget of about 33 ms on a laptop CPU. If a change
  touches the per-frame path (`vanish.py`, `smoke.py`, `plate.py`, the loop in `__main__.py`),
  mention the before/after frame time in the PR.
- **Add a test for new logic.** The tests build synthetic frames with NumPy, so they don't need
  a camera. Look at `tests/` for the pattern.
- **One change per PR**, with a clear title. Small, focused commits are easier to review.
- **Match the existing style:** short functions, comments that explain *why* rather than *what*,
  no new dependencies unless they are clearly worth it.
- **Visual changes** (smoke, transitions) should include a short GIF or screenshot in the PR.
  Please don't include other people's faces without their consent.

## Where to start

- Issues labelled [`good first issue`](https://github.com/Ahmed8881/shunshin-cam/labels/good%20first%20issue)
- Ideas that would make great PRs: more effect styles (leaf swirl, lightning), a custom-trained
  jutsu hand sign, vanishing only the person who made the sign, a GPU path.

## Reporting bugs

Open an issue with the bug template. Include your OS, Python version, webcam resolution, the
app you were calling with (Zoom, Meet...) and the full console output.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By taking part you agree to it.
