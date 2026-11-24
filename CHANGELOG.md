# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-11-24

### Added
- `--background IMAGE`: built-in virtual background that keeps the smoke visible (Zoom's and
  Meet's own backgrounds erase it). Works with Zoom's saved background files as-is.
- Closed fist (✊) triggers the vanish too; `--gesture` accepts several signs.
- Standalone `ShunshinCam.exe` for Windows, built on every release. Drop an image on it to use
  it as the background.

### Fixed
- Webcam auto-exposure jumps no longer get mistaken for a bumped camera.
- Clear warning when another app is holding the webcam, instead of a false one at startup.

## [1.0.0] - 2026-11-24

### Added
- Hand-sign triggered vanish with procedural anime smoke.
- Self-learning clean plate with exposure matching, bump detection, hole fill and grain matching.
- Virtual camera output for Zoom, Google Meet, Teams and Discord, with a preview HUD.
- One-step OBS install offer on Windows and in-app Zoom/Meet setup steps.
- `shunshin-cam` command via `pip install`.
