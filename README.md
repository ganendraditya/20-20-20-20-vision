# 420vision (20/20 20/20 Vision)

> An ultra-lightweight, native Menu Bar / System Tray reminder & assistant for dry eye sufferers using on-device computer vision.

[![CI & Verification](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The 20-20-20-20 Rule

- **20 Minutes:** Screen time interval before eye fatigue sets in.
- **20 Feet (6 meters):** Optical infinity distance that allows the eye's ciliary focusing muscles to fully relax.
- **20 Seconds:** Minimum time required for the muscles to disengage.
- **20 Blinks:** Conscious blinks target to restore the tear film.

The legitimate medical guideline from the **American Optometric Association (AOA)** is strictly the **20-20-20 rule** (take a 20-second break every 20 minutes to look at something 20 feet away). The fourth 20 is borrowed from normal human resting blink rate (~15–20 blinks/min per the [AAO](https://www.aao.org/eye-health/tips-prevention/computer-usage) and [Bentivoglio et al.](https://pubmed.ncbi.nlm.nih.gov/9399231/)), which drops by up to 60% while staring at screens.

### How the System Validates the 20-20-20 Rule

The system verifies optical distance breaks using deterministic physical cues rather than intrusive surveillance:

- **Physical Departure (Step Away):** Standing up, stretching, or stepping away from your desk (`face not detected`) credits the break.
- **Side Head Pose (Look Left/Right):** Turning your head toward a side window or across the room ($|\text{Yaw}| \ge 20^\circ$) advances the countdown.
- **Upward Gaze (Look Up / Stretch):** Tilting your head upward ($\text{Pitch} \ge 15^\circ$) toward the ceiling or a high window advances the countdown.
- **Freeze-on-Screen (Look at Monitor):** Looking directly at the screen ($|\text{Yaw}| < 20^\circ$ and $\text{Pitch} < 15^\circ$) freezes the countdown until you resume looking away (times out after 60s of continuous staring).
- **Downward Gaze Rejection:** Glancing down at a keyboard or phone does not count as looking 6 meters away.
- **Eyelid Closure vs. Optical Distance:** Eyelid rest relieves dry eye, while true 20-20-20 optical muscle disengagement requires distant fixation with eyes open.

---

## References

1. **American Optometric Association (AOA):** [Computer Vision Syndrome & The 20-20-20 Rule](https://www.aoa.org/healthy-eyes/eye-and-vision-conditions/computer-vision-syndrome)
2. **American Academy of Ophthalmology (AAO):** [Blink Rate Reduction in Digital Eye Strain](https://www.aao.org/eye-health/tips-prevention/computer-usage)
3. **Bentivoglio et al. (PubMed):** [Analysis of Blink Rate Patterns in Normal Subjects](https://pubmed.ncbi.nlm.nih.gov/9399231/)
4. **Soukupová & Čech (CVWW 2016):** [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

---

## System Requirements

- **Camera Device (Required):** Built-in laptop webcam or external USB camera (recommended: 720p or 640x480 at 15–30 FPS for optimal tracking accuracy).
- **Supported Operating Systems:**
  - **macOS:** macOS 11+ (Apple Silicon & Intel, verified in automated CI).
  - **Windows:** Windows 10 / 11 64-bit (verified in automated CI).
- **Permissions:** Camera access permission must be granted to the application when prompted by the operating system.

---

## Privacy by Design

- **100% On-Device Processing:** Computer vision inference (UltraFace & FaceMesh) executes strictly in volatile memory (RAM).
- **Zero Video Leakage:** No video feeds, frames, or photos are ever saved to disk or transmitted across any network socket.
- **Always-Yield Etiquette:** The camera stream is instantly released when Zoom, Google Meet, Microsoft Teams, or FaceTime requests the device.

---

## Quick Install (Terminal)

### macOS
Run in your terminal:
```bash
curl -fsSL https://raw.githubusercontent.com/ganendraditya/20-20-20-20-vision/main/install.sh | bash
```
Installs to `~/.local/share/420vision`, links launcher to `~/.local/bin/420vision`, and configures LaunchAgent auto-start at login.

### Windows (PowerShell)
Run in PowerShell:
```powershell
irm https://raw.githubusercontent.com/ganendraditya/20-20-20-20-vision/main/install.ps1 | iex
```
Installs to `%LOCALAPPDATA%\420vision`, adds to User `PATH`, and sets up Windows Startup tray launch.

---

## Uninstalling

### macOS
```bash
# Clean removal
bash <(curl -fsSL https://raw.githubusercontent.com/ganendraditya/20-20-20-20-vision/main/uninstall.sh)

# To also purge local database and configs
bash <(curl -fsSL https://raw.githubusercontent.com/ganendraditya/20-20-20-20-vision/main/uninstall.sh) --purge
```

### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/ganendraditya/20-20-20-20-vision/main/uninstall.ps1 | iex

# Purge all local data
irm https://raw.githubusercontent.com/ganendraditya/20-20-20-20-vision/main/uninstall.ps1 | iex -RemoveAllData
```

---

## Architecture & System Design

- **Engine:** Pure Rust native daemon compiled with zero garbage collector.
- **Vision Inference:** Two-stage edge pipeline combining UltraFace RFB-320 gatekeeper and MediaPipe FaceMesh (468 3D landmarks) running on ONNX Runtime (`ort`).
- **Mathematical EAR:** [Soukupová & Čech (2016)](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf) canonical eye aspect ratio formulation with Asymmetric Impulse EMA filter to eliminate glasses glare jitter and capture fast blinks.
- **Multi-Subject Stability:** Spatial inertia and hysteresis tracking to prevent flickering in crowded rooms.
- **Strict Always-Yield Etiquette:** Never monopolizes the webcam; instantly yields when Zoom, Meet, Teams, or FaceTime requests the camera.
- **Minimalist Tray Popover UI:** Native OS webview with 4 tabs:
  1. **Home:** Big master toggle, live BPM, and 20-min countdown.
  2. **Camera Test:** PhotoBooth sandbox with isolated validation counter.
  3. **Stats:** Local SQLite (`analytics.db`) compliance trends.
  4. **Settings:** Device selector, 5-second automatic eye shape calibration, and Smart Keep-Awake toggle.
- **Power & Resource Budget:** $<30$ MB RAM, $<3\%$ CPU (throttled to 15 FPS, drops to 5 FPS when away). Zero video recording, 100% private.

---

## Development Setup

```bash
# 1. Clone repository
git clone https://github.com/ganendraditya/20-20-20-20-vision.git
cd 20-20-20-20-vision

# 2. Install dependencies
npm install

# 3. Run development mode
npm run tauri dev

# 4. Run test suite
cargo test --manifest-path src-tauri/Cargo.toml
```

---

## License

MIT
