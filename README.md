# 420vision (20-20-20-20 Vision)

> An ultra-lightweight, native Menu Bar / System Tray assistant for dry eye sufferers using on-device computer vision.

[![CI & Verification](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Background & Clinical Context

### The Standard 20-20-20 Rule
Originally developed by California optometrist **Dr. Jeffrey Anshel** and endorsed by both the **American Optometric Association (AOA)** and the **American Academy of Ophthalmology (AAO)**, the **20-20-20 rule** is an ergonomic guideline to alleviate *Computer Vision Syndrome (CVS)* and *Digital Eye Strain (DES)*:
> **Every 20 minutes, look at an object at least 20 feet (~6 meters) away for at least 20 seconds.**

- **Why 20 feet?** At roughly 6 meters, optical infinity is achieved—allowing the eye's *ciliary muscles* (which constantly contract to maintain near-focus on digital screens) to completely relax.
- **Why 20 seconds?** It takes approximately 20 seconds for the ciliary muscles to disengage and for the tear film to restabilize.

### The 4th "20": Why 420vision?
While the medical rule stops at three 20s, prolonged digital focus causes another severe issue: **blink rate reduction**. Studies show people blink **up to 60% less frequently** when staring at digital monitors (dropping from a healthy 15–20 blinks/min down to 4–7 blinks/min), leading to rapid tear film evaporation and dry eye disease.

The name **420vision** (20-20-20-20) adds a purposeful fourth component:
1. **20 Minutes:** Screen presence accumulated via webcam (pauses when away).
2. **20 Feet (6m):** Distance to look into to relax the ciliary focal muscles.
3. **20 Seconds:** Dedicated resting duration.
4. **20 Blinks:** 20 conscious, deliberate blinks during the break—*plus a tongue-in-cheek reference to Tyler, The Creator's "20/20, 20/20 vision" lyric*—to replenish the corneal lipid layer and rehydrate dry eyes.

---

## References & Further Reading

1. **American Optometric Association (AOA):** [Computer Vision Syndrome & The 20-20-20 Rule](https://www.aoa.org/healthy-eyes/eye-and-vision-conditions/computer-vision-syndrome)
2. **American Academy of Ophthalmology (AAO):** [Computers, Digital Devices and Eye Strain](https://www.aao.org/eye-health/tips-prevention/computer-usage)
3. **Soukupová, T., & Čech, J. (2016):** *Real-Time Eye Blink Detection using Facial Landmarks.* Center for Machine Perception, Czech Technical University.
4. **Rosenfield, M. (2011):** *Computer vision syndrome: a review of ocular causes and potential treatments.* Ophthalmic and Physiological Optics, 31(5), 502–515.

---

## Quick Install (Terminal)

### macOS (Apple Silicon & Intel)
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
- **Vision Inference:** Quantized MediaPipe FaceMesh running on ONNX Runtime (`ort`) with hardware acceleration (DirectML / CoreML / CPU).
- **Mathematical EAR:** Soukupová & Čech (2016) formula with Exponential Moving Average (EMA) smoothing to eliminate glasses glare jitter.
- **Strict Always-Yield Etiquette:** Never monopolizes the webcam; instantly yields when Zoom, Meet, Teams, or FaceTime requests the camera.
- **Cloudflare 1.1.1.1-Style UI:** Native OS webview with 4 tabs:
  1. **Home:** Big master toggle, live BPM, and 20-min countdown.
  2. **Camera Test:** PhotoBooth sandbox with isolated validation counter.
  3. **Stats:** Local SQLite (`analytics.db`) compliance trends.
  4. **Settings:** Device selector and 5-second automatic eye shape calibration.
- **Power & Resource Budget:** $<30$ MB RAM, $<3\%$ CPU (throttled to 15 FPS, drops to 2 FPS when away). Zero video recording, 100% private.

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
