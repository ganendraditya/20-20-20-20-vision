# 420vision (20-20-20-20 Vision)

> An ultra-lightweight, native Menu Bar / System Tray assistant for dry eye sufferers using on-device computer vision.

[![CI & Verification](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The 20-20-20 Rule (and the 4th 20)

### The Real Medical Rule
Endorsed explicitly by the **American Optometric Association (AOA)** to alleviate Computer Vision Syndrome (digital eye strain):
> **Every 20 minutes, take a 20-second break to view something 20 feet (~6 meters) away.**

- **20 Minutes:** Screen time interval before eye fatigue sets in.
- **20 Feet (6m):** Optical infinity distance that allows the eye's ciliary focusing muscles to fully relax.
- **20 Seconds:** Minimum time required for the muscles to disengage.

### The 4th "20"
Medically, the rule is strictly 20-20-20. 

The 4th 20 (*20 conscious blinks*) is purely made up because I wanted to name this project **20-20-20-20 vision** after Tyler, The Creator's *"See You Again"* lyric—and because 4 × 20 = **420**.

---

## References

1. **American Optometric Association (AOA):** [Computer Vision Syndrome & The 20-20-20 Rule](https://www.aoa.org/healthy-eyes/eye-and-vision-conditions/computer-vision-syndrome)

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
