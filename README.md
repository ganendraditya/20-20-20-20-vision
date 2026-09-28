# 420vision (20-20-20-20 Vision)

> An ultra-lightweight, native Menu Bar / System Tray assistant for dry eye sufferers using on-device computer vision.

[![CI & Verification](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The "4x20" Philosophy

Inspired by the classic 20-20-20 rule + Tyler, The Creator's *"20/20, 20/20 vision"* pun:
1. **20 Minutes:** Screen time accumulated only when actively facing the display.
2. **20 Feet (6m):** Look into the distance to relax ciliary focal muscles.
3. **20 Seconds:** Dedicated break duration.
4. **20 Blinks:** Conscious full blinks to rebuild the lipid layer and soothe dry eyes.

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

MIT © [Ganendra Aditya](https://github.com/ganendraditya)
