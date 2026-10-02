# 420vision (20-20-20-20 Vision)

> An ultra-lightweight, native Menu Bar / System Tray assistant for dry eye sufferers using on-device computer vision.

[![CI & Verification](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The 20-20-20-20 Rule

- **20 Minutes:** Screen time interval before eye fatigue sets in.
- **20 Feet (6m):** Optical infinity distance that allows the eye's ciliary focusing muscles to fully relax.
- **20 Seconds:** Minimum time required for the muscles to disengage.
- **20 Blinks:** Conscious blinks target to restore the tear film.

The legitimate medical guideline from the **American Optometric Association (AOA)** is strictly the **20-20-20 rule** (take a 20-second break every 20 minutes to look at something 20 feet away). The fourth 20 is borrowed from normal human resting blink rate (~15–20 blinks/min per the [AAO](https://www.aao.org/eye-health/tips-prevention/computer-usage) and [Bentivoglio et al.](https://pubmed.ncbi.nlm.nih.gov/9399231/)), which drops by up to 60% while staring at screens—and conveniently rounded to complete the **420** pun.

---

## References

1. **American Optometric Association (AOA):** [Computer Vision Syndrome & The 20-20-20 Rule](https://www.aoa.org/healthy-eyes/eye-and-vision-conditions/computer-vision-syndrome)
2. **American Academy of Ophthalmology (AAO):** [Blink Rate Reduction in Digital Eye Strain](https://www.aao.org/eye-health/tips-prevention/computer-usage)
3. **Bentivoglio et al. (PubMed):** [Analysis of Blink Rate Patterns in Normal Subjects](https://pubmed.ncbi.nlm.nih.gov/9399231/)

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
