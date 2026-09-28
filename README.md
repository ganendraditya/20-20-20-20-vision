# 420vision (20-20-20-20 Vision)

> An ultra-lightweight, native Menu Bar / System Tray assistant for dry eye sufferers using on-device computer vision.

[![CI & Verification](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/ganendraditya/20-20-20-20-vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Background: The 20-20-20 Rule & The "4th 20" Lore

### The Legitimate 20-20-20 Rule
If you spend your life in front of a terminal, you've probably heard eye doctors (and TikTok optometrists) preach the **20-20-20 rule**:
> **Every 20 minutes, look at an object 20 feet (~6 meters) away for 20 seconds.**

Devised by California optometrist **Dr. Jeffrey Anshel** and recognized by organizations like the **American Optometric Association (AOA)**, the science here is real:
- **20 feet:** Gives your eye's overworked *ciliary muscles* a break by shifting into optical infinity (distance vision).
- **20 seconds:** The time needed for those focusing muscles to actually disengage and reset.

### ...And The 4th "20" (The Confession)
Let's be 100% honest here: **there is no 4th 20 in medical science.**

As a chronic dry eye sufferer, I desperately needed a reminder to not just look away, but to actually **blink** (screen glare makes humans forget to blink, dropping our blink rate by up to 60%). 

So why **20-20-20-20**?
1. I was listening to Tyler, The Creator's *"See You Again"* (*"20/20, 20/20 vision — cupid hit me with precision"*).
2. Four 20s together make **420**.
3. It was too funny of a pun and project name to pass up.

So yes, the **4th 20 (20 conscious blinks during break)** is completely made up by me to fit the name and justify calling this app **420vision**. But hey—your dry eyes will thank you for those extra blinks anyway.

---

## References

1. **American Optometric Association (AOA):** [Computer Vision Syndrome & The 20-20-20 Rule](https://www.aoa.org/healthy-eyes/eye-and-vision-conditions/computer-vision-syndrome)
2. **American Academy of Ophthalmology (AAO):** [Computers, Digital Devices and Eye Strain](https://www.aao.org/eye-health/tips-prevention/computer-usage)
3. **Soukupová, T., & Čech, J. (2016):** *Real-Time Eye Blink Detection using Facial Landmarks.* Center for Machine Perception, Czech Technical University.
4. **Tyler, The Creator (2017):** *See You Again (feat. Kali Uchis).* Flower Boy, Columbia Records (the unofficial scientific inspiration for the 4th 20).

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
