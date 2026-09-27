# 420vision: Architecture & Implementation Plan

> **Codename:** 420vision (20-20-20-20 Vision)  
> **Repository:** `ganendraditya/20-20-20-20-vision`  
> **Core Mission:** An ultra-lightweight, high-performance Menu Bar / System Tray assistant for dry eye sufferers using native computer vision and ONNX runtime.

---

## 1. Product Philosophy & The "4x20" Rule

Inspired by the classic ophthalmology rule + Tyler, The Creator's *"20/20, 20/20 vision"* pun:
1. **20 Minutes:** Screen time accumulated only when the user is actively in front of the screen.
2. **20 Feet (6 Meters):** Distance to look away to relax the ciliary eye muscles.
3. **20 Seconds:** Minimum rest break duration.
4. **20 Blinks (The 4th 20):** Conscious full blinks during the break / target 20 blinks per minute (BPM) to restore the tear film lipid layer.

---

## 2. Core Architecture & Tech Stack (Locked for Maximum Performance)

### A. Performance & Resource Target
* **Language:** **Pure Rust** (Compiled native binary, zero garbage collector).
* **UI Shell:** **Tauri v2** using native OS webview (WebKit on macOS, WebView2 on Windows) — **NOT Chromium/Electron**.
* **RAM Footprint:** **~25–35 MB total** (industry-leading efficiency for 24/7 background apps).
* **CPU Usage:** **< 2–3%** (10–15 FPS active tracking, auto-throttled to 2 FPS when idle/away).
* **Privacy:** 100% on-device local processing; zero cloud transmission, zero video saved to disk.

### B. Distribution & UX (Cloudflare 1.1.1.1 Model)
* **Single-Line Terminal Installation:**
  * **macOS / Linux:**
    ```bash
    curl -fsSL https://raw.githubusercontent.com/ganendraditya/20-20-20-20-vision/main/install.sh | bash
    ```
    Downloads compiled native binary to `~/.local/share/420vision`, links launcher to `~/.local/bin/420vision`, and installs LaunchAgent for auto-start at login.
  * **Windows (PowerShell):**
    ```powershell
    irm https://raw.githubusercontent.com/ganendraditya/20-20-20-20-vision/main/install.ps1 | iex
    ```
    Downloads compiled native `.exe` to `%LOCALAPPDATA%\420vision`, creates `420vision.cmd` in user PATH, and creates a shortcut in the Windows Startup folder.
* **Menu Bar / System Tray Resident:**
  * Once installed via terminal, the terminal is closed and never needed again.
  * The app lives permanently in the macOS Menu Bar and Windows System Tray.
  * Clicking the icon toggles a clean popover window (Cloudflare 1.1.1.1 style).

---

## 3. User Interface Structure (Simple 1.1.1.1 Style)

```text
┌────────────────────────────────────────────────────────┐
│  🌸 420vision                             [ - ] [ X ]  │
├────────────────────────────────────────────────────────┤
│  [ Home ]    [ 📹 Camera Test ]    [ 📊 Stats ]        │
├────────────────────────────────────────────────────────┤
│                                                        │
│                     (  ON  )                           │
│                 Monitoring Active                      │
│                                                        │
│               Blink Rate: 16 BPM                       │
│            Next Break in: 14m 20s                      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

1. **Tab 1: Home (Main / 1.1.1.1 Style):**
   * Big clean ON/OFF master toggle.
   * Current status badge (`● Monitoring`, `⏸ Paused (Away)`, `⚠️ Camera Busy`).
   * Live Blinks Per Minute (BPM) meter and countdown to next 20-20-20 break.
2. **Tab 2: Camera Test (PhotoBooth Style for Live Trust & Validation):**
   * Live camera preview showing face/eye landmark tracking.
   * Real-time blink counter that increments on screen whenever the user blinks.
   * Users can see themselves: blink once -> counter +1; close eyes for 10s -> paused -> open -> counter +1.
3. **Tab 3: Analytics & Compliance:**
   * Daily average BPM trends and 20-20-20 break compliance history.
4. **Tab 4: Settings:**
   * Camera selector (e.g. `FaceTime HD Camera`, `USB External Cam`).
   * Toggle stare alert (sound / native notification).
   * One-click 5-second calibration button.

---

## 4. Vision Engine & Behavioral Logic (Locked)

1. **Vision Inference:**
   * MediaPipe FaceMesh running via **ONNX Runtime (`ort`)** with hardware acceleration (DirectML on Windows, CoreML/Metal on macOS).
   * Mathematical Eye Aspect Ratio (EAR) across eyelid landmarks.
2. **Always Yield Principle (Camera Etiquette):**
   * The app yields to any external app needing the camera (Zoom, Teams, Google Meet, FaceTime).
   * Instantly releases the capture device and silently pauses (`⏸ Camera Paused (Resource Busy)`), polling gently every 5-8 seconds to resume once the camera is free.
3. **Multi-Camera & Zero-Camera Support:**
   * If user has multiple cameras, apps using different cameras run concurrently without conflict.
   * If zero cameras exist, app stays idle (`⚠️ No Camera Detected`) and auto-detects USB plug-in.
4. **Biological Blink vs Prolonged Closure (Resting Eyes):**
   * **Blink (0.1s - 0.8s):** Counts as 1 blink event.
   * **Prolonged Eye Rest (> 1.0s closed):**
     * Counts as 1 blink on closure.
     * While eyes are closed, stare timers are paused.
     * Resets and resumes counting once eyes reopen.
5. **Head Angles & Peripheral Reading:**
   * As long as eyes are visible to the camera (e.g. looking down at phone or physical books), continue tracking blinks and accumulating screen time.
6. **Away Time Auto-Reset:**
   * No face detected: timer pauses immediately.
   * Away for $> 5$ minutes: 20-minute timer resets to 0 (eyes naturally rested).
7. **System Sleep & Lock Screen Lifecycle (Power Efficiency):**
   * **Immediate Shutdown:** On OS lock screen (`Win+L` / macOS Lock) or lid close / sleep event, the camera stream stops immediately to eliminate battery drain.
   * **Smart Resume:** Upon unlock/wake:
     - If slept/locked for $> 5$ minutes: resets 20-minute timer to 0.
     - If slept/locked for $< 5$ minutes: resumes from previous remaining time.
   * Re-initializes camera smoothly on wake without requiring user intervention.
8. **Storage Architecture (SQLite + JSON):**
   * **Location:** Standard OS config directory:
     - macOS/Linux: `~/.config/420vision/`
     - Windows: `%APPDATA%\420vision\`
   * **Analytics Database (`analytics.db`):** Embedded SQLite via `rusqlite` for querying daily/weekly compliance and BPM trends.
   * **In-Memory Buffering:** Live calculations stay strictly in RAM. Writes to disk only happen at the end of a 20-minute cycle, break event, or app shutdown (zero SSD wear).
   * **Configuration File (`config.json`):** Human-readable settings (camera index, audio toggles, calibrated EAR threshold).
   * **Zero Cloud:** 100% offline, local-only storage; zero telemetry.
9. **Camera Test (PhotoBooth Sandbox) Lifecycle:**
   * **Conditional Rendering:** Frame rendering to the UI only occurs when the user is actively viewing the Camera Test tab. When minimized or hidden in the tray, UI video rendering shuts down completely, and the engine runs strictly on lightweight numerical landmark coordinates.
   * **Sandbox Counter Reset:** Whenever the user opens the Camera Test tab, the test counter initializes at 0 (with a manual `[🔄 Reset]` button). This provides an immediate, satisfying sandbox feedback loop for the user to validate blinks in real-time, while persistent daily analytics continue tracking uninterrupted in the background.
10. **Native Notification Interaction Behavior:**
    * **Non-Blocking Auto-Dismiss:** Notifications (20-min break or stare alert) appear via native OS banners (macOS Notification Center / Windows Action Center) with a gentle chime and automatically fade out after 5-7 seconds if ignored, ensuring zero workflow interruption.
    * **Click-to-Focus Action:** If the user clicks the notification banner, the app window immediately springs to the foreground, opening directly to the 20-second break countdown & Camera Test tab so they can practice their 20 conscious blinks.
11. **Automated Cross-Platform CI/CD Release Pipeline:**
    * **Trigger:** Push to version tags (`v*.*.*`).
    * **Matrix Runners:** Concurrent GitHub Actions runners for `windows-latest` (producing `.exe`) and `macos-latest` (producing Apple Silicon & Intel Universal binary).
    * **Packaging & Publishing:** Bundles native binaries into compressed archives (`420vision-windows-x64.zip`, `420vision-macos-universal.tar.gz`) and automatically attaches them to the tagged GitHub Release, feeding directly into `install.sh` and `install.ps1`.

---

## 5. Project Directory Structure

```text
20-20-20-20-vision/
├── src-tauri/                 # Rust Core & Tauri v2 backend
│   ├── Cargo.toml             # Rust dependencies (ort, nokhwa, tauri, etc.)
│   ├── tauri.conf.json        # Tauri configuration & tray menu specs
│   └── src/
│       ├── main.rs            # Entrypoint & Tauri app builder
│       ├── capture/           # Native camera stream (nokhwa / MSMF / AVFoundation)
│       ├── vision/            # ONNX Runtime FaceMesh & EAR calculations
│       ├── detector/          # Blink state machine & stare tracker
│       ├── timer/             # 20-minute presence tracker & 5-min auto-reset
│       ├── storage/           # Local SQLite / JSON analytics store
│       └── notifier/          # Native OS notifications & in-memory chime
├── ui/                        # Frontend Webview (HTML + Tailwind + TypeScript)
│   ├── index.html             # Main 1.1.1.1 popover window
│   ├── src/
│   │   ├── main.ts            # Tab navigation & Tauri IPC listeners
│   │   ├── tabs/              # Home, Camera Test, Analytics, Settings views
│   │   └── styles/            # Minimal dark-mode Tailwind CSS
│   └── package.json           # Frontend tooling
├── models/                    # Quantized ONNX FaceMesh models
├── install.sh                 # Single-line installer (macOS/Linux)
├── install.ps1                # Single-line installer (Windows)
├── uninstall.sh               # Clean uninstaller (macOS/Linux)
├── uninstall.ps1              # Clean uninstaller (Windows)
├── PLAN.md                    # Architecture and specification tracking
└── README.md                  # Public documentation & install instructions
```

---

## 6. Execution Milestones

- [ ] **Milestone 1: Rust Core Setup & Native Camera Grabber**
  - Initialize Tauri v2 project structure.
  - Implement native camera capture with graceful yield (Zoom/Meet conflict handling).
- [ ] **Milestone 2: ONNX Runtime Vision & Blink Engine**
  - Integrate ONNX FaceMesh model for real-time eyelid landmarks.
  - Implement EAR calculation, blink counter state machine, and prolonged closure logic.
- [ ] **Milestone 3: Menu Bar / System Tray & 1.1.1.1 UI Popover**
  - Implement native tray menu and background daemon lifecycle.
  - Build the 4-tab UI: Home (1.1.1.1 toggle), Camera Test (PhotoBooth live feed), Analytics, Settings.
- [ ] **Milestone 4: Native Notifications & In-Memory Audio**
  - Native macOS banner / Windows Action Center toast notifications.
  - In-memory synthesized gentle chime cues.
- [ ] **Milestone 5: Terminal One-Line Installers & CI/CD**
  - GitHub Actions automated release pipeline to build native binaries for macOS (Universal) and Windows (`.exe`).
  - Implement `install.sh` and `install.ps1` curl/irm terminal installers.
  - Verification & comprehensive README documentation.
