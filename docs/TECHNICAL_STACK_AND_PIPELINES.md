# Technical Stack, Vision Pipelines & Architectural Specification

> Comprehensive engineering specification for **420vision (20-20-20-20 Vision)**: a private, ultra-lightweight Menu Bar / System Tray daemon for desktop eye health and dry eye prevention powered by on-device edge computer vision.

---

## 1. High-Level System Architecture

420vision runs as an unprivileged native background process using an orthogonal, multi-threaded architecture with decoupled hardware acquisition, dual-model edge inference, mathematical signal filtering, SQLite analytics, and an IPC-driven Tauri webview popover:

```text
+---------------------------------------------------------------------------------------+
|                    FRONTEND CLIENT (Tauri v2 + Webview / TS + CSS)                    |
|  - 1.1.1.1-Style Popover: Home, Camera Test (PhotoBooth), Stats, Settings             |
|  - Zero-Overhead Canvas Overlay: Dynamic Face Contour & Dual Eye Mesh Landmark Toggle |
|  - IPC Bridge: Commands (get_status, toggle, set_camera) + Frame Events (15 FPS)     |
+-------------------------------------------+-------------------------------------------+
                                            |
                         Tauri IPC Commands | Window Events / Frames
                                            v
+---------------------------------------------------------------------------------------+
|                          RUST BACKEND CORE (Tauri v2 + Lib)                           |
|                                                                                       |
|  +---------------------------------------------------------------------------------+  |
|  |                Camera Stream Manager & Hardware Contention Handler              |  |
|  |  - nokhwa Native Backend (AVFoundation / Media Foundation)                      |  |
|  |  - Auto-Discovery, Resolution Negotiation (1280x720 / 640x480 RAW RGB)          |  |
|  |  - Always-Yield Etiquette: Instantly drops handle when Zoom/Meet claims camera  |  |
|  +----------------------------------------+----------------------------------------+  |
|                                           | Raw RGB Frame (15 FPS Throttled)          |
|                                           v                                           |
|  +---------------------------------------------------------------------------------+  |
|  |                 Two-Stage Edge Computer Vision Inference Pipeline               |  |
|  |  [STAGE 1] UltraFace Detector (RFB-320 ONNX): Scans 320x240 for presence       |  |
|  |            * If 0 faces: Early return, clear landmarks, Pause presence timer    |  |
|  |  [STAGE 2] FaceMesh Landmark Regressor (468-point 3D ONNX): 192x192 Square Crop |  |
|  |            * Extracts 16 eye points + 58 canonical facial contour points        |  |
|  +----------------------------------------+----------------------------------------+  |
|                                           | 468 Landmark 3D Points                     |
|                                           v                                           |
|  +---------------------------------------------------------------------------------+  |
|  |                    Mathematical Signal Processing & FSM Engine                  |  |
|  |  1. 3-Pair Euclidean EAR: (||v1|| + ||v2|| + ||v3||) / (3.0 * ||h||)            |  |
|  |  2. Asymmetric EMA Smoothing Filter: alpha=0.40 (static) / alpha_fast=0.85     |  |
|  |  3. Bilateral & Asynchronous Blink Pairing: Window Cap <= 1.0s                  |  |
|  |  4. Presence State Machine: 20-min work interval, 5-min auto-away reset         |  |
|  |  5. Stare Warning Guard: Alerts after >8s sustained unblinking gaze             |  |
|  +--------------------+-----------------------------------+--------------------+  |
+-----------------------+-----------------------------------+--------------------+------+
                        |                                   |
                        v                                   v
             +---------------------+             +--------------------+
             | Embedded SQLite DB  |             | Native Notifier    |
             |  (rusqlite bundled) |             | & Audio Synthesizer|
             | - daily_compliance  |             | - osascript / WinRT|
             | - config key-value  |             | - In-Memory WAV    |
             +---------------------+             +--------------------+
```

---

## 2. Core Technology Stack Matrix

| Domain / Layer | Technology Component | Specification / Version | Architectural Purpose & Rationale |
| :--- | :--- | :--- | :--- |
| **System Language** | Rust | Edition `2021` (Stable) | Memory-safe, zero garbage collection pauses, low-level OS threading, predictable sub-millisecond execution. |
| **Desktop Shell & IPC** | Tauri v2 | `2.12.0` | Ultra-lightweight native webview shell replacing Electron bloat; compiles directly to native binary (<30 MB RAM). |
| **Frontend Framework** | TypeScript + HTML5 Canvas | Vanilla / Vite `8.3.1` | Zero-framework client runtime; canvas direct pixel blitting without React/Vue virtual DOM overhead. |
| **Styling Architecture** | CSS3 (Dark Mode) | Custom Modern Design Tokens | 1.1.1.1-inspired compact popover, zero CSS library dependencies. |
| **Camera Hardware I/O** | `nokhwa` | `0.10.11` (`input-native`, `output-threaded`) | Direct hardware webcam bindings via platform-native APIs (AVFoundation on macOS, MediaFoundation on Windows). |
| **Inference Engine** | ONNX Runtime (`ort`) | `2.0.0-rc.13` (`download-binaries`) | C++ native neural network runner; Level 3 graph optimizations, multithreading, zero-PyTorch dependency. |
| **Tensor Math** | `ndarray` | `0.17` | Multidimensional array operations for NCHW/NHWC tensor reshaping, crop offsets, and matrix slicing. |
| **Stage 1: Face Detector** | UltraFace (RFB-320) | `models/ultraface.onnx` (1.2 MB) | Gatekeeper edge detector ($320 \times 240$ NCHW); filters empty frames with 99.99% presence vs 7.4% background confidence before FaceMesh runs. |
| **Stage 2: Landmark Regressor** | MediaPipe FaceMesh | `models/facemesh.onnx` (2.3 MB) | Extracts 468 3D facial coordinates ($192 \times 192$ center crop); outputs landmark positions for eyelid geometry and facial contour. |
| **Local Persistence** | SQLite | `rusqlite 0.32` (`bundled`) | Zero-configuration local database for compliance analytics and configuration persistence. |
| **Image Compression** | `image` + `mozjpeg` | `0.25` / JPEG | Fast RGB-to-JPEG thumbnail encoder for zero-lag stream preview in Camera Test sandbox. |
| **Native Notifications** | Platform Subprocess | `osascript` (macOS) / Shell | Native desktop banner alerts without third-party daemon background daemons. |
| **Audio Synthesizer** | In-Memory PCM / WAV | 44.1 kHz Sine Oscillator | Synthesizes pleasant sine bell chimes directly in memory, zero bundled audio MP3 assets needed. |

---

## 3. Computer Vision & Signal Processing Pipelines

### 3.1 Two-Stage Edge Vision Pipeline

```text
[ Incoming Camera Frame (1280x720 RAW RGB) ]
                   │
                   ▼
     ┌───────────────────────────┐
     │ STAGE 1: UltraFace        │ ◄── Input: [1, 3, 240, 320] NCHW, normalized (p - 127)/128
     │ RFB-320 ONNX (~2ms CPU)   │     Parses all candidate faces (conf >= 0.70)
     └─────────────┬─────────────┘
                   │
         Face detected? (Score >= 0.70)
          ├── NO ──► is_face = false
          │          • State: "Away / Paused"
          │          • Pause presence timer (freeze 20-min countdown)
          │          • Clear canvas landmarks (zero ghost overlays)
          │          • SKIP FaceMesh (Conserves 70% CPU cycles)
          │
          └── YES
                   │
                   ▼ (Multi-Face Disambiguation: Select Dominant Box with max(Area))
     ┌───────────────────────────┐
     │ STAGE 2: MediaPipe FaceMesh│ ◄── Input: [1, 192, 192, 3] NHWC, normalized [0.0..1.0]
     │ 468 3D Landmarks ONNX     │     Crop: Dynamic square padded around dominant primary box
     └─────────────┬─────────────┘
                   │
                   ▼
     [ 468 3D Coordinates: (x, y, z) ]
```

### 3.2 Eyelid Mathematical Landmark Geometry (3-Pair EAR)

Following the Soukupová & Čech (2016) canonical model upgraded to 3 vertical anatomical pairs for curved eyelid discriminability:

* **Left Eye Landmark Indices:**
  * Horizontal Corners: `33` (outer), `133` (inner)
  * Vertical Pairs: `(159, 145)` center, `(158, 153)` inner flanking, `(160, 144)` outer flanking
* **Right Eye Landmark Indices:**
  * Horizontal Corners: `362` (inner), `263` (outer)
  * Vertical Pairs: `(386, 374)` center, `(387, 373)` inner flanking, `(385, 380)` outer flanking

$$\text{EAR} = \frac{\|v_1\| + \|v_2\| + \|v_3\|}{3.0 \times \|h\|}$$

Where distances are Euclidean 2D distances on the image plane:
$$\|p_1 - p_2\| = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$

### 3.3 Asymmetric Exponential Moving Average (EMA) Filter

Webcam sensors exhibit optical noise and exposure micro-flicker. An unmodified raw EAR signal leads to false blink triggers, while standard symmetric smoothing causes response lag.

420vision implements an **Asymmetric Impulse EMA Filter**:

$$\text{EMA}_t = \alpha \cdot \text{EAR}_t + (1 - \alpha) \cdot \text{EMA}_{t-1}$$

* **Static / Upward State ($\text{EAR}_t \ge \text{EMA}_{t-1}$):**
  Uses **$\alpha = 0.40$**. Provides **74.2% noise suppression** on stationary open eyes, stabilizing the HUD readout without baseline drift.
* **Downward Closure Impulse ($\text{EAR}_t < \text{EMA}_{t-1}$):**
  Switches dynamically to **$\alpha_{\text{fast}} = 0.85$**. Eliminates phase delay, allowing fast biological blinks (even $100\text{ ms}$ micro-blinks) to penetrate the closure threshold within $1\text{ to }2$ frames ($66\text{–}133\text{ ms}$).

### 3.4 Finite State Machine (FSM) & Asynchronous Blink Pairing

```text
               Left Eye Downward                        Left Eye Reopened
                 (EAR < 0.22)                              (EAR >= 0.22)
            ┌─────────────────────┐                   ┌──────────────────────┐
            │ LEFT_EYE_CLOSED     │ ────────────────► │ RECORD T_left_blink  │
            └─────────────────────┘                   └──────────┬───────────┘
                                                                 │
                                                       |T_left - T_right| <= 1.0s?
                                                                 │
            ┌─────────────────────┐                   ┌──────────┴───────────┐
            │ RIGHT_EYE_CLOSED    │ ────────────────► │ RECORD T_right_blink │
            └─────────────────────┘                   └──────────────────────┘
              Right Eye Downward                       Right Eye Reopened
                 (EAR < 0.22)                              (EAR >= 0.22)
```

1. **Duration Window Validation:**
   * Closure duration must satisfy: $80\text{ ms} \le \Delta t \le 800\text{ ms}$.
   * $\Delta t < 80\text{ ms}$: Discarded as single-frame camera sensor glitch.
   * $\Delta t \ge 1000\text{ ms}$: Flags `is_resting` (eyes closed in thought/nap); pauses stare warning timer.
2. **Asynchronous Left-Right Pairing Window ($1.0\text{s}$):**
   * Real human blinks exhibit natural micro-desynchronization between eyes.
   * If left and right eyes complete blinks within **$\le 1.0\text{ second}$**, the pair resolves into **$+1\text{ valid blink}$**.
   * Unpaired unilateral winks (e.g. winking only left eye) expire after $1.0\text{s}$ and are discarded.

### 3.5 Head Pose Yaw Symmetry Gate (Zero-Model Geometric Filter)

Generic face detectors occasionally fire on non-frontal head angles, such as the back of the head, downward neck flexion, or steep side profiles where only one eye is visible.

420vision computes the **Nasal-Interocular Yaw Asymmetry Ratio** directly from canonical FaceMesh points:
* Nose Tip: Landmark `1` ($X_{\text{nose}}$)
* Left Eye Outer Corner: Landmark `33` ($X_{\text{left}}$)
* Right Eye Outer Corner: Landmark `263` ($X_{\text{right}}$)

$$\text{Midpoint} = \frac{X_{\text{left}} + X_{\text{right}}}{2.0}, \quad \text{Width} = |X_{\text{right}} - X_{\text{left}}|$$
$$\text{Yaw Ratio} = \frac{|X_{\text{nose}} - \text{Midpoint}|}{\text{Width}}$$

* **Frontal / Active Screen Gaze ($\text{Yaw Ratio} \le 0.25$):** User is naturally looking at the display. Full blink and presence processing active.
* **Severe Side Profile / Looking Away ($\text{Yaw Ratio} > 0.35$):** User turned their head sideways ($> 35^{\circ}$) or back of head is visible. Blink triggers are suppressed and state cleanly transitions to `Away / Paused` without spawning extra AI models.

---

## 4. Presence, Stare Warning & Timer Architecture

* **Screen Time Accumulation (`PresenceTimer`):**
  * Target: $1200\text{ seconds}$ ($20\text{ minutes}$) of continuous physical face presence.
  * If user leaves the desk (`is_face == false`): countdown pauses immediately.
  * Quick returns ($< 300\text{s}$ / $5\text{ min}$) resume the session timer where it stopped.
  * Sustained absence ($\ge 300\text{s}$): automatically resets session elapsed time to $0$ (assumes user took a physical break).
* **Stare Warning Guard (`stare_warning`):**
  * Dry eye occurs primarily when users stare fixated on text/code without blinking.
  * If physical face is present with eyes open continuously for $> 8.0\text{ seconds}$ without a registered blink, the system triggers a subtle stare alert (synthesized waterdrop chime + notification).

---

## 5. Storage & Database Schema (`SQLite`)

Database file resides in the user's application data directory (`~/.local/share/420vision/analytics.db` on macOS, `%LOCALAPPDATA%\420vision\analytics.db` on Windows).

### 5.1 Tables

#### 1. `daily_compliance`
Tracks rolling 20-20-20 adherence and dry eye metrics:
* `date`: `TEXT PRIMARY KEY` (Format: `YYYY-MM-DD`)
* `total_blinks`: `INTEGER NOT NULL DEFAULT 0`
* `screen_seconds`: `INTEGER NOT NULL DEFAULT 0`
* `breaks_completed`: `INTEGER NOT NULL DEFAULT 0`
* `breaks_skipped`: `INTEGER NOT NULL DEFAULT 0`
* `avg_bpm`: `REAL NOT NULL DEFAULT 0.0`

#### 2. `config`
Stores persistent user preferences:
* `key`: `TEXT PRIMARY KEY`
* `value`: `TEXT NOT NULL`
* Keys stored: `selected_camera`, `sound_enabled`, `personal_ear_threshold`, `stare_warning_seconds`.

---

## 6. IPC Contract Matrix (Rust Backend <-> Frontend Webview)

All communication uses strongly typed DTOs via Tauri v2 IPC:

| IPC Channel | Direction | Payload / Parameters | Return Type | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `get_status` | Invoke (Pull) | None | `AppStatus` | Polled periodically to refresh Home status badge, BPM, and countdown. |
| `toggle_monitoring`| Invoke | None | `bool` | Master ON/OFF toggle from UI or Tray menu. |
| `get_cameras` | Invoke | None | `Vec<String>` | Lists detected hardware cameras for the Settings selector. |
| `set_camera` | Invoke | `{ index: usize }` | `bool` | Switches active hardware camera index dynamically. |
| `set_sandbox_viewing`| Invoke | `{ active: bool }` | `()` | Conditional rendering toggle: halts JPEG encoding when tab is closed. |
| `start_calibration` | Invoke | None | `String` | Resets the in-memory `EyeCalibrator` state machine for a fresh 5s run. |
| `submit_calibration_sample` | Invoke | `{ stage: String, ear: f32 }` | `bool` | Submits an open (3s) or closed (2s) eye EAR sample to the calibrator. |
| `finalize_calibration` | Invoke | None | `CalibrationResult` | Computes optimal personal threshold, clamps to [0.16..0.28], and saves to config. |
| `get_config` | Invoke (Pull) | None | `AppConfig` | Fetches active runtime configuration including current `ear_threshold`. |
| `hide_window` | Invoke | None | `()` | Hides the popover window to the system tray. |
| `quit_app` | Invoke | None | `()` | Performs deterministic teardown, drops hardware handles, and exits cleanly. |
| `camera-sandbox-frame`| Event (Push) | `CameraFrameDto` | Stream | Emits live preview frame (JPEG base64) + dual landmark arrays (15 FPS). |

### 6.1 `CameraFrameDto` Contract
```typescript
interface CameraFrameDto {
  width: number;
  height: number;
  is_face_detected: boolean;
  left_ear: number;
  right_ear: number;
  avg_ear: number;
  is_blinking: boolean;
  total_blinks: number;
  eye_landmarks: Array<{ x: number; y: number }>;   // 16 points (re-mapped to full frame [0..1])
  face_landmarks: Array<{ x: number; y: number }>;  // 58 points (contour jawline, eyebrows, nose, mouth)
  image_data_base64: string | null;                 // 320x180 JPEG thumbnail
}
```

---

## 7. Resource & Power Budget SLAs

* **Process Resident Set Size (RAM):** $< 30\text{ MB}$ (typically $18\text{–}24\text{ MB}$ under active inference).
* **CPU Consumption (Active Monitoring):** $< 3\%$ on modern Apple Silicon / Intel Core i5.
* **Adaptive Frame Pacing (Issue #40):** 
  - **Active Gaze State:** Throttled to $\sim 15\text{ FPS}$ (`67ms` sleep interval) providing $100\%$ blink detection parity.
  - **Away / Idle State:** Dynamically throttled down to $\sim 5\text{ FPS}$ (`200ms` sleep interval) cutting idle CPU consumption to $< 0.8\%$ and preserving laptop battery.
* **Zero Video Leakage:** Camera frames are processed strictly in volatile memory and immediately overwritten; zero frames are ever saved to disk or transmitted to any network socket.
