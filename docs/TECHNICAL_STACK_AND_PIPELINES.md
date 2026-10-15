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

At startup, ONNX models are resolved by checking `models/<name>` from the current working directory, `../models/<name>`, the workspace-level `models/` directory derived from `CARGO_MANIFEST_DIR`, `~/.local/share/420vision/models/<name>`, then `<data_dir>/420vision/models/<name>`. The first existing file is loaded; if none exists, the first candidate is returned for a useful load error.

The Tauri application tracks `src-tauri/Cargo.lock` to keep native dependency resolution reproducible, including prerelease ONNX Runtime (`ort`) APIs. CI and release test jobs use `cargo test --locked` so manifest/lockfile drift fails explicitly instead of silently selecting a newer incompatible prerelease.

---

## 3. Computer Vision & Signal Processing Pipelines

### 3.1 Two-Stage Edge Vision Pipeline & Multi-Subject Sticky Tracking

```text
[ Incoming Camera Frame (1280x720 RAW RGB) ]
                   │
                   ▼
     ┌───────────────────────────┐
     │ STAGE 1: UltraFace        │ ◄── Input: [1, 3, 240, 320] NCHW, normalized (p - 127)/128
     │ RFB-320 ONNX (~5.7ms CPU) │     • Full-frame downscale (Tier 1 & Tier 2)
     │                           │     • Adaptive 4:3 RoI Zoom (<0.015 area, Tier 3) ◄── Issue #63
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
                   ▼
     ┌─────────────────────────────────────────────────────────┐
     │ Multi-Subject Sticky Tracker (Spatial Hysteresis)       │ ◄── Issue #52
     │ • IoU continuity match (IoU >= 0.30) with last lock     │
     │ • Anti-hijack margin (intrusive size <= 1.35x locked)   │
     │ • 1.5s (22 frames) graceful loss tolerance before reset │
     └─────────────┬───────────────────────────────────────────┘
                   │ Primary User Bounding Box
                   ▼
     ┌───────────────────────────┐
     │ STAGE 2: MediaPipe FaceMesh│ ◄── Input: [1, 192, 192, 3] NHWC, normalized [0.0..1.0]
     │ 468 3D Landmarks ONNX     │     Crop: Dynamic square padded around dominant primary box
     └─────────────┬─────────────┘
                   │
                   ▼
     [ 468 3D Coordinates: (x, y, z) ]
```

### 3.1.1 UltraFace Input Resolution Benchmark & Distance Sensitivity (PoC #39)

Empirical evaluation of the spatial scaling and distance sensitivity of UltraFace RFB-320 across 3 ergonomic distance tiers:

* **Hardware & Runtime Latency (Optimized Production Release Profile):**
  * Frame Preprocessing ($1280\times 720 \to 320\times 240$ NCHW): **$0.191\text{ ms}$**
  * ONNX Inference ($[1, 3, 240, 320]$): **$5.752\text{ ms}$**
  * Total Stage 1 Execution Time: **$5.943\text{ ms}$** (Consumes only **$8.92\%$** of the $66.6\text{ ms}$ budget at $15\text{ FPS}$).

* **Distance-to-Recall Benchmark Matrix:**
  * **Tier 1 (Close / Selfie, $30\text{–}50\text{ cm}$):** **$100.0\%$ Recall** ($\text{Confidence} \ge 0.776\text{–}1.000$, $\text{Box Area} \approx 0.009\text{–}0.036$).
  * **Tier 2 (Standard Ergonomic Desk, $50\text{–}80\text{ cm}$):** **$100.0\%$ Recall** ($\text{Confidence} \ge 0.825\text{–}0.993$, $\text{Box Area} \approx 0.0015\text{–}0.008$).
  * **Tier 3 (Leaning Back, $80\text{–}120\text{ cm}$):** **$66.7\%$ Recall** ($\text{Confidence} \ge 0.706\text{–}0.784$, with ultra-compact faces $\le 0.0004$ box area approaching anchor resolution limit).

* **Architectural Decision Rule:**
  The baseline $320\times 240$ input resolution is retained as the standard configuration. The UltraFace RFB-320 graph architecture utilizes fixed static anchor grids ($4,420$ anchors configured specifically for $320\times 240$). Preprocessing overhead in compiled production Rust is virtually negligible ($< 0.2\text{ ms}$), providing zero motivation to trade off Tier 2 desk recall for fractional microsecond savings.

### 3.1.2 Adaptive Far-Field Confidence Hysteresis (Issue #55)

To prevent premature detection drops when users lean back in their chairs, stretch, or recline ($80\text{–}120\text{ cm}$, Tier 3) while remaining attentive to the screen:

* **Dual-State Detection Thresholding:**
  * **Initial Un-tracked Frame ($\text{Tracker} = \text{None}$):** Enforces strict confidence threshold ($\text{Conf} \ge 0.70$) to eliminate phantom face overlays on empty chairs or background noise.
  * **Active Session Lock ($\text{Tracker} = \text{Some}(b)$):** Relaxes the candidate ingestion threshold dynamically to $\text{Conf} \ge 0.45$.
* **Benchmark Recall Verification:**
  * Boosts Tier 3 far-field recall from **$47.2\%$ ($17/36$)** to **$63.9\%$ ($23/36$)** across a comprehensive 108-case heterogeneous stress matrix spanning 12 real-world perturbations (dim lighting, glare, sensor grain, head displacement, tilt, and lateral shadows).
  * Preserves $0\%$ false positive rate on empty or neutral background frames (maximum background score recorded: $< 0.08$).

### 3.1.3 INT8 Quantization vs FP32 Precision Benchmark (PoC #38)

Empirical evaluation of Post-Training Quantization (PTQ) on UltraFace RFB-320 (`version-RFB-320-int8.onnx` vs baseline `ultraface.onnx`):

* **Performance & Footprint Comparison (100 CPU Inference Iterations):**
  * **Disk Footprint:** FP32 $1.21\text{ MB}$ $\to$ INT8 $0.44\text{ MB}$ (**$63.9\%$ storage reduction**).
  * **Inference Latency:** FP32 $5.895\text{ ms}$ $\to$ INT8 $3.845\text{ ms}$ (**$1.53\times$ speedup**).
  * **Coordinate Fidelity:** Bounding box coordinate delta is bounded at $\le 0.0012$ normalized units ($< 0.8\text{ px}$ on $640\times 480$), ensuring identical crop framing for FaceMesh.
  * **Accuracy Trade-off:** While Tier 1 and Tier 3 confidence remain stable, dense multi-face scenarios (e.g. `4faces.png` at standard desk distance) exhibited a $\sim 12\%$ confidence drop ($0.825 \to 0.705$) due to INT8 activation quantization noise.
* **Architectural Decision Rule:**
  Retain FP32 as the default production configuration. Baseline FP32 execution ($5.9\text{ ms}$) consumes $< 2.5\%$ CPU at $15\text{ FPS}$, making the $2\text{ ms}$ speedup negligible while preserving optimal detection margins across all dense multi-face configurations. INT8 models remain validated for ultra-constrained edge profiles.

### 3.1.4 Digital RoI Zoom Crop for Far-Field Small Faces (Issue #63)

When users lean back ($80\text{–}120\text{ cm}$, Tier 3) on high-resolution streams (e.g. $1280\times 720$), downscaling the entire frame down to $320\times 240$ compresses a small face ($\sim 60\text{–}90\text{ px}$) down to $\le 15\text{ px}$, causing complete detection failure ($0\%$ recall on raw full-frame downscale).

* **Aspect-Ratio Preserving 4:3 Sub-Window Cropping:**
  - When an active session is held and target area falls below far-field threshold ($\text{Area} < 0.015$), a 4:3 aspect-ratio RoI window ($\sim 4.5\times$ face width) is cropped directly from the full-resolution raw frame buffer via zero-alloc stride sampling (`preprocess_crop`).
  - Preserves native anatomical proportions matching UltraFace's $320\times 240$ grid without non-uniform stretching.
* **Empirical Recall Gains:**
  - On standard $1280\times 720$ HD webcam streams, boosts small face detection from **$0.0\%$ ($0/15$)** to **$100.0\%$ ($15/15$)** with confidence scores $\ge 0.791\text{–}0.994$.
  - Across combined HD & VGA resolutions, lifts small face recall from **$0.0\%$** to **$83.3\%$ ($25/30$)**.
  - **Fail-Safe Fallback:** If the RoI sub-window misses (e.g. abrupt torso relocation), automatically falls back to full-frame detection within the same frame cycle ($< 12\text{ ms}$ total worst-case).

### 3.1.5 Local Eye Contrast Normalization (CLAHE) for Spectacle Glare Resilience (Issue #66)

Bright overexposure and white laptop displays (e.g. Google Docs, light-mode editors, white web pages) reflect directly off spectacle lenses (specular glare), washing out pupil contrast and eyelid contour gradients ($\nabla I$). This previously caused landmark regression errors to spike by $4\times\dots 6\times$ ($0.0056 \to 0.0425$).

* **4x4 Contextual Tile Grid Architecture (`vision/clahe.rs`):**
  - The $192\times 192$ FaceMesh input crop is partitioned into a $4\times 4$ tile grid ($48\times 48\text{ px}$ per tile, $2,304\text{ px}$ per tile).
  - Crucially, tiles $(1, 1)$ and $(1, 2)$ naturally encompass the left and right eye sub-windows in canonical facial topology, ensuring local histogram equalization is computed directly on the eye regions without distorting surrounding jawline, forehead, or cheek features.
* **Contrast Limiting & Excess Redistribution:**
  - Standard histogram spikes in homogeneous glare zones are clipped at $\text{Clip Limit} = 2.5 \times \frac{M}{256} = 23$.
  - Clipped excess is redistributed uniformly across all 256 bins before constructing normalized Cumulative Distribution Function (CDF) lookup tables.
* **Bilinear Interpolation & Stack Allocation:**
  - Tile lookup tables are pre-allocated on the stack ($[4][4][256] = 4,096\text{ bytes}$), fitting entirely inside L1 data cache with zero heap allocations in the hot vision loop.
  - Smooth bilinear interpolation between tile centers eliminates artificial block boundary artifacts.
* **Adaptive Glare Trigger:**
  - Evaluates ITU-R BT.601 integer luminance ($Y = (77R + 150G + 29B) \gg 8$).
  - If ambient lighting is balanced ($\text{Mean } Y \le 135$ and bright saturation ratio $\le 8\%$), CLAHE is completely bypassed in $< 15\ \mu\text{s}$ with $0.000$ drift on clean frames.
  - When specular glare or overexposure is detected, engages with $\alpha = 0.70$ blend.
* **Empirical Landmark Precision Gains:**
  - Improves eye landmark regression accuracy on glasses glare test fixtures by **$+26.21\%$** ($0.04252 \to 0.03137$), exceeding the $\ge 25\%$ acceptance criteria.
  - Maintains $< 0.3\text{ ms}$ processing overhead per frame in production release execution.

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
   * $\Delta t < 80\text{ ms}$: Discarded as a short sensor fluctuation.
   * $\Delta t \ge 1000\text{ ms}$: Flags `is_resting` (eyes closed in thought/nap); pauses stare warning timer.
2. **Asynchronous Left-Right Pairing Window ($1.0\text{s}$):**
   * Real human blinks exhibit natural micro-desynchronization between eyes.
   * If left and right eyes complete blinks within **$\le 1.0\text{ second}$**, the pair resolves into **$+1\text{ valid blink}$**.
   * Unpaired unilateral winks (e.g. winking only left eye) expire after $1.0\text{s}$ and are discarded.

### 3.4.1 Dynamic Independent Dual-Eye Baseline Tracking (Issue #75)

To protect users experiencing unilateral swelling (styes, blepharitis, ptosis) or natural facial asymmetry without forcing manual recalibration:
* **Independent Adaptive Open Baselines:**
  - Continuously tracks independent open-eye baselines $B_L$ and $B_R$.
  - When eye EAR $> B$, baseline adapts upwards quickly ($\alpha_{\text{up}} = 0.35$).
  - When a sample remains within 10% of baseline, adapts downwards slowly ($\alpha_{\text{down}} = 0.005$). Samples below $0.90B$ do not lower the baseline, even if the detector has not yet classified them as closed.
  - Baseline updates freeze while a closure candidate is active.
* **Calibrated Absolute Closure Cutoff (14 Oct 2026):**
  $$\text{Threshold}_{\text{eye}} = \min(\text{Threshold}_{\text{config}}, B_{\text{eye}} \times 0.90)$$
  - The calibration result now determines the effective per-eye cutoff, while the $0.90B$ cap keeps the cutoff below the open-eye baseline. For example, a calibrated threshold of $0.250$ with an open baseline of $0.300$ yields a $0.250$ cutoff instead of the prior $0.195$ relative-drop cap. With the default/configured threshold at $0.220$, that cutoff is $0.220$.
  - For a low open baseline ($B_R \approx 0.20$) and a higher configured threshold ($0.22$), the cap yields $0.180$ so the eye is not permanently classified as closed.
* **Reopening Hysteresis:** The sampled closure duration ends at the first raw EAR sample reaching the midpoint between the closure cutoff and the open baseline. Confirmation still requires two consecutive samples where both raw and smoothed EAR reach that midpoint. A one-sample reopening spike therefore does not split one closure into separate blink candidates, while EMA lag does not extend the biological-duration estimate. The duration guard still accepts sampled closures in $[80, 800]$ ms.
* **Monocular Fallback Mode:**
  - If exactly one eye is continuously closed for $\ge 5.0\text{ s}$ (e.g. medical patch), the detector automatically enters Monocular Mode, allowing blinks on the single functional eye to register immediately without pairing requirements.
  - Reopening the occluded eye seamlessly restores Binocular Mode.

### 3.4.2 Velocity Slope Sampling for Low-FPS (10–12 FPS) Micro-Blinks (Issue #64)

On low-end webcams or low-light conditions running at $\le 12.5\text{ FPS}$ ($\Delta t \ge 75\text{ ms}$), fast biological micro-blinks ($60\text{--}80\text{ ms}$) often elude static thresholding because the eye is sampled midway (dipping to $\sim 0.23$ rather than full closure $< 0.20$):
* **Temporal Derivative Tracking:**
  - Calculates per-eye velocity: $v_t = \frac{\text{EAR}_t - \text{EAR}_{t-1}}{\Delta t}$.
  - Detects rapid closing descent ($v_t \le -0.40\text{ s}^{-1}$ with drop $\ge 20\%$ below baseline), retains a velocity-only candidate across a low-EAR plateau for up to $800$ ms, and confirms reopening after two consecutive raw EAR samples reach the midpoint between closure cutoff and open-eye baseline, with smoothed EAR also reaching that midpoint before completion.
  - Velocity-only fallback is disabled and any pending slope candidate is cleared while that eye's static closure state is active. The static closure path alone owns duration validation for such closures, preventing a later EAR fluctuation during a long hold from bypassing the $800$ ms maximum. Rapid transients that never enter the static closure state remain eligible for the velocity-only path.
* **Single-Sample Temporal Guard:**
  - One closed sample is sufficient at any frame rate; a static blink still requires a sampled duration of $80\text{--}800\text{ ms}$ and confirmed reopening. Low-FPS velocity sampling supplements this path.
* **Downward Reading Gaze Immunity:**
  - When head pitch is tilted down toward desk/keyboard ($\text{Pitch} < -15^\circ$), blink triggers and velocity spikes are suppressed, preventing false positive blink counts during reading.

### 3.4.3 Opt-In Blink Pipeline Diagnostics (Issue #81)

Set `VISION420_DEBUG_EAR=1` when launching the app to confirm trace activation and emit one diagnostic line per detector update (including no-face updates). With the flag unset, diagnostic logging is disabled and detector decisions are unchanged. Camera-read, decode, and FaceMesh inference failures are reported separately. Each line includes:
* Camera frame capture time (Unix milliseconds when supplied by the backend), detector update interval, and stage timings for capture, RGB decode, UltraFace, FaceMesh preprocessing/inference, and blink detection. The capture-time field lets opt-in traces be aligned against a separately timestamped camera recording; `0` means the backend supplied no capture timestamp.
* Face/gaze gates, raw and smoothed per-eye EAR, per-eye baselines and effective thresholds.
* Per-eye closure state and sampled duration, including too-short/too-long/insufficient-sample/gaze-gated candidates, monocular mode, and the detector outcome (`NoClosureCandidate`, `ClosurePending`, `AwaitingOtherEye`, or a counted blink).

The deterministic tests exercise the production `BlinkDetector` with synthetic landmark timelines: calibration-derived cutoff behavior, baseline protection, a shallow low-FPS descent held until recovery (and expired when stale), reopening-spike debounce, normal bilateral and unpaired unilateral candidates, and a 100ms blink falling entirely between samples at a 400ms cadence. The last case is expected to be unobservable; it documents the capture-sampling limit rather than claiming successful detection. These synthetic tests do not establish real-camera accuracy.

**Pre-instrumentation smoke observation (11 Oct 2026):** The MacBook Pro camera opened at a requested `1280×720 @ 30 FPS`; the existing sparse debug trace was consistent with roughly **2.5–2.8 detector updates/s**. It recorded blink events at detector frames 121 and 174, but had no per-frame timestamps or independent blink labels. Treat this as a sampling-throughput warning, not a measured blink recall/accuracy result. Use the opt-in trace above to collect intervals and stage timings for a labeled run.

**Issue #81 live smoke (11 Oct 2026):** FaceMesh and UltraFace loaded and the camera stream opened at `1280×720 @ 30 FPS`, but the Camera Test preview remained blank and the run produced no per-update diagnostic lines. Photo Booth displayed the camera after 420vision was stopped. This run therefore does not localize the failure to capture, IPC/event delivery, or rendering, and is not a blink-accuracy result.

**Follow-up diagnosis (12 Oct 2026):** During a run attributed by macOS to OpenChamber, the trace-activation marker appeared but no detector updates followed. A three-second process sample placed the capture worker in `Camera::frame()` → `AVFoundationCaptureDevice::frame()` → `flume::Receiver::recv()` for all 1,940 samples. The run was blocked awaiting camera data, before decode or inference; no yield message was emitted. TCC logs attributed the request to `dev.openchamber.desktop` and reported its missing `com.apple.security.device.camera` hardened-runtime entitlement; `codesign` confirmed the installed OpenChamber app lacks it. This establishes an OpenChamber launch-context permission problem for that run, not camera contention.

In a subsequent live trace, frames were processed but only about **2.3 detector updates/s** were observed (frames 64–115 across 21.9s; typical `detector_dt` 0.42–0.48s). Representative stage timings were decode 165–195ms, UltraFace 80–90ms, FaceMesh preprocessing 38–43ms, and inference about 1ms; `cam.frame()` itself was about 0.1ms. Thus decode and face detection, not ONNX inference or camera acquisition, dominate this run's loop time. The blink total remained zero. At this 0.42s cadence the detector's low-FPS special case (`dt` 80–160ms) is not selected, so it requires two closed samples. Trace examples show one closure sample rejected as `RejectedInsufficientSamples` (f65–66), and a later sampled closure rejected as `RejectedTooLong` at 861–1,299ms (f67–70). Those are sampled intervals, not independently labeled biological blink durations; the evidence demonstrates a sampling/state-machine mismatch but does not establish real-world recall. No blink rules have been tuned.

**Concurrent camera reference (12 Oct 2026):** A temporary AVFoundation recorder successfully read and recorded the built-in camera while 420vision was using it; all 603 recorder callbacks were written at 29.8 FPS, with no dropped callbacks. The 20.2s aligned video/log window contained 45 detector updates (median 431ms, 2.32 updates/s) and `total_blinks` stayed at zero. Visual review showed a bilateral eyelid closure/reopen between detector updates around 1.7–2.0s; the app's adjacent updates at 1.53s and 2.00s both saw open eyes, so that closure was never sampled. Another visible closure began around 0.2s; sampled detector updates remained closed at 0.20s and 0.63s, then reopened at 1.07s and rejected the 879ms sampled duration as `RejectedTooLong` (the state machine accepts up to 800ms). A later eye-squeeze around 9.6s was only one sampled closed update and was rejected as `RejectedInsufficientSamples`; it is not labeled a confirmed natural blink. This directly confirms cadence-related false negatives in this run, not a false-positive event, and is not a representative recall estimate. The initial recorder failure was caused by the helper requesting an exclusive device-configuration lock; removing that request allowed concurrent reading without changing the app's camera ownership.

**Development-build cadence correction (12 Oct 2026):** The earlier 2.32 updates/s trace came from Cargo's unoptimized default dev profile. The existing release profile measured 10.91 updates/s (median interval 91.7ms; median total processing 20.5ms). To keep `tauri dev` usable for real-time pipeline testing, `src-tauri/Cargo.toml` now sets `opt-level = 2` for the dev profile and `nokhwa-core`. The optimized dev build measured 10.41 updates/s (median interval 96.1ms; median total processing 24.5ms), with median decode 12.9ms, UltraFace 8.5ms, FaceMesh preprocessing 0.7ms, and inference 1.9ms. No camera format, capture ownership, blink threshold, or state-machine rule changed. This verifies cadence only; the unrecorded performance run is not a labeled blink-accuracy result.

**Optimized-dev labeled validation (12 Oct 2026):** During a 20.3s synchronized webcam-only clip, the recorder wrote 607/607 frames at 29.85 FPS with no dropped callbacks. The aligned app trace had 227 updates (median interval 89.5ms, 11.17 updates/s), no face-unavailable samples, and median stage timings of 6.2ms decode, 5.6ms UltraFace, 0.6ms FaceMesh preprocessing, 1.4ms inference, and 19.0ms total. All nine `CountedBinocular` transitions aligned with visible bilateral closures in the 20-FPS visual review; no obvious false-positive count appeared in this short clip. The review also showed likely missed short closures: around 6.0s, raw EAR remained above threshold (0.211/0.209 vs 0.198/0.199), and around 14.63s raw EAR fell below threshold (0.179/0.175 vs 0.185/0.188) while smoothed EAR stayed just above it (0.192/0.189); the next update saw reopening. A further closure around 17.1s is possible but visually less certain. These residual cases point to threshold/EMA behavior after cadence improved. The short, sampled visual review is not a representative recall estimate, and no detector rule was changed during this cadence fix.

**Missed-closure mechanism investigation (14 Oct 2026):** Hardware-free reproductions exercised the actual `EarCalculator`, `EyeCalibrator`, and `BlinkDetector` implementations with controlled landmark timelines. These are synthetic signal tests, not replays of the deleted live recordings and not estimates of real-world accuracy:

* **Cutoff/calibration policy mismatch:** With open EAR 0.300 and closed-sample EAR 0.220, calibration returns 0.250, but `min(configured_threshold, baseline * 0.65)` still produces an effective cutoff of 0.195. A 500ms dip to 0.220 is missed; increasing the configured ceiling to 0.280 or the normal EMA alpha to 0.70 does not recover it. A deep 0.100 control dip is counted. This policy can override the separation learned by calibration; it is not evidence that raising the configured ceiling alone solves the live misses.
* **Immediate-rebound limitation:** At 100ms cadence, the same 0.320-to-0.220-to-0.320 signal produces one count with a one-sample dip but zero with two- or five-sample dips. The velocity-spike flag is cleared on the first following sample that is neither another qualifying descent nor a qualifying rebound. A plateau above the static closure cutoff therefore loses its recovery candidate. This is a confirmed limitation of the current micro-blink fallback, not proof that every live miss used that path.
* **Open-baseline contamination mechanism:** An unrecognized 50s low-EAR plateau at 0.220 stays classified `Open` and reduces the baseline from 0.300000 to 0.226532, moving the cutoff from 0.195000 to 0.147245. The baseline freeze depends on the detector's own closed classification, not an independent open-eye observation. The mechanism is reproducible; its contribution to a specific live session requires retained per-update evidence.
* **Bilateral gating:** A measured drop from open L/R 0.300/0.310 to 0.200/0.190 yields a completed right-eye candidate but no left-eye candidate, so the total stays zero. This agrees with the independently reviewed unpaired live closure, without establishing the source of the asymmetric landmark error.
* **EMA and duration boundaries:** Downward smoothing already uses alpha 0.85; normal-alpha variants 0.40 and 0.70 both map a raw 0.220 descent from 0.300 to smoothed 0.232. The normal alpha primarily changes recovery here. An 800ms deep raw-EAR closure can remain below cutoff for an additional recovery sample and be rejected, whereas a 700ms control is counted. This duration effect is secondary and does not explain the latest live periods that stayed `Open/Open`.
* **Bounded relaxation experiment:** Changing only the experimental relative drop from 35% to 25% recovers a synthetic 500ms dip from 0.300 to 0.210 (zero counts to one). This demonstrates sensitivity to the cutoff policy, not improved precision: the same measured signal has no independent blink-versus-squint/noise label, and the previous recorded A/B review already showed duplicate counts under more permissive variants.

**Production detector follow-up (14 Oct 2026):** The effective production cutoff is now `min(configured/calibrated threshold, baseline * 0.90)`, and downward baseline adaptation is limited to samples at or above `baseline * 0.90`. Debug A/B variants retain the earlier relative closure-cutoff formulas but share the new baseline-protection guard. The low-FPS descent candidate persists through a plateau for at most 800ms and is cleared whenever a static per-eye closure is active; only closures that never enter that state use velocity-only completion. Velocity-only completion now requires two consecutive raw EAR samples above the reopening midpoint and current smoothed EAR to have reached it, avoiding EMA recovery lag without counting a partial rebound. Static closure candidates require two consecutive samples where both raw and smoothed EAR reach the midpoint; duration is measured to the first raw sample crossing that midpoint, preserving the 800ms guard.

The latest synchronized live recording (38.5s; 1,143 video frames, zero reported drops; 391 detector updates) exposed a false count during an extended closure: event 16 was `CountedBinocular` while the detector's recorded closure durations were 5,010ms and 3,342ms. A later long closure was rejected as `RejectedTooLong`. The production detector regression recreates the bypass with a multi-second static closure, a brief deeper EAR dip, and partial recovery: it failed before this guard and passes afterward with no blink count. Nine hardware-free sampling tests pass, including calibration, baseline protection, shallow low-FPS recovery, stale-candidate expiry, reopening-spike rejection, duration boundaries, and the long-closure velocity-bypass regression. The live total of 18 is not a validated accuracy score because the clip did not provide reliably identifiable phase labels. Detailed video and trace artifacts were deleted after review, retaining the session summary metrics. These changes address reproduced mechanisms, not a claim of improved real-camera precision/recall; validate against a clearly labeled live test before treating tuning as successful.

**Reopening confirmation follow-up (14 Oct 2026):** In the 58.8s synchronized session `session-XjnHU4` (1,756 video frames, zero drops, 611 aligned detector updates), a count at 44.675s occurred while the reviewed video still showed closed eyes. Smoothed EAR was 0.230/0.228: above the 0.220 static cutoff and the velocity path's former 75%-of-baseline recovery floor, but below the reopening midpoint (~0.257/~0.258). Velocity-only completion now requires two consecutive raw EAR samples above the reopening midpoint and smoothed EAR to reach it before completion. A synthetic shallow-plateau regression verifies that 0.24 EAR is not enough to count and that a subsequent confirmed reopening is. An existing 12 FPS synthetic micro-blink sweep also verifies that one-frame filter lag on reopening does not prevent a sampled blink from counting. The same video showed a bilateral ~2s hold correctly remaining uncounted and marked resting; it also showed another closure that stayed `Open/Open` because EAR remained above cutoff. That separate threshold/detection limitation is not fixed by this reopening guard. Video review sampled frames around events and is not a complete frame-by-frame accuracy estimate; only the summary metrics are retained after review.

### 3.4.4 Integrated Development Reference Recording (macOS)

Run `npm run dev:record` to build, ad-hoc sign, and launch a standalone debug `.app` with recording enabled. The launcher embeds the `com.apple.security.device.camera` entitlement; the app also has its own macOS TCC identity and `NSCameraUsageDescription`. Approve the macOS camera prompt for **420vision**. Do not use `tauri dev` launched as a child of OpenChamber for live camera tests: macOS attributes that child to OpenChamber, whose hardened-runtime identity lacks the camera entitlement. Release builds do not include the debug recorder.

While this mode is enabled, `capture/reference_macos.m` owns one AVFoundation camera session. Its sample callback writes the camera-rate stream to an H.264 `AVAssetWriter` and makes only the latest RGB frame available to the Rust detector. If a video write fails, the failure is logged once and camera frames continue to the detector; finalization also reports the failed recording. The detector waits at most 200ms per capture read, so Monitoring OFF can stop the stream and finalize the video even if the camera stops delivering frames. No second camera owner, recorder subprocess, microphone, or raw-frame IPC is introduced. The video can run at the camera's actual cadence; detector/log updates remain at the existing inference cadence. Detector thresholds and state-machine rules are unchanged.

Monitoring ON opens the selected camera and starts a private `debug-recordings/session-*/` directory in the workspace. Monitoring OFF, camera switching, camera read failure, and normal app Quit stop the AVFoundation session, finalize the video, and flush the session's detector log. Hiding the window or closing the Camera Test tab does not stop monitoring or recording. A subsequent ON creates a new session. `Ctrl+C`, forced termination, and app rebuilds are not clean Quit; turn monitoring OFF and wait for the `SAVED` line before using them.

Each session contains `video.mov`, `frames.csv`, `detector.log`, and (on successful finalization) `summary.json`. The CSV columns are `frame_index,video_seconds,capture_wall_ms`. Video presentation time is derived from the same frame's capture wall-clock timestamp; it can be aligned with `capture_wall_ms` in the detector log. Encoder backpressure drops video frames rather than building an unbounded queue; the summary records written frames, dropped frames, and first/last capture times. A failed or empty recording is logged as a failure, never as `SAVED`.

For a disposable same-frame EAR calibration, run `npm run dev:record:ab`. This debug-only mode logs four independent detector traces against each identical landmark update: the current production baseline (calibrated absolute cutoff capped at 90% of baseline, EMA alpha 0.40), threshold variant (25% relative drop, alpha 0.40), EMA variant (35% relative drop, alpha 0.70), and combined relative-drop variant (25% relative drop, alpha 0.70). Each line is tagged with `variant=baseline`, `threshold25`, `alpha70`, or `threshold25_alpha70`; all variants share the configured absolute threshold and all non-threshold detector rules. Only the baseline supplies app/UI metrics and persisted blink counts. The experimental constructor and parallel detectors are compiled into debug builds only; this mode does not change release defaults. Historical A/B totals recorded before the production-policy update are not a direct baseline for new runs. Review recorded video against each variant's count transitions and false positives before drawing accuracy conclusions.

**Same-frame EAR calibration (14 Oct 2026):** The synchronized 58.18s capture contained 1,739 video frames (29.87 FPS, no drops) and 609 detector updates per variant (10.47 updates/s); every detector timestamp matched a written video-frame timestamp exactly. Final count-event totals were baseline 12, threshold25 25, alpha70 26, and combined 32. In the visually reviewed 47.10–48.00s interval, the video showed two bilateral closure episodes: baseline emitted two counts but missed the first and duplicated the second; threshold25 emitted four, alpha70 three, and combined four. The more permissive threshold did recover the first brief closure, but all higher totals are not evidence of higher blink accuracy. The video did not provide clearly visible phase markers or an unambiguous single labeled 1.5s hold, so this run cannot support full-protocol recall/precision claims. Production defaults and state-machine timing were not changed.

**Same-frame EAR follow-up (14 Oct 2026):** The 37.407s capture contained 1,115 video frames (29.78 FPS, no drops) and 384 detector updates per variant (10.27 updates/s); all 384 timestamps per variant matched written video-frame timestamps exactly. Final count-event totals were baseline 4, threshold25 19, alpha70 7, and combined 23. A contact sheet sampled every 0.5s did not show the requested 3/1/2-finger phase markers; visible eye squeezes therefore cannot be assigned confidently to the instructed normal-blink, short-squeeze, and hold phases. The large count spread is not evidence that a more permissive variant is more accurate. This run cannot establish per-phase recall/precision or whether the detector is correct; production defaults and state-machine timing remain unchanged.

The session directory is private to the user and ignored by Git. Recordings contain sensitive webcam imagery and must be deleted along with detailed traces after visual review; retain only non-image summary metrics. These debug-only recordings are the explicit exception to the normal no-video-persistence policy.

Hardware-free regression tests feed synthetic RGB frames through the native writer, decode the completed video, verify timestamp/log persistence, exercise camera-release/restart session separation, test bounded no-frame waits, and reject empty sessions. These tests verify recording mechanics only; live camera FPS and real-world blink accuracy still require a recorded, visually reviewed session.

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
* **20-Second Break Verification State Machine (Issue #71 & #73):**
  * Upon reaching $1200\text{ s}$ ($20\text{ min}$), `PresenceTimer` transitions into `BreakPhase::BreakPending` with an active $20\text{ s}$ countdown (`break_remaining_seconds`).
  * Triggers initial break prompt notification and chime (`play_break_chime`).
  * **Net Rest Accumulation with Freeze-on-Screen (Issue #73):**
    - The 20-second countdown advances **only** when the user is genuinely resting (turned away, gazing upward, or out of frame).
    - When the user stares at the monitor/code (`is_face_present == true`), the countdown **instantly freezes/pauses**.
    - If the user ignores the break and continues staring at the screen for $> 60\text{ s}$, the break times out and rolls back 5 minutes (snooze).
  * **3D Head Pose Gaze Gate (Issue #73):**
    - Evaluates dual-axis 3D orientation:
      1. *Horizontal Yaw:* $|X| \ge 20^\circ$ (`yaw_ratio > 0.35`, turned left or right toward room/window).
      2. *Vertical Pitch:* Upward tilt $\ge 15^\circ$ (`pitch_ratio \ge 0.15`, gazing at ceiling or high window).
      3. *Downward Gaze Rejection:* Looking downward at keyboard/desk/phone is explicitly rejected from resting credit.
    - User qualifies as resting if turned away, looking upward, or completely absent from frame.
  * Upon completing $20\text{ s}$ of verified net rest, fires `play_break_completed_chime`, resets `active_screen_seconds = 0.0`, records `breaks_completed` in `daily_analytics`, and transitions back to `BreakPhase::Monitoring`.
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
| `get_config` | Invoke (Pull) | None | `AppConfig` | Fetches active runtime configuration including current `ear_threshold` and `keep_awake_enabled`. |
| `update_config` | Invoke | `{ sound_enabled?: bool, stare_alert_enabled?: bool, keep_awake_enabled?: bool }` | `AppConfig` | Atomically updates and persists user settings from UI toggles. |
| `hide_window` | Invoke | None | `()` | Hides the popover window to the system tray. |
| `quit_app` | Invoke | None | `()` | Performs deterministic teardown, drops hardware handles, and exits cleanly. |
| `camera-sandbox-frame`| Event (Push) | `CameraFrameDto` | Stream | While the Camera Test tab is open, emits a JPEG thumbnail capped at 320×180 and landmark arrays (up to 15 FPS; throttled to 5 FPS when no face is detected). |

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
  total_blinks: number;                             // Authoritative cumulative detector count
  blink_diagnostics?: BlinkDiagnosticsDto | null;   // Additive, versioned feedback; null if no detector update
  eye_landmarks: Array<{ x: number; y: number }>;   // 16 points (re-mapped to full frame [0..1])
  face_landmarks: Array<{ x: number; y: number }>;  // 75 selected feature points (jawline, brows, nose, lips)
  face_mesh_landmarks?: Array<{ x: number; y: number }>; // 468 sandbox wireframe points; absent in older payloads
  image_data_base64: string | null;                 // 320x180 JPEG thumbnail
  yaw_deg: number;                                  // Head horizontal turn angle (degrees)
  pitch_deg: number;                                // Head vertical pitch angle (degrees)
  roll_deg: number;                                 // In-plane head tilt angle (degrees)
  distance_cm: number;                              // Estimated distance to camera (cm)
  is_resting_gaze: boolean;                         // Meets optical distance rest criteria
}

interface BlinkDiagnosticsDto {
  version: 1;
  decision: "face_unavailable" | "looking_away" | "no_closure_candidate"
    | "closure_pending" | "awaiting_other_eye" | "counted_binocular" | "counted_monocular";
  left_status: EyeBlinkStatusDto;
  right_status: EyeBlinkStatusDto;
  smoothed_left_ear: number | null;
  smoothed_right_ear: number | null;
  threshold_left: number | null;
  threshold_right: number | null;
  left_closure_ms: number;
  right_closure_ms: number;
  is_resting: boolean;
}

type EyeBlinkStatusDto = "open" | "closure_pending" | "candidate_completed"
  | "rejected_too_short" | "rejected_too_long" | "rejected_insufficient_samples"
  | "rejected_downward_gaze";
```

**Camera Test feedback and count (14 Oct 2026):** Test Blinks is the backend's cumulative `total_blinks` minus the view's reset offset, not a frontend rising-edge count of `is_blinking`. Entering Camera Test snapshots `get_status.total_blinks_today` before enabling preview events; out-of-order status responses from previous entries are discarded by a per-entry request generation. Reset changes only the local offset. Repeated totals cannot add counts, consecutive counted updates remain distinct, and skipped preview events are recovered by the next cumulative total. The worker's cumulative total is monotonic; lower preview totals queued before the status snapshot are ignored rather than interpreted as new counts. A new app process initializes a new view counter. The preview labels its existing average EAR as raw; below it, version-1 `blink_diagnostics` displays the actual smoothed per-eye EAR and effective cutoffs, current detector state, and the last counted/not-counted result. Long closures use neutral "not counted as a blink" wording, with active closures of at least one second shown as resting. The last result remains visible between transient events and is cleared on test entry or Reset. Closure durations are detector-sampled durations, not independently measured biological durations. Missing feedback is shown as unavailable rather than inferred from raw EAR. The DTO is mapped in `capture/frame.rs`, is emitted only with the existing Camera Test preview, and is not persisted. Detector thresholds, smoothing, temporal guards, velocity recovery, and pairing rules are unchanged. Run `node --experimental-strip-types --test tests/camera-feedback.test.mjs` for hardware-free UI logic regressions.

**Live Camera Test / recording alignment (14 Oct 2026):** In session `session-XIeBsu`, 1,060 video frames were written over 47.701s (22.20 FPS, no drops); 503/503 unique detector-update timestamps matched a recorded video-frame timestamp exactly (10.55 updates/s, median interval 100ms). The backend count reached four at video times 8.819s, 23.149s, 23.912s, and 27.086s; each increment aligned with a visible closure/reopening period. Visual review also found clear squeezed-eye periods that did not increment: at 13.75–14.00s only the right-eye EAR crossed its cutoff, leaving the detector waiting for the other eye; at 15.00–15.25s the eyes were visibly closed while the closest sampled smoothed EAR (0.216/0.211) remained above cutoffs (0.194/0.198); at 16.25–17.00s the sampled status remained `Open/Open`. The requested hand phase markers were not clearly visible, so these events cannot be assigned to the intended normal/squeeze/hold phases. This confirms the recorder and logs are aligned and demonstrates missed closures in this individual run, but is not a population recall estimate. The Camera Test counter now follows the backend cumulative count. Production thresholds and detector rules were not changed.
`face_mesh_landmarks` is an additive sandbox-only field with a Serde default for older payloads. The frontend renders the official MediaPipe 468-point tessellation (1,322 unique edges) as a low-opacity wireframe, then overlays lighter feature contours and a cyan/red eye outline. The static topology and its Apache-2.0 attribution live in `src/faceMeshTopology.ts`. The daemon sends these normalized points only while the sandbox tab is viewed; the inference model, detector, and production monitoring path are unchanged.

Sandbox frame emission is gated by both Camera Test tab selection and main-window focus. Losing focus stops preview work; regaining focus resumes it only when Camera Test remains selected.

---

## 7. Resource & Power Budget SLAs

* **Process Resident Set Size (RAM):** $< 30\text{ MB}$ (typically $18\text{–}24\text{ MB}$ under active inference).
* **CPU Consumption (Active Monitoring):** $< 3\%$ on modern Apple Silicon / Intel Core i5.
* **Adaptive Frame Pacing (Issue #40):** 
  - **Active Gaze State:** Throttled to $\sim 15\text{ FPS}$ (`67ms` sleep interval) providing $100\%$ blink detection parity.
  - **Away / Idle State:** Dynamically throttled down to $\sim 5\text{ FPS}$ (`200ms` sleep interval) cutting idle CPU consumption to $< 0.8\%$ and preserving laptop battery.
* **Presence-Aware Smart Keep-Awake (Issue #62):**
  - **Active Attention:** When the user is looking at the screen (`is_face == true`), native OS sleep assertion (`IOPMAssertionCreateWithName` on macOS, `SetThreadExecutionState` on Windows) prevents display sleep, allowing reading articles or documents without touching mouse/keyboard.
  - **Instant Release:** The assertion is immediately released when the user looks away or leaves (`is_face == false`), allowing normal OS energy timeouts to resume without battery waste. Configurable via `keep_awake_enabled` toggle in Settings.
* **Zero Video Leakage (Normal/Release Operation):** Camera frames are processed in memory and are not persisted as video. The explicit macOS debug reference mode in Section 3.4.4 saves local video for human-reviewed validation; it does not upload footage.

## 8. Desktop Window Activation (Tauri/macOS)

The main webview starts hidden so 420vision can run from the menu bar without opening its dashboard. On macOS, a Dock reopen event always restores, shows, and focuses the main window. The menu-bar Open command and icon's left-click do the same; neither toggles visibility. The header Close action hides the window to the tray.

## 9. Local Development Server Ports

The 420vision Vite development server and Tauri `devUrl` use port `1422`; Vite HMR uses port `1423` when `TAURI_DEV_HOST` is set. Keeping these ports aligned and separate from the default Tauri development port (`1420`) prevents local projects from loading another app's frontend when multiple `tauri dev` sessions are running.
