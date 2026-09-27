# 420vision: Engineering Laws & Anti-Slop Principles

> Inspired by Prateek's 20 Engineering Laws, tailored specifically for **420vision** (Local Edge Computer Vision, Native Rust/Tauri Daemon, Real-Time Hardware Processing).

---

## 1. Core Engineering Laws for 420vision

### 1. Gall's Law
> *"A complex system that works is invariably found to have evolved from a simple system that worked. A complex system designed from scratch never works."*
* **In 420vision:** Don't attempt to build an end-to-end multi-modal deep neural network from day one. Build the deterministic mathematical EAR pipeline first. Once blinks are tracked reliably, evolve features incrementally.

### 2. Postel's Law (The Robustness Principle)
> *"Be conservative in what you send, liberal in what you accept."*
* **In 420vision:**
  * **Liberal Inward:** Accept raw, noisy video frames from low-end webcams, poor lighting, tilted faces, and reflective glasses without throwing panics.
  * **Conservative Outward:** Send strictly formatted, validated numerical coordinates, sanitized IPC state payloads, and bounded EAR metrics to the UI.

### 3. Murphy's Law & Designing Failure Paths
> *"Anything that can go wrong will go wrong. Design the failure path, not just the happy path."*
* **In 420vision:**
  * External apps (Zoom, Teams, FaceTime) will snatch the webcam at unexpected times. We must **Always Yield** gracefully and never crash.
  * Zero-camera laptops and hot-plugged USB cameras must be handled without panics.
  * System sleep/lock events must immediately freeze timers and release hardware.

### 4. Little's Law & Frame Pacing ($L = \lambda W$)
> *"Load drives latency."*
* **In 420vision:** Video frame queues must never buffer stale frames. If the inference engine takes 30ms to process a frame, the webcam capture loop must drop intermediate frames rather than queueing them up. Processing must always operate on the *latest* frame ($L \approx 1$).

### 5. Amdahl's Law (The Serial Bottleneck)
> *"The serial part caps your speedup."*
* **In 420vision:** Inference tensor calculation may be fast, but serial pixel copying and UI thread blocking will ruin the app. Camera capture and ONNX inference live in a dedicated background worker thread; the UI thread only receives lightweight event updates.

### 6. Wirth's Law
> *"Software slows down faster than hardware speeds up."*
* **In 420vision:** We strictly avoid heavy runtimes (no Chromium/Electron, no multi-hundred-megabyte Python wheels). Compiled Rust + Tauri native webview ensures $<30$ MB RAM and $<3\%$ CPU usage. Never let background utility bloat devour user battery.

### 7. Hyrum's Law
> *"Every observable behavior becomes someone's dependency."*
* **In 420vision:** The IPC contracts between Rust backend and frontend webview (`get_status`, `toggle_monitoring`, etc.) are explicit public APIs. Any change in data shape breaks UI state polling. Keep DTO structs versioned and strict.

### 8. Pareto Principle (80/20 Rule)
> *"20% of the paths cause 80% of issues."*
* **In 420vision:** 80% of detection failures come from:
  1. Lighting glare bouncing off spectacles.
  2. Looking down at phones/books with extreme downward eyelids.
  Focus engineering effort and smoothing filters (EMA) on these specific edge conditions.

### 9. Parkinson's Law
> *"Unbounded scope always gets consumed."*
* **In 420vision:** Resist feature creep. This is a dedicated dry eye & blink assistant, not a full facial recognition security suite. Every prospective feature must first reside in the **Ideas** Kanban column before touching production code.

### 10. Leaky Abstraction
> *"The abstraction you ignore will fail at 3am."*
* **In 420vision:** ONNX Runtime (`ort`) and webcam drivers (`nokhwa`) are not magical black boxes. Hardware drivers fail, DirectShow / AVFoundation handle locks leak if not explicitly dropped. Always write explicit cleanup routines and error boundary guards.

---

## 2. Development & Review Protocol (The OCR Rule)

To maintain absolute code quality and prevent regressions:

1. **Dedicated Issue Branches:**
   * Every task must be developed on an isolated branch: `feature/<task-name>` or `fix/<bug-name>`.
   * Never commit unreviewed feature code directly to `main`.
2. **Pre-Merge Open Code Review (OCR):**
   * Before opening or merging any Pull Request, run:
     ```bash
     ocr review --audience agent --background "<task context>" --commit HEAD
     ```
   * All `critical`, `high`, and actionable `medium` severity findings must be resolved on the branch.
   * Only merge to `main` when review passes cleanly with 0 actionable defects.
3. **Automated Testing Gate:**
   * Rust unit and integration tests (`cargo test`) must pass cleanly before any merge.
