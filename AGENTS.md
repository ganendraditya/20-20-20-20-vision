# 420vision: Engineering Laws, Design Architecture & Anti-Slop Principles

> Unified engineering discipline synthesized from **Prateek's 20 Engineering Laws**, **Core System Architecture Principles** (derived from `subtitle-translator`, `not-notebooklm`, and `koma`), and **Karpathy's Execution Guidelines**, specifically tailored for **420vision** (Local Edge Computer Vision, Native Rust/Tauri Daemon, Real-Time Hardware Processing).

---

## Part 1: Core Architectural Principles (SOLID, Orthogonality, & Boundaries)

These principles govern all Rust backend, ONNX inference, and Tauri UI implementations in this repository:

### 1. Ortogonalitas (Orthogonality & Loose Coupling)
Modules must be designed to stand independently without hidden side-effects:
- Modifying the webcam capture layer (`capture`) must **never** break or leak into the FaceMesh ONNX engine (`vision`).
- Updating the EAR math or blink state machine (`detector`) must **never** mutate UI state directly.
- Changes in the Tauri webview UI (HTML/CSS/TS) must **never** alter domain contracts or backend Rust state.
- Database persistence (`storage`) must remain completely orthogonal to the live frame loop.

### 2. Dependency Inversion (DIP) & Program Against Abstractions
- High-level orchestration must depend strictly on abstract contracts, not raw concrete internals.
- UI layer only consumes versioned, typed DTOs (`AppStatus`, `DailyCompliance`) via explicit Tauri IPC commands.
- Never pass raw camera byte arrays or raw ONNX tensor outputs directly to UI or storage layers.

### 3. Encapsulate What Varies
- Platform-specific camera handling (AVFoundation on macOS vs Media Foundation/DirectShow on Windows) is strictly isolated inside `capture/`.
- Platform-specific notifications (macOS `osascript` vs Windows Action Center) are strictly isolated inside `notifier/`.
- The core eye-tracking math, blink detector, and 20-min presence logic remain 100% platform-agnostic.

### 4. Tell, Don't Ask & Law of Demeter
- Components command collaborators to execute domain operations rather than inspecting their internal state to make procedural decisions externally.
- *Good:* `detector.update(landmarks)`
- *Avoid:* External code reaching into detector internals, manually checking frame buffer lengths, subtracting timestamps, and mutating blink counters from outside.

### 5. Single Responsibility (SRP) & Interface Segregation (ISP)
- Every file, struct, and module has exactly one reason to change:
  - `capture`: Hardware device stream lifecycle and buffer fetching.
  - `vision`: Tensor preprocessing, ONNX model loading, and 468 3D landmark extraction.
  - `detector`: Geometric EAR calculation, EMA smoothing, and temporal blink state machine.
  - `timer`: 20-minute screen presence accumulation and auto-reset logic.
  - `storage`: SQLite analytics queries and configuration persistence.
  - `ui`: Presentation and user interaction.

### 6. DRY (Don't Repeat Yourself) Without Indirection
- Maintain single sources of truth for landmark indices (e.g. eye landmark pairs), EAR formulas, and DTO contracts.
- Avoid duplicate calculation logic across modules.
- Avoid empty re-exports or useless pass-through layers that add navigation friction without providing abstraction value.

### 7. YAGNI (You Aren't Gonna Need It)
- Build strictly for the active task's acceptance criteria.
- No speculative configurations, unused trait abstractions for single-use structs, or premature plugin hooks until actually needed.

### 8. Reversibility & Fail-Safe Defaults
- All background threads, camera streams, and state timers must have deterministic teardown and graceful cancellation paths.
- If camera capture fails or is snatched by Zoom, the system must fail gracefully into a safe `Paused` state without panicking or leaking hardware handles.

### 9. Clean Code & Surgical Edits
- Functions must remain concise, strongly typed, and well-bounded.
- Touch only what you must. Every modified line must trace directly to the active issue requirements. Never reformat or rewrite adjacent working code unnecessarily.

---

## Part 2: Engineering Laws (Tailored for 420vision)

### 1. Gall's Law
> *"A complex system that works is invariably found to have evolved from a simple system that worked. A complex system designed from scratch never works."*
* **In 420vision:** Build the deterministic mathematical EAR pipeline first. Once blinks are tracked reliably, evolve features incrementally.

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

## Part 3: UI & Craftsmanship Filter (Anti-Slop)

Whenever building or refining the Tauri popover UI (HTML/Tailwind/TS):
- **Purpose over decoration:** No random glows, generic pill badges, or template fluff without a clear functional reason.
- **Resilience:** All states must be handled gracefully (`Monitoring Active`, `Paused (Away)`, `Camera Busy`, `No Camera Detected`).
- **Conditional Resource Usage:** When the UI window is minimized or closed in the tray, frame rendering to the webview stops completely.
- **Functional completeness:** No dead buttons, non-functional toggles, or dummy placeholders.

---

## Part 4: Development & Review Protocol (The OCR Rule)

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
