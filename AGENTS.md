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
- UI layer only consumes versioned, typed DTOs (`AppStatus`, `DailyCompliance`, `CameraFrameDto`) via explicit Tauri IPC commands.
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
  - `capture`: Hardware device stream lifecycle, buffer fetching, and frame throttling.
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

## Part 2: Engineering Execution Guidelines (Karpathy)

### 1. Think Before Coding
- **State assumptions explicitly.** If uncertain, ask rather than guess.
- **Surface tradeoffs.** If multiple approaches exist, compare them objectively.
- **Push back when warranted.** If a simpler, more reliable approach exists, speak up.
- **Stop when confused.** Never write speculative code based on ambiguous requirements.

### 2. Simplicity First
- **Minimum code that solves the problem.** Nothing speculative.
- No features beyond what was requested.
- No abstractions for single-use code.
- If 200 lines could be 50, simplify it.
- Ask: *"Would a senior engineer say this is overcomplicated?"* If yes, simplify.

### 3. Surgical Changes
- **Touch only what you must.** Clean up only your own mess.
- Don't "improve" adjacent code, comments, or formatting unnecessarily.
- Don't refactor things that aren't broken.
- Every changed line must trace directly to the user's request.

### 4. Goal-Driven Execution
- **Define success criteria. Loop until verified.**
- Transform tasks into verifiable criteria (write test/smoke script, verify, commit).
- Loop independently against tangible feedback before declaring done.

---

## Part 3: Agent Guardrails & Operating Rules (Anti-Sycophancy & Zero-Assumption)

Derived from `koma` and `not-notebooklm`:

### 1. Zero Hallucination & Anti-Sycophancy in Code Reviews
- **Never claim a technical bug/error based on assumption or memory alone:** Verify before reporting. Always verify against concrete code, tests, or official documentation.
- **No Sycophancy (Stand on Facts, Not Pleasing the User):**
  - Do NOT reflexively fold, apologize, or claim "I was wrong" just because the user challenges a finding.
  - Re-verify facts objectively: if a point is technically correct, stand by it with proof; if it is genuinely flawed, explain transparently why without subservient fluff.
- **Differentiate facts from suggestions:** Distinguish hard breaking bugs from constructive styling/resilience suggestions.

### 2. Don't Assume — Ask First (Communication & Clarification)
- **Do not fill in the blanks with silent assumptions:** When requirements, implementation details, edge cases, user preferences, or error handling behaviors are ambiguous or unspecified, **STOP and ASK the user**.
- **Proactive confirmation on gaps:** Highlight architectural trade-offs, edge cases, or breaking implications before writing code.
- **Ask via the `question` tool:** Use the interactive question tool with clear options rather than guessing the user's intent.
- **Confirm before irreversible actions:** Never delete files, change major architectural patterns, rewrite public contracts, or auto-merge without explicit confirmation.

### 3. Strict Git Invariant: FORBIDDEN AUTO-COMMIT / AUTO-PUSH / NAKED MERGE
- **Strict User Authorization Gate:** The agent is **STRICTLY FORBIDDEN** from running `git commit`, `git push`, creating tags, or merging branches autonomously without an **explicit instruction** from the user ("commit now", "push now", "ok commit", etc.).
- **Mandatory Pull Request Lifecycle (No Naked Merges into `main`):**
  - Every non-trivial feature, refactor, or bugfix **MUST** transition through a formal GitHub Pull Request (`gh pr create`) before being merged into `main`. Direct or naked branch merges into `main` without an associated PR are strictly forbidden.
  - **Standard Engineering Lifecycle:**
    1. **Branch & Implement:** Develop on an isolated branch (`feature/<name>-#<id>`, `fix/<name>-#<id>`).
    2. **Local Verification:** Run full test suites (`cargo test`, `npm run tauri build -- --no-bundle`).
    3. **Push & Open PR:** Push the feature branch and open a PR via `gh pr create` linking the relevant issue (`Closes #<id>`).
    4. **AI Code Review on PR:** Execute `ocr review --audience agent --from main --to <branch>` to review the PR diff cleanly.
    5. **Dialectical Verification & Scorecard:** Present structured findings to the user (Confirmed Bugs vs False Positives). Never apply fixes silently or sycophantically.
    6. **Surgical Resolution on Branch:** With user approval, apply verified fixes with surgical precision on the branch.
    7. **User Authorization Gate:** Present the clean PR status and await explicit user instruction to merge.
    8. **Merge & Sync:** Merge via `gh pr merge --squash --delete-branch`, synchronize `docs/TECHNICAL_STACK_AND_PIPELINES.md`, and pull updated `main`.
- **Report Status First:** Upon task completion, present a concise summary of changes, test suite results, and linter status, and await user instruction. Inquiring for confirmation ("Would you like to commit?") is permitted, but executing commit/push without explicit confirmation is prohibited.
- **Living Architecture Specification Synchronization:** Whenever system architecture, engine dependencies, or core operational workflows are committed or merged into `main`, `docs/TECHNICAL_STACK_AND_PIPELINES.md` **MUST BE SYNCHRONIZED** within the same PR so the documentation remains an accurate reference for continuous study.
- **Strict English Consistency Across Repository Artefacts:** All documentation files (`*.md`), technical specifications, GitHub Issues, Pull Request descriptions, Git commit messages, and GitHub Release notes **MUST BE WRITTEN EXCLUSIVELY IN CLEAR, CONCISE ENGLISH**. Maintain strict language consistency across all repository artefacts for international open-source parity.

### 4. Code Review Scientific Verification Protocol (Anti-Hallucinated Findings)
When conducting AI Code Reviews (via `ocr review`, dual LLM evaluations, or manual diff inspection), the agent **MUST NOT ACCEPT REVIEWER FINDINGS AT FACE VALUE OR ACT AS A SYCOPHANT TO REVIEW BOTS**. Follow a mandatory, evidence-backed verification protocol before touching any code:

- **Mandatory User Presentation Before Applying Changes:**
  - The agent is **STRICTLY FORBIDDEN** from unilaterally modifying code, committing, or merging fixes immediately after receiving automated review comments without first presenting the findings dialectically to the user.
  - Present a structured scorecard: categorize items into **Hard Blockers / Confirmed Bugs** vs **False Positives / Rejected Claims** vs **Architectural Optimizations**, complete with reproduction proof.

- **Step 1: Problem Validity Verification (Is this a genuine defect or a hallucination/misunderstanding?):**
  - **Never Assume Validity:** Treat reviewer comments with healthy skepticism. LLM reviewers frequently misread token-truncated code, misunderstand project conventions, or flag stylistic non-issues as critical bugs.
  - **Define the Concrete Failure Scenario:** *"Under what exact inputs, camera angles, or concurrency state does this failure occur, and what is the exact stack trace or measurable impact?"*
  - **Execute an Empirical Reproduction Script:** Run a minimal terminal script, synthetic frame assertion, or benchmark harness to test the failure hypothesis.
  - **Classification:**
    - If reproduction confirms an actual error, crash, memory leak, or measurable accuracy degradation: classify as **CONFIRMED REAL ISSUE** with log/terminal evidence.
    - If reproduction passes cleanly, or the claim is based on truncated files, obsolete syntax, or false assumptions: reject the finding dialectically with proof as **FALSE POSITIVE / REJECTED**. Do not modify code for rejected items.

- **Step 2: Solution Validity & Orthogonality Verification:**
  - **Never Blindly Apply Suggested Diff:** Review bot fix suggestions are often naive, incomplete, or break neighboring invariants. Critically evaluate whether the suggested fix genuinely addresses the root cause or just silences a linter.
  - **Surgical Implementation:** Apply the verified solution with minimal footprint.
  - **Dual Verification:**
    1. Re-run the reproduction script from Step 1 to verify the defect is genuinely eliminated.
    2. Run full test suites (`cargo test`, `npm run tauri build -- --no-bundle`) to verify zero regressions across neighboring systems.

### 5. Execution, Timeouts & Heavy Workflows (CLI & Review)
- **Harness Shell Timeout Vigilance:** The default harness shell timeout (120s / 2 minutes) is strictly inadequate for heavy tasks, reasoning models (Gemini Pro, Claude Sonnet/Opus), or large builds.
  - When invoking `ocr review`, builds, or test suites, **explicitly pass `timeout: 300000` to `600000` (5–10 minutes)**. Never let default 120s cutoff waste tokens or interrupt reasoning mid-stream.
  - For `ocr review`, pass `--effort low` and `--exclude 'src-tauri/tests/*,README.md'` to prevent unbounded roundtrips while keeping token usage bounded.
  - Always direct heavy CLI outputs to a persistent file (`--output <path>`) so results are safely preserved.

---

## Part 4: Core Engineering Laws (Tailored for 420vision)

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

## Part 5: UI & Craftsmanship Filter (Anti-Slop)

Whenever building or refining the Tauri popover UI (HTML/Tailwind/TS):
- **Purpose over decoration:** No random glows, generic pill badges, or template fluff without a clear functional reason.
- **Resilience:** All states must be handled gracefully (`Monitoring Active`, `Paused (Away)`, `Camera Busy`, `No Camera Detected`).
- **Conditional Resource Usage:** When the UI window is minimized or closed in the tray, frame rendering to the webview stops completely.
- **Functional completeness:** No dead buttons, non-functional toggles, or dummy placeholders.

---

## Part 6: Mandatory Technical Stack & Architecture Sync Rule

> **LAW OF SYSTEM SPECIFICATION FRESHNESS:**
> Any change—whether minor architectural refinement, pipeline modification, model tuning, mathematical adjustment (such as EMA coefficients or state machine timing), or IPC contract alteration—**MUST be immediately synchronized and documented in `docs/TECHNICAL_STACK_AND_PIPELINES.md`**.
>
> 1. Never let architectural documentation lag behind code commits.
> 2. Document the *exact* formulas, landmark topologies, and data contracts that exist in production code.
> 3. Future agents must inspect `docs/TECHNICAL_STACK_AND_PIPELINES.md` before altering vision pipelines, threshold calculations, or model configurations.
