import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface AppStatus {
  is_running: boolean;
  is_camera_active: boolean;
  bpm: number;
  next_break_seconds: number;
  total_blinks_today: number;
  status_text: string;
}

interface LandmarkPoint {
  x: number;
  y: number;
}

interface CameraFrameDto {
  width: number;
  height: number;
  is_face_detected: boolean;
  left_ear: number;
  right_ear: number;
  avg_ear: number;
  is_blinking: boolean;
  total_blinks: number;
  eye_landmarks: LandmarkPoint[];
  face_landmarks: LandmarkPoint[];
  image_data_base64: string | null;
  yaw_deg: number;
  pitch_deg: number;
  roll_deg: number;
  distance_cm: number;
  is_resting_gaze: boolean;
}

interface DailyCompliance {
  date: string;
  avg_bpm: number;
  breaks_completed: number;
  breaks_skipped: number;
  screen_minutes: number;
}

interface AppConfig {
  ear_threshold: number;
  stare_limit_secs: number;
  break_target_minutes: number;
  break_duration_seconds: number;
  selected_camera_index: number;
  sound_enabled: boolean;
  stare_alert_enabled: boolean;
  keep_awake_enabled: boolean;
}

interface CalibrationResult {
  success: boolean;
  threshold: number;
  message: string;
}

let sandboxBlinks = 0;
let wasBlinking = false;
let currentEar = 0.0;
let isFaceDetected = false;
let isCalibrating = false;
let unlistenCameraFrames: (() => void) | null = null;

window.addEventListener("beforeunload", () => {
  if (unlistenCameraFrames) {
    unlistenCameraFrames();
    unlistenCameraFrames = null;
  }
});

function init() {
  setupTabs();
  setupIPC();
  loadCameras();
  startStatusPoller();
  listenToCameraFrames();
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

function setupTabs() {
  const tabButtons = document.querySelectorAll<HTMLButtonElement>(".tab-btn");
  const tabPanes = document.querySelectorAll<HTMLElement>(".tab-pane");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      if (!targetTab) return;

      tabButtons.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const pane = document.getElementById(targetTab);
      if (pane) pane.classList.add("active");

      // Reset sandbox counter and notify backend if switching to Camera Test tab
      if (targetTab === "tab-camera") {
        sandboxBlinks = 0;
        updateSandboxUI();
        invoke("set_sandbox_viewing", { active: true }).catch(console.error);
      } else {
        invoke("set_sandbox_viewing", { active: false }).catch(console.error);
      }

      // Refresh stats if switching to Stats tab
      if (targetTab === "tab-stats") {
        loadStats();
      }
    });
  });

  // Close / hide window button
  const closeBtn = document.getElementById("btn-close");
  closeBtn?.addEventListener("click", async () => {
    try {
      await invoke("set_sandbox_viewing", { active: false });
      await invoke("hide_window");
    } catch (e) {
      console.error("Failed to hide window:", e);
    }
  });

  // Sandbox reset button
  const resetBtn = document.getElementById("btn-reset-sandbox");
  resetBtn?.addEventListener("click", () => {
    sandboxBlinks = 0;
    updateSandboxUI();
  });
}

function updateSandboxUI() {
  const lbl = document.getElementById("val-sandbox-blinks");
  if (lbl) lbl.textContent = sandboxBlinks.toString();
}

function setupIPC() {
  // Master toggle button
  const masterToggle = document.getElementById("btn-master-toggle");
  masterToggle?.addEventListener("click", async () => {
    try {
      const isRunning = await invoke<boolean>("toggle_monitoring");
      updateToggleUI(isRunning);
    } catch (e) {
      console.error("Failed to toggle monitoring:", e);
    }
  });

  // Quit button
  const quitBtn = document.getElementById("btn-quit");
  quitBtn?.addEventListener("click", async () => {
    try {
      await invoke("quit_app");
    } catch (e) {
      console.error("Failed to quit app:", e);
    }
  });

  // Calibrate button & 5-Second Interactive Wizard Flow
  const calibrateBtn = document.getElementById("btn-calibrate");
  const calBox = document.getElementById("calibration-status-box");
  const calInstruction = document.getElementById("calibration-instruction");
  const calFill = document.getElementById("calibration-progress-fill");
  const calResult = document.getElementById("calibration-result-text");

  // Settings Checkboxes (Sound, Stare Alert, Smart Keep-Awake)
  const chkSound = document.getElementById("chk-sound") as HTMLInputElement | null;
  const chkStare = document.getElementById("chk-stare") as HTMLInputElement | null;
  const chkKeepAwake = document.getElementById("chk-keep-awake") as HTMLInputElement | null;

  // Consolidated initial configuration fetch
  invoke<AppConfig>("get_config").then((cfg) => {
    if (calResult) calResult.textContent = `Current threshold: ${cfg.ear_threshold.toFixed(3)}`;
    if (chkSound) chkSound.checked = cfg.sound_enabled;
    if (chkStare) chkStare.checked = cfg.stare_alert_enabled;
    if (chkKeepAwake) chkKeepAwake.checked = cfg.keep_awake_enabled;
  }).catch(console.error);

  const saveSettings = async () => {
    try {
      await invoke("update_config", {
        soundEnabled: chkSound?.checked ?? null,
        stareAlertEnabled: chkStare?.checked ?? null,
        keepAwakeEnabled: chkKeepAwake?.checked ?? null,
      });
    } catch (e) {
      console.error("Failed to save settings:", e);
    }
  };

  chkSound?.addEventListener("change", saveSettings);
  chkStare?.addEventListener("change", saveSettings);
  chkKeepAwake?.addEventListener("change", saveSettings);

  calibrateBtn?.addEventListener("click", async () => {
    if (isCalibrating) return;
    isCalibrating = true;

    try {
      if (calBox) calBox.style.display = "flex";
      if (calibrateBtn) calibrateBtn.setAttribute("disabled", "true");
      await invoke("start_calibration");

      // Stage 1: Keep Eyes Open Naturally for 3 seconds (collect sample every 100ms)
      if (calInstruction) calInstruction.textContent = "1/2: Look at screen naturally (Eyes Open)...";
      if (calFill) {
        calFill.style.background = "var(--accent)";
        calFill.style.width = "0%";
      }

      for (let i = 1; i <= 30; i++) {
        await new Promise((r) => setTimeout(r, 100));
        if (isFaceDetected && currentEar > 0.05) {
          await invoke("submit_calibration_sample", { stage: "open", ear: currentEar });
        }
        if (calFill) calFill.style.width = `${(i / 50) * 100}%`;
      }

      // Stage 2: Close Eyes Comfortably for 2 seconds (collect sample every 100ms)
      if (calInstruction) calInstruction.textContent = "2/2: Close eyes comfortably (Eyes Closed)...";
      if (calFill) calFill.style.background = "var(--green)";

      for (let i = 31; i <= 50; i++) {
        await new Promise((r) => setTimeout(r, 100));
        if (isFaceDetected && currentEar > 0.01) {
          await invoke("submit_calibration_sample", { stage: "closed", ear: currentEar });
        }
        if (calFill) calFill.style.width = `${(i / 50) * 100}%`;
      }

      // Finalize calibration and display personalized threshold
      const res = await invoke<CalibrationResult>("finalize_calibration");
      if (calInstruction) {
        calInstruction.textContent = res.success ? "Calibration Complete" : "Calibration Failed";
      }
      if (calResult) {
        calResult.textContent = res.success
          ? `Personal Threshold: ${res.threshold.toFixed(3)} (Saved)`
          : res.message;
      }
    } catch (e) {
      console.error("Calibration failed:", e);
      if (calInstruction) calInstruction.textContent = "Error during calibration";
    } finally {
      isCalibrating = false;
      if (calibrateBtn) calibrateBtn.removeAttribute("disabled");
    }
  });

  // Camera selector change
  const camSelect = document.getElementById("sel-camera") as HTMLSelectElement | null;
  camSelect?.addEventListener("change", async () => {
    const idx = parseInt(camSelect.value, 10);
    try {
      await invoke("set_camera", { index: idx });
    } catch (e) {
      console.error("Failed to set camera:", e);
    }
  });
}

async function loadCameras() {
  const sel = document.getElementById("sel-camera") as HTMLSelectElement | null;
  if (!sel) return;

  try {
    const cameras = await invoke<string[]>("get_cameras");
    sel.innerHTML = "";
    cameras.forEach((cam, idx) => {
      const opt = document.createElement("option");
      opt.value = idx.toString();
      opt.textContent = cam;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error("Failed to load cameras:", e);
  }
}

async function loadStats() {
  try {
    const history = await invoke<DailyCompliance[]>("get_stats");
    if (history.length === 0) return;

    const today = history[0];
    const totalBreaks = today.breaks_completed + today.breaks_skipped;
    const complianceEl = document.getElementById("stat-compliance");
    if (complianceEl) {
      if (totalBreaks === 0) {
        complianceEl.textContent = "\u2014 (No breaks yet)";
      } else {
        const rate = Math.round((today.breaks_completed / totalBreaks) * 100);
        complianceEl.textContent = `${rate}% (${today.breaks_completed} / ${totalBreaks})`;
      }
    }

    const hoursEl = document.getElementById("stat-screen-hours");
    if (hoursEl) {
      const hrs = Math.floor(today.screen_minutes / 60);
      const mins = today.screen_minutes % 60;
      hoursEl.textContent = hrs > 0 ? `${hrs}h ${mins}m` : `${mins}m`;
    }

    const bpmEl = document.getElementById("stat-avg-bpm");
    if (bpmEl) {
      bpmEl.textContent = `${today.avg_bpm.toFixed(1)} BPM`;
    }

    const streakEl = document.getElementById("stat-streak-msg");
    if (streakEl) {
      if (today.breaks_completed > 0) {
        streakEl.textContent = `${today.breaks_completed} break(s) completed today!`;
      } else {
        streakEl.textContent = `Ready to protect your eyes today`;
      }
    }
  } catch (e) {
    console.error("Failed to load compliance stats:", e);
  }
}

async function listenToCameraFrames() {
  try {
    const videoImg = document.getElementById("cam-video-stream") as HTMLImageElement | null;
    const canvas = document.getElementById("cam-mesh-canvas") as HTMLCanvasElement | null;
    const ctx = canvas?.getContext("2d");
    const placeholder = document.getElementById("cam-loading-placeholder");
    const hudTop = document.getElementById("cam-hud-top");
    const hudBottom = document.getElementById("cam-hud-bottom");
    const earText = document.getElementById("cam-hud-ear");
    const triggerText = document.getElementById("cam-hud-trigger");
    const faceText = document.getElementById("cam-hud-face");
    const yawText = document.getElementById("cam-hud-yaw");
    const pitchText = document.getElementById("cam-hud-pitch");
    const rollText = document.getElementById("cam-hud-roll");
    const distText = document.getElementById("cam-hud-dist");
    const chkShowMesh = document.getElementById("chk-show-mesh") as HTMLInputElement | null;
    const chkShowFace = document.getElementById("chk-show-face-contour") as HTMLInputElement | null;

    unlistenCameraFrames = await listen<CameraFrameDto>("camera-sandbox-frame", (event) => {
      const data = event.payload;

      // Update image stream
      if (data.image_data_base64 && videoImg) {
        videoImg.src = data.image_data_base64;
        videoImg.style.display = "block";
        if (placeholder) placeholder.style.display = "none";
        if (hudTop) hudTop.style.display = "flex";
        if (hudBottom) hudBottom.style.display = "flex";
      }

      // Draw Landmark Overlay on canvas
      if (canvas && ctx) {
        const cw = canvas.parentElement?.clientWidth || 320;
        const ch = canvas.parentElement?.clientHeight || 230;
        if (canvas.width !== cw || canvas.height !== ch) {
          canvas.width = cw;
          canvas.height = ch;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const showEyeMesh = chkShowMesh ? chkShowMesh.checked : true;
        const showFaceContour = chkShowFace ? chkShowFace.checked : true;

        let hasDrawn = false;

        // 1. Draw Full Face Contour (jaw, eyebrows, nose, mouth)
        if (showFaceContour && data.face_landmarks && data.face_landmarks.length > 0) {
          hasDrawn = true;
          ctx.fillStyle = "#a855f7"; // Vibrant purple for facial contours
          for (const pt of data.face_landmarks) {
            const px = (1.0 - pt.x) * canvas.width;
            const py = pt.y * canvas.height;
            ctx.beginPath();
            ctx.arc(px, py, 1.8, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        // 2. Draw Eye Mesh Overlay
        if (showEyeMesh && data.eye_landmarks && data.eye_landmarks.length > 0) {
          hasDrawn = true;
          ctx.fillStyle = data.is_blinking ? "#ef4444" : "#38bdf8"; // Red when blinking, cyan when open
          for (const pt of data.eye_landmarks) {
            const px = (1.0 - pt.x) * canvas.width;
            const py = pt.y * canvas.height;
            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        canvas.style.display = hasDrawn ? "block" : "none";
      }

      currentEar = data.avg_ear;
      isFaceDetected = data.is_face_detected;

      // 1. EAR & Blink status
      if (earText) {
        earText.textContent = `EAR: ${data.avg_ear.toFixed(2)}`;
        earText.className = data.is_blinking ? "hud-pill hud-pill-danger" : "hud-pill";
      }

      if (triggerText) {
        triggerText.textContent = "Trigger: Δ 35%";
      }

      // 2. Face detection badge
      if (faceText) {
        faceText.textContent = data.is_face_detected ? "Face: Locked" : "Face: Searching";
        faceText.className = data.is_face_detected ? "hud-pill hud-pill-active" : "hud-pill hud-pill-danger";
      }

      // 3. 3D Orientation (Yaw / Pitch / Roll) with authoritative backend resting state
      if (yawText) {
        const roundedYaw = Math.round(data.yaw_deg) || 0;
        const absYaw = Math.abs(roundedYaw);
        const sign = roundedYaw > 0 ? "+" : "";
        if (data.is_resting_gaze && absYaw >= 15) {
          yawText.textContent = `Yaw: ${sign}${roundedYaw}° 🟢 (Resting)`;
          yawText.className = "hud-pill hud-pill-active";
        } else {
          yawText.textContent = `Yaw: ${sign}${roundedYaw}° (Screen)`;
          yawText.className = "hud-pill";
        }
      }

      if (pitchText) {
        const roundedPitch = Math.round(data.pitch_deg) || 0;
        const sign = roundedPitch > 0 ? "+" : "";
        if (data.is_resting_gaze && data.pitch_deg >= 5) {
          pitchText.textContent = `Pitch: ${sign}${roundedPitch}° 🟢 (Upward Rest)`;
          pitchText.className = "hud-pill hud-pill-active";
        } else if (roundedPitch < -15) {
          pitchText.textContent = `Pitch: ${roundedPitch}° (Desk/Phone)`;
          pitchText.className = "hud-pill hud-pill-warning";
        } else {
          pitchText.textContent = `Pitch: ${sign}${roundedPitch}°`;
          pitchText.className = "hud-pill";
        }
      }

      if (rollText) {
        const roundedRoll = Math.round(data.roll_deg) || 0;
        const sign = roundedRoll > 0 ? "+" : "";
        rollText.textContent = `Roll: ${sign}${roundedRoll}°`;
        rollText.className = "hud-pill";
      }

      // 4. Physical distance estimation
      if (distText) {
        if (data.is_face_detected && data.distance_cm > 0) {
          const d = Math.round(data.distance_cm);
          if (d < 40) {
            distText.textContent = `Dist: ~${d} cm (Too Close ⚠️)`;
            distText.className = "hud-pill hud-pill-warning";
          } else if (d > 85) {
            distText.textContent = `Dist: ~${d} cm (Far Field)`;
            distText.className = "hud-pill hud-pill-info";
          } else {
            distText.textContent = `Dist: ~${d} cm (Optimal)`;
            distText.className = "hud-pill hud-pill-active";
          }
        } else {
          distText.textContent = "Dist: -- cm";
          distText.className = "hud-pill";
        }
      }

      // Rising-edge trigger: count only on transition to prevent multi-frame duplicate increments
      if (data.is_blinking) {
        if (!wasBlinking) {
          sandboxBlinks += 1;
          updateSandboxUI();
        }
        wasBlinking = true;
      } else {
        wasBlinking = false;
      }
    });
  } catch (e) {
    console.error("Failed to register camera frame listener:", e);
  }
}

function updateToggleUI(isRunning: boolean) {
  const btn = document.getElementById("btn-master-toggle");
  const stateLbl = document.getElementById("lbl-toggle-state");
  const statusLbl = document.getElementById("lbl-status-text");
  const dot = document.getElementById("status-indicator-dot");

  if (isRunning) {
    btn?.classList.add("active");
    if (stateLbl) stateLbl.textContent = "ON";
    if (statusLbl) statusLbl.textContent = "Monitoring Active";
    if (dot) dot.style.background = "#10b981";
  } else {
    btn?.classList.remove("active");
    if (stateLbl) stateLbl.textContent = "OFF";
    if (statusLbl) statusLbl.textContent = "Monitoring Paused";
    if (dot) dot.style.background = "#6b7280";
  }
}

async function startStatusPoller() {
  async function poll() {
    try {
      const status = await invoke<AppStatus>("get_status");
      updateToggleUI(status.is_running);

      const statusLbl = document.getElementById("lbl-status-text");
      if (statusLbl && status.is_running) {
        statusLbl.textContent = status.status_text;
      }

      const bpmLbl = document.getElementById("val-bpm");
      if (bpmLbl) bpmLbl.innerHTML = `${Math.round(status.bpm)} <small>BPM</small>`;

      const breakLbl = document.getElementById("val-next-break");
      if (breakLbl) {
        const mins = Math.floor(status.next_break_seconds / 60);
        const secs = status.next_break_seconds % 60;
        breakLbl.textContent = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
      }
    } catch (e) {
      // Backend not yet ready or polling error
    }
  }

  // Poll initially and then every 1 second
  await poll();
  setInterval(poll, 1000);
}
