import { invoke } from "@tauri-apps/api/core";

interface AppStatus {
  is_running: boolean;
  is_camera_active: boolean;
  bpm: number;
  next_break_seconds: number;
  total_blinks_today: number;
  status_text: string;
}

interface DailyCompliance {
  date: string;
  avg_bpm: number;
  breaks_completed: number;
  breaks_skipped: number;
  screen_minutes: number;
}

let sandboxBlinks = 0;

window.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupIPC();
  loadCameras();
  startStatusPoller();
});

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

      // Reset sandbox counter if switching to Camera Test tab
      if (targetTab === "tab-camera") {
        sandboxBlinks = 0;
        updateSandboxUI();
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

  // Calibrate button
  const calibrateBtn = document.getElementById("btn-calibrate");
  calibrateBtn?.addEventListener("click", async () => {
    try {
      const res = await invoke<string>("start_calibration");
      alert(res);
    } catch (e) {
      console.error("Failed to start calibration:", e);
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
        streakEl.textContent = `🔥 ${today.breaks_completed} break(s) completed today!`;
      } else {
        streakEl.textContent = `🌱 Ready to protect your eyes today`;
      }
    }
  } catch (e) {
    console.error("Failed to load compliance stats:", e);
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
