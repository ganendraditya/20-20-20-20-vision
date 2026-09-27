import { invoke } from "@tauri-apps/api/core";

interface AppStatus {
  is_running: boolean;
  is_camera_active: boolean;
  bpm: number;
  next_break_seconds: number;
  total_blinks_today: number;
  status_text: string;
}

let sandboxBlinks = 0;

window.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupIPC();
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
}

function updateToggleUI(isRunning: boolean) {
  const btn = document.getElementById("btn-master-toggle");
  const stateLbl = document.getElementById("lbl-toggle-state");
  const statusLbl = document.getElementById("lbl-status-text");

  if (isRunning) {
    btn?.classList.add("active");
    if (stateLbl) stateLbl.textContent = "ON";
    if (statusLbl) statusLbl.textContent = "Monitoring Active";
  } else {
    btn?.classList.remove("active");
    if (stateLbl) stateLbl.textContent = "OFF";
    if (statusLbl) statusLbl.textContent = "Monitoring Paused";
  }
}

async function startStatusPoller() {
  async function poll() {
    try {
      const status = await invoke<AppStatus>("get_status");
      updateToggleUI(status.is_running);

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
