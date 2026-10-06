pub mod capture;
pub mod vision;
pub mod detector;
pub mod timer;
pub mod storage;
pub mod notifier;

use std::sync::Mutex;
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager, State,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppStatus {
    pub is_running: bool,
    pub is_camera_active: bool,
    pub bpm: f32,
    pub next_break_seconds: u32,
    pub total_blinks_today: u32,
    pub status_text: String,
}

use crate::storage::AppConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub success: bool,
    pub threshold: f32,
    pub message: String,
}

pub struct AppState {
    pub status: AppStatus,
    pub selected_camera_index: usize,
    pub is_sandbox_viewing: bool,
    pub eye_calibrator: detector::EyeCalibrator,
    pub config: AppConfig,
}

impl Default for AppState {
    fn default() -> Self {
        let config = AppConfig::load();
        Self {
            status: AppStatus {
                is_running: true,
                is_camera_active: true,
                bpm: 0.0,
                next_break_seconds: 1200, // 20 minutes (20 * 60)
                total_blinks_today: 0,
                status_text: "Monitoring Active".to_string(),
            },
            selected_camera_index: config.selected_camera_index,
            is_sandbox_viewing: false,
            eye_calibrator: detector::EyeCalibrator::new(),
            config,
        }
    }
}

// ============================================================================
// IPC Commands (Contracts between Rust backend and Frontend Webview)
// ============================================================================

#[tauri::command]
fn get_status(state: State<'_, Mutex<AppState>>) -> AppStatus {
    let state = state.lock().unwrap();
    state.status.clone()
}

#[tauri::command]
fn toggle_monitoring(state: State<'_, Mutex<AppState>>) -> bool {
    let mut state = state.lock().unwrap();
    state.status.is_running = !state.status.is_running;
    if state.status.is_running {
        state.status.status_text = "Monitoring Active".to_string();
    } else {
        state.status.status_text = "Paused".to_string();
    }
    println!("[420vision::ipc] toggle_monitoring -> {}", state.status.is_running);
    state.status.is_running
}

#[tauri::command]
fn set_sandbox_viewing(active: bool, state: State<'_, Mutex<AppState>>) {
    let mut state = state.lock().unwrap();
    state.is_sandbox_viewing = active;
    println!("[420vision::ipc] set_sandbox_viewing -> {}", active);
}

#[tauri::command]
fn get_cameras(_state: State<'_, Mutex<AppState>>) -> Vec<String> {
    let list = capture::CameraManager::list_cameras();
    println!("[420vision::ipc] get_cameras -> {:?}", list);
    list
}

#[tauri::command]
fn set_camera(index: usize, state: State<'_, Mutex<AppState>>) -> bool {
    let mut state = state.lock().unwrap();
    state.selected_camera_index = index;
    state.config.selected_camera_index = index;
    let _ = state.config.save();
    println!("[420vision::ipc] set_camera -> {}", index);
    true
}

#[tauri::command]
fn start_calibration(state: State<'_, Mutex<AppState>>) -> String {
    let mut state = state.lock().unwrap();
    state.eye_calibrator = detector::EyeCalibrator::new();
    println!("[420vision::ipc] start_calibration -> calibrator reset and ready");
    "Calibration started".to_string()
}

#[tauri::command]
fn submit_calibration_sample(stage: String, ear: f32, state: State<'_, Mutex<AppState>>) -> bool {
    if !ear.is_finite() {
        return false;
    }
    let mut state = state.lock().unwrap();
    match stage.as_str() {
        "open" => state.eye_calibrator.add_open_sample(ear),
        "closed" => state.eye_calibrator.add_closed_sample(ear),
        _ => return false,
    }
    true
}

#[tauri::command]
fn finalize_calibration(state: State<'_, Mutex<AppState>>) -> CalibrationResult {
    let (res, config_to_save) = {
        let mut state = state.lock().unwrap();
        match state.eye_calibrator.finalize() {
            Some(threshold) => {
                // Apply bounds safety clamping [0.16..0.28]
                let clamped = threshold.clamp(0.16, 0.28);
                state.config.ear_threshold = clamped;
                let cfg = state.config.clone();
                println!("[420vision::ipc] finalize_calibration -> Success! Personal threshold: {:.3}", clamped);
                (
                    CalibrationResult {
                        success: true,
                        threshold: clamped,
                        message: format!("Calibration complete! Optimal threshold set to {:.3}", clamped),
                    },
                    Some(cfg),
                )
            }
            None => {
                println!("[420vision::ipc] finalize_calibration -> Failed to extract sufficient variance");
                (
                    CalibrationResult {
                        success: false,
                        threshold: state.config.ear_threshold,
                        message: "Calibration incomplete or invalid. Retaining previous threshold.".to_string(),
                    },
                    None,
                )
            }
        }
    }; // Lock released here before disk I/O

    if let Some(cfg) = config_to_save {
        let _ = cfg.save();
    }

    res
}

#[tauri::command]
fn get_config(state: State<'_, Mutex<AppState>>) -> AppConfig {
    let state = state.lock().unwrap();
    state.config.clone()
}

#[tauri::command]
fn update_config(
    sound_enabled: Option<bool>,
    stare_alert_enabled: Option<bool>,
    keep_awake_enabled: Option<bool>,
    state: State<'_, Mutex<AppState>>,
) -> Result<AppConfig, String> {
    let config_to_save = {
        let mut state = state.lock().unwrap();
        if let Some(sound) = sound_enabled {
            state.config.sound_enabled = sound;
        }
        if let Some(stare) = stare_alert_enabled {
            state.config.stare_alert_enabled = stare;
        }
        if let Some(keep_awake) = keep_awake_enabled {
            state.config.keep_awake_enabled = keep_awake;
        }
        state.config.clone()
    };

    config_to_save.save().map_err(|e| format!("Failed to save config: {}", e))?;
    println!("[420vision::ipc] Config updated: keep_awake={}", config_to_save.keep_awake_enabled);
    Ok(config_to_save)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyCompliance {
    pub date: String,
    pub avg_bpm: f32,
    pub breaks_completed: u32,
    pub breaks_skipped: u32,
    pub screen_minutes: u32,
}

#[tauri::command]
fn get_stats() -> Vec<DailyCompliance> {
    let stats = match storage::AnalyticsDb::open() {
        Ok(db) => match db.get_compliance_history(7) {
            Ok(history) if !history.is_empty() => history,
            _ => vec![
                DailyCompliance {
                    date: "Today".to_string(),
                    avg_bpm: 16.0,
                    breaks_completed: 0,
                    breaks_skipped: 0,
                    screen_minutes: 0,
                },
            ],
        },
        Err(e) => {
            eprintln!("[420vision] Failed to query analytics DB: {}", e);
            vec![]
        }
    };
    println!("[420vision::ipc] get_stats returned {} items", stats.len());
    stats
}

#[tauri::command]
fn hide_window(app: AppHandle) {
    let state = app.state::<Mutex<AppState>>();
    let mut state = state.lock().unwrap();
    state.is_sandbox_viewing = false;
    println!("[420vision::ipc] hide_window called");

    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
}

#[tauri::command]
fn quit_app(app: AppHandle) {
    println!("[420vision::ipc] quit_app called -> exiting cleanly");
    app.exit(0);
}

// ============================================================================
// Application Entrypoint
// ============================================================================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(AppState::default()))
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Start the background camera capture daemon
            capture::run_capture_loop(app.handle().clone());

            // Build Tray Menu items with clear labels
            let open_item = MenuItem::with_id(app, "open_window", "Open 420vision", true, None::<&str>)?;
            let toggle_item = MenuItem::with_id(app, "toggle_pause", "Turn Monitoring OFF / ON", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit 420vision", true, None::<&str>)?;
            
            let tray_menu = Menu::with_items(app, &[
                &open_item,
                &toggle_item,
                &quit_item,
            ])?;

            // Build Tray Icon purely programmatically with explicit icon
            let tray_icon_image = app.default_window_icon().cloned()
                .unwrap_or_else(|| tauri::image::Image::from_bytes(include_bytes!("../icons/32x32.png")).unwrap());

            let _tray = TrayIconBuilder::with_id("main-tray")
                .icon(tray_icon_image)
                .icon_as_template(true)
                .menu(&tray_menu)
                .show_menu_on_left_click(false)
                .tooltip("420vision: 20-20-20-20 Vision Assistant")
                .on_menu_event(|app, event| {
                    match event.id.as_ref() {
                        "open_window" => {
                            if let Some(window) = app.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                        "toggle_pause" => {
                            let state = app.state::<Mutex<AppState>>();
                            let mut state = state.lock().unwrap();
                            state.status.is_running = !state.status.is_running;
                            if state.status.is_running {
                                state.status.status_text = "Monitoring Active".to_string();
                            } else {
                                state.status.status_text = "Paused".to_string();
                            }
                            println!("[420vision::tray] Monitoring toggled -> {}", state.status.is_running);
                        }
                        "quit" => {
                            println!("[420vision::tray] Quit clicked -> exiting cleanly");
                            app.exit(0);
                        }
                        _ => {}
                    }
                })
                .on_tray_icon_event(|tray, event| {
                    match event {
                        // Left-click: toggle popover window
                        TrayIconEvent::Click {
                            button: MouseButton::Left,
                            button_state: MouseButtonState::Up,
                            ..
                        } => {
                            let app = tray.app_handle();
                            if let Some(window) = app.get_webview_window("main") {
                                if window.is_visible().unwrap_or(false) {
                                    let state = app.state::<Mutex<AppState>>();
                                    let mut state = state.lock().unwrap();
                                    state.is_sandbox_viewing = false;
                                    let _ = window.hide();
                                } else {
                                    let _ = window.show();
                                    let _ = window.set_focus();
                                }
                            }
                        }
                        _ => {}
                    }
                })
                .build(app)?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            toggle_monitoring,
            get_cameras,
            set_camera,
            start_calibration,
            submit_calibration_sample,
            finalize_calibration,
            get_config,
            update_config,
            get_stats,
            hide_window,
            quit_app,
            set_sandbox_viewing
        ])
        .run(tauri::generate_context!())
        .expect("error while running 420vision application");
}
