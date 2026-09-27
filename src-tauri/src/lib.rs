pub mod capture;
pub mod vision;
pub mod detector;

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

pub struct AppState {
    pub status: AppStatus,
    pub selected_camera_index: usize,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            status: AppStatus {
                is_running: true,
                is_camera_active: true,
                bpm: 0.0,
                next_break_seconds: 1200, // 20 minutes (20 * 60)
                total_blinks_today: 0,
                status_text: "Monitoring Active".to_string(),
            },
            selected_camera_index: 0,
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
    state.status.is_running
}

#[tauri::command]
fn get_cameras(_state: State<'_, Mutex<AppState>>) -> Vec<String> {
    capture::CameraManager::list_cameras()
}

#[tauri::command]
fn set_camera(index: usize, state: State<'_, Mutex<AppState>>) -> bool {
    let mut state = state.lock().unwrap();
    state.selected_camera_index = index;
    true
}

#[tauri::command]
fn start_calibration() -> String {
    "Calibration initiated".to_string()
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
    // Initial schema for analytics stats
    vec![
        DailyCompliance {
            date: "Today".to_string(),
            avg_bpm: 16.5,
            breaks_completed: 4,
            breaks_skipped: 1,
            screen_minutes: 100,
        },
    ]
}

#[tauri::command]
fn hide_window(app: AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
}

#[tauri::command]
fn quit_app(app: AppHandle) {
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

            // Build Tray Menu
            let toggle_item = MenuItem::with_id(app, "toggle_window", "Open 420vision", true, None::<&str>)?;
            let pause_item = MenuItem::with_id(app, "toggle_pause", "Pause / Resume Monitoring", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit 420vision", true, None::<&str>)?;
            
            let tray_menu = Menu::with_items(app, &[
                &toggle_item,
                &pause_item,
                &quit_item,
            ])?;

            // Build Tray Icon linking to the one configured in tauri.conf.json
            let _tray = TrayIconBuilder::with_id("main-tray")
                .menu(&tray_menu)
                .show_menu_on_left_click(false)
                .tooltip("420vision: 20-20-20-20 Vision Assistant")
                .on_menu_event(|app, event| {
                    match event.id.as_ref() {
                        "toggle_window" => {
                            if let Some(window) = app.get_webview_window("main") {
                                if window.is_visible().unwrap_or(false) {
                                    let _ = window.hide();
                                } else {
                                    let _ = window.show();
                                    let _ = window.set_focus();
                                }
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
                        }
                        "quit" => {
                            app.exit(0);
                        }
                        _ => {}
                    }
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            if window.is_visible().unwrap_or(false) {
                                let _ = window.hide();
                            } else {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
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
            get_stats,
            hide_window,
            quit_app
        ])
        .run(tauri::generate_context!())
        .expect("error while running 420vision application");
}
