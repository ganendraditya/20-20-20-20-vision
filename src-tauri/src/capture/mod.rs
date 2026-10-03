pub mod frame;

pub use frame::CameraFrameDto;
use nokhwa::{
    native_api_backend,
    query,
    utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution},
    Camera,
};
use nokhwa::pixel_format::RgbFormat;
use std::sync::Mutex;
use std::time::Duration;
use std::thread;
use tauri::{AppHandle, Emitter, Manager};
use crate::AppState;

pub struct CameraManager {
    camera: Option<Camera>,
    is_paused_by_conflict: bool,
}

impl CameraManager {
    pub fn new() -> Self {
        Self {
            camera: None,
            is_paused_by_conflict: false,
        }
    }

    /// List all available cameras on the system
    pub fn list_cameras() -> Vec<String> {
        let backend = native_api_backend().unwrap_or(nokhwa::utils::ApiBackend::Auto);
        let cameras = query(backend).unwrap_or_else(|_| vec![]);
        
        let mut list = Vec::new();
        for (i, cam) in cameras.into_iter().enumerate() {
            list.push(format!("Device {}: {}", i, cam.human_name()));
        }
        
        if list.is_empty() {
            list.push("No Camera Detected".to_string());
        }
        
        list
    }

    /// Initialize a specific camera device with fallback formats
    pub fn init_camera(&mut self, index: usize) -> Result<(), String> {
        let idx = CameraIndex::Index(index as u32);
        println!("[420vision::camera] Attempting to initialize camera at index {}", index);

        // Try standard formats in priority order
        // 1. macOS AVFoundation native RAWRGB 720p @ 30FPS
        // 2. YUYV 640x480 @ 15FPS (common USB webcams & Windows MSMF)
        // 3. NV12 1080p
        let candidate_formats = vec![
            CameraFormat::new(Resolution::new(1280, 720), FrameFormat::RAWRGB, 30),
            CameraFormat::new(Resolution::new(640, 480), FrameFormat::YUYV, 15),
            CameraFormat::new(Resolution::new(1920, 1080), FrameFormat::NV12, 30),
            CameraFormat::new(Resolution::new(640, 480), FrameFormat::NV12, 30),
        ];

        for fmt in candidate_formats {
            let req = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt));
            if let Ok(mut cam) = Camera::new(idx.clone(), req) {
                if let Ok(_) = cam.open_stream() {
                    println!("[420vision::camera] ✅ Stream opened successfully with format: {:?}", fmt);
                    self.camera = Some(cam);
                    self.is_paused_by_conflict = false;
                    return Ok(());
                }
            }
        }

        Err(format!("Could not open camera stream at index {} with any supported format", index))
    }

    /// Stop the camera stream and release the device completely
    pub fn release_camera(&mut self) {
        if let Some(mut cam) = self.camera.take() {
            let _ = cam.stop_stream();
            println!("[420vision::camera] Camera stream released.");
        }
    }
}

/// The background daemon loop that manages the camera stream
pub fn run_capture_loop(app_handle: AppHandle) {
    thread::spawn(move || {
        let mut cam_manager = CameraManager::new();
        let mut current_cam_index = 9999; // Force init on first pass
        
        loop {
            // Check state
            let (is_running, selected_index, is_sandbox_viewing) = {
                let state_mutex = app_handle.state::<Mutex<AppState>>();
                let state = state_mutex.lock().unwrap();
                (
                    state.status.is_running,
                    state.selected_camera_index,
                    state.is_sandbox_viewing,
                )
            };

            if !is_running {
                // If user toggled OFF via master switch, yield camera completely
                cam_manager.release_camera();
                current_cam_index = usize::MAX;
                thread::sleep(Duration::from_millis(1000));
                continue;
            }

            // Handle Camera Selection Changes, initial start, or resume
            if current_cam_index != selected_index || cam_manager.camera.is_none() {
                cam_manager.release_camera();
                match cam_manager.init_camera(selected_index) {
                    Ok(_) => {
                        current_cam_index = selected_index;
                        update_status_text(&app_handle, "Monitoring Active");
                        println!("[420vision::camera] Camera {} active and monitoring.", selected_index);
                    }
                    Err(e) => {
                        println!("[420vision::camera] Init failed: {}", e);
                    }
                }
            }

            // If camera was yielded due to conflict (Zoom/Meet), try to re-acquire gently
            if cam_manager.is_paused_by_conflict {
                if cam_manager.init_camera(selected_index).is_ok() {
                    cam_manager.is_paused_by_conflict = false;
                    update_status_text(&app_handle, "Monitoring Active");
                    println!("[420vision::camera] Camera re-acquired after external app released it.");
                } else {
                    // Still busy, wait 5 seconds before retrying
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }
            }

            // Capture Frame
            if let Some(cam) = &mut cam_manager.camera {
                match cam.frame() {
                    Ok(frame) => {
                        // Conditional rendering: emit frame event ONLY when user is viewing Camera Test tab
                        if is_sandbox_viewing {
                            let (w, h) = (frame.resolution().width(), frame.resolution().height());
                            let dto = CameraFrameDto {
                                width: w,
                                height: h,
                                is_face_detected: true,
                                left_ear: 0.30,
                                right_ear: 0.30,
                                avg_ear: 0.30,
                                is_blinking: false,
                                total_blinks: 0,
                                image_data_base64: None,
                            };
                            let _ = app_handle.emit("camera-sandbox-frame", dto);
                        }

                        // Throttle frame rate manually to ~15 FPS to save CPU (1000ms / 15 ≈ 67ms)
                        thread::sleep(Duration::from_millis(67));
                    }
                    Err(nokhwa::NokhwaError::ReadFrameError(_)) 
                    | Err(nokhwa::NokhwaError::OpenDeviceError(_, _)) => {
                        // Conflict detected! Camera stolen by another app.
                        // IMPLEMENT ALWAYS-YIELD PRINCIPLE
                        cam_manager.release_camera();
                        cam_manager.is_paused_by_conflict = true;
                        update_status_text(&app_handle, "⚠️ Camera Paused (In Use)");
                        println!("[420vision::camera] Hardware contention detected (Zoom/Meet/FaceTime active). Yielding camera.");
                    }
                    Err(_) => {
                        // Other errors, back off briefly
                        thread::sleep(Duration::from_millis(500));
                    }
                }
            } else {
                // No camera available at all
                update_status_text(&app_handle, "⚠️ No Camera Detected");
                thread::sleep(Duration::from_secs(5));
            }
        }
    });
}

fn update_status_text(app: &AppHandle, text: &str) {
    let state_mutex = app.state::<Mutex<AppState>>();
    let mut state = state_mutex.lock().unwrap();
    state.status.status_text = text.to_string();
}
