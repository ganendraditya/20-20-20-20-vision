pub mod frame;

pub use frame::CameraFrameDto;
use nokhwa::{
    native_api_backend,
    query,
    utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution},
    Camera,
};
use nokhwa::pixel_format::YuyvFormat;
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

    /// Initialize a specific camera device
    pub fn init_camera(&mut self, index: usize) -> Result<(), String> {
        let _backend = native_api_backend().unwrap_or(nokhwa::utils::ApiBackend::Auto);
        let idx = CameraIndex::Index(index as u32);
        
        // Request low resolution (e.g. 640x480) for performance
        let format = RequestedFormat::new::<YuyvFormat>(
            RequestedFormatType::Closest(CameraFormat::new(
                Resolution::new(640, 480),
                FrameFormat::YUYV,
                15, // 15 FPS
            ))
        );

        match Camera::new(idx, format) {
            Ok(mut cam) => {
                if let Err(e) = cam.open_stream() {
                    return Err(format!("Failed to open camera stream: {}", e));
                }
                self.camera = Some(cam);
                self.is_paused_by_conflict = false;
                Ok(())
            }
            Err(e) => Err(format!("Failed to initialize camera: {}", e)),
        }
    }

    /// Stop the camera stream and release the device completely
    pub fn release_camera(&mut self) {
        if let Some(mut cam) = self.camera.take() {
            let _ = cam.stop_stream();
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
                if cam_manager.init_camera(selected_index).is_ok() {
                    current_cam_index = selected_index;
                }
            }

            // If camera was yielded due to conflict (Zoom/Meet), try to re-acquire gently
            if cam_manager.is_paused_by_conflict {
                if cam_manager.init_camera(selected_index).is_ok() {
                    cam_manager.is_paused_by_conflict = false;
                    update_status_text(&app_handle, "Monitoring Active");
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
