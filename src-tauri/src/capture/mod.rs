pub mod frame;

pub use frame::{CameraFrameDto, EyeLandmarkPoint};

use nokhwa::{
    native_api_backend,
    query,
    utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution},
    Camera,
};
use nokhwa::pixel_format::RgbFormat;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::thread;
use tauri::{AppHandle, Emitter, Manager};

use crate::detector::BlinkDetector;
use crate::timer::PresenceTimer;
use crate::vision::{BlazeFaceEngine, FaceMeshEngine};
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

/// Locate or fallback FaceMesh model path
fn get_facemesh_model_path() -> PathBuf {
    // 1. Check local models directory relative to current working directory
    let local = PathBuf::from("models/facemesh.onnx");
    if local.exists() {
        return local;
    }
    // 2. Check installed bundle directory
    if let Some(share) = dirs::data_dir() {
        let installed = share.join("420vision/models/facemesh.onnx");
        if installed.exists() {
            return installed;
        }
    }
    local
}

/// Locate or fallback BlazeFace detector model path
fn get_blazeface_model_path() -> PathBuf {
    // 1. Check local models directory relative to current working directory
    let local = PathBuf::from("models/blazeface.onnx");
    if local.exists() {
        return local;
    }
    // 2. Check installed bundle directory
    if let Some(share) = dirs::data_dir() {
        let installed = share.join("420vision/models/blazeface.onnx");
        if installed.exists() {
            return installed;
        }
    }
    local
}

/// The background daemon loop that manages the camera stream & live vision inference
pub fn run_capture_loop(app_handle: AppHandle) {
    thread::spawn(move || {
        let mut cam_manager = CameraManager::new();
        let mut current_cam_index = 9999; // Force init on first pass

        // Initialize FaceMeshEngine
        let model_path = get_facemesh_model_path();
        let mut vision_engine = match FaceMeshEngine::new(&model_path) {
            Ok(engine) => {
                println!("[420vision::vision] ✅ Loaded FaceMesh model from {:?}", model_path);
                Some(engine)
            }
            Err(e) => {
                eprintln!("[420vision::vision] ⚠️ Could not load FaceMesh model ({:?}): {}", model_path, e);
                None
            }
        };

        // Initialize BlazeFace detector
        let blaze_path = get_blazeface_model_path();
        let mut face_detector = match BlazeFaceEngine::new(&blaze_path) {
            Ok(detector) => {
                println!("[420vision::vision] ✅ Loaded BlazeFace detector from {:?}", blaze_path);
                Some(detector)
            }
            Err(e) => {
                eprintln!("[420vision::vision] ⚠️ Could not load BlazeFace detector ({:?}): {}", blaze_path, e);
                None
            }
        };

        // Initialize BlinkDetector and PresenceTimer
        let mut blink_detector = BlinkDetector::new(0.22, 8.0);
        let mut presence_timer = PresenceTimer::new(1200.0, 300.0);

        loop {
            let now = Instant::now();

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
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }
            }

            // Capture Frame
            if let Some(cam) = &mut cam_manager.camera {
                match cam.frame() {
                    Ok(frame) => {
                        let (w, h) = (frame.resolution().width() as usize, frame.resolution().height() as usize);
                        
                        // Decode raw buffer to RGB
                        let mut is_face = false;
                        let mut left_ear = 0.0f32;
                        let mut right_ear = 0.0f32;
                        let mut avg_ear = 0.0f32;
                        let mut is_blinking = false;
                        let mut total_blinks = 0u32;
                        let mut current_bpm = 0.0f32;
                        let mut landmarks_cache = None;

                        if let Ok(rgb_img) = frame.decode_image::<RgbFormat>() {
                            let raw_bytes = rgb_img.as_raw();

                            // Gatekeeper: Verify facial presence using BlazeFace detector
                            let face_present = if let Some(detector) = &mut face_detector {
                                let blaze_input = detector.preprocess(raw_bytes, w, h);
                                match detector.detect_face(blaze_input) {
                                    Ok(detected) => detected,
                                    Err(e) => {
                                        eprintln!("[420vision::vision] BlazeFace detection error: {}", e);
                                        true // Fail-safe: fallback to FaceMesh on detector error
                                    }
                                }
                            } else {
                                true // Detector not available, fallback
                            };

                            if face_present {
                                // Run ONNX FaceMesh inference
                                if let Some(engine) = &mut vision_engine {
                                    let preprocessed = engine.preprocess(raw_bytes, w, h);
                                    if let Ok(landmarks) = engine.infer(preprocessed) {
                                        is_face = true;

                                        // Run blink detector
                                        let event = blink_detector.update(&landmarks, now);
                                        is_blinking = event.is_blink;
                                        total_blinks = event.total_blinks;
                                        current_bpm = event.current_bpm;
                                        left_ear = event.left_ear;
                                        right_ear = event.right_ear;
                                        avg_ear = event.avg_ear;
                                        landmarks_cache = Some(landmarks);

                                        // Trigger stare warning notification if prolonged staring
                                        if event.stare_warning {
                                            println!("[420vision::alert] 👁️ Stare warning triggered (>8s without blink)");
                                            crate::notifier::Notifier::notify_stare_warning();
                                            crate::notifier::AudioPlayer::play_stare_warning();
                                        }
                                    } else {
                                        blink_detector.update(&[], now);
                                    }
                                }
                            } else {
                                is_face = false;
                                landmarks_cache = None;
                                blink_detector.update(&[], now);
                            }

                            // Run presence timer
                            let presence_state = presence_timer.update(is_face, now);
                            if presence_state.break_triggered {
                                println!("[420vision::alert] ✨ 20-20-20 Break time triggered!");
                                crate::notifier::Notifier::notify_break_time();
                                crate::notifier::AudioPlayer::play_break_chime();
                            }

                            // Update global AppState metrics
                            {
                                let state_mutex = app_handle.state::<Mutex<AppState>>();
                                let mut state = state_mutex.lock().unwrap();
                                state.status.bpm = current_bpm;
                                state.status.total_blinks_today = total_blinks;
                                state.status.next_break_seconds = presence_state.next_break_seconds;
                                if is_face {
                                    state.status.status_text = "Monitoring Active".to_string();
                                } else {
                                    state.status.status_text = "Away / Paused".to_string();
                                }
                            }

                            // Conditional rendering: emit image stream ONLY when user views Camera Test tab
                            if is_sandbox_viewing {
                                // Extract eye contour landmarks and map from crop-space to full-frame coordinates
                                let eye_indices = [
                                    33, 133, 159, 145, 158, 153, 160, 144, // Left eye
                                    362, 263, 386, 374, 387, 373, 385, 380, // Right eye
                                ];
                                let mut eye_points = Vec::with_capacity(eye_indices.len());
                                if is_face {
                                    if let Some(last_lm) = &landmarks_cache {
                                        let side = w.min(h);
                                        let crop_x = (w - side) / 2;
                                        let crop_y = (h - side) / 2;

                                        for &idx in &eye_indices {
                                            if let Some(lm) = last_lm.get(idx) {
                                                // Remap [0..1] crop coordinate back to [0..1] full-frame space
                                                let full_x = (crop_x as f32 + lm.x * side as f32) / w as f32;
                                                let full_y = (crop_y as f32 + lm.y * side as f32) / h as f32;
                                                eye_points.push(EyeLandmarkPoint { x: full_x, y: full_y });
                                            }
                                        }
                                    }
                                }

                                // Downscale preview image to 320x180 JPEG for zero-lag transmission
                                let thumb = image::imageops::thumbnail(&rgb_img, 320, 180);
                                let mut jpeg_bytes = Vec::new();
                                let mut cursor = Cursor::new(&mut jpeg_bytes);
                                let base64_str = if thumb.write_to(&mut cursor, image::ImageFormat::Jpeg).is_ok() {
                                    Some(format!("data:image/jpeg;base64,{}", BASE64_STANDARD.encode(&jpeg_bytes)))
                                } else {
                                    None
                                };

                                let dto = CameraFrameDto {
                                    width: w as u32,
                                    height: h as u32,
                                    is_face_detected: is_face,
                                    left_ear,
                                    right_ear,
                                    avg_ear,
                                    is_blinking,
                                    total_blinks,
                                    eye_landmarks: eye_points,
                                    image_data_base64: base64_str,
                                };
                                let _ = app_handle.emit("camera-sandbox-frame", dto);
                            }
                        }

                        // Throttle frame rate manually to ~15 FPS to save CPU
                        thread::sleep(Duration::from_millis(67));
                    }
                    Err(nokhwa::NokhwaError::ReadFrameError(_)) 
                    | Err(nokhwa::NokhwaError::OpenDeviceError(_, _)) => {
                        cam_manager.release_camera();
                        cam_manager.is_paused_by_conflict = true;
                        update_status_text(&app_handle, "⚠️ Camera Paused (In Use)");
                        println!("[420vision::camera] Hardware contention detected (Zoom/Meet/FaceTime active). Yielding camera.");
                    }
                    Err(_) => {
                        thread::sleep(Duration::from_millis(500));
                    }
                }
            } else {
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
