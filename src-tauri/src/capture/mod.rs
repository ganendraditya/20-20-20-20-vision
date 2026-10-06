pub mod frame;

pub use frame::{CameraFrameDto, LandmarkPoint};

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
use crate::vision::{FaceBoundingBox, FaceDetectorEngine, FaceMeshEngine};
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

/// Locate or fallback FaceDetector model path
fn get_facedetector_model_path() -> PathBuf {
    // 1. Check local models directory relative to current working directory
    let local = PathBuf::from("models/ultraface.onnx");
    if local.exists() {
        return local;
    }
    // 2. Check installed bundle directory
    if let Some(share) = dirs::data_dir() {
        let installed = share.join("420vision/models/ultraface.onnx");
        if installed.exists() {
            return installed;
        }
    }
    local
}

// Canonical MediaPipe landmark indices for eye contours and essential face structure
pub const EYE_INDICES: [usize; 16] = [
    33, 133, 159, 145, 158, 153, 160, 144, // Left eye
    362, 263, 386, 374, 387, 373, 385, 380, // Right eye
];

pub const FACE_CONTOUR_INDICES: [usize; 75] = [
    // Jawline (17 points)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    // Left eyebrow (5 points)
    70, 63, 105, 66, 107,
    // Right eyebrow (5 points)
    336, 296, 334, 293, 300,
    // Nose bridge & tip (9 points)
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
    // Outer lips (12 points)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
];

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

        // Initialize UltraFace detector
        let detector_path = get_facedetector_model_path();
        let mut face_detector = match FaceDetectorEngine::new(&detector_path) {
            Ok(detector) => {
                println!("[420vision::vision] ✅ Loaded UltraFace detector from {:?}", detector_path);
                Some(detector)
            }
            Err(e) => {
                eprintln!("[420vision::vision] ⚠️ Could not load UltraFace detector ({:?}): {}", detector_path, e);
                None
            }
        };

        // Initialize BlinkDetector and PresenceTimer with loaded config threshold
        let initial_config = {
            let state_mutex = app_handle.state::<Mutex<AppState>>();
            let state = state_mutex.lock().unwrap();
            state.config.clone()
        };
        let mut blink_detector = BlinkDetector::new(initial_config.ear_threshold, initial_config.stare_limit_secs);
        let mut presence_timer = PresenceTimer::new(1200.0, 300.0);
        let mut face_tracker = crate::vision::FaceTracker::default();

        // Pre-allocated landmark buffers to prevent repeated heap re-allocations in hot loop
        let mut eye_points_buf = Vec::with_capacity(EYE_INDICES.len());
        let mut face_points_buf = Vec::with_capacity(FACE_CONTOUR_INDICES.len());

        loop {
            let now = Instant::now();

            // Check state & dynamically synchronize threshold from config
            let (is_running, selected_index, is_sandbox_viewing, active_threshold) = {
                let state_mutex = app_handle.state::<Mutex<AppState>>();
                let state = state_mutex.lock().unwrap();
                (
                    state.status.is_running,
                    state.selected_camera_index,
                    state.is_sandbox_viewing,
                    state.config.ear_threshold,
                )
            };
            blink_detector.set_threshold(active_threshold);

            if !is_running {
                cam_manager.release_camera();
                current_cam_index = usize::MAX;
                face_tracker.reset();
                thread::sleep(Duration::from_millis(1000));
                continue;
            }

            // Handle Camera Selection Changes, initial start, or resume
            if current_cam_index != selected_index || cam_manager.camera.is_none() {
                cam_manager.release_camera();
                face_tracker.reset(); // Clear stale spatial state from previous camera
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

                            // Gatekeeper: Verify facial presence & extract primary dominant bounding box with Sticky Tracking (Issue #43 & #52)
                            let (face_present, dominant_bbox) = if let Some(detector) = &mut face_detector {
                                let det_input = detector.preprocess(raw_bytes, w, h);
                                match detector.detect_faces_and_track(det_input, Some(&mut face_tracker)) {
                                    Ok((detected, _, bbox)) => (detected, bbox),
                                    Err(e) => {
                                        eprintln!("[420vision::vision] FaceDetector error: {}", e);
                                        (true, None) // Fail-safe: fallback to FaceMesh on detector error
                                    }
                                }
                            } else {
                                (true, None) // Detector not available, fallback
                            };

                            // Track the active crop origin and scale for mapping landmarks back to full-frame space
                            let mut active_crop = None;

                            if face_present {
                                // Run ONNX FaceMesh inference prioritizing the dominant user
                                if let Some(engine) = &mut vision_engine {
                                    let crop = FaceBoundingBox::compute_crop_region(dominant_bbox.as_ref(), w, h);
                                    active_crop = Some(crop);

                                    let preprocessed = engine.preprocess(raw_bytes, w, h, dominant_bbox.as_ref());
                                    if let Ok(landmarks) = engine.infer(preprocessed) {
                                        // Evaluate Head Pose Yaw Gate:
                                        // If user is severely turned away (yaw_ratio > 0.35), consider them "Away / Not looking"
                                        let is_facing = crate::detector::EarCalculator::estimate_head_pose(&landmarks)
                                            .map(|pose| pose.is_facing_camera)
                                            .unwrap_or(true);

                                        is_face = is_facing;

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
                                eye_points_buf.clear();
                                face_points_buf.clear();

                                if is_face {
                                    if let Some(last_lm) = &landmarks_cache {
                                        let (crop_x, crop_y, side) = active_crop.unwrap_or_else(|| {
                                            let s = w.min(h);
                                            ((w - s) / 2, (h - s) / 2, s)
                                        });

                                        for &idx in &EYE_INDICES {
                                            if let Some(lm) = last_lm.get(idx) {
                                                let full_x = (crop_x as f32 + lm.x * side as f32) / w as f32;
                                                let full_y = (crop_y as f32 + lm.y * side as f32) / h as f32;
                                                eye_points_buf.push(LandmarkPoint { x: full_x, y: full_y });
                                            }
                                        }

                                        for &idx in &FACE_CONTOUR_INDICES {
                                            if let Some(lm) = last_lm.get(idx) {
                                                let full_x = (crop_x as f32 + lm.x * side as f32) / w as f32;
                                                let full_y = (crop_y as f32 + lm.y * side as f32) / h as f32;
                                                face_points_buf.push(LandmarkPoint { x: full_x, y: full_y });
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
                                    eye_landmarks: eye_points_buf.clone(),
                                    face_landmarks: face_points_buf.clone(),
                                    image_data_base64: base64_str,
                                };
                                let _ = app_handle.emit("camera-sandbox-frame", dto);
                            }
                        }

                        // Adaptive Frame Pacing (Issue #40):
                        // When user is actively present and facing camera: 15 FPS (~67ms sleep)
                        // When user is Away / Paused or looking away: throttle to 5 FPS (~200ms sleep) to conserve CPU & battery
                        let throttle_ms = if is_face { 67 } else { 200 };
                        thread::sleep(Duration::from_millis(throttle_ms));
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
