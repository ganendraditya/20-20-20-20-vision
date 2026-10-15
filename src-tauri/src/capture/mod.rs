pub mod frame;
#[cfg(all(debug_assertions, target_os = "macos"))]
mod reference;

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
    #[cfg(all(debug_assertions, target_os = "macos"))]
    reference_camera: Option<reference::ReferenceRecorder>,
    is_paused_by_conflict: bool,
}

impl CameraManager {
    pub fn new() -> Self {
        Self {
            camera: None,
            #[cfg(all(debug_assertions, target_os = "macos"))]
            reference_camera: None,
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

        #[cfg(all(debug_assertions, target_os = "macos"))]
        if reference::enabled() {
            let backend = native_api_backend().unwrap_or(nokhwa::utils::ApiBackend::Auto);
            let cameras = query(backend).map_err(|e| e.to_string())?;
            let camera = cameras.get(index).ok_or("Selected camera is unavailable")?;
            self.reference_camera = Some(reference::ReferenceRecorder::start(&camera.misc())?);
            self.is_paused_by_conflict = false;
            return Ok(());
        }

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
                    println!("[420vision::camera] Stream opened successfully with format: {:?}", fmt);
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
        #[cfg(all(debug_assertions, target_os = "macos"))]
        if let Some(camera) = self.reference_camera.take() {
            drop(camera); // Stops AVFoundation before finalizing the recording.
            println!("[420vision::camera] Reference camera stream released.");
        }
    }

    fn has_camera(&self) -> bool {
        #[cfg(all(debug_assertions, target_os = "macos"))]
        if self.reference_camera.is_some() {
            return true;
        }
        self.camera.is_some()
    }

    fn frame(&mut self) -> Result<Option<nokhwa::Buffer>, nokhwa::NokhwaError> {
        #[cfg(all(debug_assertions, target_os = "macos"))]
        if let Some(camera) = &mut self.reference_camera {
            return camera.frame();
        }
        self.camera
            .as_mut()
            .ok_or_else(|| nokhwa::NokhwaError::ReadFrameError("Camera is not open".into()))?
            .frame()
            .map(Some)
    }

    fn save_detector_log(&mut self, line: &str) {
        #[cfg(all(debug_assertions, target_os = "macos"))]
        if let Some(camera) = &mut self.reference_camera {
            camera.log_detector(line);
        }
        #[cfg(not(all(debug_assertions, target_os = "macos")))]
        let _ = line;
    }
}

#[cfg(all(debug_assertions, target_os = "macos"))]
static REFERENCE_STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(all(debug_assertions, target_os = "macos"))]
static REFERENCE_WORKER: Mutex<Option<thread::JoinHandle<()>>> = Mutex::new(None);

/// Normal Quit must finalize an opt-in recording before the process exits.
#[cfg(all(debug_assertions, target_os = "macos"))]
pub(crate) fn stop_reference_capture() {
    if reference::enabled() {
        REFERENCE_STOP.store(true, std::sync::atomic::Ordering::Release);
        if let Some(worker) = REFERENCE_WORKER.lock().unwrap().take() {
            let _ = worker.join();
        }
    }
}

/// Resolve an ONNX model file by name.
///
/// Search order (first existing file wins):
/// 1. `models/<name>` relative to the working directory (`cargo run` / `tauri dev` from repo root)
/// 2. `../models/<name>` (`tauri dev` runs the binary from `src-tauri/`)
/// 3. `<CARGO_MANIFEST_DIR>/../models/<name>` (workspace root, independent of process cwd)
/// 4. `~/.local/share/420vision/models/<name>` (written by `install.sh`)
/// 5. `<data_dir>/420vision/models/<name>` (platform default, e.g. Application Support on macOS)
///
/// If nothing exists, the first candidate is returned so the caller's error names a real path.
fn resolve_model_path(name: &str) -> PathBuf {
    let mut candidates = vec![
        PathBuf::from("models").join(name),
        PathBuf::from("../models").join(name),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../models")
            .join(name),
    ];
    if let Some(home) = dirs::home_dir() {
        candidates.push(home.join(".local/share/420vision/models").join(name));
    }
    if let Some(data) = dirs::data_dir() {
        candidates.push(data.join("420vision/models").join(name));
    }

    candidates
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .unwrap_or_else(|| candidates[0].clone())
}

/// Locate FaceMesh model path
fn get_facemesh_model_path() -> PathBuf {
    resolve_model_path("facemesh.onnx")
}

/// Locate FaceDetector model path
fn get_facedetector_model_path() -> PathBuf {
    resolve_model_path("ultraface.onnx")
}

#[derive(Clone, Copy, Default)]
struct DebugFrameTimings {
    capture_ms: f32,
    decode_ms: f32,
    face_detector_ms: f32,
    face_mesh_preprocess_ms: f32,
    face_mesh_inference_ms: f32,
    blink_detector_ms: f32,
}

fn elapsed_ms(start: Option<Instant>) -> f32 {
    start
        .map(|instant| instant.elapsed().as_secs_f32() * 1000.0)
        .unwrap_or(0.0)
}

fn log_blink_diagnostics(
    variant: &str,
    frame_number: u64,
    app_elapsed_secs: f32,
    capture_wall_ms: Option<u128>,
    capture_interval_ms: f32,
    frame_started: Option<Instant>,
    timings: &DebugFrameTimings,
    face_detected: bool,
    total_blinks: u32,
    diagnostics: crate::detector::state_machine::BlinkDiagnostics,
) -> String {
    let line = format!(
        "[420vision::debug] variant={} f={} t={:.1}s capture_wall_ms={} frame_dt={:.1}ms detector_dt={:.1}ms stage_ms[capture={:.1},decode={:.1},face={:.1},mesh_pre={:.1},mesh_infer={:.1},blink={:.1},total={:.1}] face={} facing={} pitch_down={} raw=({:.3},{:.3}) smooth=({:.3},{:.3}) baseline=({:.3},{:.3}) threshold=({:.3},{:.3}) eye=({:?},{:?}) closure_ms=({:.0},{:.0}) mono={} resting={} decision={:?} total_blinks={}",
        variant,
        frame_number,
        app_elapsed_secs,
        capture_wall_ms.unwrap_or(0),
        capture_interval_ms,
        diagnostics.frame_interval_ms,
        timings.capture_ms,
        timings.decode_ms,
        timings.face_detector_ms,
        timings.face_mesh_preprocess_ms,
        timings.face_mesh_inference_ms,
        timings.blink_detector_ms,
        elapsed_ms(frame_started),
        face_detected,
        diagnostics.is_facing_camera,
        diagnostics.is_downward_reading_gaze,
        diagnostics.raw_left_ear.unwrap_or(-1.0),
        diagnostics.raw_right_ear.unwrap_or(-1.0),
        diagnostics.smoothed_left_ear.unwrap_or(-1.0),
        diagnostics.smoothed_right_ear.unwrap_or(-1.0),
        diagnostics.baseline_left.unwrap_or(-1.0),
        diagnostics.baseline_right.unwrap_or(-1.0),
        diagnostics.threshold_left.unwrap_or(-1.0),
        diagnostics.threshold_right.unwrap_or(-1.0),
        diagnostics.left_status,
        diagnostics.right_status,
        diagnostics.left_closure_ms,
        diagnostics.right_closure_ms,
        diagnostics.is_monocular,
        diagnostics.is_resting,
        diagnostics.decision,
        total_blinks,
    );
    println!("{line}");
    line
}

// Canonical MediaPipe landmark indices for eye contours and essential face structure
pub const FACE_MESH_LANDMARK_COUNT: usize = 468;

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
    let worker = thread::spawn(move || {
        let mut cam_manager = CameraManager::new();
        let mut current_cam_index = 9999; // Force init on first pass

        // Initialize FaceMeshEngine
        let model_path = get_facemesh_model_path();
        let mut vision_engine = match FaceMeshEngine::new(&model_path) {
            Ok(engine) => {
                println!("[420vision::vision] Loaded FaceMesh model from {:?}", model_path);
                Some(engine)
            }
            Err(e) => {
                eprintln!("[420vision::vision] Could not load FaceMesh model ({:?}): {}. Place it in models/ or run the installer for your OS.", model_path, e);
                None
            }
        };

        // Initialize UltraFace detector
        let detector_path = get_facedetector_model_path();
        let mut face_detector = match FaceDetectorEngine::new(&detector_path) {
            Ok(detector) => {
                println!("[420vision::vision] Loaded UltraFace detector from {:?}", detector_path);
                Some(detector)
            }
            Err(e) => {
                eprintln!("[420vision::vision] Could not load UltraFace detector ({:?}): {}. Place it in models/ or run the installer for your OS.", detector_path, e);
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
        // Diagnostics are opt-in because emitting one line per processed frame is intentionally verbose.
        let debug_ear = std::env::var("VISION420_DEBUG_EAR")
            .map(|value| value == "1")
            .unwrap_or(false);
        #[cfg(all(debug_assertions, target_os = "macos"))]
        let debug_ear = debug_ear || reference::enabled();
        #[cfg(all(debug_assertions, target_os = "macos"))]
        let mut ear_ab_detectors: Vec<(&'static str, BlinkDetector)> =
            if reference::ear_ab_test_enabled() {
                println!(
                    "[420vision::debug] EAR A/B enabled: baseline drop=35% alpha=0.40; threshold-only drop=25%; EMA-only alpha=0.70; combined drop=25% alpha=0.70"
                );
                vec![
                    (
                        "threshold25",
                        BlinkDetector::with_experimental_parameters(
                            initial_config.ear_threshold,
                            initial_config.stare_limit_secs,
                            0.40,
                            0.25,
                        ),
                    ),
                    (
                        "alpha70",
                        BlinkDetector::with_experimental_parameters(
                            initial_config.ear_threshold,
                            initial_config.stare_limit_secs,
                            0.70,
                            crate::detector::state_machine::RELATIVE_BLINK_DROP,
                        ),
                    ),
                    (
                        "threshold25_alpha70",
                        BlinkDetector::with_experimental_parameters(
                            initial_config.ear_threshold,
                            initial_config.stare_limit_secs,
                            0.70,
                            0.25,
                        ),
                    ),
                ]
            } else {
                Vec::new()
            };
        if debug_ear {
            println!("[420vision::debug] EAR trace enabled (VISION420_DEBUG_EAR=1)");
        }
        let debug_started = debug_ear.then(Instant::now);
        let mut debug_frame_counter = 0u64;
        let mut last_debug_capture_at: Option<Instant> = None;
        let mut presence_timer = PresenceTimer::new(1200.0, 300.0);
        let mut total_blinks: u32 = 0;
        let mut current_bpm: f32 = 0.0;
        let mut last_recorded_blinks: u32 = 0;
        let mut face_tracker = crate::vision::FaceTracker::default();
        let mut sleep_blocker = crate::notifier::SleepBlocker::new();

        // Pre-allocated landmark buffers to prevent repeated heap re-allocations in hot loop
        let mut eye_points_buf = Vec::with_capacity(EYE_INDICES.len());
        let mut face_points_buf = Vec::with_capacity(FACE_CONTOUR_INDICES.len());
        let mut face_mesh_points_buf = Vec::with_capacity(FACE_MESH_LANDMARK_COUNT);

        loop {
            #[cfg(all(debug_assertions, target_os = "macos"))]
            if REFERENCE_STOP.load(std::sync::atomic::Ordering::Acquire) {
                cam_manager.release_camera();
                sleep_blocker.release();
                break;
            }
            let now = Instant::now();

            // Check state & dynamically synchronize threshold and power config
            let (is_running, selected_index, is_sandbox_viewing, active_threshold, keep_awake_enabled) = {
                let state_mutex = app_handle.state::<Mutex<AppState>>();
                let state = state_mutex.lock().unwrap();
                (
                    state.status.is_running,
                    state.selected_camera_index,
                    state.sandbox_view.is_viewing(),
                    state.config.ear_threshold,
                    state.config.keep_awake_enabled,
                )
            };
            blink_detector.set_threshold(active_threshold);
            #[cfg(all(debug_assertions, target_os = "macos"))]
            for (_, detector) in &mut ear_ab_detectors {
                detector.set_threshold(active_threshold);
            }

            if !is_running {
                cam_manager.release_camera();
                current_cam_index = usize::MAX;
                face_tracker.reset();
                sleep_blocker.release();
                thread::sleep(Duration::from_millis(1000));
                continue;
            }

            // Handle Camera Selection Changes, initial start, or resume
            if current_cam_index != selected_index || !cam_manager.has_camera() {
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
            if cam_manager.has_camera() {
                let frame_started = debug_ear.then(Instant::now);
                let capture_interval_ms = match (frame_started, last_debug_capture_at) {
                    (Some(current), Some(previous)) => {
                        current.duration_since(previous).as_secs_f32() * 1000.0
                    }
                    _ => 0.0,
                };
                if debug_ear {
                    last_debug_capture_at = frame_started;
                }
                let mut timings = DebugFrameTimings::default();
                let frame_result = cam_manager.frame();
                timings.capture_ms = elapsed_ms(frame_started);

                match frame_result {
                    Ok(Some(frame)) => {
                        let (w, h) = (frame.resolution().width() as usize, frame.resolution().height() as usize);
                        let capture_wall_ms = frame
                            .capture_timestamp()
                            .map(|timestamp| timestamp.as_millis());
                        
                        // Decode raw buffer to RGB
                        let mut is_face = false;
                        let mut left_ear = 0.0f32;
                        let mut right_ear = 0.0f32;
                        let mut avg_ear = 0.0f32;
                        let mut is_blinking = false;
                        let mut landmarks_cache = None;
                        let mut head_pose_cache = None;
                        let mut distance_cm = 0.0f32;
                        let mut detector_updated = false;
                        #[cfg(all(debug_assertions, target_os = "macos"))]
                        let mut ear_ab_blink_ms = Vec::with_capacity(ear_ab_detectors.len());

                        let decode_started = debug_ear.then(Instant::now);
                        let decoded_frame = frame.decode_image::<RgbFormat>();
                        timings.decode_ms = elapsed_ms(decode_started);
                        if debug_ear && decoded_frame.is_err() {
                            eprintln!("[420vision::debug] RGB frame decode failed");
                        }
                        if let Ok(rgb_img) = decoded_frame {
                            let raw_bytes = rgb_img.as_raw();

                            // Gatekeeper: Verify facial presence & extract primary dominant bounding box with Adaptive RoI Zoom (Issue #43, #52, #63)
                            let face_detector_started = debug_ear.then(Instant::now);
                            let (face_present, dominant_bbox) = if let Some(detector) = &mut face_detector {
                                match detector.detect_with_adaptive_roi(raw_bytes, w, h, Some(&mut face_tracker)) {
                                    Ok((detected, bbox)) => (detected, bbox),
                                    Err(e) => {
                                        eprintln!("[420vision::vision] FaceDetector error: {}", e);
                                        (true, None) // Fail-safe: fallback to FaceMesh on detector error
                                    }
                                }
                            } else {
                                (true, None) // Detector not available, fallback
                            };
                            timings.face_detector_ms = elapsed_ms(face_detector_started);

                            // Track the active crop origin and scale for mapping landmarks back to full-frame space
                            let mut active_crop = None;

                            if face_present {
                                if let Some(bbox) = dominant_bbox.as_ref() {
                                    let face_w_norm = (bbox.xmax - bbox.xmin).abs().clamp(0.05, 1.0);
                                    distance_cm = (11.0 / face_w_norm).clamp(20.0, 150.0);
                                }

                                // Run ONNX FaceMesh inference prioritizing the dominant user
                                if let Some(engine) = &mut vision_engine {
                                    let crop = FaceBoundingBox::compute_crop_region(dominant_bbox.as_ref(), w, h);
                                    active_crop = Some(crop);

                                    let mesh_preprocess_started = debug_ear.then(Instant::now);
                                    let preprocessed = engine.preprocess(raw_bytes, w, h, dominant_bbox.as_ref());
                                    timings.face_mesh_preprocess_ms = elapsed_ms(mesh_preprocess_started);
                                    let mesh_inference_started = debug_ear.then(Instant::now);
                                    let inference_result = engine.infer(preprocessed);
                                    timings.face_mesh_inference_ms = elapsed_ms(mesh_inference_started);
                                    if let Err(error) = &inference_result {
                                        if debug_ear {
                                            eprintln!(
                                                "[420vision::debug] FaceMesh inference failed: {}",
                                                error
                                            );
                                        }
                                    }
                                    if let Ok(landmarks) = inference_result {
                                        // Evaluate 3D Head Pose Gaze Gate (Issue #48 & #73):
                                        // User is considered facing screen ONLY if not turned away (yaw <= 0.35)
                                        // and not gazing far upward (pitch < 0.15).
                                        let pose_opt = crate::detector::EarCalculator::estimate_head_pose(&landmarks);
                                        let is_facing = pose_opt.map(|pose| pose.is_facing_camera).unwrap_or(true);

                                        is_face = is_facing;
                                        head_pose_cache = pose_opt;

                                        // Run blink detector
                                        let blink_detector_started = debug_ear.then(Instant::now);
                                        let event = blink_detector.update(&landmarks, now);
                                        timings.blink_detector_ms =
                                            elapsed_ms(blink_detector_started);
                                        #[cfg(all(debug_assertions, target_os = "macos"))]
                                        for (_, detector) in &mut ear_ab_detectors {
                                            let started = Instant::now();
                                            detector.update(&landmarks, now);
                                            ear_ab_blink_ms.push(elapsed_ms(Some(started)));
                                        }
                                        detector_updated = true;
                                        is_blinking = event.is_blink;
                                        total_blinks = event.total_blinks;
                                        current_bpm = event.current_bpm;
                                        left_ear = event.left_ear;
                                        right_ear = event.right_ear;
                                        avg_ear = event.avg_ear;
                                        landmarks_cache = Some(landmarks);

                                        // Trigger stare warning notification if prolonged staring
                                        if event.stare_warning {
                                            println!("[420vision::alert] Stare warning triggered (>8s without blink)");
                                            crate::notifier::Notifier::notify_stare_warning();
                                            crate::notifier::AudioPlayer::play_stare_warning();
                                        }
                                    } else {
                                        blink_detector.update(&[], now);
                                        #[cfg(all(debug_assertions, target_os = "macos"))]
                                        for (_, detector) in &mut ear_ab_detectors {
                                            let started = Instant::now();
                                            detector.update(&[], now);
                                            ear_ab_blink_ms.push(elapsed_ms(Some(started)));
                                        }
                                        detector_updated = true;
                                    }
                                }
                            } else {
                                is_face = false;
                                landmarks_cache = None;
                                blink_detector.update(&[], now);
                                #[cfg(all(debug_assertions, target_os = "macos"))]
                                for (_, detector) in &mut ear_ab_detectors {
                                    let started = Instant::now();
                                    detector.update(&[], now);
                                    ear_ab_blink_ms.push(elapsed_ms(Some(started)));
                                }
                                detector_updated = true;
                            }

                            // Presence-Aware Smart Keep-Awake (Issue #62):
                            // Hold display awake only while user is actively looking at screen.
                            // Immediately release assertion when user looks away or leaves.
                            if is_face && keep_awake_enabled {
                                sleep_blocker.acquire();
                            } else {
                                sleep_blocker.release();
                            }

                            // Run presence timer
                            let presence_state = presence_timer.update(is_face, now);
                            if presence_state.break_triggered {
                                println!("[420vision::alert] 20-20-20 Break time triggered!");
                                crate::notifier::Notifier::notify_break_time();
                                crate::notifier::AudioPlayer::play_break_chime();
                            }
                            if presence_state.break_completed {
                                println!("[420vision::alert] 20-20-20 Break completed!");
                                crate::notifier::AudioPlayer::play_break_completed_chime();

                                let delta_blinks = total_blinks.saturating_sub(last_recorded_blinks);
                                last_recorded_blinks = total_blinks;

                                if let Ok(storage) = crate::storage::AnalyticsDb::open() {
                                    let _ = storage.record_break_today(current_bpm, delta_blinks, true, false, 1200.0);
                                }
                            }

                            // Update global AppState metrics
                            {
                                let state_mutex = app_handle.state::<Mutex<AppState>>();
                                let mut state = state_mutex.lock().unwrap();
                                state.status.bpm = current_bpm;
                                state.status.total_blinks_today = total_blinks;
                                state.status.next_break_seconds = if presence_state.break_phase == crate::timer::BreakPhase::BreakPending {
                                    presence_state.break_remaining_seconds
                                } else {
                                    presence_state.next_break_seconds
                                };
                                if presence_state.break_phase == crate::timer::BreakPhase::BreakPending {
                                    if !is_face {
                                        state.status.status_text = format!("Break Counting Down ({}s remaining) 🟢", presence_state.break_remaining_seconds);
                                    } else {
                                        state.status.status_text = format!("Break Paused ({}s remaining) - Look away or up", presence_state.break_remaining_seconds);
                                    }
                                } else if is_face {
                                    state.status.status_text = "Monitoring Active".to_string();
                                } else {
                                    state.status.status_text = "Away / Paused".to_string();
                                }
                            }

                            // Conditional rendering: emit image stream ONLY when user views Camera Test tab
                            if is_sandbox_viewing {
                                eye_points_buf.clear();
                                face_points_buf.clear();
                                face_mesh_points_buf.clear();

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

                                        for lm in last_lm.iter().take(FACE_MESH_LANDMARK_COUNT) {
                                            let full_x =
                                                (crop_x as f32 + lm.x * side as f32) / w as f32;
                                            let full_y =
                                                (crop_y as f32 + lm.y * side as f32) / h as f32;
                                            face_mesh_points_buf.push(LandmarkPoint {
                                                x: full_x,
                                                y: full_y,
                                            });
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

                                let (yaw_deg, pitch_deg, roll_deg, is_resting_gaze) = match head_pose_cache {
                                    Some(p) => (
                                        (p.yaw_deg * 10.0).round() / 10.0,
                                        (p.pitch_deg * 10.0).round() / 10.0,
                                        (p.roll_deg * 10.0).round() / 10.0,
                                        p.is_resting_gaze,
                                    ),
                                    None => (0.0, 0.0, 0.0, false),
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
                                    blink_diagnostics: detector_updated.then(|| {
                                        frame::BlinkDiagnosticsDto::from(blink_detector.diagnostics())
                                    }),
                                    eye_landmarks: eye_points_buf.clone(),
                                    face_landmarks: face_points_buf.clone(),
                                    face_mesh_landmarks: face_mesh_points_buf.clone(),
                                    image_data_base64: base64_str,
                                    yaw_deg,
                                    pitch_deg,
                                    roll_deg,
                                    distance_cm: (distance_cm * 10.0).round() / 10.0,
                                    is_resting_gaze,
                                };
                                let _ = app_handle.emit("camera-sandbox-frame", dto);
                            }
                        }

                        if debug_ear && detector_updated {
                            debug_frame_counter += 1;
                            let line = log_blink_diagnostics(
                                "baseline",
                                debug_frame_counter,
                                debug_started
                                    .map(|started| started.elapsed().as_secs_f32())
                                    .unwrap_or(0.0),
                                capture_wall_ms,
                                capture_interval_ms,
                                frame_started,
                                &timings,
                                is_face,
                                total_blinks,
                                blink_detector.diagnostics(),
                            );
                            cam_manager.save_detector_log(&line);
                            #[cfg(all(debug_assertions, target_os = "macos"))]
                            for (index, (variant, detector)) in
                                ear_ab_detectors.iter().enumerate()
                            {
                                let mut variant_timings = timings;
                                variant_timings.blink_detector_ms =
                                    ear_ab_blink_ms.get(index).copied().unwrap_or_default();
                                let line = log_blink_diagnostics(
                                    variant,
                                    debug_frame_counter,
                                    debug_started
                                        .map(|started| started.elapsed().as_secs_f32())
                                        .unwrap_or(0.0),
                                    capture_wall_ms,
                                    capture_interval_ms,
                                    frame_started,
                                    &variant_timings,
                                    is_face,
                                    detector.total_blinks(),
                                    detector.diagnostics(),
                                );
                                cam_manager.save_detector_log(&line);
                            }
                        }

                        // Adaptive Frame Pacing (Issue #40):
                        // When user is actively present and facing camera: 15 FPS (~67ms sleep)
                        // When user is Away / Paused or looking away: throttle to 5 FPS (~200ms sleep) to conserve CPU & battery
                        let throttle_ms = if is_face { 67 } else { 200 };
                        thread::sleep(Duration::from_millis(throttle_ms));
                    }
                    Ok(None) => continue,
                    Err(error @ nokhwa::NokhwaError::ReadFrameError(_))
                    | Err(error @ nokhwa::NokhwaError::OpenDeviceError(_, _)) => {
                        if debug_ear {
                            eprintln!(
                                "[420vision::debug] camera frame acquisition failed; yielding device: {}",
                                error
                            );
                        }
                        cam_manager.release_camera();
                        cam_manager.is_paused_by_conflict = true;
                        sleep_blocker.release();
                        update_status_text(&app_handle, "Camera Paused (In Use)");
                        println!("[420vision::camera] Hardware contention detected (Zoom/Meet/FaceTime active). Yielding camera.");
                    }
                    Err(error) => {
                        if debug_ear {
                            eprintln!(
                                "[420vision::debug] camera frame acquisition failed: {}",
                                error
                            );
                        }
                        sleep_blocker.release();
                        thread::sleep(Duration::from_millis(500));
                    }
                }
            } else {
                sleep_blocker.release();
                update_status_text(&app_handle, "No Camera Detected");
                thread::sleep(Duration::from_secs(5));
            }
        }
    });
    #[cfg(all(debug_assertions, target_os = "macos"))]
    if reference::enabled() {
        *REFERENCE_WORKER.lock().unwrap() = Some(worker);
    }
    #[cfg(not(all(debug_assertions, target_os = "macos")))]
    let _ = worker;
}

fn update_status_text(app: &AppHandle, text: &str) {
    let state_mutex = app.state::<Mutex<AppState>>();
    let mut state = state_mutex.lock().unwrap();
    state.status.status_text = text.to_string();
}
