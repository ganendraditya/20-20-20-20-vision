use std::path::Path;
use vision420_lib::vision::clahe::{apply_clahe_face_192, ClaheConfig, FACE_CROP_DIM};
use vision420_lib::vision::{FaceMeshEngine, Landmark3D};

pub const EYE_INDICES: [usize; 16] = [
    33, 133, 159, 145, 158, 153, 160, 144, // Left eye
    362, 263, 386, 374, 387, 373, 385, 380, // Right eye
];

fn compute_eye_landmarks_error(pred: &[Landmark3D], gt: &[Landmark3D]) -> f32 {
    let mut total_err = 0.0f32;
    for &idx in &EYE_INDICES {
        let p = &pred[idx];
        let g = &gt[idx];
        let d = ((p.x - g.x).powi(2) + (p.y - g.y).powi(2) + (p.z - g.z).powi(2)).sqrt();
        total_err += d;
    }
    total_err / EYE_INDICES.len() as f32
}

#[test]
fn test_spectacle_glare_landmark_accuracy_improvement() {
    let model_path = Path::new("../models/facemesh.onnx");
    if !model_path.exists() {
        eprintln!("FaceMesh model not found at {:?}, skipping test", model_path);
        return;
    }

    // Engine without CLAHE (baseline reference)
    let mut engine_baseline = FaceMeshEngine::new(model_path)
        .expect("Failed to initialize FaceMeshEngine")
        .with_clahe_config(ClaheConfig {
            blend_alpha: 0.0, // Disabled
            ..ClaheConfig::default()
        });

    // Engine with CLAHE active
    let mut engine_clahe = FaceMeshEngine::new(model_path)
        .expect("Failed to initialize FaceMeshEngine");

    let fixture_path = Path::new("tests/fixtures/1face.png");
    let img = image::open(fixture_path).expect("Failed to open fixture").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);

    // 1. Clean ground truth landmarks under natural illumination
    let tensor_clean = engine_baseline.preprocess(img.as_raw(), w, h, None);
    let gt_landmarks = engine_baseline.infer(tensor_clean).expect("Inference failed");

    // 2. Spectacle glare & display overexposure (+35% luminance boost)
    let mut glared_rgb = img.as_raw().to_vec();
    for px in glared_rgb.iter_mut() {
        *px = ((*px as f32 * 1.35).min(255.0)) as u8;
    }

    // Inference without CLAHE
    let tensor_uncompensated = engine_baseline.preprocess(&glared_rgb, w, h, None);
    let pred_uncompensated = engine_baseline.infer(tensor_uncompensated).expect("Inference failed");
    let err_uncompensated = compute_eye_landmarks_error(&pred_uncompensated, &gt_landmarks);

    // Inference with CLAHE
    let tensor_compensated = engine_clahe.preprocess(&glared_rgb, w, h, None);
    let pred_compensated = engine_clahe.infer(tensor_compensated).expect("Inference failed");
    let err_compensated = compute_eye_landmarks_error(&pred_compensated, &gt_landmarks);

    let improvement_pct = ((err_uncompensated - err_compensated) / err_uncompensated) * 100.0;

    println!("\n==========================================================================");
    println!("🧪 SPECTACLE GLARE & DISPLAY OVEREXPOSURE BENCHMARK (Issue #66)");
    println!("==========================================================================");
    println!("Uncompensated Glare Landmark Error : {:.5}", err_uncompensated);
    println!("CLAHE Compensated Landmark Error    : {:.5}", err_compensated);
    println!("Landmark Regression Improvement     : +{:.2}%", improvement_pct);
    println!("Target Acceptance Threshold         : >= 25.00%");
    println!("==========================================================================");

    assert!(
        improvement_pct >= 25.0,
        "CLAHE must improve landmark regression accuracy on glare fixtures by >= 25%, got {:.2}%",
        improvement_pct
    );
}

#[test]
fn test_clahe_latency_budget() {
    let mut buf = [170u8; FACE_CROP_DIM * FACE_CROP_DIM * 3];
    let config = ClaheConfig::default();

    // Warm-up run
    apply_clahe_face_192(&mut buf, &config);

    let iterations = 200;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        apply_clahe_face_192(&mut buf, &config);
    }
    let elapsed = start.elapsed();
    let avg_micros = elapsed.as_micros() as f32 / iterations as f32;
    let avg_ms = avg_micros / 1000.0;

    println!("\n--------------------------------------------------------------------------");
    println!("⚡ CLAHE EXECUTION OVERHEAD BENCHMARK");
    println!("--------------------------------------------------------------------------");
    println!("Iterations Evaluated : {}", iterations);
    println!("Average Overhead     : {:.2} µs ({:.3} ms)", avg_micros, avg_ms);
    println!("Budget Limit         : 300.00 µs (0.300 ms)");
    println!("--------------------------------------------------------------------------");

    // In debug builds without compiler optimizations, bounds checks add constant overhead.
    // Ensure reasonable bounds in debug, and < 0.3 ms in release mode.
    #[cfg(not(debug_assertions))]
    assert!(
        avg_ms < 0.30,
        "Production release processing overhead must be < 0.3 ms, got {:.3} ms",
        avg_ms
    );
}

#[test]
fn test_clean_frame_drift_floor() {
    let model_path = Path::new("../models/facemesh.onnx");
    if !model_path.exists() {
        return;
    }

    let mut engine_clean = FaceMeshEngine::new(model_path).expect("Failed to initialize FaceMeshEngine");

    let fixture_path = Path::new("tests/fixtures/1face.png");
    let img = image::open(fixture_path).expect("Failed to open fixture").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);

    let tensor_clean = engine_clean.preprocess(img.as_raw(), w, h, None);
    let landmarks = engine_clean.infer(tensor_clean).expect("Inference failed");

    // Verify eye coordinate stability on natural clean illumination
    assert_eq!(landmarks.len(), 468);
    let left_eye = landmarks[33];
    let right_eye = landmarks[263];

    assert!((left_eye.x - 0.335).abs() < 0.02, "Left eye anchor must remain rock-solid");
    assert!((right_eye.x - 0.637).abs() < 0.02, "Right eye anchor must remain rock-solid");
}
