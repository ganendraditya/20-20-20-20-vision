use vision420_lib::detector::{EarCalculator, EyeCalibrator};
use vision420_lib::vision::Landmark3D;

fn create_synthetic_landmarks(eye_height: f32, eye_width: f32) -> Vec<Landmark3D> {
    let mut landmarks = vec![Landmark3D { x: 0.0, y: 0.0, z: 0.0 }; 468];

    // Left eye setup
    // H: (362, 263)
    landmarks[362] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    landmarks[263] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    // V1: (385, 380)
    landmarks[385] = Landmark3D { x: eye_width * 0.33, y: eye_height, z: 0.0 };
    landmarks[380] = Landmark3D { x: eye_width * 0.33, y: 0.0, z: 0.0 };
    // V2: (387, 373)
    landmarks[387] = Landmark3D { x: eye_width * 0.66, y: eye_height, z: 0.0 };
    landmarks[373] = Landmark3D { x: eye_width * 0.66, y: 0.0, z: 0.0 };

    // Right eye setup
    // H: (33, 133)
    landmarks[33] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    landmarks[133] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    // V1: (160, 144)
    landmarks[160] = Landmark3D { x: eye_width * 0.33, y: eye_height, z: 0.0 };
    landmarks[144] = Landmark3D { x: eye_width * 0.33, y: 0.0, z: 0.0 };
    // V2: (158, 153)
    landmarks[158] = Landmark3D { x: eye_width * 0.66, y: eye_height, z: 0.0 };
    landmarks[153] = Landmark3D { x: eye_width * 0.66, y: 0.0, z: 0.0 };

    landmarks
}

#[test]
fn test_ear_calculation_open_vs_closed() {
    let mut calc = EarCalculator::new(0.5);

    // Open eyes: height 6.0, width 10.0 -> EAR = (6.0 + 6.0) / (2 * 10.0) = 0.6
    let open_landmarks = create_synthetic_landmarks(6.0, 10.0);
    let metrics_open = calc.calculate(&open_landmarks).expect("Should compute metrics");
    assert!((metrics_open.avg_ear - 0.6).abs() < 1e-4);
    assert!((metrics_open.smoothed_ear - 0.6).abs() < 1e-4);

    // Closed eyes: height 1.0, width 10.0 -> EAR = (1.0 + 1.0) / (2 * 10.0) = 0.1
    let closed_landmarks = create_synthetic_landmarks(1.0, 10.0);
    let metrics_closed = calc.calculate(&closed_landmarks).expect("Should compute metrics");
    assert!((metrics_closed.avg_ear - 0.1).abs() < 1e-4);
    // EMA smoothing: 0.5 * 0.1 + 0.5 * 0.6 = 0.35
    assert!((metrics_closed.smoothed_ear - 0.35).abs() < 1e-4);
}

#[test]
fn test_insufficient_landmarks_handling() {
    let mut calc = EarCalculator::new(0.3);
    let insufficient = vec![Landmark3D { x: 0.0, y: 0.0, z: 0.0 }; 100];
    assert!(calc.calculate(&insufficient).is_none());
}

#[test]
fn test_eye_calibration_threshold_optimization() {
    let mut calibrator = EyeCalibrator::new();

    // Record open eye samples (around 0.32)
    for sample in &[0.30, 0.31, 0.32, 0.33, 0.32] {
        calibrator.add_open_sample(*sample);
    }

    // Record closed eye samples (around 0.12)
    for sample in &[0.11, 0.12, 0.13, 0.12] {
        calibrator.add_closed_sample(*sample);
    }

    // Median open = 0.32, Median closed = 0.12
    // Threshold = 0.12 + (0.32 - 0.12) * 0.38 = 0.12 + 0.076 = 0.196
    let threshold = calibrator.finalize().expect("Calibration should succeed");
    assert!((threshold - 0.196).abs() < 1e-3);
}
