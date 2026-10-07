use vision420_lib::detector::{EarCalculator, EyeCalibrator};
use vision420_lib::vision::Landmark3D;

fn create_synthetic_landmarks(eye_height: f32, eye_width: f32) -> Vec<Landmark3D> {
    let mut landmarks = vec![Landmark3D { x: 0.0, y: 0.0, z: 0.0 }; 468];

    // Left eye setup (33, 133, 159, 145, 158, 153, 160, 144)
    landmarks[33] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    landmarks[133] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    landmarks[159] = Landmark3D { x: eye_width * 0.5, y: eye_height, z: 0.0 };
    landmarks[145] = Landmark3D { x: eye_width * 0.5, y: 0.0, z: 0.0 };
    landmarks[158] = Landmark3D { x: eye_width * 0.3, y: eye_height, z: 0.0 };
    landmarks[153] = Landmark3D { x: eye_width * 0.3, y: 0.0, z: 0.0 };
    landmarks[160] = Landmark3D { x: eye_width * 0.7, y: eye_height, z: 0.0 };
    landmarks[144] = Landmark3D { x: eye_width * 0.7, y: 0.0, z: 0.0 };

    // Right eye setup (362, 263, 386, 374, 387, 373, 385, 380)
    landmarks[362] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    landmarks[263] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    landmarks[386] = Landmark3D { x: eye_width * 0.5, y: eye_height, z: 0.0 };
    landmarks[374] = Landmark3D { x: eye_width * 0.5, y: 0.0, z: 0.0 };
    landmarks[387] = Landmark3D { x: eye_width * 0.3, y: eye_height, z: 0.0 };
    landmarks[373] = Landmark3D { x: eye_width * 0.3, y: 0.0, z: 0.0 };
    landmarks[385] = Landmark3D { x: eye_width * 0.7, y: eye_height, z: 0.0 };
    landmarks[380] = Landmark3D { x: eye_width * 0.7, y: 0.0, z: 0.0 };

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

#[test]
fn test_head_pose_yaw_symmetry_gate() {
    let mut frontal_landmarks = vec![Landmark3D { x: 0.0, y: 0.0, z: 0.0 }; 468];
    // Outer eye corners: 33 at x=2.0, 263 at x=8.0 (width = 6.0, mid = 5.0)
    frontal_landmarks[33] = Landmark3D { x: 2.0, y: 3.0, z: 0.0 };
    frontal_landmarks[263] = Landmark3D { x: 8.0, y: 3.0, z: 0.0 };
    // Nose tip: exactly at x=5.1 (slight natural asymmetry, offset = 0.1, yaw = 0.1 / 6.0 = 0.016)
    frontal_landmarks[1] = Landmark3D { x: 5.1, y: 5.0, z: 0.0 };

    let frontal_pose = EarCalculator::estimate_head_pose(&frontal_landmarks).expect("Should estimate pose");
    assert!(frontal_pose.is_facing_camera, "Frontal face must be approved");
    assert!(frontal_pose.yaw_ratio < 0.10);

    // Profile landmarks: user turned head significantly to the side
    let mut profile_landmarks = frontal_landmarks.clone();
    // Nose tip shifted far to x=7.8 (offset = 2.8, yaw = 2.8 / 6.0 = 0.466 > 0.35)
    profile_landmarks[1] = Landmark3D { x: 7.8, y: 5.0, z: 0.0 };

    let profile_pose = EarCalculator::estimate_head_pose(&profile_landmarks).expect("Should estimate pose");
    assert!(!profile_pose.is_facing_camera, "Side profile head angle must be rejected");
    assert!(profile_pose.yaw_ratio > 0.40);
    assert!(profile_pose.is_resting_gaze, "Turned head qualifies as resting gaze");

    // Upward pitch gaze: user tilts head up to look at ceiling / high window
    let mut upward_landmarks = frontal_landmarks.clone();
    // In frontal pose, eye midpoint y=3.0, baseline nose y=5.7 (downward offset = 2.7 = 0.45 * 6.0)
    // When tilted up, nose tip moves closer to or above eye level (e.g. y=3.2)
    upward_landmarks[1] = Landmark3D { x: 5.0, y: 3.2, z: 0.0 };
    let upward_pose = EarCalculator::estimate_head_pose(&upward_landmarks).expect("Should estimate pose");
    assert!(!upward_pose.is_facing_camera, "Looking upward must not be counted as facing screen");
    assert!(upward_pose.is_resting_gaze, "Looking upward qualifies as resting gaze");
    assert!(upward_pose.pitch_ratio >= 0.15);
}
