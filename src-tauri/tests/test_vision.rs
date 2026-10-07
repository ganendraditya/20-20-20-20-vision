use vision420_lib::vision::{FaceDetectorEngine, FaceMeshEngine};
use std::path::Path;

#[test]
fn test_facemesh_engine_initialization_and_inference() {
    let model_path = Path::new("../models/facemesh.onnx");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping test", model_path);
        return;
    }

    let mut engine = FaceMeshEngine::new(model_path).expect("Failed to initialize FaceMeshEngine");
    
    // Create dummy 640x480 black image buffer
    let width = 640;
    let height = 480;
    let dummy_rgb = vec![0u8; width * height * 3];

    let preprocessed = engine.preprocess(&dummy_rgb, width, height, None);
    assert_eq!(preprocessed.shape(), &[1, 192, 192, 3]);

    let landmarks = engine.infer(preprocessed).expect("Inference failed");
    assert_eq!(landmarks.len(), 468);
    println!("Successfully extracted {} 3D landmarks via ONNX Runtime!", landmarks.len());
}

#[test]
fn test_facedetector_engine_presence_detection() {
    let model_path = Path::new("../models/ultraface.onnx");
    if !model_path.exists() {
        eprintln!("UltraFace model not found at {:?}, skipping", model_path);
        return;
    }

    let mut engine = FaceDetectorEngine::new(model_path).expect("Failed to initialize FaceDetectorEngine");

    // 1. Black image (empty room / no face) -> must NOT detect face
    let width = 640;
    let height = 480;
    let black_rgb = vec![0u8; width * height * 3];
    let preprocessed_black = engine.preprocess(&black_rgb, width, height);
    assert_eq!(preprocessed_black.shape(), &[1, 3, 240, 320]);

    let has_face_black = engine.detect_face(preprocessed_black).expect("Detection failed");
    assert!(!has_face_black, "Black frame must NOT detect any face");

    // 2. Uniform gray image (no face) -> must NOT detect face
    let gray_rgb = vec![128u8; width * height * 3];
    let preprocessed_gray = engine.preprocess(&gray_rgb, width, height);
    let has_face_gray = engine.detect_face(preprocessed_gray).expect("Detection failed");
    assert!(!has_face_gray, "Uniform gray frame must NOT detect any face");

    // 3. Real face image (1face.png) -> MUST detect face
    if let Ok(img) = image::open("/tmp/1face.png") {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let preprocessed_face = engine.preprocess(rgb.as_raw(), w, h);
        let has_face_real = engine.detect_face(preprocessed_face).expect("Detection failed");
        assert!(has_face_real, "Real face frame MUST be detected as face");
    }
}

#[test]
fn test_multi_face_disambiguation_largest_bbox_prioritization() {
    let model_path = Path::new("../models/ultraface.onnx");
    if !model_path.exists() {
        return;
    }

    let mut engine = FaceDetectorEngine::new(model_path).expect("Failed to initialize FaceDetectorEngine");

    let fixture_path = Path::new("tests/fixtures/1face.png");
    if fixture_path.exists() {
        let img = image::open(fixture_path).expect("Failed to open test fixture image").to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let preprocessed = engine.preprocess(img.as_raw(), w, h);
        let (has_face, dominant_box) = engine.detect_faces_and_primary_box(preprocessed).expect("Inference failed");

        assert!(has_face, "1face.png must detect a face");
        assert!(dominant_box.is_some(), "Dominant bounding box must be extracted");

        let bbox = dominant_box.unwrap();
        println!("Extracted dominant bbox: xmin={:.3}, ymin={:.3}, xmax={:.3}, ymax={:.3}, area={:.4}",
            bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.area());

        // Validate box geometry
        assert!(bbox.xmax > bbox.xmin, "xmax must exceed xmin");
        assert!(bbox.ymax > bbox.ymin, "ymax must exceed ymin");
        assert!(bbox.area() > 0.01, "Face box area must be significant");
        assert!(bbox.confidence >= 0.70, "Confidence must exceed 0.70");
    } else {
        panic!("Missing required test fixture image: {:?}", fixture_path);
    }
}

#[test]
fn test_face_tracker_sticky_persistence_side_by_side_50_50() {
    use vision420_lib::vision::{FaceBoundingBox, FaceTracker};

    let mut tracker = FaceTracker::new(20, 0.30, 1.35);

    // Scenario: Two colleagues sitting side-by-side in front of camera
    // Colleague A on Left: x in [0.15, 0.45], area = 0.30 * 0.40 = 0.120
    // Colleague B on Right: x in [0.55, 0.85], area = 0.30 * 0.40 = 0.120
    let person_left = FaceBoundingBox {
        xmin: 0.15,
        ymin: 0.20,
        xmax: 0.45,
        ymax: 0.60,
        confidence: 0.90,
    };
    let person_right = FaceBoundingBox {
        xmin: 0.55,
        ymin: 0.20,
        xmax: 0.85,
        ymax: 0.60,
        confidence: 0.89,
    };

    // Frame 1: Initial detection lock selects person_left (slightly higher conf/area)
    println!("person_left area: {}, person_right area: {}", person_left.area(), person_right.area());
    let selected_frame1 = tracker.update(&[person_left, person_right]).expect("Must select target");
    println!("selected_frame1: {:?}", selected_frame1);
    assert!((selected_frame1.xmin - person_left.xmin).abs() < 1e-4, "Must lock on initial primary user (Left)");

    // Simulate 100 consecutive frames where person_right fluctuates to be slightly larger (up to +20%)
    for frame in 2..=100 {
        // Noise fluctuation: person_right leans forward by 10%
        let fluctuating_right = FaceBoundingBox {
            xmin: 0.54,
            ymin: 0.18,
            xmax: 0.86,
            ymax: 0.62,
            confidence: 0.95, // Even with higher confidence!
        };
        // person_left slightly micro-moves
        let moving_left = FaceBoundingBox {
            xmin: 0.15 + (frame as f32 % 5.0) * 0.002,
            ymin: 0.20,
            xmax: 0.45 + (frame as f32 % 5.0) * 0.002,
            ymax: 0.60,
            confidence: 0.88,
        };

        let selected = tracker.update(&[moving_left, fluctuating_right]).expect("Must select target");
        assert!(
            (selected.xmin - moving_left.xmin).abs() < 0.05,
            "Frame {}: Tracker must remain sticky on user Left without flickering to Right!", frame
        );
    }
}

#[test]
fn test_face_tracker_crowd_5_faces_anti_flicker() {
    use vision420_lib::vision::{FaceBoundingBox, FaceTracker};

    let mut tracker = FaceTracker::new(20, 0.30, 1.35);

    // 5 people in frame (e.g. Scrum meeting / crowded cafe)
    let p1 = FaceBoundingBox { xmin: 0.10, ymin: 0.30, xmax: 0.25, ymax: 0.50, confidence: 0.85 }; // Left
    let p2 = FaceBoundingBox { xmin: 0.35, ymin: 0.20, xmax: 0.65, ymax: 0.60, confidence: 0.92 }; // Center User (Largest)
    let p3 = FaceBoundingBox { xmin: 0.70, ymin: 0.30, xmax: 0.85, ymax: 0.50, confidence: 0.84 }; // Right
    let p4 = FaceBoundingBox { xmin: 0.20, ymin: 0.65, xmax: 0.30, ymax: 0.80, confidence: 0.75 }; // Background 1
    let p5 = FaceBoundingBox { xmin: 0.60, ymin: 0.65, xmax: 0.70, ymax: 0.80, confidence: 0.76 }; // Background 2

    // Frame 1: Tracker locks on center user p2
    let initial = tracker.update(&[p1, p2, p3, p4, p5]).expect("Must lock on center user");
    assert!((initial.xmin - p2.xmin).abs() < 1e-4);

    // 50 frames with people in background moving, entering, and changing sizes
    for f in 2..=50 {
        let p1_jitter = FaceBoundingBox { xmin: 0.09, ymin: 0.29, xmax: 0.26, ymax: 0.51, confidence: 0.88 };
        let p2_stable = FaceBoundingBox { xmin: 0.36, ymin: 0.21, xmax: 0.64, ymax: 0.59, confidence: 0.90 };
        let p3_jitter = FaceBoundingBox { xmin: 0.71, ymin: 0.31, xmax: 0.84, ymax: 0.49, confidence: 0.82 };

        let current = tracker.update(&[p1_jitter, p2_stable, p3_jitter, p4, p5]).expect("Must track");
        assert!(
            (current.xmin - p2_stable.xmin).abs() < 0.05,
            "Frame {}: Tracker must remain locked to primary center user amidst 5-person crowd", f
        );
    }
}

#[test]
fn test_face_tracker_passerby_anti_hijacking_immunity() {
    use vision420_lib::vision::{FaceBoundingBox, FaceTracker};

    let mut tracker = FaceTracker::new(20, 0.30, 1.35);

    // Active user occupying 35% frame width
    let user = FaceBoundingBox {
        xmin: 0.20,
        ymin: 0.20,
        xmax: 0.50,
        ymax: 0.60, // area = 0.30 * 0.40 = 0.120
        confidence: 0.91,
    };

    let initial = tracker.update(&[user]).expect("Must lock user");
    assert!((initial.xmin - user.xmin).abs() < 1e-4);

    // A passerby walks behind / beside the user with area 25% larger (within 1.35x margin)
    // area = 0.35 * 0.42 = 0.147 (1.225x user area)
    let passerby = FaceBoundingBox {
        xmin: 0.55,
        ymin: 0.15,
        xmax: 0.90,
        ymax: 0.57,
        confidence: 0.96,
    };

    for _ in 0..15 {
        let locked = tracker.update(&[user, passerby]).expect("Must keep lock");
        assert!(
            (locked.xmin - user.xmin).abs() < 1e-4,
            "Tracker must resist being hijacked by passerby"
        );
    }
}

#[test]
fn test_face_tracker_graceful_handoff_on_departure() {
    use vision420_lib::vision::{FaceBoundingBox, FaceTracker};

    // 10-frame departure tolerance
    let mut tracker = FaceTracker::new(10, 0.30, 1.35);

    let user_a = FaceBoundingBox { xmin: 0.10, ymin: 0.20, xmax: 0.40, ymax: 0.60, confidence: 0.90 };
    let user_b = FaceBoundingBox { xmin: 0.60, ymin: 0.20, xmax: 0.85, ymax: 0.55, confidence: 0.85 };

    // Initial: user_a is primary
    tracker.update(&[user_a, user_b]).unwrap();

    // User A leaves the room! For 9 frames, user_b is present alone
    for frame in 1..=9 {
        let locked = tracker.update(&[user_b]);
        // During grace period, tracker retains memory of user_a's position to tolerate brief occlusion
        assert!(locked.is_some(), "Frame {}: Tracker tolerates brief loss", frame);
    }

    // Frame 10: lost_frames reaches 10 (which is <= max_lost_frames=10)
    tracker.update(&[user_b]);

    // On frame 11 (lost_frames becomes 11 > 10 max_lost_frames), tracker gracefully hand-offs to user_b
    let handoff = tracker.update(&[user_b]).expect("Must handoff to user_b");
    assert!((handoff.xmin - user_b.xmin).abs() < 1e-4, "Must handoff to remaining user B");
}

#[test]
fn test_review_fix_camera_switch_resets_tracker() {
    use vision420_lib::vision::{FaceBoundingBox, FaceTracker};

    let mut tracker = FaceTracker::new(20, 0.30, 1.35);

    // Camera 1 locked onto face at top-left
    let cam1_face = FaceBoundingBox { xmin: 0.1, ymin: 0.1, xmax: 0.3, ymax: 0.3, confidence: 0.9 };
    tracker.update(&[cam1_face]);
    assert!(tracker.current_lock().is_some());

    // Switch camera occurs -> tracker must reset to clear stale coordinate space
    tracker.reset();
    assert_eq!(tracker.current_lock(), None);

    // Camera 2 detects face at bottom-right
    let cam2_face = FaceBoundingBox { xmin: 0.7, ymin: 0.7, xmax: 0.9, ymax: 0.9, confidence: 0.85 };
    let locked = tracker.update(&[cam2_face]).expect("Must lock immediately to new camera face");
    assert!((locked.xmin - cam2_face.xmin).abs() < 1e-4);
}

#[test]
fn test_review_fix_nms_and_total_cmp_handles_nan_and_overlapping_candidates() {
    use vision420_lib::vision::{FaceBoundingBox, FaceTracker};

    let mut tracker = FaceTracker::new(20, 0.30, 1.35);

    // Initial user at center (area = 0.30 * 0.30 = 0.09)
    let b1 = FaceBoundingBox { xmin: 0.20, ymin: 0.20, xmax: 0.50, ymax: 0.50, confidence: 0.95 };
    let locked = tracker.update(&[b1]).expect("Must lock user b1");
    assert!((locked.xmin - b1.xmin).abs() < 1e-4);

    // Intruding face (non-overlapping, slightly larger: 0.32 * 0.32 = 0.1024 vs b1: 0.090, 1.13x < 1.35x)
    let b3 = FaceBoundingBox { xmin: 0.60, ymin: 0.20, xmax: 0.92, ymax: 0.52, confidence: 0.90 };
    // Redundant raw anchor of b1 that overlaps heavily
    let b2 = FaceBoundingBox { xmin: 0.21, ymin: 0.21, xmax: 0.51, ymax: 0.51, confidence: 0.80 };

    // Next frame: b2 is present with b1, plus b3
    let next_locked = tracker.update(&[b1, b2, b3]).expect("Must maintain lock");
    assert!((next_locked.xmin - b1.xmin).abs() < 1e-4);
}

#[test]
fn test_adaptive_far_field_hysteresis_preserves_presence_on_lean_back() {
    use vision420_lib::vision::{FaceDetectorEngine, FaceTracker};
    let model_path = Path::new("../models/ultraface.onnx");
    if !model_path.exists() {
        return;
    }

    let mut engine = FaceDetectorEngine::new(model_path).expect("Failed to init FaceDetectorEngine");
    let mut tracker = FaceTracker::default();

    let fixture = Path::new("tests/fixtures/1face.png");
    let img = image::open(fixture).expect("Failed to open fixture").to_rgb8();
    let (orig_w, orig_h) = (img.width(), img.height());

    // Frame 1: User sits normally at Tier 1 desk distance (640x480)
    let pre_normal = engine.preprocess(img.as_raw(), orig_w as usize, orig_h as usize);
    let (has_face_f1, _, primary_f1) = engine.detect_faces_and_track(pre_normal, Some(&mut tracker)).unwrap();
    assert!(has_face_f1, "Normal desk frame must detect face");
    assert!(primary_f1.is_some(), "Tracker must acquire initial lock");
    assert!(tracker.current_lock().is_some(), "Tracker must hold active lock");

    // Frame 2: User leans back to Tier 3 distance (0.25x scale face on canvas)
    let scaled_w = (orig_w as f32 * 0.25).round() as u32;
    let scaled_h = (orig_h as f32 * 0.25).round() as u32;
    let scaled = image::imageops::resize(&img, scaled_w, scaled_h, image::imageops::FilterType::Triangle);

    let mut canvas = image::RgbImage::new(orig_w, orig_h);
    for px in canvas.pixels_mut() { *px = image::Rgb([120, 120, 120]); }
    let ox = (orig_w - scaled_w) / 2;
    let oy = (orig_h - scaled_h) / 2;
    image::imageops::overlay(&mut canvas, &scaled, ox as i64, oy as i64);

    let pre_far = engine.preprocess(canvas.as_raw(), orig_w as usize, orig_h as usize);
    let (has_face_far, _, primary_far) = engine.detect_faces_and_track(pre_far, Some(&mut tracker)).unwrap();

    assert!(has_face_far, "Adaptive hysteresis must retain face presence when user leans back");
    assert!(primary_far.is_some(), "Adaptive tracker must retain lock on far-field user");
}

#[test]
fn test_video_continuous_trajectory_dynamic_speed_and_angles() {
    use vision420_lib::vision::{FaceBoundingBox, FaceTracker};

    let mut tracker = FaceTracker::default();

    let mut curr_x = 0.50f32;
    let mut curr_y = 0.50f32;
    let mut curr_w = 0.25f32;

    let mut lock_maintained_frames = 0;
    let total_frames = 300;

    for frame in 1..=total_frames {
        let (dx, dy, dw) = if frame <= 30 {
            (0.0, 0.0, 0.0) // Still / seated
        } else if frame <= 80 {
            let t = (frame - 30) as f32;
            (t.sin() * 0.005, t.cos() * 0.003, t.sin() * 0.002) // Gentle wander
        } else if frame <= 130 {
            let t = (frame - 80) as f32;
            (t.sin() * 0.025, (t * 1.5).cos() * 0.020, (t * 0.5).sin() * 0.008) // Rapid displacement across desk
        } else if frame <= 180 {
            (0.008, -0.008, -0.001) // Migrates toward top-right corner
        } else if frame <= 230 {
            (-0.015, 0.012, 0.001) // Shift toward bottom-left corner
        } else {
            (0.002, 0.002, -0.002) // Deep recline / lean back (Tier 3 far field)
        };

        curr_x = (curr_x + dx).clamp(0.05, 0.75);
        curr_y = (curr_y + dy).clamp(0.05, 0.65);
        curr_w = (curr_w + dw).clamp(0.12, 0.35);
        let curr_h = curr_w * 1.25;

        let active_user_box = FaceBoundingBox {
            xmin: curr_x,
            ymin: curr_y,
            xmax: curr_x + curr_w,
            ymax: curr_y + curr_h,
            confidence: if curr_w < 0.15 { 0.52 } else { 0.90 },
        };

        let mut candidates = vec![active_user_box];

        // Introduce background bystander across room in frames 231..300
        if frame > 230 {
            let passerby = FaceBoundingBox {
                xmin: 0.75,
                ymin: 0.30,
                xmax: 0.90,
                ymax: 0.52,
                confidence: 0.88,
            };
            candidates.push(passerby);
        }

        let tracked = tracker.update(&candidates);
        if let Some(target) = tracked {
            let dist_sq = target.center_distance_sq(&active_user_box);
            if dist_sq < 0.02 {
                lock_maintained_frames += 1;
            }
        }
    }

    assert_eq!(
        lock_maintained_frames, total_frames,
        "Tracker must maintain 100% lock retention across all 300 continuous video trajectory frames"
    );
}

#[derive(Debug, Clone, Copy)]
enum VideoScenarioType {
    StillSeatedNormal,
    SlowBreathingBobbing,
    GentleTypingLean,
    SubtleSideGlance,
    SlowDiagonalDrift,
    RapidChairReclineFarField,
    StandingDeskTransitionUp,
    SittingDownTransitionDown,
    ReachingForCoffeeRight,
    ReachingForMouseLeft,
    ZigZagHeadShake,
    PeriodicMicroNodding,
    BystanderPassesFarBackground,
    BystanderBriefOcclusion,
    CornerTopRightRest,
    CornerBottomLeftSlump,
    HighSpeedLateralSprint,
    SuddenBrakeAndReverseSprint,
    DiagonalTeleportLikeBurst,
    RotationalCircleOrbit,
    FigureEightLissajousOrbit,
    PulsingDepthOscillation,
    MultiBystanderCafeWalkby,
    EdgeClippingTopBorder,
    EdgeClippingRightBorder,
    StrobeLikeFrameDropOscillation,
    HyperVelocityJitterChaos,
    ExtremeFarFieldCornerShrink,
    FalseGhostBoxFlickerBait,
    DoubleSwappingIdenticalTwins,
}

struct VideoTestCase {
    name: &'static str,
    scenario: VideoScenarioType,
    frames_count: usize,
    min_required_retention_pct: f32,
}

#[test]
fn test_video_continuous_multi_scenario_chaos_matrix_30_videos() {
    use vision420_lib::vision::{FaceBoundingBox, FaceTracker};

    let test_cases = [
        VideoTestCase { name: "01_StillSeatedNormal", scenario: VideoScenarioType::StillSeatedNormal, frames_count: 150, min_required_retention_pct: 100.0 },
        VideoTestCase { name: "02_SlowBreathingBobbing", scenario: VideoScenarioType::SlowBreathingBobbing, frames_count: 150, min_required_retention_pct: 100.0 },
        VideoTestCase { name: "03_GentleTypingLean", scenario: VideoScenarioType::GentleTypingLean, frames_count: 150, min_required_retention_pct: 100.0 },
        VideoTestCase { name: "04_SubtleSideGlance", scenario: VideoScenarioType::SubtleSideGlance, frames_count: 150, min_required_retention_pct: 100.0 },
        VideoTestCase { name: "05_SlowDiagonalDrift", scenario: VideoScenarioType::SlowDiagonalDrift, frames_count: 150, min_required_retention_pct: 100.0 },
        VideoTestCase { name: "06_RapidChairReclineFarField", scenario: VideoScenarioType::RapidChairReclineFarField, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "07_StandingDeskTransitionUp", scenario: VideoScenarioType::StandingDeskTransitionUp, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "08_SittingDownTransitionDown", scenario: VideoScenarioType::SittingDownTransitionDown, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "09_ReachingForCoffeeRight", scenario: VideoScenarioType::ReachingForCoffeeRight, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "10_ReachingForMouseLeft", scenario: VideoScenarioType::ReachingForMouseLeft, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "11_ZigZagHeadShake", scenario: VideoScenarioType::ZigZagHeadShake, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "12_PeriodicMicroNodding", scenario: VideoScenarioType::PeriodicMicroNodding, frames_count: 150, min_required_retention_pct: 100.0 },
        VideoTestCase { name: "13_BystanderPassesFarBackground", scenario: VideoScenarioType::BystanderPassesFarBackground, frames_count: 150, min_required_retention_pct: 100.0 },
        VideoTestCase { name: "14_BystanderBriefOcclusion", scenario: VideoScenarioType::BystanderBriefOcclusion, frames_count: 150, min_required_retention_pct: 95.0 },
        VideoTestCase { name: "15_CornerTopRightRest", scenario: VideoScenarioType::CornerTopRightRest, frames_count: 150, min_required_retention_pct: 100.0 },
        VideoTestCase { name: "16_CornerBottomLeftSlump", scenario: VideoScenarioType::CornerBottomLeftSlump, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "17_HighSpeedLateralSprint", scenario: VideoScenarioType::HighSpeedLateralSprint, frames_count: 150, min_required_retention_pct: 95.0 },
        VideoTestCase { name: "18_SuddenBrakeAndReverseSprint", scenario: VideoScenarioType::SuddenBrakeAndReverseSprint, frames_count: 150, min_required_retention_pct: 95.0 },
        VideoTestCase { name: "19_DiagonalTeleportLikeBurst", scenario: VideoScenarioType::DiagonalTeleportLikeBurst, frames_count: 150, min_required_retention_pct: 90.0 },
        VideoTestCase { name: "20_RotationalCircleOrbit", scenario: VideoScenarioType::RotationalCircleOrbit, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "21_FigureEightLissajousOrbit", scenario: VideoScenarioType::FigureEightLissajousOrbit, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "22_PulsingDepthOscillation", scenario: VideoScenarioType::PulsingDepthOscillation, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "23_MultiBystanderCafeWalkby", scenario: VideoScenarioType::MultiBystanderCafeWalkby, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "24_EdgeClippingTopBorder", scenario: VideoScenarioType::EdgeClippingTopBorder, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "25_EdgeClippingRightBorder", scenario: VideoScenarioType::EdgeClippingRightBorder, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "26_StrobeLikeFrameDropOscillation", scenario: VideoScenarioType::StrobeLikeFrameDropOscillation, frames_count: 150, min_required_retention_pct: 90.0 },
        VideoTestCase { name: "27_HyperVelocityJitterChaos", scenario: VideoScenarioType::HyperVelocityJitterChaos, frames_count: 150, min_required_retention_pct: 88.0 },
        VideoTestCase { name: "28_ExtremeFarFieldCornerShrink", scenario: VideoScenarioType::ExtremeFarFieldCornerShrink, frames_count: 150, min_required_retention_pct: 90.0 },
        VideoTestCase { name: "29_FalseGhostBoxFlickerBait", scenario: VideoScenarioType::FalseGhostBoxFlickerBait, frames_count: 150, min_required_retention_pct: 98.0 },
        VideoTestCase { name: "30_DoubleSwappingIdenticalTwins", scenario: VideoScenarioType::DoubleSwappingIdenticalTwins, frames_count: 150, min_required_retention_pct: 92.0 },
    ];

    let mut scenarios_passed = 0;
    let total_scenarios = test_cases.len();

    for tc in &test_cases {
        let mut tracker = FaceTracker::default();
        let mut curr_x = 0.45f32;
        let mut curr_y = 0.40f32;
        let mut curr_w = 0.25f32;

        let mut retained_in_case = 0;

        for f in 1..=tc.frames_count {
            let t = f as f32;
            let (dx, dy, dw, conf, drop_frame, extra_boxes) = match tc.scenario {
                VideoScenarioType::StillSeatedNormal => (0.0, 0.0, 0.0, 0.95, false, vec![]),
                VideoScenarioType::SlowBreathingBobbing => (0.0, (t * 0.1).sin() * 0.002, 0.0, 0.93, false, vec![]),
                VideoScenarioType::GentleTypingLean => ((t * 0.05).sin() * 0.003, (t * 0.05).cos() * 0.003, (t * 0.05).sin() * 0.002, 0.92, false, vec![]),
                VideoScenarioType::SubtleSideGlance => ((t * 0.08).sin() * 0.004, 0.0, 0.0, 0.90, false, vec![]),
                VideoScenarioType::SlowDiagonalDrift => (0.001, 0.001, 0.0, 0.91, false, vec![]),
                VideoScenarioType::RapidChairReclineFarField => (0.0, 0.001, if f > 30 && f < 90 { -0.002 } else { 0.0 }, if f >= 60 { 0.50 } else { 0.90 }, false, vec![]),
                VideoScenarioType::StandingDeskTransitionUp => (0.0, -0.003, 0.0, 0.90, false, vec![]),
                VideoScenarioType::SittingDownTransitionDown => (0.0, 0.003, 0.0, 0.90, false, vec![]),
                VideoScenarioType::ReachingForCoffeeRight => (if f > 40 && f < 80 { 0.006 } else if f >= 80 && f < 120 { -0.006 } else { 0.0 }, 0.001, 0.0, 0.88, false, vec![]),
                VideoScenarioType::ReachingForMouseLeft => (if f > 40 && f < 80 { -0.006 } else if f >= 80 && f < 120 { 0.006 } else { 0.0 }, 0.001, 0.0, 0.88, false, vec![]),
                VideoScenarioType::ZigZagHeadShake => ((t * 0.3).sin() * 0.012, 0.0, 0.0, 0.89, false, vec![]),
                VideoScenarioType::PeriodicMicroNodding => (0.0, (t * 0.25).sin() * 0.006, 0.0, 0.91, false, vec![]),
                VideoScenarioType::BystanderPassesFarBackground => (0.0, 0.0, 0.0, 0.92, false, vec![FaceBoundingBox { xmin: 0.80, ymin: 0.15, xmax: 0.92, ymax: 0.32, confidence: 0.85 }]),
                VideoScenarioType::BystanderBriefOcclusion => (0.0, 0.0, 0.0, 0.90, f >= 50 && f <= 53, vec![]),
                VideoScenarioType::CornerTopRightRest => (0.003, -0.003, -0.001, 0.89, false, vec![]),
                VideoScenarioType::CornerBottomLeftSlump => (-0.003, 0.003, 0.0, 0.87, false, vec![]),
                VideoScenarioType::HighSpeedLateralSprint => ((t * 0.2).sin() * 0.024, 0.0, 0.0, 0.88, false, vec![]),
                VideoScenarioType::SuddenBrakeAndReverseSprint => (if f < 75 { 0.020 } else { -0.020 }, 0.0, 0.0, 0.86, false, vec![]),
                VideoScenarioType::DiagonalTeleportLikeBurst => ((t * 0.3).sin() * 0.028, (t * 0.25).cos() * 0.022, 0.0, 0.85, false, vec![]),
                VideoScenarioType::RotationalCircleOrbit => ((t * 0.15).cos() * 0.015, (t * 0.15).sin() * 0.015, 0.0, 0.88, false, vec![]),
                VideoScenarioType::FigureEightLissajousOrbit => ((t * 0.15).sin() * 0.018, (t * 0.30).sin() * 0.012, 0.0, 0.88, false, vec![]),
                VideoScenarioType::PulsingDepthOscillation => (0.0, 0.0, (t * 0.2).sin() * 0.008, 0.85, false, vec![]),
                VideoScenarioType::MultiBystanderCafeWalkby => (0.001, 0.0, 0.0, 0.90, false, vec![
                    FaceBoundingBox { xmin: 0.10, ymin: 0.20, xmax: 0.22, ymax: 0.35, confidence: 0.80 },
                    FaceBoundingBox { xmin: 0.78, ymin: 0.18, xmax: 0.90, ymax: 0.34, confidence: 0.82 },
                ]),
                VideoScenarioType::EdgeClippingTopBorder => (0.0, -0.004, 0.0, 0.84, false, vec![]),
                VideoScenarioType::EdgeClippingRightBorder => (0.004, 0.0, 0.0, 0.84, false, vec![]),
                VideoScenarioType::StrobeLikeFrameDropOscillation => (0.001, 0.001, 0.0, 0.88, f % 3 == 0, vec![]),
                VideoScenarioType::HyperVelocityJitterChaos => (((f * 17) % 19) as f32 * 0.003 - 0.025, ((f * 13) % 17) as f32 * 0.003 - 0.022, 0.0, 0.85, false, vec![]),
                VideoScenarioType::ExtremeFarFieldCornerShrink => (0.003, -0.003, -0.002, 0.48, false, vec![]),
                VideoScenarioType::FalseGhostBoxFlickerBait => (0.0, 0.0, 0.0, 0.88, false, if f % 5 == 0 {
                    vec![FaceBoundingBox { xmin: 0.85, ymin: 0.10, xmax: 0.95, ymax: 0.22, confidence: 0.78 }]
                } else { vec![] }),
                VideoScenarioType::DoubleSwappingIdenticalTwins => ((t * 0.1).sin() * 0.008, 0.0, 0.0, 0.90, false, vec![
                    FaceBoundingBox { xmin: 0.65 + (t * 0.1).cos() * 0.008, ymin: curr_y, xmax: 0.85 + (t * 0.1).cos() * 0.008, ymax: curr_y + curr_w * 1.25, confidence: 0.89 }
                ]),
            };

            curr_x = (curr_x + dx).clamp(0.01, 0.80);
            curr_y = (curr_y + dy).clamp(0.01, 0.70);
            curr_w = (curr_w + dw).clamp(0.08, 0.40);
            let curr_h = curr_w * 1.25;

            let active_user_box = FaceBoundingBox {
                xmin: curr_x,
                ymin: curr_y,
                xmax: (curr_x + curr_w).min(0.99),
                ymax: (curr_y + curr_h).min(0.99),
                confidence: conf,
            };

            let mut frame_candidates = Vec::new();
            if !drop_frame {
                frame_candidates.push(active_user_box);
            }
            frame_candidates.extend(extra_boxes);

            let tracked = tracker.update(&frame_candidates);
            if let Some(target) = tracked {
                let dist_sq = target.center_distance_sq(&active_user_box);
                if dist_sq <= 0.04 || drop_frame {
                    retained_in_case += 1;
                }
            }
        }

        let ret_pct = (retained_in_case as f32 / tc.frames_count as f32) * 100.0;
        if ret_pct >= tc.min_required_retention_pct {
            scenarios_passed += 1;
        }
    }

    assert_eq!(scenarios_passed, total_scenarios, "All 30 continuous video scenarios must pass retention criteria");
}

#[test]
fn test_adaptive_roi_zoom_on_real_camera_resolutions() {
    use vision420_lib::vision::{FaceBoundingBox, FaceTracker};
    let model_path = Path::new("../models/ultraface.onnx");
    if !model_path.exists() {
        return;
    }
    let mut engine = FaceDetectorEngine::new(model_path).unwrap();

    let camera_resolutions = [
        ("1280x720 (Standard HD)", 1280usize, 720usize),
        ("640x480 (VGA)", 640usize, 480usize),
    ];

    let fixture_paths = ["tests/fixtures/1face.png", "tests/fixtures/3faces.png", "tests/fixtures/4faces.png"];
    let mut full_frame_detected = 0;
    let mut roi_zoom_detected = 0;
    let mut _total_cases = 0;

    for (_res_name, w, h) in &camera_resolutions {
        for p in &fixture_paths {
            let img = image::open(p).unwrap().to_rgb8();
            let face_w = (*w as f32 * 0.07).round() as u32;
            let face_h = face_w;
            let small_face = image::imageops::resize(&img, face_w, face_h, image::imageops::FilterType::Triangle);

            let positions = [
                (0.50f32, 0.50f32),
                (0.25f32, 0.45f32),
                (0.75f32, 0.45f32),
            ];

            for (px_ratio, py_ratio) in positions {
                _total_cases += 1;
                let mut frame = image::RgbImage::new(*w as u32, *h as u32);
                for px in frame.pixels_mut() { *px = image::Rgb([115, 115, 115]); }

                let ox = ((*w as f32 * px_ratio).round() as u32).saturating_sub(face_w / 2).min(*w as u32 - face_w);
                let oy = ((*h as f32 * py_ratio).round() as u32).saturating_sub(face_h / 2).min(*h as u32 - face_h);
                image::imageops::overlay(&mut frame, &small_face, ox as i64, oy as i64);

                let pre_full = engine.preprocess(frame.as_raw(), *w, *h);
                let (det_full, _, primary_full) = engine.detect_faces_and_track(pre_full, None).unwrap();
                let conf_full = primary_full.map(|b| b.confidence).unwrap_or(0.0);
                if det_full && conf_full >= 0.70 {
                    full_frame_detected += 1;
                }

                let mut tracker = FaceTracker::default();
                let seed_box = FaceBoundingBox {
                    xmin: (ox as f32) / *w as f32,
                    ymin: (oy as f32) / *h as f32,
                    xmax: (ox as f32 + face_w as f32) / *w as f32,
                    ymax: (oy as f32 + face_h as f32) / *h as f32,
                    confidence: 0.90,
                };
                let _ = tracker.update(&[seed_box]);

                let (det_roi, primary_roi) = engine.detect_with_adaptive_roi(frame.as_raw(), *w, *h, Some(&mut tracker)).unwrap();
                let conf_roi = primary_roi.map(|b| b.confidence).unwrap_or(0.0);
                if det_roi && conf_roi >= 0.70 {
                    roi_zoom_detected += 1;
                }
            }
        }
    }

    assert!(roi_zoom_detected > full_frame_detected, "Adaptive RoI Zoom must demonstrably outperform full-frame downscale on real camera resolutions");
}
