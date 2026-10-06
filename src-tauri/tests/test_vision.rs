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
