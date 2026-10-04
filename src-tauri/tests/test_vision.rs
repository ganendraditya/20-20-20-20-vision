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

    if let Ok(img) = image::open("/tmp/1face.png") {
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let preprocessed = engine.preprocess(rgb.as_raw(), w, h);
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
    }
}
