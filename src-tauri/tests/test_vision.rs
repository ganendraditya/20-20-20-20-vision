use vision420_lib::vision::{BlazeFaceEngine, FaceMeshEngine};
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

    let preprocessed = engine.preprocess(&dummy_rgb, width, height);
    assert_eq!(preprocessed.shape(), &[1, 192, 192, 3]);

    let landmarks = engine.infer(preprocessed).expect("Inference failed");
    assert_eq!(landmarks.len(), 468);
    println!("Successfully extracted {} 3D landmarks via ONNX Runtime!", landmarks.len());
}

#[test]
fn test_blazeface_engine_face_presence_detection() {
    let model_path = Path::new("../models/blazeface.onnx");
    if !model_path.exists() {
        eprintln!("BlazeFace model not found at {:?}, skipping", model_path);
        return;
    }

    let mut engine = BlazeFaceEngine::new(model_path).expect("Failed to initialize BlazeFaceEngine");

    // 1. Black image (no face)
    let width = 640;
    let height = 480;
    let black_rgb = vec![0u8; width * height * 3];
    let preprocessed = engine.preprocess(&black_rgb, width, height);
    assert_eq!(preprocessed.shape(), &[1, 3, 128, 128]);

    let has_face = engine.detect_face(preprocessed).expect("BlazeFace detection failed");
    assert!(!has_face, "Black image must not detect any face");

    // 2. Uniform gray image (no face)
    let gray_rgb = vec![128u8; width * height * 3];
    let preprocessed_gray = engine.preprocess(&gray_rgb, width, height);
    let has_face_gray = engine.detect_face(preprocessed_gray).expect("BlazeFace detection failed");
    assert!(!has_face_gray, "Uniform gray image must not detect any face");
}
