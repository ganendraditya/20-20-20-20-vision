use std::path::Path;
use std::time::Instant;
use ort::session::Session;
use vision420_lib::vision::FaceDetectorEngine;

#[test]
fn benchmark_int8_vs_fp32_ultraface() {
    let fp32_path = Path::new("../models/ultraface.onnx");
    let int8_path = Path::new("/tmp/version-RFB-320-int8.onnx");

    if !fp32_path.exists() || !int8_path.exists() {
        eprintln!("Model files not found, skipping");
        return;
    }

    let fp32_size = std::fs::metadata(fp32_path).unwrap().len();
    let int8_size = std::fs::metadata(int8_path).unwrap().len();

    println!("\n==========================================================================================");
    println!(" [PoC/Perf #38] INT8 vs FP32 Quantization Benchmark: UltraFace RFB-320");
    println!(" Evaluating Disk Footprint, Latency (300 runs), and Detection Confidence Stability");
    println!("==========================================================================================");

    println!("Model Footprint (Disk):");
    println!("  FP32 Model Size : {:.2} MB ({} bytes)", fp32_size as f64 / 1_048_576.0, fp32_size);
    println!("  INT8 Model Size : {:.2} MB ({} bytes)", int8_size as f64 / 1_048_576.0, int8_size);
    println!("  Footprint Reduction: {:.1}%", (1.0 - (int8_size as f64 / fp32_size as f64)) * 100.0);

    let mut fp32_engine = FaceDetectorEngine::new(fp32_path).unwrap();
    let mut int8_engine = FaceDetectorEngine::new(int8_path).unwrap();

    let fixture = Path::new("tests/fixtures/1face.png");
    let img = image::open(fixture).unwrap().to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pre = fp32_engine.preprocess(img.as_raw(), w, h);

    // Warmup
    for _ in 0..10 {
        let _ = fp32_engine.detect_faces_and_primary_box(pre.clone());
        let _ = int8_engine.detect_faces_and_primary_box(pre.clone());
    }

    let iterations = 100;

    let start_fp32 = Instant::now();
    for _ in 0..iterations {
        let _ = fp32_engine.detect_faces_and_primary_box(pre.clone());
    }
    let fp32_latency = start_fp32.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let start_int8 = Instant::now();
    for _ in 0..iterations {
        let _ = int8_engine.detect_faces_and_primary_box(pre.clone());
    }
    let int8_latency = start_int8.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let (_, fp32_box) = fp32_engine.detect_faces_and_primary_box(pre.clone()).unwrap();
    let (_, int8_box) = int8_engine.detect_faces_and_primary_box(pre.clone()).unwrap();

    println!("\nInference Latency (CPU, {} runs mean):", iterations);
    println!("  FP32 Latency : {:.3} ms", fp32_latency);
    println!("  INT8 Latency : {:.3} ms", int8_latency);
    let speedup = fp32_latency / int8_latency;
    println!("  Speedup      : {:.2}x ({})", speedup, if speedup >= 1.0 { "Faster" } else { "Slower" });

    println!("\nBounding Box & Confidence Fidelity on 1face.png:");
    if let (Some(b_fp32), Some(b_int8)) = (fp32_box, int8_box) {
        println!("  FP32 Confidence: {:.4} | Box: [{:.3}, {:.3}, {:.3}, {:.3}]", b_fp32.confidence, b_fp32.xmin, b_fp32.ymin, b_fp32.xmax, b_fp32.ymax);
        println!("  INT8 Confidence: {:.4} | Box: [{:.3}, {:.3}, {:.3}, {:.3}]", b_int8.confidence, b_int8.xmin, b_int8.ymin, b_int8.xmax, b_int8.ymax);
        let coord_delta = (b_fp32.xmin - b_int8.xmin).abs().max((b_fp32.ymin - b_int8.ymin).abs());
        println!("  Max Coordinate Shift: {:.4} normalized units (~{:.1} px on 640x480)", coord_delta, coord_delta * 640.0);
    }
}
