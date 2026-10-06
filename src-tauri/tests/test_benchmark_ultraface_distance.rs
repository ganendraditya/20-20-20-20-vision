use std::path::Path;
use vision420_lib::vision::FaceDetectorEngine;

#[test]
fn benchmark_ultraface_distance_and_latency() {
    let model_path = Path::new("../models/ultraface.onnx");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping benchmark", model_path);
        return;
    }

    let mut engine = FaceDetectorEngine::new(model_path).expect("Failed to init FaceDetectorEngine");

    // Load available base fixtures
    let fixture_paths = ["tests/fixtures/1face.png", "tests/fixtures/3faces.png", "tests/fixtures/4faces.png"];
    let mut base_images = Vec::new();
    for p in &fixture_paths {
        if let Ok(img) = image::open(p) {
            base_images.push((*p, img.to_rgb8()));
        }
    }

    assert!(!base_images.is_empty(), "Must have test fixtures loaded");

    println!("\n==========================================================================");
    println!(" [PoC/Vision #39] Comprehensive Multi-Condition Distance Benchmark Matrix");
    println!(" Matrix: 3 Base Fixtures x 3 Tiers x 5 Perturbation Conditions = 45 Cases");
    println!("==========================================================================");

    let perturbations = [
        "Neutral / Center",
        "Dim Lighting (-40% lum)",
        "Overexposed (+35% lum)",
        "Off-Center Left (-20% x)",
        "Off-Center Right (+20% x)",
    ];

    let tiers = [
        ("Tier 1: Close (30-50cm)", 1.0f32),
        ("Tier 2: Standard Desk (50-80cm)", 0.5f32),
        ("Tier 3: Leaning Back (80-120cm)", 0.25f32),
    ];

    let mut tier1_detected = 0;
    let mut tier2_detected = 0;
    let mut tier3_detected = 0;
    let mut total_per_tier = 0;

    for (name, img) in &base_images {
        let (orig_w, orig_h) = (img.width(), img.height());

        for (tier_idx, (tier_name, scale)) in tiers.iter().enumerate() {
            let scaled_w = ((orig_w as f32 * scale).round() as u32).max(10);
            let scaled_h = ((orig_h as f32 * scale).round() as u32).max(10);
            let scaled_img = image::imageops::resize(img, scaled_w, scaled_h, image::imageops::FilterType::Triangle);

            for (p_idx, p_name) in perturbations.iter().enumerate() {
                if tier_idx == 0 {
                    total_per_tier += 1;
                }

                // Create full-frame canvas
                let mut canvas = image::RgbImage::new(orig_w, orig_h);
                let bg_lum = match p_idx {
                    1 => 70,  // Dim
                    2 => 180, // Bright
                    _ => 120, // Neutral
                };
                for pixel in canvas.pixels_mut() {
                    *pixel = image::Rgb([bg_lum, bg_lum, bg_lum]);
                }

                // Apply lighting to face image
                let mut face_copy = scaled_img.clone();
                for pixel in face_copy.pixels_mut() {
                    let r = pixel[0] as f32;
                    let g = pixel[1] as f32;
                    let b = pixel[2] as f32;
                    let (nr, ng, nb) = match p_idx {
                        1 => (r * 0.60, g * 0.60, b * 0.60),        // Dim
                        2 => ((r * 1.35).min(255.0), (g * 1.35).min(255.0), (b * 1.35).min(255.0)), // Bright
                        _ => (r, g, b),
                    };
                    *pixel = image::Rgb([nr as u8, ng as u8, nb as u8]);
                }

                // Apply positioning offset
                let base_x = (orig_w.saturating_sub(scaled_w)) / 2;
                let offset_x = match p_idx {
                    3 => (base_x as f32 * 0.40) as i64, // Shift left
                    4 => (base_x as f32 * 1.60) as i64, // Shift right
                    _ => base_x as i64,
                };
                let offset_y = ((orig_h.saturating_sub(scaled_h)) / 2) as i64;

                image::imageops::overlay(&mut canvas, &face_copy, offset_x, offset_y);

                let pre = engine.preprocess(canvas.as_raw(), orig_w as usize, orig_h as usize);
                let (detected, dominant) = engine.detect_faces_and_primary_box(pre).unwrap();

                let (conf, _area) = if let Some(b) = dominant {
                    (b.confidence, b.area())
                } else {
                    (0.0, 0.0)
                };

                let passed = detected && conf >= 0.70;
                if passed {
                    match tier_idx {
                        0 => tier1_detected += 1,
                        1 => tier2_detected += 1,
                        2 => tier3_detected += 1,
                        _ => {}
                    }
                } else if tier_idx == 1 {
                    println!("  [NOTE] Missed case in Tier 2: Fixture={}, P={} (Conf: {:.3})", name, p_name, conf);
                }
            }
        }
    }

    println!("\n==========================================================================");
    println!(" SUMMARY EVALUATION: 45 HETEROGENEOUS PERTURBATION SCENARIOS");
    println!("==========================================================================");
    println!(" Tier 1 Recall (30-50cm) : {}/{} ({:.1}%)", tier1_detected, total_per_tier, tier1_detected as f32 / total_per_tier as f32 * 100.0);
    println!(" Tier 2 Recall (50-80cm) : {}/{} ({:.1}%)", tier2_detected, total_per_tier, tier2_detected as f32 / total_per_tier as f32 * 100.0);
    println!(" Tier 3 Recall (80-120cm): {}/{} ({:.1}%)", tier3_detected, total_per_tier, tier3_detected as f32 / total_per_tier as f32 * 100.0);

    assert_eq!(tier1_detected, total_per_tier, "Tier 1 Close distance MUST have 100% recall across all perturbations");
    assert!(tier2_detected as f32 / total_per_tier as f32 >= 0.90, "Tier 2 Standard desk distance must maintain >= 90% recall under extreme lighting/positioning perturbations");
}
