use std::path::Path;
use vision420_lib::vision::FaceDetectorEngine;

#[derive(Debug, Clone, Copy)]
pub enum PerturbationType {
    Neutral,
    DimLighting,            // -35% luminance (night coding / dim office)
    BrightLighting,         // +30% luminance (moderately bright office window)
    OffCenterLeft,          // -20% horizontal shift
    OffCenterRight,         // +20% horizontal shift
    OffCenterHigh,          // +20% vertical shift (lower laptop screen angle)
    OffCenterLow,           // -20% vertical shift (higher monitor)
    SensorNoise,            // Low-amplitude camera sensor ISO grain
    HorizontalFlip,         // Mirrored webcam horizontal orientation
    MildMotionJitter,       // Slight head displacement jitter (1px blur)
    SoftShadow,             // Gentle lateral room lighting gradient
    IndoorColorCast,        // Soft warm indoor lighting cast
}

#[test]
fn benchmark_ultraface_distance_and_latency() {
    let model_path = Path::new("../models/ultraface.onnx");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping benchmark", model_path);
        return;
    }

    let mut engine = FaceDetectorEngine::new(model_path).expect("Failed to init FaceDetectorEngine");

    let fixture_paths = ["tests/fixtures/1face.png", "tests/fixtures/3faces.png", "tests/fixtures/4faces.png"];
    let mut base_images = Vec::new();
    for p in &fixture_paths {
        if let Ok(img) = image::open(p) {
            base_images.push((*p, img.to_rgb8()));
        }
    }

    assert!(!base_images.is_empty(), "Must have test fixtures loaded");

    println!("\n==========================================================================================");
    println!(" [PoC/Vision #55] 108-Case Empirical Distance Matrix (3 Fixtures x 3 Tiers x 12 Perturbations)");
    println!(" Matrix evaluates: Lighting variations, Positions, ISO noise, Mirrored axes, Blur & Shadows");
    println!("==========================================================================================");

    let perturbations = [
        ("Neutral / Clean", PerturbationType::Neutral),
        ("Dim Lighting (-35%)", PerturbationType::DimLighting),
        ("Bright Lighting (+30%)", PerturbationType::BrightLighting),
        ("Off-Center Left (-20%)", PerturbationType::OffCenterLeft),
        ("Off-Center Right (+20%)", PerturbationType::OffCenterRight),
        ("Off-Center High (+20%)", PerturbationType::OffCenterHigh),
        ("Off-Center Low (-20%)", PerturbationType::OffCenterLow),
        ("Webcam ISO Sensor Grain", PerturbationType::SensorNoise),
        ("Mirrored Perspective", PerturbationType::HorizontalFlip),
        ("Head Motion Jitter", PerturbationType::MildMotionJitter),
        ("Soft Lateral Shadow", PerturbationType::SoftShadow),
        ("Warm Lighting Cast", PerturbationType::IndoorColorCast),
    ];

    let tiers = [
        ("Tier 1: Close (30-50cm)", 1.0f32),
        ("Tier 2: Standard Desk (50-80cm)", 0.5f32),
        ("Tier 3: Leaning Back (80-120cm)", 0.25f32),
    ];

    let mut tier1_baseline_passed = 0;
    let mut tier2_baseline_passed = 0;
    let mut tier3_baseline_passed = 0;
    let mut tier3_adaptive_passed = 0;
    let mut tier3_roi_passed = 0;

    let cases_per_tier = base_images.len() * perturbations.len(); // 36 per tier = 108 total

    for (_fixture_name, img) in &base_images {
        let (orig_w, orig_h) = (img.width(), img.height());

        for (tier_idx, (_tier_name, scale)) in tiers.iter().enumerate() {
            let scaled_w = ((orig_w as f32 * scale).round() as u32).max(10);
            let scaled_h = ((orig_h as f32 * scale).round() as u32).max(10);
            let scaled_img = image::imageops::resize(img, scaled_w, scaled_h, image::imageops::FilterType::Triangle);

            for (_p_idx, (_p_name, p_type)) in perturbations.iter().enumerate() {
                let mut canvas = image::RgbImage::new(orig_w, orig_h);
                let bg_lum = match p_type {
                    PerturbationType::DimLighting => 70,
                    PerturbationType::BrightLighting => 170,
                    PerturbationType::SoftShadow => 100,
                    _ => 120,
                };
                for px in canvas.pixels_mut() {
                    *px = image::Rgb([bg_lum, bg_lum, bg_lum]);
                }

                let mut face_copy = scaled_img.clone();

                match p_type {
                    PerturbationType::Neutral => {}
                    PerturbationType::DimLighting => {
                        for px in face_copy.pixels_mut() {
                            px[0] = (px[0] as f32 * 0.65) as u8;
                            px[1] = (px[1] as f32 * 0.65) as u8;
                            px[2] = (px[2] as f32 * 0.65) as u8;
                        }
                    }
                    PerturbationType::BrightLighting => {
                        for px in face_copy.pixels_mut() {
                            px[0] = (px[0] as f32 * 1.30).min(255.0) as u8;
                            px[1] = (px[1] as f32 * 1.30).min(255.0) as u8;
                            px[2] = (px[2] as f32 * 1.30).min(255.0) as u8;
                        }
                    }
                    PerturbationType::SensorNoise => {
                        for (i, px) in face_copy.pixels_mut().enumerate() {
                            let noise = ((i * 13 + 3) % 17) as i16 - 8;
                            px[0] = (px[0] as i16 + noise).clamp(0, 255) as u8;
                            px[1] = (px[1] as i16 + noise).clamp(0, 255) as u8;
                            px[2] = (px[2] as i16 + noise).clamp(0, 255) as u8;
                        }
                    }
                    PerturbationType::HorizontalFlip => {
                        face_copy = image::imageops::flip_horizontal(&face_copy);
                    }
                    PerturbationType::MildMotionJitter => {
                        face_copy = image::imageops::blur(&face_copy, 0.7);
                    }
                    PerturbationType::SoftShadow => {
                        let mid_x = face_copy.width() / 2;
                        for (x, _y, px) in face_copy.enumerate_pixels_mut() {
                            if x > mid_x {
                                px[0] = (px[0] as f32 * 0.65) as u8;
                                px[1] = (px[1] as f32 * 0.65) as u8;
                                px[2] = (px[2] as f32 * 0.65) as u8;
                            } else {
                                px[0] = (px[0] as f32 * 1.15).min(255.0) as u8;
                                px[1] = (px[1] as f32 * 1.15).min(255.0) as u8;
                                px[2] = (px[2] as f32 * 1.15).min(255.0) as u8;
                            }
                        }
                    }
                    PerturbationType::IndoorColorCast => {
                        for px in face_copy.pixels_mut() {
                            px[0] = (px[0] as f32 * 1.15).min(255.0) as u8;
                            px[1] = (px[1] as f32 * 1.05).min(255.0) as u8;
                            px[2] = (px[2] as f32 * 0.85) as u8;
                        }
                    }
                    _ => {}
                }

                let base_x = (orig_w.saturating_sub(scaled_w)) / 2;
                let base_y = (orig_h.saturating_sub(scaled_h)) / 2;

                let offset_x = match p_type {
                    PerturbationType::OffCenterLeft => (base_x as f32 * 0.45) as i64,
                    PerturbationType::OffCenterRight => (base_x as f32 * 1.55) as i64,
                    _ => base_x as i64,
                };
                let offset_y = match p_type {
                    PerturbationType::OffCenterHigh => (base_y as f32 * 0.45) as i64,
                    PerturbationType::OffCenterLow => (base_y as f32 * 1.55) as i64,
                    _ => base_y as i64,
                };

                image::imageops::overlay(&mut canvas, &face_copy, offset_x, offset_y);

                let pre = engine.preprocess(canvas.as_raw(), orig_w as usize, orig_h as usize);

                // Run 1: Un-tracked Stateless Baseline (Conf >= 0.70)
                let (detected_stateless, dominant_stateless) = engine.detect_faces_and_primary_box(pre.clone()).unwrap();
                let conf_stateless = dominant_stateless.map(|b| b.confidence).unwrap_or(0.0);
                let passed_stateless = detected_stateless && conf_stateless >= 0.70;

                match tier_idx {
                    0 => if passed_stateless { tier1_baseline_passed += 1; },
                    1 => if passed_stateless { tier2_baseline_passed += 1; },
                    2 => if passed_stateless { tier3_baseline_passed += 1; },
                    _ => {}
                }

                // Run 2: Active Session Tracker with Adaptive Far-Field Hysteresis (Issue #55)
                // Run 3: Active Session Tracker with Adaptive RoI Zoom (Issue #63)
                if tier_idx == 2 {
                    let mut tracker_hysteresis = vision420_lib::vision::FaceTracker::default();
                    let pre_seed = engine.preprocess(img.as_raw(), orig_w as usize, orig_h as usize);
                    let _ = engine.detect_faces_and_track(pre_seed, Some(&mut tracker_hysteresis));

                    let (detected_adaptive, _, dominant_adaptive) = engine.detect_faces_and_track(pre, Some(&mut tracker_hysteresis)).unwrap();
                    let passed_adaptive = detected_adaptive && dominant_adaptive.is_some();
                    if passed_adaptive {
                        tier3_adaptive_passed += 1;
                    }

                    let mut tracker_roi = vision420_lib::vision::FaceTracker::default();
                    let _ = engine.detect_with_adaptive_roi(img.as_raw(), orig_w as usize, orig_h as usize, Some(&mut tracker_roi));

                    let (det_roi, dominant_roi) = engine.detect_with_adaptive_roi(canvas.as_raw(), orig_w as usize, orig_h as usize, Some(&mut tracker_roi)).unwrap();
                    if det_roi && dominant_roi.is_some() {
                        tier3_roi_passed += 1;
                    }
                }
            }
        }
    }

    println!("\n==========================================================================================");
    println!(" FINAL AUDIT SCORECARD: 108 HETEROGENEOUS EMPIRICAL TEST CASES");
    println!("==========================================================================================");
    println!(" Total Cases Evaluated      : 108 scenarios (36 per tier)");
    println!(" Tier 1 (Close / 30-50 cm)  : {}/{} ({:.1}%)", tier1_baseline_passed, cases_per_tier, (tier1_baseline_passed as f32 / cases_per_tier as f32) * 100.0);
    println!(" Tier 2 (Standard / 50-80cm): {}/{} ({:.1}%)", tier2_baseline_passed, cases_per_tier, (tier2_baseline_passed as f32 / cases_per_tier as f32) * 100.0);
    println!(" Tier 3 (Far / 80-120cm)    : {}/{} ({:.1}%) [Stateless Baseline @ 0.70 Threshold]", tier3_baseline_passed, cases_per_tier, (tier3_baseline_passed as f32 / cases_per_tier as f32) * 100.0);
    println!(" Tier 3 (Far / 80-120cm)    : {}/{} ({:.1}%) [Active Adaptive Lock @ 0.45 Hysteresis]", tier3_adaptive_passed, cases_per_tier, (tier3_adaptive_passed as f32 / cases_per_tier as f32) * 100.0);
    println!(" Tier 3 (Far / 80-120cm)    : {}/{} ({:.1}%) [Active Adaptive RoI Zoom (Issue #63)]", tier3_roi_passed, cases_per_tier, (tier3_roi_passed as f32 / cases_per_tier as f32) * 100.0);
    println!("==========================================================================================");

    assert_eq!(tier1_baseline_passed, cases_per_tier, "Tier 1 Close distance MUST have 100% recall across all 12 perturbations");
    assert!(tier2_baseline_passed as f32 / cases_per_tier as f32 >= 0.85, "Tier 2 Standard desk distance must maintain >= 85% recall");
    assert!(tier3_adaptive_passed > tier3_baseline_passed, "Adaptive Hysteresis must demonstrably outperform stateless baseline in Tier 3");
}
