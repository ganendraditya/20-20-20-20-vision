use std::time::{Duration, Instant};
use vision420_lib::detector::BlinkDetector;
use vision420_lib::vision::Landmark3D;

fn create_synthetic_landmarks_with_pitch(eye_height: f32, eye_width: f32, pitch_deg: f32) -> Vec<Landmark3D> {
    let mut lm = vec![Landmark3D { x: 0.0, y: 0.0, z: 0.0 }; 468];

    // Left eye (33, 133, 159, 145, 158, 153, 160, 144)
    lm[33] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    lm[133] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    lm[159] = Landmark3D { x: eye_width * 0.5, y: eye_height, z: 0.0 };
    lm[145] = Landmark3D { x: eye_width * 0.5, y: 0.0, z: 0.0 };
    lm[158] = Landmark3D { x: eye_width * 0.3, y: eye_height, z: 0.0 };
    lm[153] = Landmark3D { x: eye_width * 0.3, y: 0.0, z: 0.0 };
    lm[160] = Landmark3D { x: eye_width * 0.7, y: eye_height, z: 0.0 };
    lm[144] = Landmark3D { x: eye_width * 0.7, y: 0.0, z: 0.0 };

    // Right eye (362, 263, 386, 374, 387, 373, 385, 380)
    lm[362] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    lm[263] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    lm[386] = Landmark3D { x: eye_width * 0.5, y: eye_height, z: 0.0 };
    lm[374] = Landmark3D { x: eye_width * 0.5, y: 0.0, z: 0.0 };
    lm[387] = Landmark3D { x: eye_width * 0.3, y: eye_height, z: 0.0 };
    lm[373] = Landmark3D { x: eye_width * 0.3, y: 0.0, z: 0.0 };
    lm[385] = Landmark3D { x: eye_width * 0.7, y: eye_height, z: 0.0 };
    lm[380] = Landmark3D { x: eye_width * 0.7, y: 0.0, z: 0.0 };

    // Pitch adjusts nose tip vertical position:
    // Neutral = 0.45 * eye_width
    // Downward gaze (negative pitch) moves nose down (e.g. y = 0.45 - sin(pitch) * eye_width)
    let pitch_rad = pitch_deg.to_radians();
    let base_downward_offset = 0.45 * eye_width;
    let nose_y = base_downward_offset - pitch_rad.sin() * eye_width;
    lm[1] = Landmark3D { x: eye_width * 0.5, y: nose_y, z: 0.0 };

    lm
}

#[test]
fn test_poc_baseline_empirical_sweep() {
    println!("\n==========================================================================");
    println!("📊 EMPIRICAL SWEEP: Low-FPS Micro-Blinks & Downward Gaze Verification");
    println!("==========================================================================");

    // Test cases for half-sampled fast micro-blinks at 10 FPS (100ms) and 12 FPS (83ms)
    // format: (name, fps, dip_ear, pitch_deg, is_real_blink)
    let scenarios = [
        ("10 FPS: 60ms micro-blink (half-sampled dip to 0.23)", 10, 2.3, 0.0, true),
        ("10 FPS: 70ms micro-blink (half-sampled dip to 0.225)", 10, 2.25, 0.0, true),
        ("10 FPS: 80ms micro-blink (half-sampled dip to 0.24)", 10, 2.4, 0.0, true),
        ("12 FPS: 60ms micro-blink (half-sampled dip to 0.23)", 12, 2.3, 0.0, true),
        ("12 FPS: 75ms micro-blink (half-sampled dip to 0.22)", 12, 2.2, 0.0, true),
        ("10 FPS: Slight eyelid drop / squint (EAR 0.32 -> 0.28, 12% drop)", 10, 2.8, 0.0, false),
        ("10 FPS: Slit eye slight drop (EAR 0.20 -> 0.175, 12% drop)", 10, 1.75, 0.0, false),
        ("10 FPS: Looking down at keyboard (-25° pitch, sustained 0.22)", 10, 2.2, -25.0, false),
        ("10 FPS: Reading notes on desk (-30° pitch, sustained 0.20)", 10, 2.0, -30.0, false),
    ];

    let mut true_positives = 0;
    let mut total_real_blinks = 0;
    let mut false_positives = 0;

    for (name, fps, dip_ear, pitch_deg, is_real_blink) in scenarios {
        let frame_dt_ms = 1000 / fps;
        let dt = Duration::from_millis(frame_dt_ms);
        let mut detector = BlinkDetector::new(0.22, 2.0);
        let mut now = Instant::now();

        let open_h = if name.contains("Slit eye") { 2.0 } else { 3.2 };
        let open_lm = create_synthetic_landmarks_with_pitch(open_h, 10.0, 0.0);
        for _ in 0..10 {
            detector.update(&open_lm, now);
            now += dt;
        }

        let dip_lm = create_synthetic_landmarks_with_pitch(dip_ear, 10.0, pitch_deg);

        if is_real_blink {
            total_real_blinks += 1;
            // 1 frame dip then reopen
            detector.update(&dip_lm, now);
            now += dt;
            detector.update(&open_lm, now);
            now += dt;
            let final_evt = detector.update(&open_lm, now);

            if final_evt.total_blinks >= 1 {
                true_positives += 1;
                println!("  ✅ [CAPTURED] {}", name);
            } else {
                println!("  ❌ [MISSED]   {}", name);
            }
        } else {
            // Sustained downward gaze for 5 frames
            for _ in 0..5 {
                detector.update(&dip_lm, now);
                now += dt;
            }
            detector.update(&open_lm, now);
            now += dt;
            let final_evt = detector.update(&open_lm, now);

            if final_evt.total_blinks > 0 {
                false_positives += 1;
                println!("  ❌ [FALSE POSITIVE] {}", name);
            } else {
                println!("  ✅ [CLEAN REJECT]   {}", name);
            }
        }
    }

    println!("\nVERIFIED PERFORMANCE:");
    println!("  Real Micro-Blinks Recall: {}/{} ({:.1}%)", true_positives, total_real_blinks, (true_positives as f32 / total_real_blinks as f32) * 100.0);
    println!("  False Positives on Downward Gaze: {}", false_positives);
    println!("==========================================================================\n");

    assert_eq!(true_positives, total_real_blinks, "All low-FPS micro-blinks must be captured");
    assert_eq!(false_positives, 0, "Zero false positives on downward reading gaze");
}
