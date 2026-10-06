use vision420_lib::detector::{BlinkDetector, EarCalculator};
use vision420_lib::vision::Landmark3D;
use std::time::{Duration, Instant};

fn create_synthetic_landmarks(left_height: f32, right_height: f32, eye_width: f32) -> Vec<Landmark3D> {
    let mut landmarks = vec![Landmark3D { x: 0.0, y: 0.0, z: 0.0 }; 468];

    // Left eye setup (33, 133, 159, 145, 158, 153, 160, 144)
    landmarks[33] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    landmarks[133] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    landmarks[159] = Landmark3D { x: eye_width * 0.5, y: left_height, z: 0.0 };
    landmarks[145] = Landmark3D { x: eye_width * 0.5, y: 0.0, z: 0.0 };
    landmarks[158] = Landmark3D { x: eye_width * 0.3, y: left_height, z: 0.0 };
    landmarks[153] = Landmark3D { x: eye_width * 0.3, y: 0.0, z: 0.0 };
    landmarks[160] = Landmark3D { x: eye_width * 0.7, y: left_height, z: 0.0 };
    landmarks[144] = Landmark3D { x: eye_width * 0.7, y: 0.0, z: 0.0 };

    // Right eye setup (362, 263, 386, 374, 387, 373, 385, 380)
    landmarks[362] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    landmarks[263] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    landmarks[386] = Landmark3D { x: eye_width * 0.5, y: right_height, z: 0.0 };
    landmarks[374] = Landmark3D { x: eye_width * 0.5, y: 0.0, z: 0.0 };
    landmarks[387] = Landmark3D { x: eye_width * 0.3, y: right_height, z: 0.0 };
    landmarks[373] = Landmark3D { x: eye_width * 0.3, y: 0.0, z: 0.0 };
    landmarks[385] = Landmark3D { x: eye_width * 0.7, y: right_height, z: 0.0 };
    landmarks[380] = Landmark3D { x: eye_width * 0.7, y: 0.0, z: 0.0 };

    // Nose tip (index 1) centered at midpoint between corners (33 and 263) for frontal head pose
    landmarks[1] = Landmark3D { x: eye_width * 0.5, y: -eye_width * 0.5, z: 0.0 };

    landmarks
}

#[test]
fn test_benchmark_ema_alpha_noise_rejection_and_step_response() {
    let alphas = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0];
    let threshold = 0.22f32;

    println!("\n=========================================================================================");
    println!("🔍 ISSUE #41 BENCHMARK: EAR EMA Alpha Calibration Analysis");
    println!("=========================================================================================");

    // -------------------------------------------------------------------------
    // TEST 1: Signal-to-Noise Ratio (SNR) on Static Open Eyes (1000 frames)
    // Base EAR = 0.32, Pseudo-random uniform noise +/- 0.04 simulating camera sensor glare & jitter
    // -------------------------------------------------------------------------
    println!("\n[TEST 1: Noise Rejection on 1000 Static Frames (Base EAR = 0.32 +/- 0.04 noise)]");
    println!("{:<8} | {:<16} | {:<16} | {:<16} | {:<16}", "Alpha", "Raw Var", "Smoothed Var", "Noise Suppress %", "Min Smoothed");
    println!("{:-<8}-|-{:-<16}-|-{:-<16}-|-{:-<16}-|-{:-<16}", "", "", "", "", "");

    // Deterministic LCG pseudo-random generator
    let mut seed = 123456789u64;
    let mut next_noise = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let normalized = ((seed >> 33) as f32) / ((u32::MAX >> 1) as f32); // -1.0 .. 1.0
        normalized * 0.04
    };

    // Generate 1000 noisy samples
    let samples: Vec<f32> = (0..1000).map(|_| 0.32 + next_noise()).collect();
    let raw_mean = samples.iter().sum::<f32>() / 1000.0;
    let raw_var = samples.iter().map(|&x| (x - raw_mean).powi(2)).sum::<f32>() / 1000.0;

    for &alpha in &alphas {
        let mut calc = EarCalculator::new(alpha);
        let mut smoothed_series = Vec::with_capacity(1000);

        for &val in &samples {
            let lm = create_synthetic_landmarks(val * 10.0, val * 10.0, 10.0);
            let metrics = calc.calculate(&lm).unwrap();
            smoothed_series.push(metrics.smoothed_ear);
        }

        let mean = smoothed_series.iter().sum::<f32>() / 1000.0;
        let var = smoothed_series.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 1000.0;
        let reduction = (1.0 - (var / raw_var)) * 100.0;
        let min_val = smoothed_series.iter().cloned().fold(f32::INFINITY, f32::min);

        println!("{:<8.2} | {:<16.6} | {:<16.6} | {:<15.1}% | {:<16.4}", alpha, raw_var, var, reduction, min_val);
    }

    // -------------------------------------------------------------------------
    // TEST 2: Step-Response Latency on Sudden Eye Closure
    // Transition from Open (0.32) to Closed (0.10) -> How many frames to cross threshold (0.22)?
    // At 15 FPS, 1 frame = 66.7 ms, 2 frames = 133.3 ms, 3 frames = 200.0 ms
    // -------------------------------------------------------------------------
    println!("\n[TEST 2: Step-Response Latency (Open 0.32 -> Closed 0.10, Threshold = 0.22)]");
    println!("{:<8} | {:<18} | {:<18} | {:<25}", "Alpha", "Frames to < 0.22", "Latency (ms @15fps)", "EAR Trajectory (first 4 frames)");
    println!("{:-<8}-|-{:-<18}-|-{:-<18}-|-{:-<25}", "", "", "", "");

    for &alpha in &alphas {
        let mut calc = EarCalculator::new(alpha);
        // Initialize with steady open state (0.32)
        let open_lm = create_synthetic_landmarks(3.2, 3.2, 10.0);
        for _ in 0..10 {
            calc.calculate(&open_lm);
        }

        // Eye snaps closed to 0.10
        let closed_lm = create_synthetic_landmarks(1.0, 1.0, 10.0);
        let mut frames_to_cross = 0;
        let mut trajectory = Vec::new();

        for frame in 1..=10 {
            let metrics = calc.calculate(&closed_lm).unwrap();
            if trajectory.len() < 4 {
                trajectory.push(format!("{:.3}", metrics.smoothed_ear));
            }
            if metrics.smoothed_ear < threshold && frames_to_cross == 0 {
                frames_to_cross = frame;
            }
        }

        let ms = frames_to_cross as f32 * 66.67;
        println!("{:<8.2} | {:<18} | {:<17.1}ms | {:?}", alpha, frames_to_cross, ms, trajectory);
    }

    // -------------------------------------------------------------------------
    // TEST 3B: Rapid 1-Frame Micro-Blink Stress Test (Extreme 66.7 ms blink at 15 FPS)
    // Frame 0: Open (0.32)
    // Frame 1: Closed (0.10)
    // Frame 2: Reopened (0.32)
    // -------------------------------------------------------------------------
    println!("\n[TEST 3B: 1-Frame Ultra-Fast Micro-Blink Stress Test (66.7 ms at 15 FPS)]");
    println!("{:<8} | {:<16} | {:<18} | {:<20}", "Alpha", "Frame 1 Reached", "Penetrates < 0.22?", "1-Frame Captured?");
    println!("{:-<8}-|-{:-<16}-|-{:-<18}-|-{:-<20}", "", "", "", "");

    for &alpha in &alphas {
        let mut calc = EarCalculator::new(alpha);
        let open_lm = create_synthetic_landmarks(3.2, 3.2, 10.0);
        for _ in 0..10 {
            calc.calculate(&open_lm);
        }

        let closed_lm = create_synthetic_landmarks(1.0, 1.0, 10.0);
        let m1 = calc.calculate(&closed_lm).unwrap();
        let captured = m1.smoothed_ear < threshold;
        println!("{:<8.2} | {:<16.4} | {:<18} | {:<20}", alpha, m1.smoothed_ear, captured, if captured { "✅ YES" } else { "❌ MISSED" });
    }

    // -------------------------------------------------------------------------
    // TEST 4: FSM BlinkDetector Integration Test (Left-Right Eye Desync at 15 FPS)
    // -------------------------------------------------------------------------
    println!("\n[TEST 4: StateMachine Blink Counting with Realistic Jitter Noise (250 ms Blink)]");
    println!("{:<8} | {:<16} | {:<16} | {:<25}", "Alpha", "Normal Blinks", "False Stare Alert?", "Status");
    println!("{:-<8}-|-{:-<16}-|-{:-<16}-|-{:-<25}", "", "", "", "");

    use vision420_lib::detector::BlinkDetector;
    use std::time::{Instant, Duration};

    for &alpha in &alphas {
        let mut detector = BlinkDetector::with_alpha(threshold, 2.0, alpha);
        let mut now = Instant::now();

        // 1. Initial 10 frames open with noise
        for _ in 0..10 {
            let n = next_noise();
            let lm = create_synthetic_landmarks((0.32 + n) * 10.0, (0.32 + n) * 10.0, 10.0);
            detector.update(&lm, now);
            now += Duration::from_millis(67);
        }

        // 2. Perform a clean 250 ms blink (4 frames closed)
        for _ in 0..4 {
            let lm = create_synthetic_landmarks(1.0, 1.0, 10.0);
            detector.update(&lm, now);
            now += Duration::from_millis(67);
        }

        // 3. Reopen eyes (2 frames open to settle)
        let open_lm = create_synthetic_landmarks(3.2, 3.2, 10.0);
        now += Duration::from_millis(67);
        detector.update(&open_lm, now);
        now += Duration::from_millis(67);
        let evt = detector.update(&open_lm, now);

        println!("{:<8.2} | {:<16} | {:<16} | {:<25}", alpha, evt.total_blinks, evt.stare_warning, if evt.total_blinks == 1 { "Clean 1 Blink" } else { "Mismatch" });
    }

    // -------------------------------------------------------------------------
    // TEST 5 (Rigor Upgrade #59): AC Mains Strobe Lighting Immunity (50Hz & 60Hz)
    // Simulates sinusoidal light intensity flicker from fluorescent tubes / cheap office LEDs
    // -------------------------------------------------------------------------
    println!("\n[TEST 5: AC Mains 50Hz & 60Hz Strobe Immunity (15 FPS Sampling)]");
    println!("{:<8} | {:<16} | {:<16} | {:<16}", "Alpha", "Strobe Freq", "Variance Red %", "Strobe Triggered Blink?");
    println!("{:-<8}-|-{:-<16}-|-{:-<16}-|-{:-<16}", "", "", "", "");

    for &freq in &[50.0f32, 60.0f32] {
        let base_ear = 0.30f32;
        let strobe_amp = 0.035f32; // +/- 0.035 optical ripple

        let mut raw_strobe_samples = Vec::with_capacity(300);
        for f in 0..300 {
            // Include micro-phase drift (66.67ms actual frame interval = ~14.999 FPS)
            let t = f as f32 * 0.066667; 
            let ripple = (2.0 * std::f32::consts::PI * freq * t).sin() * strobe_amp;
            raw_strobe_samples.push(base_ear + ripple);
        }
        let raw_strobe_mean = raw_strobe_samples.iter().sum::<f32>() / 300.0;
        let raw_strobe_var = raw_strobe_samples.iter().map(|&x| (x - raw_strobe_mean).powi(2)).sum::<f32>() / 300.0;

        for &alpha in &[0.30, 0.40, 0.50] {
            let mut detector = BlinkDetector::with_alpha(threshold, 2.0, alpha);
            let mut calc = EarCalculator::new(alpha);
            let mut smoothed = Vec::with_capacity(300);
            let mut now = Instant::now();

            let mut falsely_triggered = false;
            for &val in &raw_strobe_samples {
                let lm = create_synthetic_landmarks(val * 10.0, val * 10.0, 10.0);
                let m = calc.calculate(&lm).unwrap();
                smoothed.push(m.smoothed_ear);

                let evt = detector.update(&lm, now);
                if evt.total_blinks > 0 {
                    falsely_triggered = true;
                }
                now += Duration::from_millis(67);
            }

            let s_mean = smoothed.iter().sum::<f32>() / 300.0;
            let s_var = smoothed.iter().map(|&x| (x - s_mean).powi(2)).sum::<f32>() / 300.0;
            let red_pct = if raw_strobe_var > 1e-7 {
                (1.0 - (s_var / raw_strobe_var)) * 100.0
            } else {
                100.0 // Near zero raw ripple due to exact harmonic cancellation
            };

            println!("{:<8.2} | {:<14}Hz | {:<15.1}% | {:<20}",
                alpha, freq as u32, red_pct, if falsely_triggered { "FAIL (False Blink!)" } else { "PASS (Zero False Blinks)" });

            assert!(!falsely_triggered, "Alpha {} must never trigger false blinks on {}Hz AC strobe!", alpha, freq);
        }
    }

    // -------------------------------------------------------------------------
    // TEST 6 (Rigor Upgrade #59): Micro-Blink Duration Boundary Sweep (40ms..140ms)
    // Sweeps micro-blink duration across 30 FPS vs 15 FPS vs 10 FPS
    // -------------------------------------------------------------------------
    println!("\n[TEST 6: Micro-Blink Temporal Boundary Sweep across Frame Rates]");
    println!("{:<8} | {:<12} | {:<18} | {:<16}", "FPS", "Duration (ms)", "Closed Frames", "Capture Status");
    println!("{:-<8}-|-{:-<12}-|-{:-<18}-|-{:-<16}", "", "", "", "");

    for &fps in &[30u64, 15u64, 10u64] {
        let frame_dt = 1000 / fps;
        for &dur_ms in &[40u64, 60, 80, 100, 120, 140] {
            let mut detector = BlinkDetector::with_alpha(threshold, 2.0, 0.40);
            let mut now = Instant::now();

            // Settle open state
            let open_lm = create_synthetic_landmarks(3.2, 3.2, 10.0);
            for _ in 0..10 {
                detector.update(&open_lm, now);
                now += Duration::from_millis(frame_dt);
            }

            // Perform blink of duration dur_ms
            let closed_lm = create_synthetic_landmarks(0.8, 0.8, 10.0);
            let closed_frames = (dur_ms as f32 / frame_dt as f32).round() as u64;

            for _ in 0..closed_frames {
                detector.update(&closed_lm, now);
                now += Duration::from_millis(frame_dt);
            }

            // Settle reopened (advance by frame_dt to allow state machine to observe reopened eye)
            detector.update(&open_lm, now);
            now += Duration::from_millis(frame_dt);
            let final_evt = detector.update(&open_lm, now);

            let status = if final_evt.total_blinks == 1 { "CAPTURED" } else { "REJECTED (<2 frames guard or <80ms)" };
            println!("{:<8} | {:<12} | {:<18} | {:<16}", fps, dur_ms, closed_frames, status);

            // Verified biological invariant:
            // Measured duration in state machine is from first closed frame to first reopened frame:
            // `duration = closed_frames * frame_dt`
            let measured_dur_ms = closed_frames * frame_dt;
            if closed_frames >= 2 && measured_dur_ms >= 80 && measured_dur_ms <= 800 {
                assert_eq!(final_evt.total_blinks, 1, "Blinks meeting biological threshold must be captured");
            }
        }
    }

    println!("=========================================================================================\n");
}

#[test]
fn test_single_frame_glitch_rejection_proof() {
    let threshold = 0.22f32;
    let mut detector = BlinkDetector::with_alpha(threshold, 2.0, 0.40);
    let mut now = Instant::now();

    let open_eyes = create_synthetic_landmarks(3.2, 3.2, 10.0); // EAR = 0.32
    let glitch_noise_closed = create_synthetic_landmarks(1.0, 1.0, 10.0); // EAR = 0.10 (Sudden 1-frame camera drop)

    // 1. User is staring with steady open eyes
    for _ in 0..10 {
        detector.update(&open_eyes, now);
        now += Duration::from_millis(67);
    }

    // 2. Camera sensor suffers an extreme single-frame optical drop/glitch (only 1 frame, 67ms)
    let glitch_evt = detector.update(&glitch_noise_closed, now);
    assert!(!glitch_evt.is_blink);
    now += Duration::from_millis(67);

    // 3. Next frame camera immediately recovers to open eyes
    let recover_evt = detector.update(&open_eyes, now);
    
    println!("\n[PROOF TEST: Single-Frame Optical Glitch Rejection]");
    println!("Total Blinks registered after 1-frame camera glitch: {}", recover_evt.total_blinks);
    assert_eq!(recover_evt.total_blinks, 0, "Single-frame camera glitch must be rejected with 0 blinks!");
    assert!(!recover_evt.is_blink, "Single-frame glitch must not trigger a blink event!");
}
