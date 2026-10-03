use vision420_lib::detector::EarCalculator;
use vision420_lib::vision::Landmark3D;

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

        println!("{:<8.2} | {:<16} | {:<16} | {:<25}", alpha, evt.total_blinks, evt.stare_warning, if evt.total_blinks == 1 { "✅ Clean 1 Blink" } else { "❌ Mismatch" });

        println!("{:<8.2} | {:<16} | {:<16} | {:<25}", alpha, evt.total_blinks, evt.stare_warning, if evt.total_blinks == 1 { "✅ Clean 1 Blink" } else { "❌ Mismatch" });
    }

    println!("=========================================================================================\n");
}
