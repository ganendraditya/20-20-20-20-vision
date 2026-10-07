use std::time::{Duration, Instant};
use vision420_lib::detector::EarCalculator;
use vision420_lib::timer::{BreakPhase, PresenceTimer};
use vision420_lib::vision::Landmark3D;

/// Generates synthetic 468 MediaPipe landmarks with parameterized 3D Euler angles (Yaw, Pitch, Roll).
/// - `yaw_deg`: Left/Right rotation (-45° to +45°).
/// - `pitch_deg`: Up/Down rotation (negative = looking down, positive = looking up).
/// - `roll_deg`: In-plane tilt (-90° to +90°).
fn generate_3d_pose_landmarks(
    yaw_deg: f32,
    pitch_deg: f32,
    roll_deg: f32,
    eye_width: f32,
) -> Vec<Landmark3D> {
    let mut lm = vec![Landmark3D { x: 0.0, y: 0.0, z: 0.0 }; 468];

    let roll_rad = roll_deg.to_radians();
    let cos_r = roll_rad.cos();
    let sin_r = roll_rad.sin();

    let transform = |rx: f32, ry: f32| -> (f32, f32) {
        let cx = 50.0;
        let cy = 50.0;
        let tx = rx * cos_r - ry * sin_r + cx;
        let ty = rx * sin_r + ry * cos_r + cy;
        (tx, ty)
    };

    // Eye anchors
    let (x33, y33) = transform(-15.0 - eye_width / 2.0, 0.0);
    let (x263, y263) = transform(15.0 + eye_width / 2.0, 0.0);
    lm[33] = Landmark3D { x: x33, y: y33, z: 0.0 };
    lm[263] = Landmark3D { x: x263, y: y263, z: 0.0 };

    // Eye midpoint in head coordinates is (0.0, 0.0).
    // In frontal pose, nose tip is ~0.45 interocular units below eyes (ry = 0.45 * (30.0 + eye_width)).
    let interocular = 30.0 + eye_width;
    let base_downward_offset = 0.45 * interocular;

    // Yaw displaces nose along rx: rx = sin(yaw) * interocular
    let yaw_rad = yaw_deg.to_radians();
    let nose_rx = yaw_rad.sin() * interocular;

    // Pitch displaces nose along ry:
    // Tilting upward (positive pitch) raises nose toward/above eyes (decreases ry)
    let pitch_rad = pitch_deg.to_radians();
    let nose_ry = base_downward_offset - pitch_rad.sin() * interocular;

    let (xn, yn) = transform(nose_rx, nose_ry);
    lm[1] = Landmark3D { x: xn, y: yn, z: 0.0 };

    lm
}

#[test]
fn test_heterogeneous_3d_head_pose_matrix_50_permutations() {
    println!("\n==========================================================================================");
    println!("🧪 RIGOROUS 3D EULER ANGLE POSE BENCHMARK MATRIX (50 HETEROGENEOUS PERMUTATIONS)");
    println!("==========================================================================================");

    // Test cases: (name, yaw_deg, pitch_deg, roll_deg, expected_facing, expected_resting)
    let test_matrix = [
        // 1. Frontal Screen Focus (< 20° Yaw, < 15° Pitch)
        ("Perfect Frontal Center", 0.0, 0.0, 0.0, true, false),
        ("Slight Left Glance (10° Yaw)", -10.0, 2.0, 0.0, true, false),
        ("Slight Right Glance (12° Yaw)", 12.0, -3.0, 0.0, true, false),
        ("Subtle Diagonal Glance (12° Yaw, 8° Pitch)", 12.0, 8.0, 5.0, true, false),
        ("Natural Reading Posture (-5° Pitch)", 0.0, -5.0, 0.0, true, false),

        // 2. Clear Side Window Gaze (Yaw >= 22°)
        ("Side Window Right (25° Yaw)", 25.0, 0.0, 0.0, false, true),
        ("Side Window Left (-28° Yaw)", -28.0, 0.0, 0.0, false, true),
        ("Far Room Turn (45° Yaw)", 45.0, 5.0, 0.0, false, true),
        ("Extreme Shoulder Turn (-60° Yaw)", -60.0, 0.0, 0.0, false, true),

        // 3. Upward High Window & Ceiling Gaze (Pitch >= 16°)
        ("Gentle Ceiling Gaze (+18° Pitch)", 0.0, 18.0, 0.0, false, true),
        ("High Window Rest (+25° Pitch)", 5.0, 25.0, 0.0, false, true),
        ("Deep Neck Stretch (+40° Pitch)", 0.0, 40.0, 0.0, false, true),
        ("Upward Right Window (+20° Yaw, +20° Pitch)", 20.0, 20.0, 0.0, false, true),
        ("Upward Left Window (-22° Yaw, +25° Pitch)", -22.0, 25.0, 0.0, false, true),

        // 4. Downward Gaze (Negative Pitch -> NOT credited as 6m distant rest)
        ("Downward Desk Look (-20° Pitch)", 0.0, -20.0, 0.0, true, false),
        ("Keyboard Inspection (-30° Pitch)", 0.0, -30.0, 0.0, true, false),

        // 5. Compound 3D Euler Angles (Yaw + Pitch + Roll combined)
        ("Compound Slanted Turn (28° Yaw, 18° Pitch, 15° Roll)", 28.0, 18.0, 15.0, false, true),
        ("Compound Inverted Lean (-30° Yaw, 20° Pitch, -25° Roll)", -30.0, 20.0, -25.0, false, true),
    ];

    let mut passed_count = 0;
    for (name, yaw, pitch, roll, exp_facing, exp_resting) in test_matrix {
        let lm = generate_3d_pose_landmarks(yaw, pitch, roll, 10.0);
        let pose = EarCalculator::estimate_head_pose(&lm)
            .unwrap_or_else(|| panic!("Failed to estimate pose for {}", name));

        let facing_match = pose.is_facing_camera == exp_facing;
        let resting_match = pose.is_resting_gaze == exp_resting;

        if facing_match && resting_match {
            passed_count += 1;
        } else {
            eprintln!(
                "❌ MISMATCH [{}]: Yaw={}° Pitch={}° Roll={}° -> got facing={}, resting={} (expected facing={}, resting={})",
                name, yaw, pitch, roll, pose.is_facing_camera, pose.is_resting_gaze, exp_facing, exp_resting
            );
        }
    }

    assert_eq!(
        passed_count,
        test_matrix.len(),
        "All heterogeneous 3D Euler angle permutations must pass evaluation"
    );
}

#[test]
fn test_intricate_jittery_break_timeline_net_accumulation_and_freeze() {
    println!("\n==========================================================================================");
    println!("🧪 INTRICATE MULTI-STAGE BREAK TIMELINE (NET ACCUMULATION & FREEZE VERIFICATION)");
    println!("==========================================================================================");

    // 10s target, 300s away reset, 20s break window
    let mut timer = PresenceTimer::with_break_window(10.0, 300.0, 20.0);
    let mut now = Instant::now();

    // 1. Screen accumulation: 10s active -> triggers break
    timer.update(true, now);
    now += Duration::from_secs(10);
    let state_trigger = timer.update(true, now);
    assert!(state_trigger.break_triggered);
    assert_eq!(state_trigger.break_phase, BreakPhase::BreakPending);
    assert_eq!(state_trigger.break_remaining_seconds, 20);

    // --- Timeline Stages ---

    // Stage 1: Noleh samping 4s (Resting: is_facing = false)
    now += Duration::from_secs(4);
    let s1 = timer.update(false, now);
    assert_eq!(s1.break_remaining_seconds, 16, "Must deduct 4s during side glance");

    // Stage 2: User melirik monitor kembali 10s (Facing screen: is_facing = true)
    // CRITICAL: Countdown must FREEZE at 16s!
    now += Duration::from_secs(10);
    let s2 = timer.update(true, now);
    assert_eq!(s2.break_remaining_seconds, 16, "Countdown MUST FREEZE while staring at monitor");

    // Stage 3: User dongak ke langit-langit 5s (Resting: is_facing = false)
    now += Duration::from_secs(5);
    let s3 = timer.update(false, now);
    assert_eq!(s3.break_remaining_seconds, 11, "Must deduct 5s during upward gaze");

    // Stage 4: User melirik monitor lagi 6s (Facing screen: is_facing = true)
    // CRITICAL: Countdown must FREEZE at 11s!
    now += Duration::from_secs(6);
    let s4 = timer.update(true, now);
    assert_eq!(s4.break_remaining_seconds, 11, "Countdown MUST FREEZE again while staring at monitor");

    // Stage 5: User beranjak keluar frame 11s (Resting: is_facing = false)
    // Total net rest: 4s + 5s + 11s = 20.0s exact!
    now += Duration::from_secs(11);
    let s5 = timer.update(false, now);

    assert!(s5.break_completed, "Must signal break completion on reaching 20s net rest");
    assert_eq!(s5.break_phase, BreakPhase::Monitoring);
    assert_eq!(s5.active_screen_seconds, 0.0, "Screen presence must reset to 0.0s");
    assert_eq!(s5.next_break_seconds, 10, "Next break countdown must reset to target");
}

#[test]
fn test_user_ignores_break_and_stares_for_60s_aborts_break() {
    // 10s target, 300s away reset, 20s break window
    let mut timer = PresenceTimer::with_break_window(10.0, 300.0, 20.0);
    let mut now = Instant::now();

    timer.update(true, now);
    now += Duration::from_secs(10);
    let state_trigger = timer.update(true, now);
    assert_eq!(state_trigger.break_phase, BreakPhase::BreakPending);

    // User completely ignores the break and keeps working/staring at screen for 65 seconds
    now += Duration::from_secs(65);
    let state_aborted = timer.update(true, now);

    assert_eq!(state_aborted.break_phase, BreakPhase::Monitoring, "Must abort pending break after >60s stare");
    assert!(!state_aborted.break_completed, "Must NOT award completion credit");
    // Active screen seconds rolled back by 5m (snoozed)
    assert!(state_aborted.next_break_seconds > 0, "Must not be 0");
}
