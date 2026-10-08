use std::time::{Duration, Instant};
use vision420_lib::detector::BlinkDetector;
use vision420_lib::vision::Landmark3D;

fn create_synthetic_landmarks(eye_height: f32, eye_width: f32) -> Vec<Landmark3D> {
    let mut landmarks = vec![Landmark3D { x: 0.0, y: 0.0, z: 0.0 }; 468];

    // Left eye setup (33, 133, 159, 145, 158, 153, 160, 144)
    landmarks[33] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    landmarks[133] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    landmarks[159] = Landmark3D { x: eye_width * 0.5, y: eye_height, z: 0.0 };
    landmarks[145] = Landmark3D { x: eye_width * 0.5, y: 0.0, z: 0.0 };
    landmarks[158] = Landmark3D { x: eye_width * 0.3, y: eye_height, z: 0.0 };
    landmarks[153] = Landmark3D { x: eye_width * 0.3, y: 0.0, z: 0.0 };
    landmarks[160] = Landmark3D { x: eye_width * 0.7, y: eye_height, z: 0.0 };
    landmarks[144] = Landmark3D { x: eye_width * 0.7, y: 0.0, z: 0.0 };

    // Right eye setup (362, 263, 386, 374, 387, 373, 385, 380)
    landmarks[362] = Landmark3D { x: 0.0, y: 0.0, z: 0.0 };
    landmarks[263] = Landmark3D { x: eye_width, y: 0.0, z: 0.0 };
    landmarks[386] = Landmark3D { x: eye_width * 0.5, y: eye_height, z: 0.0 };
    landmarks[374] = Landmark3D { x: eye_width * 0.5, y: 0.0, z: 0.0 };
    landmarks[387] = Landmark3D { x: eye_width * 0.3, y: eye_height, z: 0.0 };
    landmarks[373] = Landmark3D { x: eye_width * 0.3, y: 0.0, z: 0.0 };
    landmarks[385] = Landmark3D { x: eye_width * 0.7, y: eye_height, z: 0.0 };
    landmarks[380] = Landmark3D { x: eye_width * 0.7, y: 0.0, z: 0.0 };

    // Nose tip setup (index 1) positioned symmetrically at midpoint of eyes for frontal face
    landmarks[1] = Landmark3D { x: eye_width * 0.5, y: 5.0, z: 0.0 };

    landmarks
}

#[test]
fn test_blink_lifecycle_and_stare_warning() {
    let mut detector = BlinkDetector::new(0.25, 2.0); // 2.0s stare limit for fast testing
    let mut now = Instant::now();

    let open_eyes = create_synthetic_landmarks(6.0, 10.0);   // EAR = 0.6
    let closed_eyes = create_synthetic_landmarks(1.0, 10.0); // EAR = 0.1

    // Frame 1: Eyes open
    let event1 = detector.update(&open_eyes, now);
    assert!(!event1.is_blink);
    assert!(!event1.stare_warning);

    // Frame 2: Eyes close
    now += Duration::from_millis(100);
    let event2 = detector.update(&closed_eyes, now);
    assert!(!event2.is_blink);

    // Frame 3: Eyes still closed for 150ms total
    now += Duration::from_millis(50);
    let event3 = detector.update(&closed_eyes, now);
    assert!(!event3.is_blink);

    // Frame 4: Eyes reopen -> valid blink completed!
    now += Duration::from_millis(50);
    let event4 = detector.update(&open_eyes, now);
    assert!(event4.is_blink);
    assert_eq!(event4.total_blinks, 1);
    assert_eq!(event4.current_bpm, 1.0);

    // Simulate prolonged stare: advance time past 2.0s without blinking
    now += Duration::from_millis(2100);
    let event5 = detector.update(&open_eyes, now);
    assert!(event5.stare_warning, "Should trigger stare warning after limit");

    // Next frame: warning shouldn't re-trigger continuously (no spamming)
    now += Duration::from_millis(100);
    let event6 = detector.update(&open_eyes, now);
    assert!(!event6.stare_warning);
}

#[test]
fn test_prolonged_eye_rest_pauses_stare_warning() {
    let mut detector = BlinkDetector::new(0.25, 2.0);
    let mut now = Instant::now();

    let open_eyes = create_synthetic_landmarks(6.0, 10.0);
    let closed_eyes = create_synthetic_landmarks(1.0, 10.0);

    // Start with open eyes
    detector.update(&open_eyes, now);

    // Eyes close and stay closed for 3.0s (resting/nap)
    now += Duration::from_millis(100);
    detector.update(&closed_eyes, now);

    now += Duration::from_millis(2900); // 3.0s total closure
    let event = detector.update(&closed_eyes, now);
    assert!(event.is_resting, "Should recognize prolonged rest");
    assert!(!event.stare_warning, "Must NOT issue stare alert while resting eyes");

    // Eyes reopen
    now += Duration::from_millis(100);
    let event_reopen = detector.update(&open_eyes, now);
    assert!(!event_reopen.is_blink, "Prolonged rest reopening must not count as a blink");
    assert_eq!(event_reopen.total_blinks, 0);
    assert!(!event_reopen.stare_warning);
    assert_eq!(event_reopen.stare_duration_secs, 0.0);
}

fn create_asymmetric_landmarks(left_height: f32, right_height: f32, eye_width: f32) -> Vec<Landmark3D> {
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

    // Nose tip setup (index 1) positioned symmetrically at midpoint of eyes for frontal face
    landmarks[1] = Landmark3D { x: eye_width * 0.5, y: 5.0, z: 0.0 };

    landmarks
}

#[test]
fn test_winking_single_eye_does_not_count_as_blink() {
    let mut detector = BlinkDetector::new(0.25, 2.0);
    let mut now = Instant::now();

    let both_open = create_synthetic_landmarks(6.0, 10.0); // EAR = 0.6
    let left_wink = create_asymmetric_landmarks(1.0, 6.0, 10.0); // Left EAR = 0.1, Right EAR = 0.6
    let right_wink = create_asymmetric_landmarks(6.0, 1.0, 10.0); // Left EAR = 0.6, Right EAR = 0.1

    // Initial state: both open
    detector.update(&both_open, now);

    // 1. Test isolated winking left eye only (single eye without right eye follow-up)
    now += Duration::from_millis(100);
    detector.update(&left_wink, now);
    now += Duration::from_millis(100);
    detector.update(&left_wink, now);
    now += Duration::from_millis(50);
    let evt_reopen1 = detector.update(&both_open, now);
    assert!(!evt_reopen1.is_blink);
    // Advance past the 1.0s pairing window
    now += Duration::from_millis(1100);
    let evt_expired1 = detector.update(&both_open, now);
    assert!(!evt_expired1.is_blink, "Isolated left wink must NOT count as blink");
    assert_eq!(evt_expired1.total_blinks, 0);

    // 2. Test isolated winking right eye only (single eye without left eye follow-up)
    now += Duration::from_millis(100);
    detector.update(&right_wink, now);
    now += Duration::from_millis(100);
    detector.update(&right_wink, now);
    now += Duration::from_millis(50);
    let evt_reopen2 = detector.update(&both_open, now);
    assert!(!evt_reopen2.is_blink);
    now += Duration::from_millis(1100);
    let evt_expired2 = detector.update(&both_open, now);
    assert!(!evt_expired2.is_blink, "Isolated right wink must NOT count as blink");
    assert_eq!(evt_expired2.total_blinks, 0);

    // 3. Test asynchronous sequential winking within 1.0s window:
    // Left eye winks (2 frames closed, 100ms each), then 400ms later right eye winks (2 frames closed)
    now += Duration::from_millis(100);
    detector.update(&left_wink, now);
    now += Duration::from_millis(100);
    detector.update(&left_wink, now);
    now += Duration::from_millis(50);
    detector.update(&both_open, now); // Left wink completed

    now += Duration::from_millis(400); // 400ms gap (< 1.0s)
    detector.update(&right_wink, now);
    now += Duration::from_millis(100);
    detector.update(&right_wink, now);
    now += Duration::from_millis(50);
    let evt_async_blink = detector.update(&both_open, now); // Right wink completed
    assert!(evt_async_blink.is_blink, "Sequential winking within 1.0s window MUST register as valid blink");
    assert_eq!(evt_async_blink.total_blinks, 1);

    // 4. Test simultaneous bilateral blink (both eyes close together)
    let both_closed = create_synthetic_landmarks(1.0, 10.0); // EAR = 0.1
    now += Duration::from_millis(100);
    detector.update(&both_closed, now);
    now += Duration::from_millis(100);
    detector.update(&both_closed, now);
    now += Duration::from_millis(50);
    let evt_blink = detector.update(&both_open, now);
    assert!(evt_blink.is_blink, "Simultaneous closure of both eyes MUST register as valid blink");
    assert_eq!(evt_blink.total_blinks, 2);
}

#[test]
fn test_swollen_eyelid_asymmetric_blink_resilience() {
    // Issue #75: Swollen eye with resting EAR 0.20 (below static 0.22 threshold)
    // Left eye normal: open height 3.2, EAR ~0.32
    // Right eye swollen: open height 2.0, EAR ~0.20 (delta = 0.12 > 0.08)
    let mut detector = BlinkDetector::new(0.22, 2.0);
    let mut now = Instant::now();

    let open_asymmetric = create_asymmetric_landmarks(3.2, 2.0, 10.0);
    let closed_asymmetric = create_asymmetric_landmarks(0.8, 0.7, 10.0);

    // 1. Establish initial open baseline across 10 frames
    for _ in 0..10 {
        detector.update(&open_asymmetric, now);
        now += Duration::from_millis(67); // 15 FPS
    }

    let (bl, br) = detector.baselines();
    assert!(bl.is_some() && br.is_some());
    let base_left = bl.unwrap();
    let base_right = br.unwrap();
    assert!(base_left > 0.30, "Left eye baseline must reflect normal eye (~0.32)");
    assert!(base_right < 0.25, "Right eye baseline must reflect swollen eye (~0.20)");

    // 2. Execute 5 consecutive natural blinks
    let mut detected_blinks = 0;
    for _ in 0..5 {
        // Closed for 2 frames (~134 ms)
        now += Duration::from_millis(67);
        detector.update(&closed_asymmetric, now);
        now += Duration::from_millis(67);
        detector.update(&closed_asymmetric, now);

        // Reopen (2 frames to settle EMA smoothing)
        now += Duration::from_millis(67);
        detector.update(&open_asymmetric, now);
        now += Duration::from_millis(67);
        let evt = detector.update(&open_asymmetric, now);
        if evt.is_blink {
            detected_blinks += 1;
        }

        // Open pause between blinks
        for _ in 0..8 {
            now += Duration::from_millis(67);
            detector.update(&open_asymmetric, now);
        }
    }

    assert_eq!(
        detected_blinks, 5,
        "Swollen eye (EAR 0.20) must maintain 100% blink recall without being locked out"
    );
    assert_eq!(detector.total_blinks(), 5);
}

#[test]
fn test_asymmetric_winking_immunity() {
    // Swollen right eye (0.20), normal left eye (0.32)
    // Winking either eye must NOT register as a blink
    let mut detector = BlinkDetector::new(0.22, 2.0);
    let mut now = Instant::now();

    let open_asymmetric = create_asymmetric_landmarks(3.2, 2.0, 10.0);
    let left_wink = create_asymmetric_landmarks(0.8, 2.0, 10.0); // Left closes, right stays 0.20
    let right_wink = create_asymmetric_landmarks(3.2, 0.7, 10.0); // Left stays 0.32, right closes

    // Establish baseline
    for _ in 0..10 {
        detector.update(&open_asymmetric, now);
        now += Duration::from_millis(67);
    }

    // 1. Unilateral normal eye wink
    now += Duration::from_millis(67);
    detector.update(&left_wink, now);
    now += Duration::from_millis(67);
    detector.update(&left_wink, now);
    now += Duration::from_millis(67);
    let evt1 = detector.update(&open_asymmetric, now);
    assert!(!evt1.is_blink, "Normal eye wink must NOT register as blink");

    // Advance 1.1s to expire pairing window
    now += Duration::from_millis(1100);
    detector.update(&open_asymmetric, now);

    // 2. Unilateral swollen eye wink
    now += Duration::from_millis(67);
    detector.update(&right_wink, now);
    now += Duration::from_millis(67);
    detector.update(&right_wink, now);
    now += Duration::from_millis(67);
    let evt2 = detector.update(&open_asymmetric, now);
    assert!(!evt2.is_blink, "Swollen eye wink must NOT register as blink");

    assert_eq!(detector.total_blinks(), 0, "Zero false blinks on unilateral winks");
}

#[test]
fn test_monocular_fallback_mode_and_recovery() {
    // Right eye is patched or completely swollen shut (> 2.0s simulated timeout)
    let mut detector = BlinkDetector::new(0.22, 2.0);
    detector.set_monocular_timeout(2.0); // Set fast 2.0s timeout for test
    let mut now = Instant::now();

    let both_open = create_synthetic_landmarks(3.2, 10.0);
    let right_patched = create_asymmetric_landmarks(3.2, 0.5, 10.0); // Right eye permanently shut
    let left_blink = create_asymmetric_landmarks(0.8, 0.5, 10.0); // Left eye blinks while right is shut

    // Initial state: right eye is patched
    for _ in 0..5 {
        detector.update(&right_patched, now);
        now += Duration::from_millis(100);
    }
    assert!(!detector.is_monocular(), "Should not enter monocular mode immediately");

    // Advance time past 2.0s timeout with right eye continuously closed
    now += Duration::from_millis(2100);
    detector.update(&right_patched, now);
    assert!(detector.is_monocular(), "Must enter Monocular Mode when one eye is persistently closed");

    // Left eye blinks in Monocular Mode -> should register WITHOUT waiting for right eye!
    now += Duration::from_millis(100);
    detector.update(&left_blink, now);
    now += Duration::from_millis(100);
    detector.update(&left_blink, now);
    now += Duration::from_millis(100);
    detector.update(&right_patched, now);
    now += Duration::from_millis(100);
    let evt_mono = detector.update(&right_patched, now);

    assert!(evt_mono.is_blink, "Functional eye blink in Monocular Mode must be recognized");
    assert_eq!(detector.total_blinks(), 1);

    // Recovery: Right eye patch removed / reopens!
    now += Duration::from_millis(100);
    detector.update(&both_open, now);
    assert!(!detector.is_monocular(), "Reopening occluded eye must exit Monocular Mode back to Binocular");
}
