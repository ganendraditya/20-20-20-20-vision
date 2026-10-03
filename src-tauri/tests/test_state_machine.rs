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

    // 1. Test winking left eye only (closure for 200ms)
    now += Duration::from_millis(100);
    let evt_w1 = detector.update(&left_wink, now);
    assert!(!evt_w1.is_blink);
    now += Duration::from_millis(100);
    let evt_w2 = detector.update(&left_wink, now);
    assert!(!evt_w2.is_blink);
    now += Duration::from_millis(50);
    let evt_reopen1 = detector.update(&both_open, now);
    assert!(!evt_reopen1.is_blink, "Left eye wink must NOT register as blink");
    assert_eq!(evt_reopen1.total_blinks, 0);

    // 2. Test winking right eye only (closure for 200ms)
    now += Duration::from_millis(100);
    let evt_r1 = detector.update(&right_wink, now);
    assert!(!evt_r1.is_blink);
    now += Duration::from_millis(100);
    let evt_r2 = detector.update(&right_wink, now);
    assert!(!evt_r2.is_blink);
    now += Duration::from_millis(50);
    let evt_reopen2 = detector.update(&both_open, now);
    assert!(!evt_reopen2.is_blink, "Right eye wink must NOT register as blink");
    assert_eq!(evt_reopen2.total_blinks, 0);

    // 3. Test genuine bilateral blink (both eyes close for 200ms)
    let both_closed = create_synthetic_landmarks(1.0, 10.0); // EAR = 0.1
    now += Duration::from_millis(100);
    detector.update(&both_closed, now);
    now += Duration::from_millis(100);
    detector.update(&both_closed, now);
    now += Duration::from_millis(50);
    let evt_blink = detector.update(&both_open, now);
    assert!(evt_blink.is_blink, "Simultaneous closure of both eyes MUST register as valid blink");
    assert_eq!(evt_blink.total_blinks, 1);
}
