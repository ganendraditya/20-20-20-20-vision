use std::time::{Duration, Instant};
use vision420_lib::detector::state_machine::{BlinkDecision, EyeBlinkStatus};
use vision420_lib::detector::{BlinkDetector, EyeCalibrator};
use vision420_lib::vision::Landmark3D;

fn make_landmarks(left_ear: f32, right_ear: f32) -> Vec<Landmark3D> {
    let mut points = vec![
        Landmark3D {
            x: 0.0,
            y: 0.0,
            z: 0.0
        };
        468
    ];
    let eyes = [
        (
            left_ear,
            0.0,
            (33, 133),
            [(159, 145), (158, 153), (160, 144)],
        ),
        (
            right_ear,
            30.0,
            (362, 263),
            [(386, 374), (387, 373), (385, 380)],
        ),
    ];

    for (ear, x_offset, (corner_a, corner_b), vertical_pairs) in eyes {
        points[corner_a] = Landmark3D {
            x: x_offset,
            y: 0.0,
            z: 0.0,
        };
        points[corner_b] = Landmark3D {
            x: x_offset + 10.0,
            y: 0.0,
            z: 0.0,
        };
        for (top, bottom) in vertical_pairs {
            points[top] = Landmark3D {
                x: x_offset + 5.0,
                y: -ear * 5.0,
                z: 0.0,
            };
            points[bottom] = Landmark3D {
                x: x_offset + 5.0,
                y: ear * 5.0,
                z: 0.0,
            };
        }
    }

    // Neutral frontal pose: nose is 0.45 interocular widths below the eye line.
    points[1] = Landmark3D {
        x: 20.0,
        y: 18.0,
        z: 0.0,
    };
    points
}

#[test]
fn normal_two_eye_blink_reports_detector_evidence_and_count() {
    let mut detector = BlinkDetector::new(0.22, 8.0);
    let open = make_landmarks(0.30, 0.30);
    let closed = make_landmarks(0.10, 0.10);
    let mut now = Instant::now();

    for _ in 0..10 {
        detector.update(&open, now);
        now += Duration::from_millis(67);
    }

    // A 200ms closure sampled for three frames at roughly 15 FPS.
    for _ in 0..3 {
        detector.update(&closed, now);
        now += Duration::from_millis(67);
    }
    let mut event = detector.update(&open, now);
    for _ in 0..8 {
        now += Duration::from_millis(67);
        event = detector.update(&open, now);
        if event.total_blinks == 1 {
            break;
        }
    }
    let diagnostic = detector.diagnostics();

    assert_eq!(event.total_blinks, 1);
    assert_eq!(diagnostic.decision, BlinkDecision::CountedBinocular);
    assert_eq!(diagnostic.left_status, EyeBlinkStatus::CandidateCompleted);
    assert_eq!(diagnostic.right_status, EyeBlinkStatus::CandidateCompleted);
    assert!((diagnostic.threshold_left.unwrap() - 0.22).abs() < 1e-4);
    assert!(diagnostic.frame_interval_ms > 60.0);
    assert!(diagnostic.raw_left_ear.unwrap() > 0.25);
    assert!(diagnostic.smoothed_left_ear.unwrap() < diagnostic.raw_left_ear.unwrap());
}

#[test]
fn eye_candidate_waiting_for_other_eye_is_visible_in_diagnostics() {
    let mut detector = BlinkDetector::new(0.22, 8.0);
    let open = make_landmarks(0.30, 0.30);
    let left_closed = make_landmarks(0.10, 0.30);
    let mut now = Instant::now();

    for _ in 0..10 {
        detector.update(&open, now);
        now += Duration::from_millis(67);
    }
    for _ in 0..3 {
        detector.update(&left_closed, now);
        now += Duration::from_millis(67);
    }
    for _ in 0..8 {
        detector.update(&open, now);
        now += Duration::from_millis(67);
        if detector.diagnostics().left_status == EyeBlinkStatus::CandidateCompleted {
            break;
        }
    }

    let diagnostic = detector.diagnostics();
    assert_eq!(diagnostic.left_status, EyeBlinkStatus::CandidateCompleted);
    assert_eq!(diagnostic.decision, BlinkDecision::AwaitingOtherEye);
    assert_eq!(detector.total_blinks(), 0);
}

#[test]
fn blink_falling_between_400ms_samples_is_not_observable() {
    let mut detector = BlinkDetector::new(0.22, 8.0);
    let open = make_landmarks(0.30, 0.30);
    let mut now = Instant::now();

    for _ in 0..10 {
        detector.update(&open, now);
        now += Duration::from_millis(400);
    }

    // Label the physical blink as closed from +100ms through +200ms. At a
    // 400ms cadence, every actual observation still lands in the open interval.
    let blink_start = Duration::from_millis(100);
    let blink_end = Duration::from_millis(200);
    let mut closed_samples = 0;
    for sample_index in 0..3 {
        let sample_time = Duration::from_millis(sample_index * 400);
        let physical_eye_is_closed = sample_time >= blink_start && sample_time < blink_end;
        let sampled_landmarks = if physical_eye_is_closed {
            closed_samples += 1;
            make_landmarks(0.10, 0.10)
        } else {
            open.clone()
        };
        detector.update(&sampled_landmarks, now);
        now += Duration::from_millis(400);
    }

    assert_eq!(
        closed_samples, 0,
        "the labeled blink must fall between all camera samples"
    );
    assert_eq!(detector.total_blinks(), 0);
    assert_eq!(
        detector.diagnostics().decision,
        BlinkDecision::NoClosureCandidate
    );
}

#[test]
fn calibrated_cutoff_is_used_but_cannot_reach_the_open_eye_baseline() {
    let mut calibration = EyeCalibrator::new();
    for _ in 0..10 {
        calibration.add_open_sample(0.30);
        calibration.add_closed_sample(0.22);
    }
    let calibrated_threshold = calibration.finalize().unwrap();
    assert!((calibrated_threshold - 0.25).abs() < 1e-4);

    let mut detector = BlinkDetector::new(calibrated_threshold, 8.0);
    let open = make_landmarks(0.30, 0.30);
    let closed = make_landmarks(0.22, 0.22);
    let mut now = Instant::now();

    for _ in 0..10 {
        detector.update(&open, now);
        now += Duration::from_millis(100);
    }
    assert!((detector.diagnostics().threshold_left.unwrap() - 0.25).abs() < 1e-4);

    for _ in 0..5 {
        detector.update(&closed, now);
        now += Duration::from_millis(100);
    }
    let mut event = detector.update(&open, now);
    for _ in 0..8 {
        now += Duration::from_millis(100);
        event = detector.update(&open, now);
        if event.total_blinks > 0 {
            break;
        }
    }
    assert_eq!(event.total_blinks, 1);

    let mut capped = BlinkDetector::new(0.28, 8.0);
    for _ in 0..10 {
        capped.update(&open, now);
        now += Duration::from_millis(100);
    }
    assert!((capped.diagnostics().threshold_left.unwrap() - 0.27).abs() < 1e-4);
    assert!(capped.diagnostics().threshold_left.unwrap() < capped.baselines().0.unwrap());
}

#[test]
fn ambiguous_low_ear_plateau_does_not_erode_the_open_eye_baseline() {
    let mut detector = BlinkDetector::new(0.22, 8.0);
    let open = make_landmarks(0.30, 0.30);
    let plateau = make_landmarks(0.24, 0.24);
    let mut now = Instant::now();

    for _ in 0..10 {
        detector.update(&open, now);
        now += Duration::from_millis(100);
    }
    for _ in 0..500 {
        detector.update(&plateau, now);
        now += Duration::from_millis(100);
    }

    let (left, right) = detector.baselines();
    assert!(left.unwrap() > 0.299);
    assert!(right.unwrap() > 0.299);
    assert_eq!(detector.diagnostics().left_status, EyeBlinkStatus::Open);
    assert_eq!(detector.total_blinks(), 0);
}

#[test]
fn low_fps_descent_candidate_survives_a_plateau_until_stable_rebound() {
    let mut detector = BlinkDetector::new(0.22, 8.0);
    let open = make_landmarks(0.32, 0.32);
    let shallow_closed = make_landmarks(0.22, 0.22);
    let mut now = Instant::now();

    for _ in 0..10 {
        detector.update(&open, now);
        now += Duration::from_millis(100);
    }
    for _ in 0..5 {
        detector.update(&shallow_closed, now);
        now += Duration::from_millis(100);
    }
    assert_eq!(detector.total_blinks(), 0);

    let mut event = detector.update(&open, now);
    for _ in 0..8 {
        now += Duration::from_millis(100);
        event = detector.update(&open, now);
        if event.total_blinks > 0 {
            break;
        }
    }
    assert_eq!(event.total_blinks, 1);

    let mut stale = BlinkDetector::new(0.22, 8.0);
    let mut now = Instant::now();
    for _ in 0..10 {
        stale.update(&open, now);
        now += Duration::from_millis(100);
    }
    for _ in 0..10 {
        stale.update(&shallow_closed, now);
        now += Duration::from_millis(100);
    }
    for _ in 0..8 {
        stale.update(&open, now);
        now += Duration::from_millis(100);
    }
    assert_eq!(
        stale.total_blinks(),
        0,
        "a descent without timely reopening must expire"
    );
}

#[test]
fn low_fps_velocity_candidate_waits_for_confirmed_reopening() {
    let mut detector = BlinkDetector::new(0.22, 8.0);
    let open = make_landmarks(0.30, 0.30);
    let shallow_closure = make_landmarks(0.22, 0.22);
    let partial_reopening = make_landmarks(0.24, 0.24);
    let mut now = Instant::now();

    for _ in 0..10 {
        detector.update(&open, now);
        now += Duration::from_millis(100);
    }
    detector.update(&shallow_closure, now);
    now += Duration::from_millis(100);

    // This shallow plateau is above the static cutoff and the old 75%-baseline
    // rebound floor, but below the midpoint reopening threshold (0.26).
    for _ in 0..3 {
        detector.update(&partial_reopening, now);
        now += Duration::from_millis(100);
    }
    assert_eq!(detector.total_blinks(), 0);
    assert_eq!(
        detector.diagnostics().decision,
        BlinkDecision::ClosurePending,
        "a partial reopening must not complete a velocity-only candidate"
    );

    let mut event = detector.update(&open, now);
    for _ in 0..5 {
        now += Duration::from_millis(100);
        event = detector.update(&open, now);
        if event.total_blinks > 0 {
            break;
        }
    }
    assert_eq!(event.total_blinks, 1);
    assert_eq!(
        detector.diagnostics().decision,
        BlinkDecision::CountedBinocular
    );
}

#[test]
fn low_fps_velocity_rebound_cannot_count_a_long_static_closure() {
    let mut detector = BlinkDetector::new(0.22, 8.0);
    let open = make_landmarks(0.30, 0.30);
    let held_closed = make_landmarks(0.20, 0.20);
    let brief_deeper_dip = make_landmarks(0.08, 0.08);
    let partial_reopen = make_landmarks(0.24, 0.24);
    let mut now = Instant::now();

    for _ in 0..10 {
        detector.update(&open, now);
        now += Duration::from_millis(100);
    }
    for _ in 0..20 {
        detector.update(&held_closed, now);
        now += Duration::from_millis(100);
    }

    detector.update(&brief_deeper_dip, now);
    now += Duration::from_millis(100);
    for _ in 0..8 {
        detector.update(&partial_reopen, now);
        now += Duration::from_millis(100);
    }

    let diagnostic = detector.diagnostics();
    assert_eq!(detector.total_blinks(), 0);
    assert_eq!(diagnostic.left_status, EyeBlinkStatus::ClosurePending);
    assert_eq!(diagnostic.right_status, EyeBlinkStatus::ClosurePending);
    assert!(diagnostic.left_closure_ms > 800.0);
    assert!(diagnostic.right_closure_ms > 800.0);
}

#[test]
fn reopening_hysteresis_prevents_one_eye_closure_from_splitting_on_a_single_spike() {
    let mut detector = BlinkDetector::new(0.22, 8.0);
    let open = make_landmarks(0.30, 0.30);
    let closed = make_landmarks(0.10, 0.10);
    let one_frame_open_spike = make_landmarks(0.65, 0.65);
    let mut now = Instant::now();

    for _ in 0..10 {
        detector.update(&open, now);
        now += Duration::from_millis(50);
    }
    for _ in 0..3 {
        detector.update(&closed, now);
        now += Duration::from_millis(50);
    }
    let event = detector.update(&one_frame_open_spike, now);
    assert_eq!(
        event.total_blinks, 0,
        "one recovered sample must not finish a closure"
    );
    now += Duration::from_millis(50);

    for _ in 0..3 {
        detector.update(&closed, now);
        now += Duration::from_millis(50);
    }
    let mut event = detector.update(&open, now);
    for _ in 0..8 {
        now += Duration::from_millis(50);
        event = detector.update(&open, now);
        if event.total_blinks > 0 {
            break;
        }
    }

    assert_eq!(event.total_blinks, 1);
    assert_eq!(
        detector.diagnostics().decision,
        BlinkDecision::CountedBinocular
    );
}

#[test]
fn reopening_confirmation_keeps_the_800ms_limit_on_sampled_raw_recovery() {
    fn count_after_closed_samples(closed_samples: usize) -> u32 {
        let mut detector = BlinkDetector::new(0.22, 8.0);
        let open = make_landmarks(0.30, 0.30);
        let closed = make_landmarks(0.10, 0.10);
        let mut now = Instant::now();

        for _ in 0..10 {
            detector.update(&open, now);
            now += Duration::from_millis(100);
        }
        for _ in 0..closed_samples {
            detector.update(&closed, now);
            now += Duration::from_millis(100);
        }
        for _ in 0..8 {
            detector.update(&open, now);
            now += Duration::from_millis(100);
        }
        detector.total_blinks()
    }

    assert_eq!(count_after_closed_samples(8), 1);
    assert_eq!(count_after_closed_samples(9), 0);
}
