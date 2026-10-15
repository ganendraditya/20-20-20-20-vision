use vision420_lib::capture::frame::{BlinkDecisionDto, BlinkDiagnosticsDto};
use vision420_lib::capture::{CameraFrameDto, LandmarkPoint};
use vision420_lib::detector::state_machine::{BlinkDecision, BlinkDiagnostics, EyeBlinkStatus};

#[test]
fn test_camera_frame_dto_serialization() {
    let dto = CameraFrameDto {
        width: 640,
        height: 480,
        is_face_detected: true,
        left_ear: 0.28,
        right_ear: 0.29,
        avg_ear: 0.285,
        is_blinking: false,
        total_blinks: 12,
        blink_diagnostics: Some(BlinkDiagnostics::default().into()),
        eye_landmarks: vec![],
        face_landmarks: vec![],
        face_mesh_landmarks: vec![LandmarkPoint { x: 0.25, y: 0.75 }; 468],
        image_data_base64: None,
        yaw_deg: 4.2,
        pitch_deg: -1.5,
        roll_deg: 0.8,
        distance_cm: 55.0,
        is_resting_gaze: false,
    };

    let serialized = serde_json::to_string(&dto).expect("Serialization failed");
    let deserialized: CameraFrameDto = serde_json::from_str(&serialized).expect("Deserialization failed");

    assert_eq!(deserialized.width, 640);
    assert_eq!(deserialized.height, 480);
    assert!(deserialized.is_face_detected);
    assert_eq!(deserialized.total_blinks, 12);
    assert_eq!(deserialized.face_mesh_landmarks.len(), 468);
    assert!((deserialized.face_mesh_landmarks[0].x - 0.25).abs() < 1e-4);
    assert!((deserialized.yaw_deg - 4.2).abs() < 1e-4);
    assert!((deserialized.distance_cm - 55.0).abs() < 1e-4);
    assert!(!deserialized.is_resting_gaze);
    let diagnostics = deserialized.blink_diagnostics.unwrap();
    assert_eq!(diagnostics.version, 1);
    assert_eq!(diagnostics.decision, BlinkDecisionDto::FaceUnavailable);
    assert!(diagnostics.smoothed_left_ear.is_none());

    // Existing senders without the additive feedback field remain readable.
    let mut legacy = serde_json::to_value(&dto).unwrap();
    legacy.as_object_mut().unwrap().remove("blink_diagnostics");
    let legacy: CameraFrameDto = serde_json::from_value(legacy).unwrap();
    assert!(legacy.blink_diagnostics.is_none());
}

#[test]
fn camera_frame_dto_defaults_missing_face_mesh_landmarks() {
    let dto = CameraFrameDto {
        width: 640,
        height: 480,
        is_face_detected: false,
        left_ear: 0.0,
        right_ear: 0.0,
        avg_ear: 0.0,
        is_blinking: false,
        total_blinks: 0,
        blink_diagnostics: None,
        eye_landmarks: vec![],
        face_landmarks: vec![],
        face_mesh_landmarks: vec![LandmarkPoint { x: 0.25, y: 0.75 }; 468],
        image_data_base64: None,
        yaw_deg: 0.0,
        pitch_deg: 0.0,
        roll_deg: 0.0,
        distance_cm: 0.0,
        is_resting_gaze: false,
    };

    let mut legacy_json = serde_json::to_value(&dto).expect("Serialization failed");
    legacy_json
        .as_object_mut()
        .expect("CameraFrameDto should serialize as an object")
        .remove("face_mesh_landmarks");
    let legacy_dto: CameraFrameDto =
        serde_json::from_value(legacy_json).expect("Legacy CameraFrameDto should deserialize");
    assert!(legacy_dto.face_mesh_landmarks.is_empty());
}

#[test]
fn blink_feedback_preserves_rejection_evidence() {
    let diagnostics = BlinkDiagnostics {
        decision: BlinkDecision::AwaitingOtherEye,
        left_status: EyeBlinkStatus::RejectedTooLong,
        right_status: EyeBlinkStatus::CandidateCompleted,
        smoothed_left_ear: Some(0.24),
        smoothed_right_ear: Some(0.23),
        threshold_left: Some(0.195),
        threshold_right: Some(0.19),
        left_closure_ms: 950.0,
        right_closure_ms: 200.0,
        ..BlinkDiagnostics::default()
    };
    let dto = BlinkDiagnosticsDto::from(diagnostics);
    let json = serde_json::to_value(dto).unwrap();
    assert_eq!(json["version"], 1);
    assert_eq!(json["decision"], "awaiting_other_eye");
    assert_eq!(json["left_status"], "rejected_too_long");
    assert_eq!(json["right_status"], "candidate_completed");
    assert_eq!(json["left_closure_ms"], 950.0);
    assert_eq!(json["right_closure_ms"], 200.0);
    let roundtrip: BlinkDiagnosticsDto = serde_json::from_value(json).unwrap();
    assert!((roundtrip.threshold_left.unwrap() - 0.195).abs() < 1e-4);
    assert!((roundtrip.smoothed_right_ear.unwrap() - 0.23).abs() < 1e-4);
}
