use vision420_lib::capture::CameraFrameDto;

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
        eye_landmarks: vec![],
        face_landmarks: vec![],
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
    assert!((deserialized.yaw_deg - 4.2).abs() < 1e-4);
    assert!((deserialized.distance_cm - 55.0).abs() < 1e-4);
    assert!(!deserialized.is_resting_gaze);
}
