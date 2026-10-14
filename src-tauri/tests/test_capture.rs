use vision420_lib::capture::{CameraFrameDto, LandmarkPoint};

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

    let mut legacy_json = serde_json::to_value(&dto).expect("Serialization failed");
    legacy_json
        .as_object_mut()
        .expect("CameraFrameDto should serialize as an object")
        .remove("face_mesh_landmarks");
    let legacy_dto: CameraFrameDto =
        serde_json::from_value(legacy_json).expect("Legacy CameraFrameDto should deserialize");
    assert!(legacy_dto.face_mesh_landmarks.is_empty());
}
