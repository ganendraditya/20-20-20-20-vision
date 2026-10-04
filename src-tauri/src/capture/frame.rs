use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraFrameDto {
    pub width: u32,
    pub height: u32,
    pub is_face_detected: bool,
    pub left_ear: f32,
    pub right_ear: f32,
    pub avg_ear: f32,
    pub is_blinking: bool,
    pub total_blinks: u32,
    pub eye_landmarks: Vec<LandmarkPoint>,
    pub face_landmarks: Vec<LandmarkPoint>,
    /// JPEG base64 or empty when window is minimized
    pub image_data_base64: Option<String>,
}
