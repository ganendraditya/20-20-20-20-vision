use crate::detector::state_machine::{BlinkDecision, BlinkDiagnostics, EyeBlinkStatus};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BlinkDecisionDto {
    FaceUnavailable,
    LookingAway,
    NoClosureCandidate,
    ClosurePending,
    AwaitingOtherEye,
    CountedBinocular,
    CountedMonocular,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EyeBlinkStatusDto {
    Open,
    ClosurePending,
    CandidateCompleted,
    RejectedTooShort,
    RejectedTooLong,
    RejectedInsufficientSamples,
    RejectedDownwardGaze,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlinkDiagnosticsDto {
    pub version: u8,
    pub decision: BlinkDecisionDto,
    pub left_status: EyeBlinkStatusDto,
    pub right_status: EyeBlinkStatusDto,
    pub smoothed_left_ear: Option<f32>,
    pub smoothed_right_ear: Option<f32>,
    pub threshold_left: Option<f32>,
    pub threshold_right: Option<f32>,
    pub left_closure_ms: f32,
    pub right_closure_ms: f32,
    pub is_resting: bool,
}

impl From<BlinkDiagnostics> for BlinkDiagnosticsDto {
    fn from(d: BlinkDiagnostics) -> Self {
        fn eye_status(status: EyeBlinkStatus) -> EyeBlinkStatusDto {
            match status {
                EyeBlinkStatus::Open => EyeBlinkStatusDto::Open,
                EyeBlinkStatus::ClosurePending => EyeBlinkStatusDto::ClosurePending,
                EyeBlinkStatus::CandidateCompleted => EyeBlinkStatusDto::CandidateCompleted,
                EyeBlinkStatus::RejectedTooShort => EyeBlinkStatusDto::RejectedTooShort,
                EyeBlinkStatus::RejectedTooLong => EyeBlinkStatusDto::RejectedTooLong,
                EyeBlinkStatus::RejectedInsufficientSamples => {
                    EyeBlinkStatusDto::RejectedInsufficientSamples
                }
                EyeBlinkStatus::RejectedDownwardGaze => EyeBlinkStatusDto::RejectedDownwardGaze,
            }
        }

        Self {
            version: 1,
            decision: match d.decision {
                BlinkDecision::FaceUnavailable => BlinkDecisionDto::FaceUnavailable,
                BlinkDecision::LookingAway => BlinkDecisionDto::LookingAway,
                BlinkDecision::NoClosureCandidate => BlinkDecisionDto::NoClosureCandidate,
                BlinkDecision::ClosurePending => BlinkDecisionDto::ClosurePending,
                BlinkDecision::AwaitingOtherEye => BlinkDecisionDto::AwaitingOtherEye,
                BlinkDecision::CountedBinocular => BlinkDecisionDto::CountedBinocular,
                BlinkDecision::CountedMonocular => BlinkDecisionDto::CountedMonocular,
            },
            left_status: eye_status(d.left_status),
            right_status: eye_status(d.right_status),
            smoothed_left_ear: d.smoothed_left_ear,
            smoothed_right_ear: d.smoothed_right_ear,
            threshold_left: d.threshold_left,
            threshold_right: d.threshold_right,
            left_closure_ms: d.left_closure_ms,
            right_closure_ms: d.right_closure_ms,
            is_resting: d.is_resting,
        }
    }
}

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
    #[serde(default)]
    pub blink_diagnostics: Option<BlinkDiagnosticsDto>,
    pub eye_landmarks: Vec<LandmarkPoint>,
    pub face_landmarks: Vec<LandmarkPoint>,
    /// Full 468-point FaceMesh for the camera-test wireframe; defaults empty for older payloads.
    #[serde(default)]
    pub face_mesh_landmarks: Vec<LandmarkPoint>,
    /// JPEG base64 or empty when window is minimized
    pub image_data_base64: Option<String>,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub roll_deg: f32,
    pub distance_cm: f32,
    pub is_resting_gaze: bool,
}
