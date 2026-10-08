pub mod state_machine;

pub use state_machine::{BlinkDetector, BlinkEvent};
use crate::vision::Landmark3D;

// Single source of truth for canonical MediaPipe 468 landmark indices around the eyes
// In MediaPipe 468 FaceMesh topology:
// Left eye (viewer's left / anatomical right eye):
//   Horizontal corners: 33 (outer), 133 (inner)
//   Vertical top-bottom pairs: (159, 145) center, (158, 153), (160, 144)
// Right eye (viewer's right / anatomical left eye):
//   Horizontal corners: 362 (inner), 263 (outer)
//   Vertical top-bottom pairs: (386, 374) center, (387, 373), (385, 380)

pub const LEFT_EYE_H: (usize, usize) = (33, 133);
pub const LEFT_EYE_V1: (usize, usize) = (159, 145);
pub const LEFT_EYE_V2: (usize, usize) = (158, 153);
pub const LEFT_EYE_V3: (usize, usize) = (160, 144);

pub const RIGHT_EYE_H: (usize, usize) = (362, 263);
pub const RIGHT_EYE_V1: (usize, usize) = (386, 374);
pub const RIGHT_EYE_V2: (usize, usize) = (387, 373);
pub const RIGHT_EYE_V3: (usize, usize) = (385, 380);

// Landmark indices for Head Pose Yaw & Pitch calculation
pub const NOSE_TIP: usize = 1;
pub const LEFT_EYE_OUTER_CORNER: usize = 33;
pub const RIGHT_EYE_OUTER_CORNER: usize = 263;

/// Maximum permissible nasal-interocular yaw asymmetry ratio for frontal gaze (< ~35 degree turn)
pub const MAX_YAW_RATIO_FRONTAL: f32 = 0.35;

/// Threshold for upward pitch gaze ratio (nose tip elevated toward/above eye plane)
/// When looking upward (> ~15 degrees), pitch_ratio >= 0.15 indicates genuine upward distance rest.
pub const MIN_PITCH_RATIO_UPWARD_REST: f32 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadPoseMetrics {
    pub is_facing_camera: bool,
    pub is_resting_gaze: bool,
    pub yaw_ratio: f32,
    pub pitch_ratio: f32,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub roll_deg: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarMetrics {
    pub left_ear: f32,
    pub right_ear: f32,
    pub avg_ear: f32,
    pub smoothed_left_ear: f32,
    pub smoothed_right_ear: f32,
    pub smoothed_ear: f32,
}

pub struct EarCalculator {
    alpha: f32, // Smoothing factor for EMA (e.g., 0.35 for responsiveness + noise rejection)
    last_smoothed_left: Option<f32>,
    last_smoothed_right: Option<f32>,
    last_smoothed_avg: Option<f32>,
}

impl EarCalculator {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            last_smoothed_left: None,
            last_smoothed_right: None,
            last_smoothed_avg: None,
        }
    }

    /// Calculate Euclidean distance between two landmarks on the 2D image plane
    /// (MediaPipe Z axis is estimated relative depth with high jitter, canonical Soukupova & Cech 2016 uses 2D)
    pub fn distance(p1: &Landmark3D, p2: &Landmark3D) -> f32 {
        let dx = p1.x - p2.x;
        let dy = p1.y - p2.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Compute Eye Aspect Ratio (EAR) for a single eye given corner & vertical landmark pairs
    pub fn compute_single_ear(
        landmarks: &[Landmark3D],
        h_pair: (usize, usize),
        v1_pair: (usize, usize),
        v2_pair: (usize, usize),
        v3_pair: (usize, usize),
    ) -> f32 {
        let (Some(p_h1), Some(p_h2)) = (landmarks.get(h_pair.0), landmarks.get(h_pair.1)) else {
            return 0.0;
        };
        let (Some(p_v1_top), Some(p_v1_bot)) = (landmarks.get(v1_pair.0), landmarks.get(v1_pair.1)) else {
            return 0.0;
        };
        let (Some(p_v2_top), Some(p_v2_bot)) = (landmarks.get(v2_pair.0), landmarks.get(v2_pair.1)) else {
            return 0.0;
        };
        let (Some(p_v3_top), Some(p_v3_bot)) = (landmarks.get(v3_pair.0), landmarks.get(v3_pair.1)) else {
            return 0.0;
        };

        let dist_h = Self::distance(p_h1, p_h2);
        if dist_h <= 1e-6 {
            return 0.0;
        }

        let dist_v1 = Self::distance(p_v1_top, p_v1_bot);
        let dist_v2 = Self::distance(p_v2_top, p_v2_bot);
        let dist_v3 = Self::distance(p_v3_top, p_v3_bot);

        // Standard 3-pair EAR formula: (||v1|| + ||v2|| + ||v3||) / (3.0 * ||h||)
        (dist_v1 + dist_v2 + dist_v3) / (3.0 * dist_h)
    }

    /// Evaluate head pose 3D yaw and pitch angles via nasal-interocular geometry.
    /// Calculates:
    /// 1. Yaw ratio: horizontal offset between nose tip (index 1) and eye midpoint normalized by interocular width.
    /// 2. Pitch ratio: vertical elevation of nose tip relative to eye level normalized by interocular width.
    ///
    /// - Frontal gaze looking at screen: yaw_ratio <= 0.35 AND pitch_ratio < 0.15
    /// - Resting gaze (Looking away or upward): yaw_ratio > 0.35 OR pitch_ratio >= 0.15 (gazing at high window/ceiling)
    pub fn estimate_head_pose(landmarks: &[Landmark3D]) -> Option<HeadPoseMetrics> {
        let (Some(p_left), Some(p_right), Some(p_nose)) = (
            landmarks.get(LEFT_EYE_OUTER_CORNER),
            landmarks.get(RIGHT_EYE_OUTER_CORNER),
            landmarks.get(NOSE_TIP),
        ) else {
            return None;
        };

        // Euclidean 2D distance between eye corners ensures roll-invariance (head tilt)
        let interocular_width = Self::distance(p_left, p_right);
        if interocular_width <= 1e-4 {
            return None;
        }

        // Eye midpoint M = (p_left + p_right) / 2
        let mx = (p_left.x + p_right.x) / 2.0;
        let my = (p_left.y + p_right.y) / 2.0;

        // Normalized unit vector along the eye line: v = (p_right - p_left) / width
        let vx = (p_right.x - p_left.x) / interocular_width;
        let vy = (p_right.y - p_left.y) / interocular_width;

        // Perpendicular normal vector pointing downwards across face: u = (-vy, vx)
        let ux = -vy;
        let uy = vx;

        // Offset vector of nose from eye midpoint: d = p_nose - M
        let dx = p_nose.x - mx;
        let dy = p_nose.y - my;

        // Yaw component is projection of d along the eye axis (horizontal relative to head):
        let nose_offset_along_eye_axis_signed = dx * vx + dy * vy;
        let nose_offset_along_eye_axis = nose_offset_along_eye_axis_signed.abs();
        let yaw_ratio = nose_offset_along_eye_axis / interocular_width;
        let yaw_deg = (nose_offset_along_eye_axis_signed / interocular_width)
            .clamp(-0.99, 0.99)
            .asin()
            .to_degrees();

        // Pitch component is projection of d along the perpendicular face axis:
        // In image coordinates, y increases downward.
        // In natural upright frontal posture, nose tip is below eye midpoint (d . u > 0, typical ~0.35-0.55).
        // When user tilts head upward, the nose tip moves closer to or above the eye line (d . u decreases).
        let normal_offset = dx * ux + dy * uy;
        // Natural neutral resting offset is approximately 0.45 interocular units below eyes.
        // An elevation delta > 0.15 indicates significant upward pitch (gazing at ceiling / distant high point).
        let baseline_downward_offset = 0.45 * interocular_width;
        let upward_elevation = (baseline_downward_offset - normal_offset)
            .clamp(-interocular_width, interocular_width);
        let pitch_ratio = upward_elevation / interocular_width;
        let pitch_deg = pitch_ratio.clamp(-0.99, 0.99).asin().to_degrees();

        // Roll component: in-plane tilt angle of the eye axis vector (vx, vy)
        let roll_deg = vy.atan2(vx).to_degrees();

        let is_turned_away = yaw_ratio > MAX_YAW_RATIO_FRONTAL;
        let is_gazing_upward = pitch_ratio >= MIN_PITCH_RATIO_UPWARD_REST;

        // Subject is facing camera if not turned away and not looking far up
        let is_facing_camera = !is_turned_away && !is_gazing_upward;

        Some(HeadPoseMetrics {
            is_facing_camera,
            is_resting_gaze: !is_facing_camera,
            yaw_ratio,
            pitch_ratio,
            yaw_deg,
            pitch_deg,
            roll_deg,
        })
    }

    /// Calculate left, right, average, and EMA-smoothed EAR from 468 facial landmarks
    pub fn calculate(&mut self, landmarks: &[Landmark3D]) -> Option<EarMetrics> {
        if landmarks.len() < 468 {
            return None;
        }

        let left_ear = Self::compute_single_ear(landmarks, LEFT_EYE_H, LEFT_EYE_V1, LEFT_EYE_V2, LEFT_EYE_V3);
        let right_ear = Self::compute_single_ear(landmarks, RIGHT_EYE_H, RIGHT_EYE_V1, RIGHT_EYE_V2, RIGHT_EYE_V3);
        let avg_ear = (left_ear + right_ear) / 2.0;

        // Asymmetric Asynchronous EMA Filter:
        // When eyes are closing (current EAR < prev EAR), we want FAST, lag-free response (alpha_fast = 0.85)
        // so real biological blinks (even 100 ms micro-blinks) are instantly recognized by the state machine.
        // When eyes are resting/static (open or closed steady), we use normal alpha (0.35)
        // to filter out sensor flicker, webcam glare, and frame noise.
        let left_alpha = match self.last_smoothed_left {
            Some(prev) if left_ear < prev => 0.85f32,
            _ => self.alpha,
        };
        let right_alpha = match self.last_smoothed_right {
            Some(prev) if right_ear < prev => 0.85f32,
            _ => self.alpha,
        };

        let smoothed_left_ear = match self.last_smoothed_left {
            Some(prev) => left_alpha * left_ear + (1.0 - left_alpha) * prev,
            None => left_ear,
        };
        let smoothed_right_ear = match self.last_smoothed_right {
            Some(prev) => right_alpha * right_ear + (1.0 - right_alpha) * prev,
            None => right_ear,
        };
        let smoothed_ear = match self.last_smoothed_avg {
            Some(prev) => self.alpha * avg_ear + (1.0 - self.alpha) * prev, // Keep canonical EMA for avg_ear display
            None => avg_ear,
        };

        self.last_smoothed_left = Some(smoothed_left_ear);
        self.last_smoothed_right = Some(smoothed_right_ear);
        self.last_smoothed_avg = Some(smoothed_ear);

        Some(EarMetrics {
            left_ear,
            right_ear,
            avg_ear,
            smoothed_left_ear,
            smoothed_right_ear,
            smoothed_ear,
        })
    }

    pub fn reset(&mut self) {
        self.last_smoothed_left = None;
        self.last_smoothed_right = None;
        self.last_smoothed_avg = None;
    }
}

/// 5-second automatic eye calibration state machine
pub struct EyeCalibrator {
    open_samples: Vec<f32>,
    closed_samples: Vec<f32>,
}

impl Default for EyeCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl EyeCalibrator {
    pub fn new() -> Self {
        Self {
            open_samples: Vec::with_capacity(50),
            closed_samples: Vec::with_capacity(30),
        }
    }

    pub fn add_open_sample(&mut self, ear: f32) {
        if ear > 0.05 {
            self.open_samples.push(ear);
        }
    }

    pub fn add_closed_sample(&mut self, ear: f32) {
        if ear > 0.01 {
            self.closed_samples.push(ear);
        }
    }

    /// Calculate optimal threshold: closed + (open - closed) * 0.38
    pub fn finalize(&self) -> Option<f32> {
        if self.open_samples.is_empty() || self.closed_samples.is_empty() {
            return None;
        }

        let mut sorted_open = self.open_samples.clone();
        sorted_open.sort_by(|a, b| a.total_cmp(b));
        let median_open = sorted_open[sorted_open.len() / 2];

        let mut sorted_closed = self.closed_samples.clone();
        sorted_closed.sort_by(|a, b| a.total_cmp(b));
        let median_closed = sorted_closed[sorted_closed.len() / 2];

        if median_open <= median_closed {
            return None;
        }

        let threshold = median_closed + (median_open - median_closed) * 0.38;
        Some((threshold * 1000.0).round() / 1000.0) // Round to 3 decimal places
    }
}
