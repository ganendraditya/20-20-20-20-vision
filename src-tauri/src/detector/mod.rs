pub mod state_machine;

pub use state_machine::{BlinkDetector, BlinkEvent};
use crate::vision::Landmark3D;

// Single source of truth for canonical MediaPipe 468 landmark indices around the eyes
// Reference: Soukupova & Cech (2016) adapted to MediaPipe topology

pub const LEFT_EYE_H: (usize, usize) = (362, 263); // Outer & inner corners
pub const LEFT_EYE_V1: (usize, usize) = (385, 380); // Top & bottom pair 1
pub const LEFT_EYE_V2: (usize, usize) = (387, 373); // Top & bottom pair 2

pub const RIGHT_EYE_H: (usize, usize) = (33, 133);  // Outer & inner corners
pub const RIGHT_EYE_V1: (usize, usize) = (160, 144); // Top & bottom pair 1
pub const RIGHT_EYE_V2: (usize, usize) = (158, 153); // Top & bottom pair 2

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarMetrics {
    pub left_ear: f32,
    pub right_ear: f32,
    pub avg_ear: f32,
    pub smoothed_ear: f32,
}

pub struct EarCalculator {
    alpha: f32, // Smoothing factor for EMA (e.g., 0.3 for responsiveness + glare stability)
    last_smoothed_ear: Option<f32>,
}

impl EarCalculator {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            last_smoothed_ear: None,
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

        let dist_h = Self::distance(p_h1, p_h2);
        if dist_h <= 1e-6 {
            return 0.0;
        }

        let dist_v1 = Self::distance(p_v1_top, p_v1_bot);
        let dist_v2 = Self::distance(p_v2_top, p_v2_bot);

        // EAR formula: (||v1|| + ||v2||) / (2 * ||h||)
        (dist_v1 + dist_v2) / (2.0 * dist_h)
    }

    /// Calculate left, right, average, and EMA-smoothed EAR from 468 facial landmarks
    pub fn calculate(&mut self, landmarks: &[Landmark3D]) -> Option<EarMetrics> {
        if landmarks.len() < 468 {
            return None;
        }

        let left_ear = Self::compute_single_ear(landmarks, LEFT_EYE_H, LEFT_EYE_V1, LEFT_EYE_V2);
        let right_ear = Self::compute_single_ear(landmarks, RIGHT_EYE_H, RIGHT_EYE_V1, RIGHT_EYE_V2);
        let avg_ear = (left_ear + right_ear) / 2.0;

        // Exponential Moving Average (EMA) smoothing:
        // EMA_t = alpha * current + (1 - alpha) * EMA_{t-1}
        let smoothed_ear = match self.last_smoothed_ear {
            Some(prev) => self.alpha * avg_ear + (1.0 - self.alpha) * prev,
            None => avg_ear,
        };

        self.last_smoothed_ear = Some(smoothed_ear);

        Some(EarMetrics {
            left_ear,
            right_ear,
            avg_ear,
            smoothed_ear,
        })
    }

    pub fn reset(&mut self) {
        self.last_smoothed_ear = None;
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
        sorted_open.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_open = sorted_open[sorted_open.len() / 2];

        let mut sorted_closed = self.closed_samples.clone();
        sorted_closed.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_closed = sorted_closed[sorted_closed.len() / 2];

        if median_open <= median_closed {
            return None;
        }

        let threshold = median_closed + (median_open - median_closed) * 0.38;
        Some((threshold * 1000.0).round() / 1000.0) // Round to 3 decimal places
    }
}
