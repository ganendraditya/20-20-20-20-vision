use std::collections::VecDeque;
use std::time::Instant;
use crate::vision::Landmark3D;
use crate::detector::EarCalculator;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlinkEvent {
    pub is_blink: bool,
    pub is_resting: bool,
    pub stare_warning: bool,
    pub current_bpm: f32,
    pub total_blinks: u32,
    pub stare_duration_secs: f32,
    pub left_ear: f32,
    pub right_ear: f32,
    pub avg_ear: f32,
}

pub struct BlinkDetector {
    ear_calculator: EarCalculator,
    threshold: f32,
    stare_limit_secs: f32,

    // Dynamic independent baseline tracking per eye (Issue #75)
    baseline_left: Option<f32>,
    baseline_right: Option<f32>,

    // Monocular fallback recovery (Issue #75)
    is_monocular: bool,
    monocular_timeout_secs: f32,

    // Temporal state tracking per eye
    left_closed_frames: u32,
    left_is_closed: bool,
    left_closure_start: Option<Instant>,
    last_left_blink_at: Option<Instant>,

    right_closed_frames: u32,
    right_is_closed: bool,
    right_closure_start: Option<Instant>,
    last_right_blink_at: Option<Instant>,

    // Stare tracking
    last_open_instant: Instant,
    stare_warning_issued: bool,

    // Rolling BPM tracking (timestamps of completed blinks within last 60s)
    blink_timestamps: VecDeque<Instant>,
    total_blinks: u32,
}

/// Relative drop from resting open baseline required to recognize eyelid closure (35% drop)
pub const RELATIVE_BLINK_DROP: f32 = 0.35;

/// Floor below which baseline EAR will not adapt downwards (protects severely hooded or swollen eyes)
pub const MIN_BASELINE_EAR: f32 = 0.12;

/// Ceiling above which baseline EAR will not adapt upwards
pub const MAX_BASELINE_EAR: f32 = 0.65;

/// Continuous closure duration on one eye before entering Monocular Fallback Mode (60 seconds)
pub const DEFAULT_MONOCULAR_TIMEOUT_SECS: f32 = 60.0;

impl BlinkDetector {
    pub fn new(threshold: f32, stare_limit_secs: f32) -> Self {
        Self::with_alpha(threshold, stare_limit_secs, 0.40)
    }

    pub fn with_alpha(threshold: f32, stare_limit_secs: f32, alpha: f32) -> Self {
        let now = Instant::now();
        Self {
            ear_calculator: EarCalculator::new(alpha),
            threshold,
            stare_limit_secs,
            baseline_left: None,
            baseline_right: None,
            is_monocular: false,
            monocular_timeout_secs: DEFAULT_MONOCULAR_TIMEOUT_SECS,
            left_closed_frames: 0,
            left_is_closed: false,
            left_closure_start: None,
            last_left_blink_at: None,
            right_closed_frames: 0,
            right_is_closed: false,
            right_closure_start: None,
            last_right_blink_at: None,
            last_open_instant: now,
            stare_warning_issued: false,
            blink_timestamps: VecDeque::with_capacity(100),
            total_blinks: 0,
        }
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Retrieve current adaptive open-eye baselines (left, right)
    pub fn baselines(&self) -> (Option<f32>, Option<f32>) {
        (self.baseline_left, self.baseline_right)
    }

    /// Check if detector is currently operating in Monocular Fallback Mode
    pub fn is_monocular(&self) -> bool {
        self.is_monocular
    }

    /// Set timeout duration for monocular fallback (used in unit tests for fast simulation)
    pub fn set_monocular_timeout(&mut self, secs: f32) {
        self.monocular_timeout_secs = secs;
    }

    /// Retrieve total recorded blinks count
    pub fn total_blinks(&self) -> u32 {
        self.total_blinks
    }

    /// Reset internal state, pairing buffers, and adaptive baselines
    pub fn reset(&mut self, now: Instant) {
        self.baseline_left = None;
        self.baseline_right = None;
        self.is_monocular = false;
        self.left_closed_frames = 0;
        self.left_is_closed = false;
        self.left_closure_start = None;
        self.last_left_blink_at = None;
        self.right_closed_frames = 0;
        self.right_is_closed = false;
        self.right_closure_start = None;
        self.last_right_blink_at = None;
        self.last_open_instant = now;
        self.stare_warning_issued = false;
        self.blink_timestamps.clear();
        self.total_blinks = 0;
    }

    /// Process a frame with 468 landmarks and return the blink / stare detection results
    pub fn update(&mut self, landmarks: &[Landmark3D], now: Instant) -> BlinkEvent {
        // Head Pose Yaw Gate (Issue #48):
        // If the subject is severely turned away (yaw_ratio > 0.35, e.g. looking away, profile, or back of head),
        // reject blink detection and pause stare tracking immediately.
        let is_facing = crate::detector::EarCalculator::estimate_head_pose(landmarks)
            .map(|pose| pose.is_facing_camera)
            .unwrap_or(true);

        // Guard against updating EMA filter with distorted EAR values while subject is looking away
        let ear_metrics_opt = if is_facing {
            self.ear_calculator.calculate(landmarks)
        } else {
            None
        };

        let ear_metrics = match ear_metrics_opt {
            Some(metrics) => metrics,
            None => {
                // If face not detected or looking away, pause stare warning timer and clear active closure states
                self.left_is_closed = false;
                self.left_closure_start = None;
                self.left_closed_frames = 0;
                self.right_is_closed = false;
                self.right_closure_start = None;
                self.right_closed_frames = 0;
                self.last_open_instant = now;
                self.stare_warning_issued = false;
                return BlinkEvent {
                    is_blink: false,
                    is_resting: false,
                    stare_warning: false,
                    current_bpm: self.get_rolling_bpm(now),
                    total_blinks: self.total_blinks,
                    stare_duration_secs: 0.0,
                    left_ear: 0.0,
                    right_ear: 0.0,
                    avg_ear: 0.0,
                };
            }
        };

        // Dynamic Independent Dual-Eye Baseline Adaptation (Issue #75)
        // Baseline adaptation only occurs while the eye is open to prevent closed-state erosion
        let b_left = match self.baseline_left {
            Some(mut b) => {
                if !self.left_is_closed {
                    let e = ear_metrics.smoothed_left_ear;
                    let thresh = b * (1.0 - RELATIVE_BLINK_DROP);
                    if e > b {
                        b = (b + (e - b) * 0.35).min(MAX_BASELINE_EAR);
                    } else if e >= thresh {
                        b = (b - (b - e) * 0.005).max(MIN_BASELINE_EAR);
                    }
                }
                self.baseline_left = Some(b);
                b
            }
            None => {
                let b = ear_metrics.smoothed_left_ear.clamp(MIN_BASELINE_EAR, MAX_BASELINE_EAR);
                self.baseline_left = Some(b);
                b
            }
        };

        let b_right = match self.baseline_right {
            Some(mut b) => {
                if !self.right_is_closed {
                    let e = ear_metrics.smoothed_right_ear;
                    let thresh = b * (1.0 - RELATIVE_BLINK_DROP);
                    if e > b {
                        b = (b + (e - b) * 0.35).min(MAX_BASELINE_EAR);
                    } else if e >= thresh {
                        b = (b - (b - e) * 0.005).max(MIN_BASELINE_EAR);
                    }
                }
                self.baseline_right = Some(b);
                b
            }
            None => {
                let b = ear_metrics.smoothed_right_ear.clamp(MIN_BASELINE_EAR, MAX_BASELINE_EAR);
                self.baseline_right = Some(b);
                b
            }
        };

        // Proportional relative-drop threshold capped at configured/test ceiling
        let thresh_left = (b_left * (1.0 - RELATIVE_BLINK_DROP)).min(self.threshold.max(0.25));
        let thresh_right = (b_right * (1.0 - RELATIVE_BLINK_DROP)).min(self.threshold.max(0.25));

        let left_eye_closed = ear_metrics.smoothed_left_ear < thresh_left;
        let right_eye_closed = ear_metrics.smoothed_right_ear < thresh_right;

        let mut left_blink_completed = false;
        let mut right_blink_completed = false;
        let mut is_resting_event = false;

        // --- Process Left Eye ---
        if left_eye_closed {
            if !self.left_is_closed {
                self.left_is_closed = true;
                self.left_closure_start = Some(now);
                self.left_closed_frames = 1;
            } else {
                self.left_closed_frames += 1;
            }

            if let Some(start) = self.left_closure_start {
                let duration = now.checked_duration_since(start).map(|d| d.as_secs_f32()).unwrap_or(0.0);
                if duration >= 1.0 {
                    is_resting_event = true;
                }
            }
        } else if self.left_is_closed {
            if let Some(start) = self.left_closure_start {
                let duration = now.checked_duration_since(start).map(|d| d.as_secs_f32()).unwrap_or(0.0);
                // Strict guard against single-frame alpha-spike glitches:
                // Require at least 2 consecutive frames under closure AND minimum 80ms duration
                if self.left_closed_frames >= 2 && (0.08..=0.8).contains(&duration) {
                    left_blink_completed = true;
                }
            }
            self.left_is_closed = false;
            self.left_closure_start = None;
            self.left_closed_frames = 0;
            self.last_open_instant = now;
            self.stare_warning_issued = false;
        }

        // --- Process Right Eye ---
        if right_eye_closed {
            if !self.right_is_closed {
                self.right_is_closed = true;
                self.right_closure_start = Some(now);
                self.right_closed_frames = 1;
            } else {
                self.right_closed_frames += 1;
            }

            if let Some(start) = self.right_closure_start {
                let duration = now.checked_duration_since(start).map(|d| d.as_secs_f32()).unwrap_or(0.0);
                if duration >= 1.0 {
                    is_resting_event = true;
                }
            }
        } else if self.right_is_closed {
            if let Some(start) = self.right_closure_start {
                let duration = now.checked_duration_since(start).map(|d| d.as_secs_f32()).unwrap_or(0.0);
                // Strict guard against single-frame alpha-spike glitches:
                // Require at least 2 consecutive frames under closure AND minimum 80ms duration
                if self.right_closed_frames >= 2 && (0.08..=0.8).contains(&duration) {
                    right_blink_completed = true;
                }
            }
            self.right_is_closed = false;
            self.right_closure_start = None;
            self.right_closed_frames = 0;
            self.last_open_instant = now;
            self.stare_warning_issued = false;
        }

        if is_resting_event {
            self.last_open_instant = now;
            self.stare_warning_issued = false;
        }

        // Monocular Fallback Recovery:
        // If exactly one eye is continuously closed for >= monocular_timeout_secs,
        // enter Monocular Mode tracking blinks on the single functional eye.
        let left_duration = self.left_closure_start
            .and_then(|start| now.checked_duration_since(start))
            .map(|d| d.as_secs_f32())
            .unwrap_or(0.0);

        let right_duration = self.right_closure_start
            .and_then(|start| now.checked_duration_since(start))
            .map(|d| d.as_secs_f32())
            .unwrap_or(0.0);

        let left_long_closed = self.left_is_closed && left_duration >= self.monocular_timeout_secs;
        let right_long_closed = self.right_is_closed && right_duration >= self.monocular_timeout_secs;

        let was_monocular = self.is_monocular;
        self.is_monocular = left_long_closed ^ right_long_closed;

        // Flush stale pending blink timestamps on mode transitions to prevent phantom pairings
        if self.is_monocular != was_monocular {
            self.last_left_blink_at = None;
            self.last_right_blink_at = None;
        }

        // --- Asynchronous Blink Matching ---
        let mut is_blink_event = false;

        if self.is_monocular {
            // In monocular fallback mode (one eye occluded/patched),
            // a completed blink on the functional eye counts immediately
            let monocular_blink = if left_long_closed {
                right_blink_completed
            } else {
                left_blink_completed
            };

            if monocular_blink {
                self.total_blinks += 1;
                self.blink_timestamps.push_back(now);
                is_blink_event = true;
                self.last_left_blink_at = None;
                self.last_right_blink_at = None;
                self.last_open_instant = now;
                self.stare_warning_issued = false;
            }
        } else {
            // Standard Binocular Mode with 1.0s Asynchronous Pairing Guard
            if left_blink_completed {
                self.last_left_blink_at = Some(now);
            }
            if right_blink_completed {
                self.last_right_blink_at = Some(now);
            }

            if let (Some(t_left), Some(t_right)) = (self.last_left_blink_at, self.last_right_blink_at) {
                let diff = if t_left > t_right {
                    t_left.checked_duration_since(t_right).map(|d| d.as_secs_f32()).unwrap_or(0.0)
                } else {
                    t_right.checked_duration_since(t_left).map(|d| d.as_secs_f32()).unwrap_or(0.0)
                };

                // If both eyes completed blink within 1.0s window cap
                if diff <= 1.0 {
                    self.total_blinks += 1;
                    self.blink_timestamps.push_back(now);
                    is_blink_event = true;

                    // Reset match trackers once paired
                    self.last_left_blink_at = None;
                    self.last_right_blink_at = None;
                    self.last_open_instant = now;
                    self.stare_warning_issued = false;
                }
            }

            // Expire unpaired single-eye blinks older than 1.0s
            if let Some(t_left) = self.last_left_blink_at {
                if now.checked_duration_since(t_left).map(|d| d.as_secs_f32()).unwrap_or(0.0) > 1.0 {
                    self.last_left_blink_at = None;
                }
            }
            if let Some(t_right) = self.last_right_blink_at {
                if now.checked_duration_since(t_right).map(|d| d.as_secs_f32()).unwrap_or(0.0) > 1.0 {
                    self.last_right_blink_at = None;
                }
            }
        }

        // Stare duration calculation:
        // In binocular mode, stare pauses if EITHER eye is closed.
        // In monocular mode, stare pauses if the functional working eye is closed.
        let active_eye_closed = if self.is_monocular {
            if left_long_closed {
                self.right_is_closed
            } else {
                self.left_is_closed
            }
        } else {
            self.left_is_closed || self.right_is_closed
        };

        let stare_duration = if active_eye_closed {
            0.0
        } else {
            now.checked_duration_since(self.last_open_instant)
                .map(|d| d.as_secs_f32())
                .unwrap_or(0.0)
        };

        let mut trigger_stare_warning = false;
        if stare_duration >= self.stare_limit_secs && !self.stare_warning_issued && !active_eye_closed {
            trigger_stare_warning = true;
            self.stare_warning_issued = true;
        }

        self.cleanup_old_blinks(now);

        BlinkEvent {
            is_blink: is_blink_event,
            is_resting: is_resting_event,
            stare_warning: trigger_stare_warning,
            current_bpm: self.get_rolling_bpm(now),
            total_blinks: self.total_blinks,
            stare_duration_secs: (stare_duration * 10.0).round() / 10.0,
            left_ear: (ear_metrics.left_ear * 1000.0).round() / 1000.0,
            right_ear: (ear_metrics.right_ear * 1000.0).round() / 1000.0,
            avg_ear: (ear_metrics.avg_ear * 1000.0).round() / 1000.0,
        }
    }

    /// Calculate blinks per minute over rolling 60 seconds
    fn get_rolling_bpm(&self, now: Instant) -> f32 {
        let sixty_secs_ago = now.checked_sub(std::time::Duration::from_secs(60)).unwrap_or(now);
        let count = self.blink_timestamps.iter().filter(|&&t| t >= sixty_secs_ago).count();
        count as f32
    }

    fn cleanup_old_blinks(&mut self, now: Instant) {
        if let Some(cutoff) = now.checked_sub(std::time::Duration::from_secs(65)) {
            while let Some(&front) = self.blink_timestamps.front() {
                if front < cutoff {
                    self.blink_timestamps.pop_front();
                } else {
                    break;
                }
            }
        }
    }
}
