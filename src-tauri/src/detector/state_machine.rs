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

    /// Process a frame with 468 landmarks and return the blink / stare detection results
    pub fn update(&mut self, landmarks: &[Landmark3D], now: Instant) -> BlinkEvent {
        let ear_metrics = match self.ear_calculator.calculate(landmarks) {
            Some(metrics) => metrics,
            None => {
                // If face not detected, pause stare warning timer and clear active closure states
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

        let left_eye_closed = ear_metrics.smoothed_left_ear < self.threshold;
        let right_eye_closed = ear_metrics.smoothed_right_ear < self.threshold;

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
                // Require at least 2 consecutive frames under closure OR minimum 80ms duration
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
                // Require at least 2 consecutive frames under closure OR minimum 80ms duration
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

        // --- Asynchronous Blink Matching with 1.0s Window ---
        if left_blink_completed {
            self.last_left_blink_at = Some(now);
        }
        if right_blink_completed {
            self.last_right_blink_at = Some(now);
        }

        let mut is_blink_event = false;
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

        // Stare duration calculation (while eyes are open)
        let any_eye_closed = self.left_is_closed || self.right_is_closed;
        let stare_duration = if any_eye_closed {
            0.0
        } else {
            now.checked_duration_since(self.last_open_instant)
                .map(|d| d.as_secs_f32())
                .unwrap_or(0.0)
        };

        let mut trigger_stare_warning = false;
        if stare_duration >= self.stare_limit_secs && !self.stare_warning_issued && !any_eye_closed {
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

    pub fn reset(&mut self, now: Instant) {
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
}
