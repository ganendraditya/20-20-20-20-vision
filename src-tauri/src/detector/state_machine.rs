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

    // Temporal state tracking
    closed_frames_count: u32,
    is_currently_closed: bool,
    closure_start_instant: Option<Instant>,

    // Stare tracking
    last_open_instant: Instant,
    stare_warning_issued: bool,

    // Rolling BPM tracking (timestamps of completed blinks within last 60s)
    blink_timestamps: VecDeque<Instant>,
    total_blinks: u32,
}

impl BlinkDetector {
    pub fn new(threshold: f32, stare_limit_secs: f32) -> Self {
        let now = Instant::now();
        Self {
            ear_calculator: EarCalculator::new(0.3),
            threshold,
            stare_limit_secs,
            closed_frames_count: 0,
            is_currently_closed: false,
            closure_start_instant: None,
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
                // If face not detected, pause stare warning timer and clear active closure state
                self.is_currently_closed = false;
                self.closure_start_instant = None;
                self.closed_frames_count = 0;
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

        let is_eye_closed = ear_metrics.avg_ear < self.threshold;
        let mut is_blink_event = false;
        let mut is_resting_event = false;

        if is_eye_closed {
            if !self.is_currently_closed {
                // Eye just closed this frame
                self.is_currently_closed = true;
                self.closure_start_instant = Some(now);
                self.closed_frames_count = 1;
            } else {
                self.closed_frames_count += 1;
            }

            // Check if eye is resting (> 1.0s continuous closure)
            if let Some(start_time) = self.closure_start_instant {
                let duration = now.checked_duration_since(start_time)
                    .map(|d| d.as_secs_f32())
                    .unwrap_or(0.0);
                if duration >= 1.0 {
                    is_resting_event = true;
                    // While resting, pause the stare timer so dry eye alert doesn't trigger
                    self.last_open_instant = now;
                    self.stare_warning_issued = false;
                }
            }
        } else {
            // Eye is open
            if self.is_currently_closed {
                // Eye just reopened! Calculate duration of closure
                if let Some(start_time) = self.closure_start_instant {
                    let closure_duration = now.checked_duration_since(start_time)
                        .map(|d| d.as_secs_f32())
                        .unwrap_or(0.0);

                    // Valid biological blink: between 0.08s and 0.8s
                    if (0.08..=0.8).contains(&closure_duration) || (self.closed_frames_count >= 2 && closure_duration < 1.0) {
                        self.total_blinks += 1;
                        self.blink_timestamps.push_back(now);
                        is_blink_event = true;
                    }
                }

                self.is_currently_closed = false;
                self.closure_start_instant = None;
                self.closed_frames_count = 0;
                
                // Reset stare timer upon completing eye closure / blink
                self.last_open_instant = now;
                self.stare_warning_issued = false;
            }
        }

        // Stare duration calculation (while eyes are open)
        let stare_duration = if self.is_currently_closed {
            0.0
        } else {
            now.checked_duration_since(self.last_open_instant)
                .map(|d| d.as_secs_f32())
                .unwrap_or(0.0)
        };

        let mut trigger_stare_warning = false;
        if stare_duration >= self.stare_limit_secs && !self.stare_warning_issued && !self.is_currently_closed {
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
        self.closed_frames_count = 0;
        self.is_currently_closed = false;
        self.closure_start_instant = None;
        self.last_open_instant = now;
        self.stare_warning_issued = false;
        self.blink_timestamps.clear();
        self.total_blinks = 0;
    }
}
