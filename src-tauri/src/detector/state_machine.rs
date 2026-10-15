use crate::detector::EarCalculator;
use crate::vision::Landmark3D;
use std::collections::VecDeque;
use std::time::Instant;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeBlinkStatus {
    Open,
    ClosurePending,
    CandidateCompleted,
    RejectedTooShort,
    RejectedTooLong,
    RejectedInsufficientSamples,
    RejectedDownwardGaze,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlinkDecision {
    FaceUnavailable,
    LookingAway,
    NoClosureCandidate,
    ClosurePending,
    AwaitingOtherEye,
    CountedBinocular,
    CountedMonocular,
}

/// Per-update detector evidence for opt-in troubleshooting; it does not affect decisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlinkDiagnostics {
    pub frame_interval_ms: f32,
    pub raw_left_ear: Option<f32>,
    pub raw_right_ear: Option<f32>,
    pub smoothed_left_ear: Option<f32>,
    pub smoothed_right_ear: Option<f32>,
    pub baseline_left: Option<f32>,
    pub baseline_right: Option<f32>,
    pub threshold_left: Option<f32>,
    pub threshold_right: Option<f32>,
    pub left_status: EyeBlinkStatus,
    pub right_status: EyeBlinkStatus,
    pub left_closure_ms: f32,
    pub right_closure_ms: f32,
    pub is_facing_camera: bool,
    pub is_downward_reading_gaze: bool,
    pub is_resting: bool,
    pub is_monocular: bool,
    pub decision: BlinkDecision,
}

impl Default for BlinkDiagnostics {
    fn default() -> Self {
        Self {
            frame_interval_ms: 0.0,
            raw_left_ear: None,
            raw_right_ear: None,
            smoothed_left_ear: None,
            smoothed_right_ear: None,
            baseline_left: None,
            baseline_right: None,
            threshold_left: None,
            threshold_right: None,
            left_status: EyeBlinkStatus::Open,
            right_status: EyeBlinkStatus::Open,
            left_closure_ms: 0.0,
            right_closure_ms: 0.0,
            is_facing_camera: false,
            is_downward_reading_gaze: false,
            is_resting: false,
            is_monocular: false,
            decision: BlinkDecision::FaceUnavailable,
        }
    }
}

pub struct BlinkDetector {
    ear_calculator: EarCalculator,
    threshold: f32,
    experimental_relative_blink_drop: Option<f32>,
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
    left_reopen_start: Option<Instant>,
    left_reopen_frames: u32,
    last_left_blink_at: Option<Instant>,

    right_closed_frames: u32,
    right_is_closed: bool,
    right_closure_start: Option<Instant>,
    right_reopen_start: Option<Instant>,
    right_reopen_frames: u32,
    last_right_blink_at: Option<Instant>,

    // Velocity slope sampling for low-FPS webcams (Issue #64)
    last_frame_instant: Option<Instant>,
    prev_left_ear: Option<f32>,
    prev_right_ear: Option<f32>,
    left_velocity_spike_since: Option<Instant>,
    right_velocity_spike_since: Option<Instant>,
    left_velocity_rebound_frames: u32,
    right_velocity_rebound_frames: u32,

    // Stare tracking
    last_open_instant: Instant,
    stare_warning_issued: bool,

    // Rolling BPM tracking (timestamps of completed blinks within last 60s)
    blink_timestamps: VecDeque<Instant>,
    total_blinks: u32,
    diagnostics: BlinkDiagnostics,
}

/// Relative-drop parameter used by debug-only experimental detector variants (35% drop)
pub const RELATIVE_BLINK_DROP: f32 = 0.35;

/// Keep the effective closure cutoff below the open-eye baseline while letting
/// the calibrated absolute threshold determine the per-eye cutoff.
const MAX_CLOSURE_CUTOFF_RATIO: f32 = 0.90;

/// Require a stable reopening before completing a closure candidate.
const REOPEN_CONFIRMATION_SAMPLES: u32 = 2;

/// Expire low-FPS descent candidates that do not recover within the blink window.
const VELOCITY_CANDIDATE_MAX_SECS: f32 = 0.8;

/// Floor below which baseline EAR will not adapt downwards (protects severely hooded or swollen eyes)
pub const MIN_BASELINE_EAR: f32 = 0.12;

/// Ceiling above which baseline EAR will not adapt upwards
pub const MAX_BASELINE_EAR: f32 = 0.65;

/// Continuous closure duration on one eye before entering Monocular Fallback Mode (5 seconds).
/// Short enough that a user with one weak or closed eye gets blink counting quickly.
pub const DEFAULT_MONOCULAR_TIMEOUT_SECS: f32 = 5.0;

impl BlinkDetector {
    pub fn new(threshold: f32, stare_limit_secs: f32) -> Self {
        Self::with_alpha(threshold, stare_limit_secs, 0.40)
    }

    pub fn with_alpha(threshold: f32, stare_limit_secs: f32, alpha: f32) -> Self {
        Self::with_alpha_and_relative_drop(threshold, stare_limit_secs, alpha, None)
    }

    #[cfg(debug_assertions)]
    pub(crate) fn with_experimental_parameters(
        threshold: f32,
        stare_limit_secs: f32,
        alpha: f32,
        relative_blink_drop: f32,
    ) -> Self {
        Self::with_alpha_and_relative_drop(
            threshold,
            stare_limit_secs,
            alpha,
            Some(relative_blink_drop),
        )
    }

    fn with_alpha_and_relative_drop(
        threshold: f32,
        stare_limit_secs: f32,
        alpha: f32,
        experimental_relative_blink_drop: Option<f32>,
    ) -> Self {
        let now = Instant::now();
        Self {
            ear_calculator: EarCalculator::new(alpha),
            threshold,
            experimental_relative_blink_drop,
            stare_limit_secs,
            baseline_left: None,
            baseline_right: None,
            is_monocular: false,
            monocular_timeout_secs: DEFAULT_MONOCULAR_TIMEOUT_SECS,
            left_closed_frames: 0,
            left_is_closed: false,
            left_closure_start: None,
            left_reopen_start: None,
            left_reopen_frames: 0,
            last_left_blink_at: None,
            right_closed_frames: 0,
            right_is_closed: false,
            right_closure_start: None,
            right_reopen_start: None,
            right_reopen_frames: 0,
            last_right_blink_at: None,
            last_frame_instant: None,
            prev_left_ear: None,
            prev_right_ear: None,
            left_velocity_spike_since: None,
            right_velocity_spike_since: None,
            left_velocity_rebound_frames: 0,
            right_velocity_rebound_frames: 0,
            last_open_instant: now,
            stare_warning_issued: false,
            blink_timestamps: VecDeque::with_capacity(100),
            total_blinks: 0,
            diagnostics: BlinkDiagnostics::default(),
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

    /// Snapshot the latest detector evidence for debug logging.
    pub fn diagnostics(&self) -> BlinkDiagnostics {
        self.diagnostics
    }

    fn closure_cutoff(&self, baseline: f32) -> f32 {
        let relative_cutoff = self
            .experimental_relative_blink_drop
            .map(|drop| baseline * (1.0 - drop))
            .unwrap_or(baseline * MAX_CLOSURE_CUTOFF_RATIO);
        self.threshold.min(relative_cutoff)
    }

    /// Reset internal state, pairing buffers, and adaptive baselines
    pub fn reset(&mut self, now: Instant) {
        self.baseline_left = None;
        self.baseline_right = None;
        self.is_monocular = false;
        self.left_closed_frames = 0;
        self.left_is_closed = false;
        self.left_closure_start = None;
        self.left_reopen_start = None;
        self.left_reopen_frames = 0;
        self.last_left_blink_at = None;
        self.right_closed_frames = 0;
        self.right_is_closed = false;
        self.right_closure_start = None;
        self.right_reopen_start = None;
        self.right_reopen_frames = 0;
        self.last_right_blink_at = None;
        self.last_frame_instant = None;
        self.prev_left_ear = None;
        self.prev_right_ear = None;
        self.left_velocity_spike_since = None;
        self.right_velocity_spike_since = None;
        self.left_velocity_rebound_frames = 0;
        self.right_velocity_rebound_frames = 0;
        self.last_open_instant = now;
        self.stare_warning_issued = false;
        self.blink_timestamps.clear();
        self.total_blinks = 0;
        self.diagnostics = BlinkDiagnostics::default();
    }

    /// Process a frame with 468 landmarks and return the blink / stare detection results
    pub fn update(&mut self, landmarks: &[Landmark3D], now: Instant) -> BlinkEvent {
        // Head Pose Yaw Gate (Issue #48 & #73 & #64):
        // If the subject is severely turned away (yaw_ratio > 0.35, e.g. looking away, profile, or back of head),
        // reject blink detection and pause stare tracking immediately.
        let pose_opt = crate::detector::EarCalculator::estimate_head_pose(landmarks);
        let is_facing = pose_opt
            .map(|pose| pose.is_facing_camera)
            .unwrap_or(true);
        let pitch_deg = pose_opt.map(|pose| pose.pitch_deg).unwrap_or(0.0);
        let is_downward_reading_gaze = pitch_deg < -15.0;

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
                let decision = if is_facing {
                    BlinkDecision::FaceUnavailable
                } else {
                    BlinkDecision::LookingAway
                };
                self.left_is_closed = false;
                self.left_closure_start = None;
                self.left_closed_frames = 0;
                self.left_reopen_start = None;
                self.left_reopen_frames = 0;
                self.right_is_closed = false;
                self.right_closure_start = None;
                self.right_closed_frames = 0;
                self.right_reopen_start = None;
                self.right_reopen_frames = 0;
                self.last_frame_instant = None;
                self.prev_left_ear = None;
                self.prev_right_ear = None;
                self.left_velocity_spike_since = None;
                self.right_velocity_spike_since = None;
                self.left_velocity_rebound_frames = 0;
                self.right_velocity_rebound_frames = 0;
                self.last_open_instant = now;
                self.stare_warning_issued = false;
                self.diagnostics = BlinkDiagnostics {
                    decision,
                    ..BlinkDiagnostics::default()
                };
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
                    if e > b {
                        b = (b + (e - b) * 0.35).min(MAX_BASELINE_EAR);
                    } else if e >= b * MAX_CLOSURE_CUTOFF_RATIO {
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
                    if e > b {
                        b = (b + (e - b) * 0.35).min(MAX_BASELINE_EAR);
                    } else if e >= b * MAX_CLOSURE_CUTOFF_RATIO {
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

        // Calibrated absolute cutoff, capped at 90% of the open-eye baseline.
        let thresh_left = self.closure_cutoff(b_left);
        let thresh_right = self.closure_cutoff(b_right);
        let reopen_thresh_left = (b_left + thresh_left) * 0.5;
        let reopen_thresh_right = (b_right + thresh_right) * 0.5;
        let left_eye_closed = ear_metrics.smoothed_left_ear < thresh_left;
        let right_eye_closed = ear_metrics.smoothed_right_ear < thresh_right;

        let dt = match self.last_frame_instant {
            Some(prev) => now.checked_duration_since(prev).map(|d| d.as_secs_f32()).unwrap_or(0.0),
            None => 0.0,
        };
        self.last_frame_instant = Some(now);
        // Low-FPS regime: frame intervals between 80ms and 160ms (approx 6 to 12.5 FPS)
        let is_low_fps = dt >= 0.08 && dt <= 0.16;

        // Velocity slope sampling for half-sampled micro-blinks on low-FPS cameras (Issue #64)
        let mut left_velocity_blink = false;
        let mut right_velocity_blink = false;

        let cur_left_ear = ear_metrics.smoothed_left_ear;
        let cur_right_ear = ear_metrics.smoothed_right_ear;

        if is_low_fps && !is_downward_reading_gaze && dt > 0.001 {
            // Check left eye velocity slope
            if left_eye_closed || self.left_is_closed {
                // Static closure owns duration validation once its state has started.
                self.left_velocity_spike_since = None;
                self.left_velocity_rebound_frames = 0;
            } else if let Some(prev_l) = self.prev_left_ear {
                let delta_l = cur_left_ear - prev_l;
                let vel_l = delta_l / dt;
                let drop_ratio = (b_left - cur_left_ear) / b_left;

                // Start a candidate on a rapid descent; retain it through a sampled plateau.
                if drop_ratio >= 0.20 && vel_l <= -0.40 {
                    self.left_velocity_spike_since.get_or_insert(now);
                    self.left_velocity_rebound_frames = 0;
                } else if let Some(started) = self.left_velocity_spike_since {
                    let age = now
                        .checked_duration_since(started)
                        .map(|d| d.as_secs_f32())
                        .unwrap_or(0.0);
                    if age > VELOCITY_CANDIDATE_MAX_SECS {
                        self.left_velocity_spike_since = None;
                        self.left_velocity_rebound_frames = 0;
                    } else if ear_metrics.left_ear >= reopen_thresh_left {
                        self.left_velocity_rebound_frames += 1;
                    } else {
                        self.left_velocity_rebound_frames = 0;
                    }
                    if self.left_velocity_rebound_frames >= REOPEN_CONFIRMATION_SAMPLES
                        && cur_left_ear >= reopen_thresh_left
                    {
                        left_velocity_blink = true;
                        self.left_velocity_spike_since = None;
                        self.left_velocity_rebound_frames = 0;
                    }
                }
            }

            // Check right eye velocity slope
            if right_eye_closed || self.right_is_closed {
                self.right_velocity_spike_since = None;
                self.right_velocity_rebound_frames = 0;
            } else if let Some(prev_r) = self.prev_right_ear {
                let delta_r = cur_right_ear - prev_r;
                let vel_r = delta_r / dt;
                let drop_ratio = (b_right - cur_right_ear) / b_right;

                if drop_ratio >= 0.20 && vel_r <= -0.40 {
                    self.right_velocity_spike_since.get_or_insert(now);
                    self.right_velocity_rebound_frames = 0;
                } else if let Some(started) = self.right_velocity_spike_since {
                    let age = now
                        .checked_duration_since(started)
                        .map(|d| d.as_secs_f32())
                        .unwrap_or(0.0);
                    if age > VELOCITY_CANDIDATE_MAX_SECS {
                        self.right_velocity_spike_since = None;
                        self.right_velocity_rebound_frames = 0;
                    } else if ear_metrics.right_ear >= reopen_thresh_right {
                        self.right_velocity_rebound_frames += 1;
                    } else {
                        self.right_velocity_rebound_frames = 0;
                    }
                    if self.right_velocity_rebound_frames >= REOPEN_CONFIRMATION_SAMPLES
                        && cur_right_ear >= reopen_thresh_right
                    {
                        right_velocity_blink = true;
                        self.right_velocity_spike_since = None;
                        self.right_velocity_rebound_frames = 0;
                    }
                }
            }
        } else {
            self.left_velocity_spike_since = None;
            self.right_velocity_spike_since = None;
            self.left_velocity_rebound_frames = 0;
            self.right_velocity_rebound_frames = 0;
        }

        self.prev_left_ear = Some(cur_left_ear);
        self.prev_right_ear = Some(cur_right_ear);

        let mut left_blink_completed = false;
        let mut right_blink_completed = false;
        let mut is_resting_event = false;
        let mut left_status =
            if left_eye_closed || self.left_is_closed || self.left_velocity_spike_since.is_some() {
                EyeBlinkStatus::ClosurePending
            } else {
                EyeBlinkStatus::Open
            };
        let mut right_status = if right_eye_closed
            || self.right_is_closed
            || self.right_velocity_spike_since.is_some()
        {
            EyeBlinkStatus::ClosurePending
        } else {
            EyeBlinkStatus::Open
        };
        let mut left_closure_secs = 0.0f32;
        let mut right_closure_secs = 0.0f32;

        // One closed sample is enough: at 10-15 FPS a real blink often spans only one frame.
        let min_closed_frames = 1;

        // --- Process Left Eye ---
        if left_eye_closed {
            if !self.left_is_closed {
                self.left_is_closed = true;
                self.left_closure_start = Some(now);
                self.left_closed_frames = 1;
            } else {
                self.left_closed_frames += 1;
            }
            if ear_metrics.left_ear >= reopen_thresh_left {
                self.left_reopen_start.get_or_insert(now);
            } else {
                self.left_reopen_start = None;
                self.left_reopen_frames = 0;
            }

            if let Some(start) = self.left_closure_start {
                let closure_end = self.left_reopen_start.unwrap_or(now);
                let duration = closure_end
                    .checked_duration_since(start)
                    .map(|d| d.as_secs_f32())
                    .unwrap_or(0.0);
                left_closure_secs = duration;
                if duration >= 1.0 {
                    is_resting_event = true;
                }
            }
        } else if self.left_is_closed {
            if let Some(start) = self.left_closure_start {
                if ear_metrics.left_ear >= reopen_thresh_left {
                    self.left_reopen_start.get_or_insert(now);
                } else {
                    self.left_reopen_start = None;
                    self.left_reopen_frames = 0;
                }

                if ear_metrics.left_ear >= reopen_thresh_left
                    && ear_metrics.smoothed_left_ear >= reopen_thresh_left
                {
                    self.left_reopen_frames += 1;
                } else {
                    self.left_reopen_frames = 0;
                }

                let closure_end = self.left_reopen_start.unwrap_or(now);
                let duration = closure_end
                    .checked_duration_since(start)
                    .map(|d| d.as_secs_f32())
                    .unwrap_or(0.0);
                left_closure_secs = duration;

                if self.left_reopen_frames >= REOPEN_CONFIRMATION_SAMPLES {
                    if self.left_closed_frames >= min_closed_frames
                        && (0.08..=0.8).contains(&duration)
                        && !is_downward_reading_gaze
                    {
                        left_blink_completed = true;
                        left_status = EyeBlinkStatus::CandidateCompleted;
                    } else if is_downward_reading_gaze {
                        left_status = EyeBlinkStatus::RejectedDownwardGaze;
                    } else if self.left_closed_frames < min_closed_frames {
                        left_status = EyeBlinkStatus::RejectedInsufficientSamples;
                    } else if duration < 0.08 {
                        left_status = EyeBlinkStatus::RejectedTooShort;
                    } else {
                        left_status = EyeBlinkStatus::RejectedTooLong;
                    }

                    self.left_is_closed = false;
                    self.left_closure_start = None;
                    self.left_closed_frames = 0;
                    self.left_reopen_start = None;
                    self.left_reopen_frames = 0;
                    self.last_open_instant = now;
                    self.stare_warning_issued = false;
                } else {
                    left_status = EyeBlinkStatus::ClosurePending;
                    if duration >= 1.0 {
                        is_resting_event = true;
                    }
                }
            }
        }

        if left_velocity_blink && !is_downward_reading_gaze {
            left_blink_completed = true;
            left_status = EyeBlinkStatus::CandidateCompleted;
            // Suppress standard closure state to prevent double-counting on reopen
            self.left_is_closed = false;
            self.left_closure_start = None;
            self.left_closed_frames = 0;
            self.left_reopen_start = None;
            self.left_reopen_frames = 0;
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
            if ear_metrics.right_ear >= reopen_thresh_right {
                self.right_reopen_start.get_or_insert(now);
            } else {
                self.right_reopen_start = None;
                self.right_reopen_frames = 0;
            }

            if let Some(start) = self.right_closure_start {
                let closure_end = self.right_reopen_start.unwrap_or(now);
                let duration = closure_end
                    .checked_duration_since(start)
                    .map(|d| d.as_secs_f32())
                    .unwrap_or(0.0);
                right_closure_secs = duration;
                if duration >= 1.0 {
                    is_resting_event = true;
                }
            }
        } else if self.right_is_closed {
            if let Some(start) = self.right_closure_start {
                if ear_metrics.right_ear >= reopen_thresh_right {
                    self.right_reopen_start.get_or_insert(now);
                } else {
                    self.right_reopen_start = None;
                    self.right_reopen_frames = 0;
                }

                if ear_metrics.right_ear >= reopen_thresh_right
                    && ear_metrics.smoothed_right_ear >= reopen_thresh_right
                {
                    self.right_reopen_frames += 1;
                } else {
                    self.right_reopen_frames = 0;
                }

                let closure_end = self.right_reopen_start.unwrap_or(now);
                let duration = closure_end
                    .checked_duration_since(start)
                    .map(|d| d.as_secs_f32())
                    .unwrap_or(0.0);
                right_closure_secs = duration;

                if self.right_reopen_frames >= REOPEN_CONFIRMATION_SAMPLES {
                    if self.right_closed_frames >= min_closed_frames
                        && (0.08..=0.8).contains(&duration)
                        && !is_downward_reading_gaze
                    {
                        right_blink_completed = true;
                        right_status = EyeBlinkStatus::CandidateCompleted;
                    } else if is_downward_reading_gaze {
                        right_status = EyeBlinkStatus::RejectedDownwardGaze;
                    } else if self.right_closed_frames < min_closed_frames {
                        right_status = EyeBlinkStatus::RejectedInsufficientSamples;
                    } else if duration < 0.08 {
                        right_status = EyeBlinkStatus::RejectedTooShort;
                    } else {
                        right_status = EyeBlinkStatus::RejectedTooLong;
                    }

                    self.right_is_closed = false;
                    self.right_closure_start = None;
                    self.right_closed_frames = 0;
                    self.right_reopen_start = None;
                    self.right_reopen_frames = 0;
                    self.last_open_instant = now;
                    self.stare_warning_issued = false;
                } else {
                    right_status = EyeBlinkStatus::ClosurePending;
                    if duration >= 1.0 {
                        is_resting_event = true;
                    }
                }
            }
        }

        if right_velocity_blink && !is_downward_reading_gaze {
            right_blink_completed = true;
            right_status = EyeBlinkStatus::CandidateCompleted;
            self.right_is_closed = false;
            self.right_closure_start = None;
            self.right_closed_frames = 0;
            self.right_reopen_start = None;
            self.right_reopen_frames = 0;
        }

        if is_downward_reading_gaze {
            left_blink_completed = false;
            right_blink_completed = false;
            if self.left_is_closed || left_status == EyeBlinkStatus::CandidateCompleted {
                left_status = EyeBlinkStatus::RejectedDownwardGaze;
            }
            if self.right_is_closed || right_status == EyeBlinkStatus::CandidateCompleted {
                right_status = EyeBlinkStatus::RejectedDownwardGaze;
            }
            self.left_closure_start = None;
            self.right_closure_start = None;
            self.left_is_closed = false;
            self.right_is_closed = false;
            self.left_closed_frames = 0;
            self.right_closed_frames = 0;
            self.left_reopen_start = None;
            self.right_reopen_start = None;
            self.left_reopen_frames = 0;
            self.right_reopen_frames = 0;
            self.left_velocity_spike_since = None;
            self.right_velocity_spike_since = None;
            self.left_velocity_rebound_frames = 0;
            self.right_velocity_rebound_frames = 0;
            self.last_open_instant = now;
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

        let decision = if is_blink_event {
            if self.is_monocular {
                BlinkDecision::CountedMonocular
            } else {
                BlinkDecision::CountedBinocular
            }
        } else if left_blink_completed || right_blink_completed {
            BlinkDecision::AwaitingOtherEye
        } else if self.last_left_blink_at.is_some() || self.last_right_blink_at.is_some() {
            BlinkDecision::AwaitingOtherEye
        } else if self.left_is_closed
            || self.right_is_closed
            || self.left_velocity_spike_since.is_some()
            || self.right_velocity_spike_since.is_some()
        {
            BlinkDecision::ClosurePending
        } else {
            BlinkDecision::NoClosureCandidate
        };

        self.diagnostics = BlinkDiagnostics {
            frame_interval_ms: dt * 1000.0,
            raw_left_ear: Some(ear_metrics.left_ear),
            raw_right_ear: Some(ear_metrics.right_ear),
            smoothed_left_ear: Some(ear_metrics.smoothed_left_ear),
            smoothed_right_ear: Some(ear_metrics.smoothed_right_ear),
            baseline_left: Some(b_left),
            baseline_right: Some(b_right),
            threshold_left: Some(thresh_left),
            threshold_right: Some(thresh_right),
            left_status,
            right_status,
            left_closure_ms: left_closure_secs * 1000.0,
            right_closure_ms: right_closure_secs * 1000.0,
            is_facing_camera: is_facing,
            is_downward_reading_gaze,
            is_resting: is_resting_event,
            is_monocular: self.is_monocular,
            decision,
        };

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
