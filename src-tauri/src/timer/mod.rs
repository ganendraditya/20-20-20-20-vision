use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresenceState {
    pub is_present: bool,
    pub active_screen_seconds: f32,
    pub next_break_seconds: u32,
    pub progress_ratio: f32,
    pub break_triggered: bool,
}

pub struct PresenceTimer {
    break_target_seconds: f32, // Default: 20 minutes = 1200.0s
    away_reset_seconds: f32,   // Default: 5 minutes = 300.0s

    active_screen_seconds: f32,
    last_update_instant: Option<Instant>,
    away_start_instant: Option<Instant>,
    break_triggered: bool,
}

impl PresenceTimer {
    pub fn new(break_target_seconds: f32, away_reset_seconds: f32) -> Self {
        Self {
            break_target_seconds,
            away_reset_seconds,
            active_screen_seconds: 0.0,
            last_update_instant: None,
            away_start_instant: None,
            break_triggered: false,
        }
    }

    /// Process a tick or frame event.
    /// `is_face_present` indicates whether face landmarks are detected.
    /// `now` is the current monotonic timestamp.
    pub fn update(&mut self, is_face_present: bool, now: Instant) -> PresenceState {
        let dt = match self.last_update_instant {
            Some(prev) => now
                .checked_duration_since(prev)
                .map(|d| d.as_secs_f32())
                .unwrap_or(0.0),
            None => 0.0,
        };
        self.last_update_instant = Some(now);

        if is_face_present {
            // User is present
            let was_away = if let Some(away_start) = self.away_start_instant.take() {
                let away_duration = now
                    .checked_duration_since(away_start)
                    .map(|d| d.as_secs_f32())
                    .unwrap_or(0.0);

                if away_duration >= self.away_reset_seconds {
                    // User was absent for >= 5 minutes: eyes rested naturally -> auto reset 20-min cycle
                    self.active_screen_seconds = 0.0;
                    self.break_triggered = false;
                }
                true
            } else {
                false
            };

            // Only add dt if user was already present on the previous update tick
            // (prevents counting throttled away duration as screen time)
            if !was_away {
                self.active_screen_seconds += dt;
            }
        } else {
            // User is away / no face detected
            if self.away_start_instant.is_none() {
                self.away_start_instant = Some(now);
            }
        }

        let remaining = (self.break_target_seconds - self.active_screen_seconds).max(0.0);
        let progress = (self.active_screen_seconds / self.break_target_seconds).clamp(0.0, 1.0);

        let mut trigger_break = false;
        if self.active_screen_seconds >= self.break_target_seconds && !self.break_triggered {
            trigger_break = true;
            self.break_triggered = true;
        }

        PresenceState {
            is_present: is_face_present,
            active_screen_seconds: (self.active_screen_seconds * 10.0).round() / 10.0,
            next_break_seconds: remaining.ceil() as u32,
            progress_ratio: progress,
            break_triggered: trigger_break,
        }
    }

    /// Reset after break is completed
    pub fn reset_break_timer(&mut self) {
        self.active_screen_seconds = 0.0;
        self.break_triggered = false;
        self.away_start_instant = None;
    }

    /// Snooze break by rolling back active seconds by `minutes`
    pub fn snooze(&mut self, minutes: f32) {
        let rollback_secs = minutes * 60.0;
        self.active_screen_seconds = (self.break_target_seconds - rollback_secs).max(0.0);
        self.break_triggered = false;
    }

    /// Handle OS lid-close, lock, or sleep event: freeze active accumulation
    pub fn handle_sleep_event(&mut self, now: Instant) {
        if self.away_start_instant.is_none() {
            self.away_start_instant = Some(now);
        }
        self.last_update_instant = Some(now);
    }

    /// Handle OS wake / unlock event
    pub fn handle_wake_event(&mut self, now: Instant) {
        if let Some(away_start) = self.away_start_instant.take() {
            let away_duration = now
                .checked_duration_since(away_start)
                .map(|d| d.as_secs_f32())
                .unwrap_or(0.0);

            if away_duration >= self.away_reset_seconds {
                // Slept for >= 5 minutes -> reset 20-minute cycle to 0
                self.active_screen_seconds = 0.0;
                self.break_triggered = false;
            }
        }
        self.last_update_instant = Some(now);
    }
}
