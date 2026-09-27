use std::time::{Duration, Instant};
use vision420_lib::timer::PresenceTimer;

#[test]
fn test_presence_accumulation_and_break_trigger() {
    let mut timer = PresenceTimer::new(5.0, 300.0); // 5s break target for fast test
    let mut now = Instant::now();

    // Initial state
    let state0 = timer.update(true, now);
    assert_eq!(state0.active_screen_seconds, 0.0);
    assert_eq!(state0.next_break_seconds, 5);
    assert!(!state0.break_triggered);

    // 2 seconds active screen time
    now += Duration::from_secs(2);
    let state1 = timer.update(true, now);
    assert_eq!(state1.active_screen_seconds, 2.0);
    assert_eq!(state1.next_break_seconds, 3);
    assert!(!state1.break_triggered);

    // 3 more seconds -> 5s total reached! Break triggered!
    now += Duration::from_secs(3);
    let state2 = timer.update(true, now);
    assert!(state2.break_triggered);
    assert_eq!(state2.next_break_seconds, 0);

    // Following frames shouldn't spam break trigger
    now += Duration::from_secs(1);
    let state3 = timer.update(true, now);
    assert!(!state3.break_triggered);
}

#[test]
fn test_away_pause_and_quick_return() {
    let mut timer = PresenceTimer::new(1200.0, 300.0); // 20m target, 5m away reset
    let mut now = Instant::now();

    // 100 seconds active
    timer.update(true, now);
    now += Duration::from_secs(100);
    let state_active = timer.update(true, now);
    assert_eq!(state_active.active_screen_seconds, 100.0);

    // User steps away for 60 seconds (< 5 minutes)
    now += Duration::from_secs(1);
    let state_away = timer.update(false, now);
    assert!(!state_away.is_present);

    now += Duration::from_secs(60);
    timer.update(false, now);

    // User returns! Active seconds should resume from 100s, not reset to 0
    now += Duration::from_secs(1);
    let state_returned = timer.update(true, now);
    assert!(state_returned.active_screen_seconds >= 100.0);
    assert!(state_returned.active_screen_seconds < 105.0);
}

#[test]
fn test_away_auto_reset_after_5_minutes() {
    let mut timer = PresenceTimer::new(1200.0, 300.0);
    let mut now = Instant::now();

    // Accumulate 600s (10 minutes)
    timer.update(true, now);
    now += Duration::from_secs(600);
    let state_active = timer.update(true, now);
    assert_eq!(state_active.active_screen_seconds, 600.0);

    // User walks away for 350 seconds (> 5 minutes = 300s)
    now += Duration::from_secs(1);
    timer.update(false, now);

    now += Duration::from_secs(350);
    timer.update(false, now);

    // User comes back! Should auto-reset active screen time to 0 (eyes rested)
    now += Duration::from_secs(1);
    let state_reset = timer.update(true, now);
    assert_eq!(state_reset.active_screen_seconds, 0.0);
    assert_eq!(state_reset.next_break_seconds, 1200);
}

#[test]
fn test_sleep_and_wake_lifecycle() {
    let mut timer = PresenceTimer::new(1200.0, 300.0);
    let mut now = Instant::now();

    // 200s active
    timer.update(true, now);
    now += Duration::from_secs(200);
    timer.update(true, now);

    // Laptop lid closed / OS lock
    timer.handle_sleep_event(now);

    // Wake after 10 minutes (600s)
    now += Duration::from_secs(600);
    timer.handle_wake_event(now);

    let state_after_wake = timer.update(true, now);
    assert_eq!(state_after_wake.active_screen_seconds, 0.0, "Should reset cycle after long sleep");

    // Also test short sleep (< away_reset_seconds): active screen time should be preserved
    let mut timer_short = PresenceTimer::new(1200.0, 300.0);
    let mut now_short = Instant::now();
    timer_short.update(true, now_short);
    now_short += Duration::from_secs(200);
    timer_short.update(true, now_short);

    timer_short.handle_sleep_event(now_short);
    now_short += Duration::from_secs(60); // 60s < 300s
    timer_short.handle_wake_event(now_short);

    let state_short = timer_short.update(true, now_short);
    assert!(state_short.active_screen_seconds >= 200.0 && state_short.active_screen_seconds < 205.0);
}
