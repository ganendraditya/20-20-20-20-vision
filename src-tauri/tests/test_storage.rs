use tempfile::tempdir;
use vision420_lib::storage::{AnalyticsDb, AppConfig};

#[test]
fn test_config_save_and_load_lifecycle() {
    let dir = tempdir().expect("Failed to create tempdir");
    let config_path = dir.path().join("config.json");

    let mut config = AppConfig::default();
    config.ear_threshold = 0.285;
    config.selected_camera_index = 2;
    config.sound_enabled = false;

    // Save
    config.save_to_path(&config_path).expect("Save should succeed");
    assert!(config_path.exists());

    // Load
    let loaded = AppConfig::load_from_path(&config_path);
    assert_eq!(loaded, config);
    assert_eq!(loaded.ear_threshold, 0.285);
    assert_eq!(loaded.selected_camera_index, 2);
    assert!(!loaded.sound_enabled);
}

#[test]
fn test_analytics_db_record_and_query_history() {
    let db = AnalyticsDb::open_in_memory().expect("Should initialize in-memory DB");

    // Record session 1 on Day 1
    db.record_break_event("2026-09-25", 15.0, 300, true, false, 1200.0)
        .expect("Record event 1");

    // Record session 2 on Day 1 (should update day's aggregate)
    db.record_break_event("2026-09-25", 17.0, 320, false, true, 1200.0)
        .expect("Record event 2");

    // Record Day 2
    db.record_break_event("2026-09-26", 18.0, 350, true, false, 1200.0)
        .expect("Record Day 2");

    let history = db.get_compliance_history(7).expect("Query history");
    assert_eq!(history.len(), 2);

    // Most recent first: 2026-09-26
    assert_eq!(history[0].date, "2026-09-26");
    assert_eq!(history[0].breaks_completed, 1);
    assert_eq!(history[0].breaks_skipped, 0);

    // Day 1: 2026-09-25 (1 completed, 1 skipped, 2400s = 40m)
    assert_eq!(history[1].date, "2026-09-25");
    assert_eq!(history[1].breaks_completed, 1);
    assert_eq!(history[1].breaks_skipped, 1);
    assert_eq!(history[1].screen_minutes, 40);
}
