use rusqlite::{params, Connection, Result};
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use crate::storage::AppConfig;
use crate::DailyCompliance;

pub struct AnalyticsDb {
    conn: Connection,
}

impl AnalyticsDb {
    pub fn get_db_path() -> PathBuf {
        AppConfig::get_config_dir().join("analytics.db")
    }

    /// Open or create database in standard OS directory
    pub fn open() -> Result<Self, String> {
        let db_path = Self::get_db_path();
        if let Some(parent) = db_path.parent() {
            let _ = create_dir_all(parent);
        }
        Self::open_path(db_path).map_err(|e| e.to_string())
    }

    pub fn open_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS daily_analytics (
                date TEXT PRIMARY KEY,
                avg_bpm REAL NOT NULL,
                total_blinks INTEGER NOT NULL,
                breaks_completed INTEGER NOT NULL,
                breaks_skipped INTEGER NOT NULL,
                active_screen_seconds REAL NOT NULL
            );",
        )?;
        Ok(())
    }

    /// Record or update analytics for a given day (date string: YYYY-MM-DD)
    pub fn record_break_event(
        &self,
        date: &str,
        bpm: f32,
        blinks: u32,
        completed: bool,
        skipped: bool,
        screen_secs: f32,
    ) -> Result<()> {
        let comp_increment = if completed { 1 } else { 0 };
        let skip_increment = if skipped { 1 } else { 0 };

        self.conn.execute(
            "INSERT INTO daily_analytics (date, avg_bpm, total_blinks, breaks_completed, breaks_skipped, active_screen_seconds)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(date) DO UPDATE SET
                avg_bpm = CASE 
                    WHEN (active_screen_seconds + ?6) > 0.0 THEN ((total_blinks + ?3) * 60.0) / (active_screen_seconds + ?6)
                    ELSE ?2
                END,
                total_blinks = total_blinks + ?3,
                breaks_completed = breaks_completed + ?4,
                breaks_skipped = breaks_skipped + ?5,
                active_screen_seconds = active_screen_seconds + ?6",
            params![date, bpm, blinks, comp_increment, skip_increment, screen_secs],
        )?;

        Ok(())
    }

    /// Record break event using local date automatically from SQLite
    pub fn record_break_today(
        &self,
        bpm: f32,
        blinks: u32,
        completed: bool,
        skipped: bool,
        screen_secs: f32,
    ) -> Result<()> {
        let comp_increment = if completed { 1 } else { 0 };
        let skip_increment = if skipped { 1 } else { 0 };

        self.conn.execute(
            "INSERT INTO daily_analytics (date, avg_bpm, total_blinks, breaks_completed, breaks_skipped, active_screen_seconds)
             VALUES (date('now', 'localtime'), ?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(date) DO UPDATE SET
                avg_bpm = CASE 
                    WHEN (active_screen_seconds + ?5) > 0.0 THEN ((total_blinks + ?2) * 60.0) / (active_screen_seconds + ?5)
                    ELSE ?1
                END,
                total_blinks = total_blinks + ?2,
                breaks_completed = breaks_completed + ?3,
                breaks_skipped = breaks_skipped + ?4,
                active_screen_seconds = active_screen_seconds + ?5",
            params![bpm, blinks, comp_increment, skip_increment, screen_secs],
        )?;

        Ok(())
    }

    /// Retrieve past N days of compliance stats
    pub fn get_compliance_history(&self, limit_days: u32) -> Result<Vec<DailyCompliance>> {
        let mut stmt = self.conn.prepare(
            "SELECT date, avg_bpm, breaks_completed, breaks_skipped, active_screen_seconds
             FROM daily_analytics
             ORDER BY date DESC
             LIMIT ?1",
        )?;

        let rows = stmt.query_map(params![limit_days], |row| {
            let screen_seconds: f32 = row.get(4)?;
            let screen_minutes = (screen_seconds / 60.0).round() as u32;

            Ok(DailyCompliance {
                date: row.get(0)?,
                avg_bpm: row.get(1)?,
                breaks_completed: row.get(2)?,
                breaks_skipped: row.get(3)?,
                screen_minutes,
            })
        })?;

        let mut history = Vec::new();
        for row in rows {
            history.push(row?);
        }

        Ok(history)
    }
}
