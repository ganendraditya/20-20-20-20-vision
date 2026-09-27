use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppConfig {
    pub ear_threshold: f32,
    pub stare_limit_secs: f32,
    pub break_target_minutes: f32,
    pub break_duration_seconds: u32,
    pub selected_camera_index: usize,
    pub sound_enabled: bool,
    pub stare_alert_enabled: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            ear_threshold: 0.22,
            stare_limit_secs: 8.0,
            break_target_minutes: 20.0,
            break_duration_seconds: 20,
            selected_camera_index: 0,
            sound_enabled: true,
            stare_alert_enabled: true,
        }
    }
}

impl AppConfig {
    /// Return standard OS configuration directory path
    /// macOS/Linux: ~/.config/420vision/config.json
    /// Windows: %APPDATA%\420vision\config.json
    pub fn get_config_dir() -> PathBuf {
        if let Some(config_base) = dirs::config_dir() {
            config_base.join("420vision")
        } else {
            PathBuf::from(".420vision")
        }
    }

    pub fn get_config_path() -> PathBuf {
        Self::get_config_dir().join("config.json")
    }

    /// Load config from file or return default
    pub fn load() -> Self {
        Self::load_from_path(Self::get_config_path())
    }

    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        if path.exists() {
            if let Ok(mut file) = File::open(path) {
                let mut contents = String::new();
                if file.read_to_string(&mut contents).is_ok() {
                    if let Ok(config) = serde_json::from_str::<AppConfig>(&contents) {
                        return config;
                    }
                }
            }
        }
        Self::default()
    }

    /// Save config to file using atomic write
    pub fn save(&self) -> Result<(), String> {
        Self::save_to_path(self, Self::get_config_path())
    }

    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            let _ = create_dir_all(parent);
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        let temp_path = path.with_extension("tmp");
        {
            let mut file = File::create(&temp_path)
                .map_err(|e| format!("Failed to create temp config file: {}", e))?;
            file.write_all(json.as_bytes())
                .map_err(|e| format!("Failed to write config file: {}", e))?;
            file.flush()
                .map_err(|e| format!("Failed to flush config file: {}", e))?;
        }

        std::fs::rename(&temp_path, path)
            .map_err(|e| format!("Failed to commit config file atomically: {}", e))?;

        Ok(())
    }
}
