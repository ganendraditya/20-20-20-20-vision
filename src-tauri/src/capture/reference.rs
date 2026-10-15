//! Opt-in macOS development capture with cancellable waits and synchronized recording.

use nokhwa::utils::{FrameFormat, Resolution};
use nokhwa::{Buffer, NokhwaError};
use serde::Serialize;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::Duration;

#[repr(C)]
#[derive(Default, Serialize)]
struct RecordingStats {
    frames: u64,
    dropped: u64,
    first_wall_ms: u64,
    last_wall_ms: u64,
}

unsafe extern "C" {
    fn v420_reference_capture_start(
        device_id: *const c_char,
        video: *const c_char,
        timestamps: *const c_char,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut c_void;
    fn v420_reference_capture_frame(
        handle: *mut c_void,
        rgb: *mut u8,
        capacity: usize,
        width: *mut u32,
        height: *mut u32,
        wall_ms: *mut u64,
        timeout_ms: u32,
    ) -> i32;
    fn v420_reference_writer_stop(
        handle: *mut c_void,
        error: *mut c_char,
        capacity: usize,
        stats: *mut RecordingStats,
    ) -> i32;
    #[cfg(test)]
    fn v420_reference_writer_start(
        video: *const c_char,
        timestamps: *const c_char,
        error: *mut c_char,
        error_capacity: usize,
    ) -> *mut c_void;
    #[cfg(test)]
    fn v420_reference_writer_append(
        handle: *mut c_void,
        rgb: *const u8,
        width: u32,
        height: u32,
        wall_ms: u64,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    #[cfg(test)]
    fn v420_reference_test_decoded_frames(video: *const c_char) -> i64;
}

pub(super) fn enabled() -> bool {
    std::env::var("VISION420_RECORD_REFERENCE").as_deref() == Ok("1")
        || std::env::args().any(|argument| argument == "--record-reference")
}

pub(super) fn ear_ab_test_enabled() -> bool {
    enabled() && std::env::args().any(|argument| argument == "--ear-ab-test")
}

fn c_path(path: &Path) -> Result<CString, String> {
    CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())
}

fn error_text(error: &[c_char; 1024]) -> String {
    unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub(super) struct ReferenceRecorder {
    handle: Option<NonNull<c_void>>,
    directory: PathBuf,
    detector_log: BufWriter<File>,
    rgb: Vec<u8>,
}

impl ReferenceRecorder {
    pub(super) fn start(device_id: &str) -> Result<Self, String> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../debug-recordings");
        fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        let directory = tempfile::Builder::new()
            .prefix("session-")
            .tempdir_in(root)
            .map_err(|e| e.to_string())?;
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))
            .map_err(|e| e.to_string())?;
        let detector_log = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(directory.path().join("detector.log"))
            .map_err(|e| e.to_string())?;
        let video = c_path(&directory.path().join("video.mov"))?;
        let timestamps = c_path(&directory.path().join("frames.csv"))?;
        let device_id = CString::new(device_id).map_err(|e| e.to_string())?;
        let mut error = [0; 1024];
        let handle = NonNull::new(unsafe {
            v420_reference_capture_start(
                device_id.as_ptr(),
                video.as_ptr(),
                timestamps.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        })
        .ok_or_else(|| error_text(&error))?;
        let directory = directory.keep();
        println!(
            "[420vision::recording] STARTED (single AVFoundation session, camera-rate reference); session={}",
            directory.display()
        );
        Ok(Self {
            handle: Some(handle),
            directory,
            detector_log: BufWriter::new(detector_log),
            rgb: vec![0; 1920 * 1080 * 3],
        })
    }

    pub(super) fn frame(&mut self) -> Result<Option<Buffer>, NokhwaError> {
        let handle = self
            .handle
            .ok_or_else(|| NokhwaError::ReadFrameError("Reference camera stopped".into()))?;
        let (mut width, mut height, mut wall_ms) = (0, 0, 0);
        let status = unsafe {
            v420_reference_capture_frame(
                handle.as_ptr(),
                self.rgb.as_mut_ptr(),
                self.rgb.len(),
                &mut width,
                &mut height,
                &mut wall_ms,
                200,
            )
        };
        if status == 0 {
            return Ok(None);
        }
        if status != 1 {
            return Err(NokhwaError::ReadFrameError(format!(
                "Invalid or oversized reference frame (status {status})"
            )));
        }
        let frame_len = width as usize * height as usize * 3;
        if frame_len > self.rgb.len() {
            return Err(NokhwaError::ReadFrameError(
                "Reference frame exceeds buffer capacity".into(),
            ));
        }
        Ok(Some(Buffer::with_timestamp(
            Resolution::new(width, height),
            &self.rgb[..frame_len],
            FrameFormat::RAWRGB,
            Some(Duration::from_millis(wall_ms)),
        )))
    }

    pub(super) fn log_detector(&mut self, line: &str) {
        let result = writeln!(self.detector_log, "{line}").and_then(|()| self.detector_log.flush());
        if let Err(error) = result {
            eprintln!("[420vision::recording] Cannot save detector log: {error}");
        }
    }

    fn finish(&mut self) -> Result<RecordingStats, String> {
        let handle = self
            .handle
            .take()
            .ok_or("Reference recorder already stopped")?;
        let mut stats = RecordingStats::default();
        let mut error = [0; 1024];
        let completed = unsafe {
            v420_reference_writer_stop(handle.as_ptr(), error.as_mut_ptr(), error.len(), &mut stats)
        };
        self.detector_log.flush().map_err(|e| e.to_string())?;
        if completed != 1 {
            return Err(error_text(&error));
        }
        let mut summary = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(self.directory.join("summary.json"))
            .map_err(|e| e.to_string())?;
        serde_json::to_writer_pretty(&mut summary, &stats).map_err(|e| e.to_string())?;
        summary.flush().map_err(|e| e.to_string())?;
        summary.sync_all().map_err(|e| e.to_string())?;
        Ok(stats)
    }
}

impl Drop for ReferenceRecorder {
    fn drop(&mut self) {
        if self.handle.is_none() {
            return;
        }
        match self.finish() {
            Ok(stats) => {
                let seconds =
                    stats.last_wall_ms.saturating_sub(stats.first_wall_ms) as f64 / 1000.0;
                let fps = if seconds > 0.0 {
                    stats.frames.saturating_sub(1) as f64 / seconds
                } else {
                    0.0
                };
                println!(
                    "[420vision::recording] SAVED session={} frames={} dropped={} fps={:.2}",
                    self.directory.display(),
                    stats.frames,
                    stats.dropped,
                    fps
                );
            }
            Err(error) => eprintln!(
                "[420vision::recording] FINALIZATION FAILED session={}: {error}",
                self.directory.display()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::thread;

    fn synthetic_recorder() -> (tempfile::TempDir, ReferenceRecorder) {
        let directory = tempfile::tempdir().unwrap();
        let detector_log = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.path().join("detector.log"))
            .unwrap();
        let video = c_path(&directory.path().join("video.mov")).unwrap();
        let timestamps = c_path(&directory.path().join("frames.csv")).unwrap();
        let mut error = [0; 1024];
        let handle = NonNull::new(unsafe {
            v420_reference_writer_start(
                video.as_ptr(),
                timestamps.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        })
        .expect("synthetic writer must initialize");
        let recorder = ReferenceRecorder {
            handle: Some(handle),
            directory: directory.path().to_path_buf(),
            detector_log: BufWriter::new(detector_log),
            rgb: vec![0; 64 * 48 * 3],
        };
        (directory, recorder)
    }

    fn synthetic_frame(index: u8) -> Vec<u8> {
        let mut rgb = vec![0; 64 * 48 * 3];
        for pixel in rgb.as_chunks_mut::<3>().0 {
            *pixel = [90 + index, 60, 30];
        }
        rgb
    }

    fn append(recorder: &mut ReferenceRecorder, index: u8, wall_ms: u64) {
        let mut error = [0; 1024];
        let result = unsafe {
            v420_reference_writer_append(
                recorder.handle.unwrap().as_ptr(),
                synthetic_frame(index).as_ptr(),
                64,
                48,
                wall_ms,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_eq!(result, 1, "{}", error_text(&error));
    }

    #[test]
    fn recorder_finalizes_camera_rate_video_and_detector_log() {
        let (directory, mut recorder) = synthetic_recorder();
        for index in 0..6 {
            append(&mut recorder, index, 1_000 + u64::from(index) * 34);
            recorder.log_detector(&format!("synthetic detector update {index}"));
            thread::sleep(Duration::from_millis(34));
        }
        let stats = recorder.finish().unwrap();
        assert_eq!(stats.frames, 6);
        assert!(directory.path().join("video.mov").metadata().unwrap().len() > 0);
        let rows = fs::read_to_string(directory.path().join("frames.csv")).unwrap();
        assert_eq!(rows.lines().count() as u64, stats.frames + 1);
        let video = c_path(&directory.path().join("video.mov")).unwrap();
        assert_eq!(
            unsafe { v420_reference_test_decoded_frames(video.as_ptr()) },
            stats.frames as i64
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("detector.log"))
                .unwrap()
                .lines()
                .count(),
            6
        );
        assert!(directory.path().join("summary.json").is_file());
    }

    #[test]
    fn video_timeline_preserves_capture_wall_time_gaps() {
        let (directory, mut recorder) = synthetic_recorder();
        for (index, wall_ms) in [8_000, 8_100, 8_300].into_iter().enumerate() {
            append(&mut recorder, index as u8, wall_ms);
        }
        assert_eq!(recorder.finish().unwrap().frames, 3);
        let rows = fs::read_to_string(directory.path().join("frames.csv")).unwrap();
        assert!(rows.contains("0,0.000000,8000"));
        assert!(rows.contains("1,0.100000,8100"));
        assert!(rows.contains("2,0.300000,8300"));
    }

    #[test]
    fn empty_session_fails_instead_of_claiming_a_video_was_saved() {
        let (_directory, mut recorder) = synthetic_recorder();
        assert!(recorder.finish().is_err());
    }

    #[test]
    fn missing_camera_frame_wait_is_bounded_and_returns_control() {
        let (_directory, mut recorder) = synthetic_recorder();
        let started = std::time::Instant::now();
        assert!(recorder.frame().unwrap().is_none());
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(recorder.finish().is_err());
    }

    #[test]
    fn stopping_reference_camera_finalizes_a_distinct_session() {
        let mut manager = super::super::CameraManager::new();
        let mut paths = Vec::new();
        let mut directories = Vec::new();
        for _ in 0..2 {
            let (directory, mut recorder) = synthetic_recorder();
            for index in 0..3 {
                append(&mut recorder, index, 10_000 + u64::from(index) * 40);
            }
            paths.push(directory.path().to_path_buf());
            directories.push(directory);
            manager.reference_camera = Some(recorder);
            manager.release_camera();
            assert!(paths.last().unwrap().join("video.mov").is_file());
            assert!(paths.last().unwrap().join("summary.json").is_file());
        }
        assert_ne!(paths[0], paths[1]);
    }
}
