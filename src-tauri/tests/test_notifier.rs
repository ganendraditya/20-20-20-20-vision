use vision420_lib::notifier::{AudioPlayer, Notifier};

#[test]
fn test_wav_audio_synthesis_valid_header() {
    let wav_bytes = AudioPlayer::synthesize_chime_wav(&[440.0], 0.1, 0.5);

    // Assert minimum RIFF WAVE format constraints
    assert!(wav_bytes.len() > 44, "WAV must contain header + data");
    assert_eq!(&wav_bytes[0..4], b"RIFF");
    assert_eq!(&wav_bytes[8..12], b"WAVE");
    assert_eq!(&wav_bytes[12..16], b"fmt ");
    assert_eq!(&wav_bytes[36..40], b"data");

    // Sample rate check: 44100 Hz in bytes 24..28
    let sample_rate = u32::from_le_bytes(wav_bytes[24..28].try_into().unwrap());
    assert_eq!(sample_rate, 44100);

    // Audio format check: PCM = 1 in bytes 20..22
    let audio_format = u16::from_le_bytes(wav_bytes[20..22].try_into().unwrap());
    assert_eq!(audio_format, 1);
}

#[test]
fn test_audio_chimes_can_be_invoked_safely() {
    AudioPlayer::set_silent_mode(true);
    // Should trigger non-blocking threads without panic
    AudioPlayer::play_stare_warning();
    AudioPlayer::play_break_chime();
}

#[test]
fn test_native_notifications_can_be_invoked_safely() {
    Notifier::set_silent_mode(true);
    // Should spawn asynchronous OS notification task without error or blocking
    Notifier::notify_stare_warning();
    Notifier::notify_break_time();
}

#[test]
fn test_sleep_blocker_acquire_and_release_lifecycle() {
    use vision420_lib::notifier::SleepBlocker;

    let mut blocker = SleepBlocker::new();
    assert!(!blocker.is_active(), "Blocker should initially be inactive");

    // 1. Acquire sleep assertion
    blocker.acquire();
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    assert!(blocker.is_active(), "Blocker must be active after acquire");

    // Idempotent acquire should not break state
    blocker.acquire();
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    assert!(blocker.is_active());

    // 2. Release sleep assertion
    blocker.release();
    assert!(!blocker.is_active(), "Blocker must be inactive after release");

    // Idempotent release
    blocker.release();
    assert!(!blocker.is_active());
}
