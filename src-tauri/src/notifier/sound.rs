use std::thread;

pub struct AudioPlayer;

impl AudioPlayer {
    /// Synthesize an in-memory PCM 16-bit Mono WAV byte buffer with exponential decay
    pub fn synthesize_chime_wav(freqs: &[f32], duration_secs: f32, volume: f32) -> Vec<u8> {
        let sample_rate = 44100u32;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let num_channels = 1u16;
        let bits_per_sample = 16u16;
        let byte_rate = sample_rate * (num_channels as u32) * (bits_per_sample as u32 / 8);
        let block_align = num_channels * (bits_per_sample / 8);
        let data_size = (num_samples * 2) as u32;

        let mut buffer = Vec::with_capacity(44 + num_samples * 2);

        // RIFF header
        buffer.extend_from_slice(b"RIFF");
        buffer.extend_from_slice(&(36 + data_size).to_le_bytes());
        buffer.extend_from_slice(b"WAVE");

        // fmt subchunk
        buffer.extend_from_slice(b"fmt ");
        buffer.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size (16 for PCM)
        buffer.extend_from_slice(&1u16.to_le_bytes());  // AudioFormat (1 for PCM)
        buffer.extend_from_slice(&num_channels.to_le_bytes());
        buffer.extend_from_slice(&sample_rate.to_le_bytes());
        buffer.extend_from_slice(&byte_rate.to_le_bytes());
        buffer.extend_from_slice(&block_align.to_le_bytes());
        buffer.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data subchunk
        buffer.extend_from_slice(b"data");
        buffer.extend_from_slice(&data_size.to_le_bytes());

        // Generate audio samples
        let pi2 = std::f32::consts::PI * 2.0;
        let decay_rate = 5.0 / duration_secs.max(0.01);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let envelope = (-decay_rate * t).exp() * volume;

            let mut sample_val = 0.0f32;
            for &freq in freqs {
                sample_val += (freq * pi2 * t).sin();
            }
            if !freqs.is_empty() {
                sample_val /= freqs.len() as f32;
            }

            let pcm = (sample_val * envelope * 32767.0).clamp(-32768.0, 32767.0) as i16;
            buffer.extend_from_slice(&pcm.to_le_bytes());
        }

        buffer
    }

    /// Play soft double-tone chime in background thread asynchronously
    pub fn play_stare_warning() {
        thread::spawn(|| {
            // Soft A5 -> D6 chime (880 Hz, 1174 Hz)
            let wav_bytes = Self::synthesize_chime_wav(&[880.0, 1174.66], 0.25, 0.4);
            Self::play_bytes(&wav_bytes);
        });
    }

    /// Play calming 4-chord progression chime for 20-20-20 break
    pub fn play_break_chime() {
        thread::spawn(|| {
            // F#m7 harmony: F#4 (370), A4 (440), C#5 (554), E5 (659)
            let wav_bytes = Self::synthesize_chime_wav(&[369.99, 440.0, 554.37, 659.25], 1.2, 0.5);
            Self::play_bytes(&wav_bytes);
        });
    }

    fn play_bytes(bytes: &[u8]) {
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            use std::io::Write;

            if let Ok(mut temp) = tempfile::NamedTempFile::new() {
                if temp.write_all(bytes).is_ok() {
                    let path = temp.path().to_string_lossy().to_string();
                    let _ = Command::new("afplay")
                        .arg(&path)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status();
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Windows native PlaySound API via powershell invoking script block with parameter binding
            use std::process::Command;
            use std::io::Write;

            if let Ok(mut temp) = tempfile::NamedTempFile::new() {
                if temp.write_all(bytes).is_ok() {
                    let path = temp.path().to_string_lossy().to_string();
                    let invoker = "& { param($p) (New-Object Media.SoundPlayer $p).PlaySync() } $args[0]";
                    let _ = Command::new("powershell")
                        .args(["-WindowStyle", "Hidden", "-Command", invoker, &path])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status();
                }
            }
        }
    }
}
