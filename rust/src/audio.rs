//! WAV audio output utilities.

use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::Path;

/// Save audio samples to a WAV file.
///
/// Args:
///     samples: Audio samples in range [-1.0, 1.0]
///     path: Output file path
///     sample_rate: Sample rate in Hz (typically 24000 for Kokoro)
pub fn save_wav(samples: &[f32], path: &Path, sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec).context("Failed to create WAV file")?;

    for &sample in samples {
        // Clamp and convert to 16-bit integer
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * i16::MAX as f32) as i16;
        writer
            .write_sample(int_sample)
            .context("Failed to write sample")?;
    }

    writer.finalize().context("Failed to finalize WAV file")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_wav() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0).sin()).collect();

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        save_wav(&samples, path, 24000).unwrap();

        // Verify file was created and has content
        let mut file = std::fs::File::open(path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        // WAV files start with "RIFF"
        assert_eq!(&buffer[0..4], b"RIFF");
    }
}
