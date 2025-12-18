//! Voice pack loading and style selection.

use anyhow::{Context, Result};
use ndarray::{Array2, s};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::path::Path;

/// Voice pack containing style embeddings.
///
/// Each voice pack is a 2D array of shape `[N, 256]` where N varies by pack.
/// The first 128 dimensions are the style vector, the last 128 are additional style info.
pub struct VoicePack {
    data: Array2<f32>,
}

impl VoicePack {
    /// Load a voice pack from a .npy file.
    ///
    /// The file should contain a float32 array of shape `[N, 256]`.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open voice pack file")?;
        let data: Array2<f32> =
            Array2::read_npy(file).context("Failed to read voice pack .npy")?;

        if data.ncols() != 256 {
            anyhow::bail!(
                "Voice pack must have 256 columns, got {}",
                data.ncols()
            );
        }

        Ok(Self { data })
    }

    /// Select a style embedding based on phoneme length.
    ///
    /// Returns a `[1, 256]` array containing the selected style embedding.
    /// Selection logic: `pack[min(phoneme_len - 1, pack.nrows() - 1)]`
    pub fn select_style(&self, phoneme_len: usize) -> Array2<f32> {
        let index = if phoneme_len == 0 {
            0
        } else {
            (phoneme_len - 1).min(self.data.nrows() - 1)
        };

        // Get the row and reshape to [1, 256]
        let row = self.data.slice(s![index, ..]);
        row.insert_axis(ndarray::Axis(0)).to_owned()
    }

    /// Get the number of style entries in this voice pack.
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Check if voice pack is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_select_style() {
        // Create a mock voice pack
        let data = Array2::<f32>::zeros((100, 256));
        let pack = VoicePack { data };

        // Test various phoneme lengths
        let style = pack.select_style(0);
        assert_eq!(style.shape(), &[1, 256]);

        let style = pack.select_style(50);
        assert_eq!(style.shape(), &[1, 256]);

        let style = pack.select_style(1000); // Larger than pack size
        assert_eq!(style.shape(), &[1, 256]);
    }
}
