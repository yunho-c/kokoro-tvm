//! Voice pack loading and style selection.

use anyhow::{Context, Result};
use ndarray::{s, Array2, Array3, ArrayD};
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
    /// Handles both 2D `[N, 256]` and 3D `[N, 1, 256]` arrays (squeezes middle dim).
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open voice pack file")?;

        // Try to load as dynamic array first to handle different shapes
        let data_dyn: ArrayD<f32> =
            ArrayD::read_npy(file).context("Failed to read voice pack .npy")?;

        let shape = data_dyn.shape();

        // Handle different shapes
        let data: Array2<f32> = match shape.len() {
            2 => {
                // Already 2D: [N, 256]
                data_dyn
                    .into_dimensionality()
                    .context("Failed to convert to 2D array")?
            }
            3 => {
                // 3D: [N, 1, 256] - squeeze middle dimension
                if shape[1] != 1 {
                    anyhow::bail!(
                        "3D voice pack should have shape [N, 1, 256], got {:?}",
                        shape
                    );
                }
                let arr3: Array3<f32> = data_dyn
                    .into_dimensionality()
                    .context("Failed to convert to 3D array")?;
                // Remove axis 1 to get [N, 256]
                arr3.index_axis_move(ndarray::Axis(1), 0)
            }
            _ => {
                anyhow::bail!(
                    "Voice pack must be 2D [N, 256] or 3D [N, 1, 256], got shape {:?}",
                    shape
                );
            }
        };

        if data.ncols() != 256 {
            anyhow::bail!("Voice pack must have 256 columns, got {}", data.ncols());
        }

        println!("  Loaded voice pack: {} entries, shape {:?}", data.nrows(), data.shape());

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

    /// Select a style embedding by explicit index.
    ///
    /// Returns a `[1, 256]` array containing the selected style embedding.
    pub fn select_style_by_index(&self, index: usize) -> Result<Array2<f32>> {
        if self.data.is_empty() {
            anyhow::bail!("Voice pack is empty");
        }
        if index >= self.data.nrows() {
            anyhow::bail!(
                "Voice index {} out of range (max {})",
                index,
                self.data.nrows().saturating_sub(1)
            );
        }

        let row = self.data.slice(s![index, ..]);
        Ok(row.insert_axis(ndarray::Axis(0)).to_owned())
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
