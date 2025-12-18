//! Preprocessing utilities for TVM inference.

use crate::constants::{STATIC_AUDIO_LEN, STATIC_TEXT_LEN};
use ndarray::{Array1, Array2, Array3, ArrayD, ArrayViewD, Axis};

/// Numerically stable sigmoid function.
pub fn sigmoid(x: &ArrayViewD<f32>) -> ArrayD<f32> {
    x.mapv(|v| {
        if v >= 0.0 {
            1.0 / (1.0 + (-v).exp())
        } else {
            let exp_v = v.exp();
            exp_v / (1.0 + exp_v)
        }
    })
}

/// Pad input IDs to static length.
///
/// Returns the padded input array and the original length.
pub fn pad_input_ids(ids: &[i64], static_len: usize) -> (Array2<i64>, usize) {
    let cur_len = ids.len().min(static_len);

    let mut padded = Array2::<i64>::zeros((1, static_len));
    for (i, &id) in ids.iter().take(cur_len).enumerate() {
        padded[[0, i]] = id;
    }

    (padded, cur_len)
}

/// Create attention mask and text mask.
///
/// - `text_mask`: True where position > length (padding positions)
/// - `attention_mask`: Inverted text_mask as i64 (1 for valid, 0 for padding)
pub fn create_masks(cur_len: usize, static_len: usize) -> (Array2<bool>, Array2<i64>) {
    let mut text_mask = Array2::<bool>::from_elem((1, static_len), false);
    let mut attention_mask = Array2::<i64>::ones((1, static_len));

    for i in cur_len..static_len {
        text_mask[[0, i]] = true;
        attention_mask[[0, i]] = 0;
    }

    (text_mask, attention_mask)
}

/// Build alignment matrix from predicted durations.
///
/// Args:
///     duration_logits: [B, seq_len, bins] duration prediction logits
///     cur_len: Actual sequence length (before padding)
///     speed: Speed multiplier (1.0 = normal)
///
/// Returns:
///     - full_aln: [1, STATIC_TEXT_LEN, STATIC_AUDIO_LEN] alignment matrix
///     - actual_audio_len: Number of frames actually used
pub fn build_alignment(
    duration_logits: &ArrayD<f32>,
    cur_len: usize,
    speed: f32,
) -> (Array3<f32>, usize) {
    // Apply sigmoid and sum over bins dimension
    let probs = sigmoid(&duration_logits.view());

    // Sum over last axis (bins) and apply speed
    // probs is [B, seq_len, bins], we want [seq_len]
    let duration_sum: Array1<f32> = probs
        .sum_axis(Axis(2))
        .into_shape_with_order(probs.shape()[1])
        .unwrap()
        .mapv(|v| v / speed);

    // Round and clamp to get integer durations
    let pred_dur: Vec<i64> = duration_sum
        .iter()
        .take(cur_len)
        .map(|&v| v.round().max(1.0) as i64)
        .collect();

    // Build alignment indices: repeat each text position by its duration
    let indices: Vec<usize> = pred_dur
        .iter()
        .enumerate()
        .flat_map(|(i, &dur)| std::iter::repeat(i).take(dur as usize))
        .collect();

    let actual_audio_len = indices.len().min(STATIC_AUDIO_LEN);

    // Create alignment matrix
    let mut pred_aln_trg = Array2::<f32>::zeros((cur_len, STATIC_AUDIO_LEN));
    for (frame_idx, &text_idx) in indices.iter().take(actual_audio_len).enumerate() {
        pred_aln_trg[[text_idx, frame_idx]] = 1.0;
    }

    // Pad to full static text length
    let mut full_aln = Array3::<f32>::zeros((1, STATIC_TEXT_LEN, STATIC_AUDIO_LEN));
    for i in 0..cur_len {
        for j in 0..STATIC_AUDIO_LEN {
            full_aln[[0, i, j]] = pred_aln_trg[[i, j]];
        }
    }

    (full_aln, actual_audio_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let x = ArrayD::from_elem(vec![2, 3], 0.0f32);
        let y = sigmoid(&x.view());
        assert!((y[[0, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pad_input_ids() {
        let ids = vec![1, 2, 3, 4, 5];
        let (padded, cur_len) = pad_input_ids(&ids, 10);

        assert_eq!(cur_len, 5);
        assert_eq!(padded.shape(), &[1, 10]);
        assert_eq!(padded[[0, 0]], 1);
        assert_eq!(padded[[0, 4]], 5);
        assert_eq!(padded[[0, 5]], 0); // Padding
    }

    #[test]
    fn test_create_masks() {
        let (text_mask, attention_mask) = create_masks(5, 10);

        assert_eq!(text_mask.shape(), &[1, 10]);
        assert!(!text_mask[[0, 4]]); // Valid position
        assert!(text_mask[[0, 5]]); // Padding position

        assert_eq!(attention_mask[[0, 4]], 1);
        assert_eq!(attention_mask[[0, 5]], 0);
    }
}
