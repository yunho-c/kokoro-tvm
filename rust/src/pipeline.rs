//! TVM inference pipeline for Kokoro TTS.

use crate::constants::{SAMPLES_PER_FRAME, STATIC_AUDIO_LEN, STATIC_TEXT_LEN};
use crate::preprocessing::{build_alignment, create_masks, pad_input_ids};
use anyhow::{Context, Result};
use ndarray::{s, Array2, Array3, ArrayD, Axis};
use std::collections::HashMap;
use std::path::Path;
use tvm_ffi::{Function, Module, Tensor};

/// TVM-based inference pipeline for Kokoro TTS.
///
/// This pipeline orchestrates the following TVM-compiled modules:
/// - BERT encoder: input_ids, attention_mask -> d_en
/// - Duration predictor: d_en, style, lengths, mask -> (duration_logits, d)
/// - F0N predictor: en, style, frame_lengths -> (F0, N)
/// - Text encoder: input_ids, lengths, mask -> t_en
/// - Decoder: asr, F0, N, style -> audio
pub struct KokoroPipeline {
    // Module functions
    f_bert: Function,
    f_duration: Function,
    f_f0n: Function,
    f_text_enc: Function,
    decoder_fns: HashMap<usize, Function>,
    decoder_bucket_lens: Vec<usize>,

    // Device info
    device: String,
}

impl KokoroPipeline {
    /// Load TVM modules from a directory.
    ///
    /// Args:
    ///     lib_dir: Directory containing compiled .so/.dylib files
    ///     device: Target device ("llvm", "metal", "cuda")
    pub fn load(lib_dir: &Path, device: &str) -> Result<Self> {
        let ext = if cfg!(target_os = "macos") {
            if device == "metal" {
                "dylib"
            } else {
                "so"
            }
        } else {
            "so"
        };

        // Load encoder modules
        let bert_path = lib_dir.join(format!("bert_compiled.{}", ext));
        let duration_path = lib_dir.join(format!("duration_compiled.{}", ext));
        let f0n_path = lib_dir.join(format!("f0n_compiled.{}", ext));
        let text_enc_path = lib_dir.join(format!("text_encoder_compiled.{}", ext));

        println!("  Loading BERT from {:?}...", bert_path);
        let bert_mod = Module::load_from_file(bert_path.to_str().unwrap())
            .context("Failed to load BERT module")?;

        println!("  Loading Duration from {:?}...", duration_path);
        let duration_mod = Module::load_from_file(duration_path.to_str().unwrap())
            .context("Failed to load Duration module")?;

        println!("  Loading F0N from {:?}...", f0n_path);
        let f0n_mod = Module::load_from_file(f0n_path.to_str().unwrap())
            .context("Failed to load F0N module")?;

        println!("  Loading Text Encoder from {:?}...", text_enc_path);
        let text_enc_mod = Module::load_from_file(text_enc_path.to_str().unwrap())
            .context("Failed to load Text Encoder module")?;

        // Get functions from modules
        // Note: The VirtualMachine API may differ - this is a best-effort implementation
        // based on tvm-ffi documentation. May need adjustment based on actual API.
        let f_bert = bert_mod
            .get_function("bert_forward")
            .context("Failed to get bert_forward function")?;
        let f_duration = duration_mod
            .get_function("duration_forward")
            .context("Failed to get duration_forward function")?;
        let f_f0n = f0n_mod
            .get_function("f0n_forward")
            .context("Failed to get f0n_forward function")?;
        let f_text_enc = text_enc_mod
            .get_function("text_encoder_forward")
            .context("Failed to get text_encoder_forward function")?;

        // Load decoder(s) - check for bucketed decoders first
        let mut decoder_fns = HashMap::new();
        let mut decoder_bucket_lens = Vec::new();

        // Try to find bucketed decoders (decoder_compiled_seq256.so, etc.)
        for entry in std::fs::read_dir(lib_dir).context("Failed to read lib_dir")? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if name_str.starts_with("decoder_compiled_seq") && name_str.ends_with(ext) {
                // Extract bucket size from filename
                let prefix = "decoder_compiled_seq";
                let suffix = format!(".{}", ext);
                if let Some(num_str) = name_str
                    .strip_prefix(prefix)
                    .and_then(|s| s.strip_suffix(&suffix))
                {
                    if let Ok(bucket_len) = num_str.parse::<usize>() {
                        println!("  Loading Decoder bucket {} from {:?}...", bucket_len, entry.path());
                        let decoder_mod =
                            Module::load_from_file(entry.path().to_str().unwrap())
                                .context(format!("Failed to load decoder bucket {}", bucket_len))?;
                        let f_decoder = decoder_mod
                            .get_function("decoder_forward")
                            .context("Failed to get decoder_forward function")?;
                        decoder_fns.insert(bucket_len, f_decoder);
                        decoder_bucket_lens.push(bucket_len);
                    }
                }
            }
        }

        // Fall back to default decoder if no buckets found
        if decoder_fns.is_empty() {
            let decoder_path = lib_dir.join(format!("decoder_compiled.{}", ext));
            println!("  Loading Decoder from {:?}...", decoder_path);
            let decoder_mod = Module::load_from_file(decoder_path.to_str().unwrap())
                .context("Failed to load Decoder module")?;
            let f_decoder = decoder_mod
                .get_function("decoder_forward")
                .context("Failed to get decoder_forward function")?;
            decoder_fns.insert(STATIC_AUDIO_LEN, f_decoder);
            decoder_bucket_lens.push(STATIC_AUDIO_LEN);
        }

        decoder_bucket_lens.sort();
        println!(
            "Loaded {} decoder bucket(s): {:?}",
            decoder_bucket_lens.len(),
            decoder_bucket_lens
        );

        Ok(Self {
            f_bert,
            f_duration,
            f_f0n,
            f_text_enc,
            decoder_fns,
            decoder_bucket_lens,
            device: device.to_string(),
        })
    }

    /// Select the smallest decoder bucket that fits the given frame count.
    fn select_decoder_bucket(&self, frames: usize) -> usize {
        for &bucket in &self.decoder_bucket_lens {
            if bucket >= frames {
                return bucket;
            }
        }
        // Return largest bucket if none fit
        *self.decoder_bucket_lens.last().unwrap_or(&STATIC_AUDIO_LEN)
    }

    /// Run full inference.
    ///
    /// Args:
    ///     input_ids: Token IDs (including start/end tokens)
    ///     ref_s: Style embedding [256] (flat slice from the [1, 256] array)
    ///     speed: Speech speed multiplier (1.0 = normal)
    ///
    /// Returns:
    ///     Audio samples as Vec<f32>
    pub fn forward(&self, input_ids: &[i64], ref_s: &[f32], speed: f32) -> Result<Vec<f32>> {
        // --- Preprocessing ---
        let (input_ids_arr, cur_len) = pad_input_ids(input_ids, STATIC_TEXT_LEN);
        let (text_mask, attention_mask) = create_masks(cur_len, STATIC_TEXT_LEN);
        let input_lengths = Array2::from_elem((1,), cur_len as i64);

        // Style embeddings: s = ref_s[:, 128:], style_128 = ref_s[:, :128]
        let style_128: Vec<f32> = ref_s[..128].to_vec();
        let s: Vec<f32> = ref_s[128..].to_vec();

        // Convert to TVM tensors
        let input_ids_tvm = Tensor::from_slice(
            input_ids_arr.as_slice().unwrap(),
            &[1, STATIC_TEXT_LEN as i64],
        )?;
        let attention_mask_tvm = Tensor::from_slice(
            attention_mask.as_slice().unwrap(),
            &[1, STATIC_TEXT_LEN as i64],
        )?;

        // --- BERT: input_ids, attention_mask -> d_en [1, 512, seq_len] ---
        let bert_out = self.f_bert.call_tuple((&input_ids_tvm, &attention_mask_tvm))?;
        let d_en_tvm: Tensor = bert_out.try_into()?;

        // --- Duration: d_en, s, lengths, mask -> (duration_logits, d) ---
        let s_tvm = Tensor::from_slice(&s, &[1, 128])?;
        let lengths_tvm = Tensor::from_slice(&[cur_len as i64], &[1])?;

        // Convert bool mask to appropriate type for TVM
        let text_mask_i8: Vec<i8> = text_mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
        let text_mask_tvm = Tensor::from_slice(&text_mask_i8, &[1, STATIC_TEXT_LEN as i64])?;

        let duration_out =
            self.f_duration
                .call_tuple((&d_en_tvm, &s_tvm, &lengths_tvm, &text_mask_tvm))?;

        // Extract duration_logits and d from output tuple
        // This is a simplification - actual tuple extraction may differ
        let (duration_logits_tvm, d_tvm): (Tensor, Tensor) = duration_out.try_into()?;

        // Convert to ndarray for alignment computation
        let duration_logits = Self::tensor_to_arrayd(&duration_logits_tvm)?;
        let d_np = Self::tensor_to_array3(&d_tvm)?;

        // --- Alignment computation ---
        let (full_aln, actual_audio_len) = build_alignment(&duration_logits, cur_len, speed);

        // Compute en = d.T @ alignment
        // d is [1, T, 640], transpose to [1, 640, T]
        let d_transposed = d_np.permuted_axes([0, 2, 1]);
        let en = Self::matmul_3d(&d_transposed, &full_aln);

        // --- F0N: en, s, frame_lengths -> (F0, N) ---
        let en_tvm = Tensor::from_slice(
            en.as_slice().unwrap(),
            &[1, 640, STATIC_AUDIO_LEN as i64],
        )?;
        let frame_lengths_tvm = Tensor::from_slice(&[actual_audio_len as i64], &[1])?;

        let f0n_out = self.f_f0n.call_tuple((&en_tvm, &s_tvm, &frame_lengths_tvm))?;
        let (f0_tvm, n_tvm): (Tensor, Tensor) = f0n_out.try_into()?;

        // --- Text Encoder: input_ids, lengths, mask -> t_en ---
        let t_en_out =
            self.f_text_enc
                .call_tuple((&input_ids_tvm, &lengths_tvm, &text_mask_tvm))?;
        let t_en_tvm: Tensor = t_en_out.try_into()?;

        // Compute asr = t_en @ alignment
        let t_en = Self::tensor_to_array3(&t_en_tvm)?;
        let asr = Self::matmul_3d(&t_en, &full_aln);

        // --- Decoder: asr, F0, N, style[:128] -> audio ---
        let bucket_len = self.select_decoder_bucket(actual_audio_len);

        // Slice tensors to bucket size
        let asr_b = asr.slice(s![.., .., ..bucket_len]).to_owned();
        let f0_np = Self::tensor_to_array2(&f0_tvm)?;
        let n_np = Self::tensor_to_array2(&n_tvm)?;
        let f0_b = f0_np.slice(s![.., ..(bucket_len * 2)]).to_owned();
        let n_b = n_np.slice(s![.., ..(bucket_len * 2)]).to_owned();

        let asr_tvm = Tensor::from_slice(
            asr_b.as_slice().unwrap(),
            &[1, 512, bucket_len as i64],
        )?;
        let f0_b_tvm = Tensor::from_slice(
            f0_b.as_slice().unwrap(),
            &[1, (bucket_len * 2) as i64],
        )?;
        let n_b_tvm = Tensor::from_slice(
            n_b.as_slice().unwrap(),
            &[1, (bucket_len * 2) as i64],
        )?;
        let style_128_tvm = Tensor::from_slice(&style_128, &[1, 128])?;

        let f_decoder = self.decoder_fns.get(&bucket_len).context(format!(
            "No decoder found for bucket size {}",
            bucket_len
        ))?;

        let audio_out =
            f_decoder.call_tuple((&asr_tvm, &f0_b_tvm, &n_b_tvm, &style_128_tvm))?;
        let audio_tvm: Tensor = audio_out.try_into()?;

        // Convert to output vector and trim
        let audio_full = Self::tensor_to_vec(&audio_tvm)?;
        let target_samples = audio_full.len().min(actual_audio_len * SAMPLES_PER_FRAME);

        Ok(audio_full[..target_samples].to_vec())
    }

    // --- Helper functions for tensor conversion ---

    fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<f32>> {
        // This is a placeholder - actual implementation depends on tvm-ffi API
        // The tensor data should be accessible via some method
        todo!("Implement tensor to vec conversion based on tvm-ffi API")
    }

    fn tensor_to_arrayd(tensor: &Tensor) -> Result<ArrayD<f32>> {
        todo!("Implement tensor to ArrayD conversion based on tvm-ffi API")
    }

    fn tensor_to_array2(tensor: &Tensor) -> Result<Array2<f32>> {
        todo!("Implement tensor to Array2 conversion based on tvm-ffi API")
    }

    fn tensor_to_array3(tensor: &Tensor) -> Result<Array3<f32>> {
        todo!("Implement tensor to Array3 conversion based on tvm-ffi API")
    }

    fn matmul_3d(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
        // Simple batch matmul: [1, M, K] @ [1, K, N] -> [1, M, N]
        let m = a.shape()[1];
        let n = b.shape()[2];
        let k = a.shape()[2];

        let mut result = Array3::<f32>::zeros((1, m, n));

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[[0, i, l]] * b[[0, l, j]];
                }
                result[[0, i, j]] = sum;
            }
        }

        result
    }
}
