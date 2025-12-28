//! Validation utilities for comparing Rust pipeline outputs with Python golden tensors.

use anyhow::{Context, Result};
use ndarray::{s, Array1, Array2, Array3, ArrayD, Ix3};
use ndarray_npy::{NpzReader, NpzWriter, ReadNpyExt, WriteNpyExt};
use std::fs::{self, File};
use std::path::Path;

use crate::PipelineTrace;

/// Statistics for comparing two arrays.
#[derive(Debug)]
pub struct ComparisonStats {
    pub mae: f32,
    pub max_abs: f32,
    pub rmse: f32,
}

impl ComparisonStats {
    /// Compute comparison metrics between two arrays.
    pub fn compute(a: &[f32], b: &[f32]) -> Self {
        if a.is_empty() || b.is_empty() {
            return Self {
                mae: f32::NAN,
                max_abs: f32::NAN,
                rmse: f32::NAN,
            };
        }
        
        let n = a.len().min(b.len());
        let mut sum_abs_diff = 0.0f64;
        let mut max_abs_diff = 0.0f32;
        let mut sum_sq_diff = 0.0f64;
        
        for i in 0..n {
            let diff = (a[i] - b[i]).abs();
            sum_abs_diff += diff as f64;
            max_abs_diff = max_abs_diff.max(diff);
            sum_sq_diff += (diff as f64) * (diff as f64);
        }
        
        Self {
            mae: (sum_abs_diff / n as f64) as f32,
            max_abs: max_abs_diff,
            rmse: (sum_sq_diff / n as f64).sqrt() as f32,
        }
    }
    
    /// Check if the comparison passes a threshold.
    pub fn passes(&self, max_mae: f32) -> bool {
        self.mae.is_finite() && self.mae < max_mae
    }
}

impl std::fmt::Display for ComparisonStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mae={:.6e}, max_abs={:.6e}, rmse={:.6e}",
            self.mae, self.max_abs, self.rmse
        )
    }
}

#[derive(Debug)]
pub struct IntComparisonStats {
    pub mismatches: usize,
    pub max_abs: i64,
    pub len_a: usize,
    pub len_b: usize,
}

impl std::fmt::Display for IntComparisonStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mismatches={} max_abs={} len_a={} len_b={}",
            self.mismatches, self.max_abs, self.len_a, self.len_b
        )
    }
}

fn compare_f32(name: &str, a: &[f32], b: &[f32], max_mae: f32) -> bool {
    if a.is_empty() || b.is_empty() {
        println!("{}: (no data to compare)", name);
        return false;
    }
    let stats = ComparisonStats::compute(a, b);
    let pass = stats.passes(max_mae);
    let status = if pass { "✓ PASS" } else { "✗ FAIL" };
    let len_note = if a.len() != b.len() {
        format!(" (len {} vs {})", a.len(), b.len())
    } else {
        String::new()
    };
    println!("{}: {} - {}{}", name, status, stats, len_note);
    pass
}

fn compare_i64(name: &str, a: &[i64], b: &[i64]) -> bool {
    let n = a.len().min(b.len());
    if n == 0 {
        println!("{}: (no data to compare)", name);
        return false;
    }

    let mut mismatches = 0usize;
    let mut max_abs = 0i64;
    for i in 0..n {
        let diff = (a[i] - b[i]).abs();
        if diff != 0 {
            mismatches += 1;
            max_abs = max_abs.max(diff);
        }
    }

    let stats = IntComparisonStats {
        mismatches,
        max_abs,
        len_a: a.len(),
        len_b: b.len(),
    };
    let pass = mismatches == 0 && a.len() == b.len();
    let status = if pass { "✓ PASS" } else { "✗ FAIL" };
    println!("{}: {} - {}", name, status, stats);
    pass
}

fn view_to_vec_f32<A: ndarray::Dimension>(view: ndarray::ArrayView<f32, A>) -> Vec<f32> {
    view.iter().copied().collect()
}

/// Golden tensors loaded from Python exports.
pub struct GoldenTensors {
    pub input_ids: Array2<i64>,
    pub ref_s: Array2<f32>,
    pub speed: f32,
    pub d_en: Option<Array3<f32>>,
    pub duration_logits: Option<ArrayD<f32>>,
    pub d: Option<Array3<f32>>,
    pub pred_dur: Option<Array1<i64>>,
    pub frames: Option<i64>,
    pub f0: Option<Array2<f32>>,
    pub n: Option<Array2<f32>>,
    pub t_en: Option<Array3<f32>>,
    pub audio: Option<Array1<f32>>,
}

impl GoldenTensors {
    /// Load golden tensors from a directory containing Python exports.
    pub fn load(dir: &Path) -> Result<Self> {
        println!("Loading golden tensors from {:?}...", dir);
        
        // Load inputs (00_inputs.npz)
        let inputs_path = dir.join("00_inputs.npz");
        let mut inputs_npz = NpzReader::new(File::open(&inputs_path)
            .context("Failed to open 00_inputs.npz")?)?;
        
        let input_ids: Array2<i64> = inputs_npz.by_name("input_ids.npy")?;
        let ref_s: Array2<f32> = inputs_npz.by_name("ref_s.npy")?;
        let speed_arr: Array1<f32> = inputs_npz.by_name("speed.npy")?;
        let speed = speed_arr[0];
        
        println!("  Loaded inputs: input_ids={:?}, ref_s={:?}, speed={}", 
                 input_ids.shape(), ref_s.shape(), speed);
        
        // Load BERT output (01_bert_out.npy)
        let d_en = Self::load_npy_opt::<f32, 3>(dir, "01_bert_out.npy")?;
        if let Some(ref arr) = d_en {
            println!("  Loaded d_en: {:?}", arr.shape());
        }
        
        // Load duration output (02_duration_out.npz)
        let (duration_logits, d) = Self::load_duration_npz(dir)?;
        
        // Load alignment (03_alignment.npz)
        let (pred_dur, frames) = Self::load_alignment_npz(dir)?;
        
        // Load F0N output (04_f0n_out.npz)
        let (f0, n) = Self::load_f0n_npz(dir)?;
        
        // Load text encoder output (05_text_enc_out.npy)
        let t_en = Self::load_npy_opt::<f32, 3>(dir, "05_text_enc_out.npy")?;
        if let Some(ref arr) = t_en {
            println!("  Loaded t_en: {:?}", arr.shape());
        }
        
        // Load audio (06_audio.npy)
        let audio = Self::load_npy_opt::<f32, 1>(dir, "06_audio.npy")?;
        if let Some(ref arr) = audio {
            println!("  Loaded audio: {} samples ({:.2}s)", arr.len(), arr.len() as f32 / 24000.0);
        }
        
        Ok(Self {
            input_ids,
            ref_s,
            speed,
            d_en,
            duration_logits,
            d,
            pred_dur,
            frames,
            f0,
            n,
            t_en,
            audio,
        })
    }
    
    fn load_npy_opt<T, const D: usize>(dir: &Path, name: &str) -> Result<Option<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<[usize; D]>>>>
    where
        T: ndarray_npy::ReadableElement,
        ndarray::Dim<[usize; D]>: ndarray::Dimension,
    {
        let path = dir.join(name);
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(&path).context(format!("Failed to open {}", name))?;
        let arr = ndarray::ArrayBase::read_npy(file).context(format!("Failed to parse {}", name))?;
        Ok(Some(arr))
    }
    
    fn load_duration_npz(dir: &Path) -> Result<(Option<ArrayD<f32>>, Option<Array3<f32>>)> {
        let path = dir.join("02_duration_out.npz");
        if !path.exists() {
            return Ok((None, None));
        }
        
        let mut npz = NpzReader::new(File::open(&path)?)?;
        
        let duration_logits: Option<ArrayD<f32>> = npz.by_name("duration_logits.npy").ok();
        let d: Option<Array3<f32>> = npz.by_name("d.npy").ok();
        
        if let Some(ref arr) = duration_logits {
            println!("  Loaded duration_logits: {:?}", arr.shape());
        }
        if let Some(ref arr) = d {
            println!("  Loaded d: {:?}", arr.shape());
        }
        
        Ok((duration_logits, d))
    }
    
    fn load_alignment_npz(dir: &Path) -> Result<(Option<Array1<i64>>, Option<i64>)> {
        let path = dir.join("03_alignment.npz");
        if !path.exists() {
            return Ok((None, None));
        }
        
        let mut npz = NpzReader::new(File::open(&path)?)?;
        
        let pred_dur: Option<Array1<i64>> = npz.by_name("pred_dur.npy").ok();
        let frames_arr: Option<Array1<i64>> = npz.by_name("frames.npy").ok();
        let frames = frames_arr.map(|arr| arr[0]);
        
        if let Some(ref arr) = pred_dur {
            println!("  Loaded pred_dur: {:?}", arr.shape());
        }
        if let Some(frames) = frames {
            println!("  Loaded frames: {}", frames);
        }
        
        Ok((pred_dur, frames))
    }
    
    fn load_f0n_npz(dir: &Path) -> Result<(Option<Array2<f32>>, Option<Array2<f32>>)> {
        let path = dir.join("04_f0n_out.npz");
        if !path.exists() {
            return Ok((None, None));
        }
        
        let mut npz = NpzReader::new(File::open(&path)?)?;
        
        let f0: Option<Array2<f32>> = npz.by_name("f0.npy").ok();
        let n: Option<Array2<f32>> = npz.by_name("n.npy").ok();
        
        if let Some(ref arr) = f0 {
            println!("  Loaded f0: {:?}", arr.shape());
        }
        if let Some(ref arr) = n {
            println!("  Loaded n: {:?}", arr.shape());
        }
        
        Ok((f0, n))
    }
}

/// Run validation comparing Rust outputs with golden tensors.
pub fn validate_against_golden(
    golden: &GoldenTensors,
    trace: &PipelineTrace,
) -> Result<bool> {
    println!("\n=== Validation Results ===\n");
    println!(
        "cur_len={} frames={} decoder_bucket_len={}",
        trace.cur_len, trace.frames, trace.decoder_bucket_len
    );

    let mut all_pass = true;
    let cur_len = trace.cur_len;
    const MAX_MAE_INTERMEDIATE: f32 = 1e-3;

    if let Some(ref golden_d_en) = golden.d_en {
        let max_len = cur_len
            .min(golden_d_en.shape()[2])
            .min(trace.d_en.shape()[2]);
        let golden_slice = golden_d_en.slice(s![.., .., ..max_len]);
        let trace_slice = trace.d_en.slice(s![.., .., ..max_len]);
        let pass = compare_f32(
            "bert.d_en[:cur_len]",
            &view_to_vec_f32(golden_slice),
            &view_to_vec_f32(trace_slice),
            MAX_MAE_INTERMEDIATE,
        );
        all_pass &= pass;
    } else {
        println!("bert.d_en: (no golden tensor to compare)");
    }

    if let Some(ref golden_logits) = golden.duration_logits {
        let golden_view = golden_logits
            .view()
            .into_dimensionality::<Ix3>()
            .map_err(|e| anyhow::anyhow!("Expected duration_logits to be 3D: {}", e))?;
        let trace_view = trace
            .duration_logits
            .view()
            .into_dimensionality::<Ix3>()
            .map_err(|e| anyhow::anyhow!("Expected duration_logits to be 3D: {}", e))?;
        let max_len = cur_len
            .min(golden_view.shape()[1])
            .min(trace_view.shape()[1]);
        let golden_slice = golden_view.slice(s![.., ..max_len, ..]);
        let trace_slice = trace_view.slice(s![.., ..max_len, ..]);
        let pass = compare_f32(
            "duration.logits[:cur_len]",
            &view_to_vec_f32(golden_slice),
            &view_to_vec_f32(trace_slice),
            MAX_MAE_INTERMEDIATE,
        );
        all_pass &= pass;
    } else {
        println!("duration.logits: (no golden tensor to compare)");
    }

    if let Some(ref golden_d) = golden.d {
        let max_len = cur_len
            .min(golden_d.shape()[1])
            .min(trace.d.shape()[1]);
        let golden_slice = golden_d.slice(s![.., ..max_len, ..]);
        let trace_slice = trace.d.slice(s![.., ..max_len, ..]);
        let pass = compare_f32(
            "duration.d[:cur_len]",
            &view_to_vec_f32(golden_slice),
            &view_to_vec_f32(trace_slice),
            MAX_MAE_INTERMEDIATE,
        );
        all_pass &= pass;
    } else {
        println!("duration.d: (no golden tensor to compare)");
    }

    if let Some(ref golden_pred_dur) = golden.pred_dur {
        let max_len = cur_len
            .min(golden_pred_dur.len())
            .min(trace.pred_dur.len());
        let golden_slice = golden_pred_dur.slice(s![..max_len]);
        let trace_slice = trace.pred_dur.slice(s![..max_len]);
        let pass = compare_i64(
            "alignment.pred_dur[:cur_len]",
            golden_slice.as_slice().unwrap_or(&[]),
            trace_slice.as_slice().unwrap_or(&[]),
        );
        all_pass &= pass;
    } else {
        println!("alignment.pred_dur: (no golden tensor to compare)");
    }

    if let Some(golden_frames) = golden.frames {
        let golden_frames = golden_frames.max(0) as usize;
        let pass = golden_frames == trace.frames;
        let status = if pass { "✓ PASS" } else { "✗ FAIL" };
        println!(
            "alignment.frames: {} - golden={} rust={}",
            status, golden_frames, trace.frames
        );
        all_pass &= pass;
    } else {
        println!("alignment.frames: (no golden frames to compare)");
    }

    let frames = golden
        .frames
        .map(|v| v.max(0) as usize)
        .unwrap_or(trace.frames)
        .min(trace.frames);
    let valid_f0n = frames.saturating_mul(2);

    if let Some(ref golden_f0) = golden.f0 {
        let max_len = valid_f0n
            .min(golden_f0.shape()[1])
            .min(trace.f0.shape()[1]);
        let golden_slice = golden_f0.slice(s![.., ..max_len]);
        let trace_slice = trace.f0.slice(s![.., ..max_len]);
        let pass = compare_f32(
            "f0[:2*frames]",
            &view_to_vec_f32(golden_slice),
            &view_to_vec_f32(trace_slice),
            MAX_MAE_INTERMEDIATE,
        );
        all_pass &= pass;
    } else {
        println!("f0: (no golden tensor to compare)");
    }

    if let Some(ref golden_n) = golden.n {
        let max_len = valid_f0n
            .min(golden_n.shape()[1])
            .min(trace.n.shape()[1]);
        let golden_slice = golden_n.slice(s![.., ..max_len]);
        let trace_slice = trace.n.slice(s![.., ..max_len]);
        let pass = compare_f32(
            "n[:2*frames]",
            &view_to_vec_f32(golden_slice),
            &view_to_vec_f32(trace_slice),
            MAX_MAE_INTERMEDIATE,
        );
        all_pass &= pass;
    } else {
        println!("n: (no golden tensor to compare)");
    }

    if let Some(ref golden_t_en) = golden.t_en {
        let max_len = cur_len
            .min(golden_t_en.shape()[2])
            .min(trace.t_en.shape()[2]);
        let golden_slice = golden_t_en.slice(s![.., .., ..max_len]);
        let trace_slice = trace.t_en.slice(s![.., .., ..max_len]);
        let pass = compare_f32(
            "text_encoder.t_en[:cur_len]",
            &view_to_vec_f32(golden_slice),
            &view_to_vec_f32(trace_slice),
            MAX_MAE_INTERMEDIATE,
        );
        all_pass &= pass;
    } else {
        println!("text_encoder.t_en: (no golden tensor to compare)");
    }

    // Compare final audio
    if let Some(ref golden_audio) = golden.audio {
        let stats = ComparisonStats::compute(golden_audio.as_slice().unwrap(), &trace.audio);
        let pass = stats.passes(0.01); // 1% tolerance for audio
        all_pass &= pass;

        let status = if pass { "✓ PASS" } else { "✗ FAIL" };
        println!("Audio: {} - {}", status, stats);

        // Compute correlation
        let corr = compute_correlation(golden_audio.as_slice().unwrap(), &trace.audio);
        println!("       correlation={:.4}", corr);
    } else {
        println!("Audio: (no golden audio to compare)");
    }

    println!();
    
    if all_pass {
        println!("=== All validations PASSED ===");
    } else {
        println!("=== Some validations FAILED ===");
    }
    
    Ok(all_pass)
}

/// Compute Pearson correlation coefficient between two arrays.
fn compute_correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return f32::NAN;
    }
    
    let n = a.len().min(b.len());
    if n < 2 {
        return f32::NAN;
    }
    
    // Compute means
    let mean_a: f64 = a[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mean_b: f64 = b[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    
    // Compute covariance and standard deviations
    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    
    for i in 0..n {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    
    if var_a < 1e-10 || var_b < 1e-10 {
        return f32::NAN;
    }
    
    (cov / (var_a.sqrt() * var_b.sqrt())) as f32
}

/// Save tensors to .npy files for inspection.
///
/// Creates files:
/// - 00_inputs.npy: input_ids as i64 array
/// - 00_ref_s.npy: style embedding as f32 array  
/// - 06_audio.npy: output audio as f32 array
pub fn save_tensors(
    output_dir: &Path,
    input_ids: &[i64],
    ref_s: &[f32],
    speed: f32,
    audio: &[f32],
) -> Result<()> {
    fs::create_dir_all(output_dir)
        .context(format!("Failed to create output directory {:?}", output_dir))?;
    
    println!("Saving tensors to {:?}...", output_dir);
    
    // Save input_ids
    let input_ids_arr = Array2::from_shape_vec((1, input_ids.len()), input_ids.to_vec())
        .context("Failed to create input_ids array")?;
    let input_ids_path = output_dir.join("00_inputs.npy");
    let input_ids_file = File::create(&input_ids_path)
        .context("Failed to create 00_inputs.npy")?;
    input_ids_arr.write_npy(input_ids_file)
        .context("Failed to write 00_inputs.npy")?;
    println!("  Saved input_ids: {:?}", input_ids_arr.shape());
    
    // Save ref_s
    let ref_s_arr = Array2::from_shape_vec((1, ref_s.len()), ref_s.to_vec())
        .context("Failed to create ref_s array")?;
    let ref_s_path = output_dir.join("00_ref_s.npy");
    let ref_s_file = File::create(&ref_s_path)
        .context("Failed to create 00_ref_s.npy")?;
    ref_s_arr.write_npy(ref_s_file)
        .context("Failed to write 00_ref_s.npy")?;
    println!("  Saved ref_s: {:?}", ref_s_arr.shape());
    
    // Save speed
    let speed_arr = Array1::from_vec(vec![speed]);
    let speed_path = output_dir.join("00_speed.npy");
    let speed_file = File::create(&speed_path)
        .context("Failed to create 00_speed.npy")?;
    speed_arr.write_npy(speed_file)
        .context("Failed to write 00_speed.npy")?;
    println!("  Saved speed: {}", speed);
    
    // Save audio
    let audio_arr = Array1::from_vec(audio.to_vec());
    let audio_path = output_dir.join("06_audio.npy");
    let audio_file = File::create(&audio_path)
        .context("Failed to create 06_audio.npy")?;
    audio_arr.write_npy(audio_file)
        .context("Failed to write 06_audio.npy")?;
    println!("  Saved audio: {} samples ({:.2}s)", audio.len(), audio.len() as f32 / 24000.0);
    
    Ok(())
}

/// Save a full trace in the same layout as validate_steps.py.
pub fn save_trace_tensors(
    output_dir: &Path,
    input_ids: &[i64],
    ref_s: &[f32],
    speed: f32,
    trace: &PipelineTrace,
) -> Result<()> {
    fs::create_dir_all(output_dir)
        .context(format!("Failed to create output directory {:?}", output_dir))?;

    println!("Saving trace tensors to {:?}...", output_dir);

    let input_ids_arr = Array2::from_shape_vec((1, input_ids.len()), input_ids.to_vec())
        .context("Failed to create input_ids array")?;
    let ref_s_arr = Array2::from_shape_vec((1, ref_s.len()), ref_s.to_vec())
        .context("Failed to create ref_s array")?;
    let speed_arr = Array1::from_vec(vec![speed]);

    let inputs_path = output_dir.join("00_inputs.npz");
    let mut inputs_npz = NpzWriter::new(File::create(&inputs_path)?);
    inputs_npz.add_array("input_ids.npy", &input_ids_arr)?;
    inputs_npz.add_array("ref_s.npy", &ref_s_arr)?;
    inputs_npz.add_array("speed.npy", &speed_arr)?;
    inputs_npz.finish()?;

    let bert_path = output_dir.join("01_bert_out.npy");
    trace
        .d_en
        .write_npy(File::create(&bert_path).context("Failed to create 01_bert_out.npy")?)?;

    let duration_path = output_dir.join("02_duration_out.npz");
    let mut duration_npz = NpzWriter::new(File::create(&duration_path)?);
    duration_npz.add_array("duration_logits.npy", &trace.duration_logits)?;
    duration_npz.add_array("d.npy", &trace.d)?;
    duration_npz.finish()?;

    let alignment_path = output_dir.join("03_alignment.npz");
    let mut alignment_npz = NpzWriter::new(File::create(&alignment_path)?);
    alignment_npz.add_array("pred_dur.npy", &trace.pred_dur)?;
    let frames_arr = Array1::from_vec(vec![trace.frames as i64]);
    alignment_npz.add_array("frames.npy", &frames_arr)?;
    alignment_npz.finish()?;

    let f0n_path = output_dir.join("04_f0n_out.npz");
    let mut f0n_npz = NpzWriter::new(File::create(&f0n_path)?);
    f0n_npz.add_array("f0.npy", &trace.f0)?;
    f0n_npz.add_array("n.npy", &trace.n)?;
    f0n_npz.finish()?;

    let t_en_path = output_dir.join("05_text_enc_out.npy");
    trace
        .t_en
        .write_npy(File::create(&t_en_path).context("Failed to create 05_text_enc_out.npy")?)?;

    let audio_arr = Array1::from_vec(trace.audio.clone());
    let audio_path = output_dir.join("06_audio.npy");
    audio_arr
        .write_npy(File::create(&audio_path).context("Failed to create 06_audio.npy")?)?;

    Ok(())
}
