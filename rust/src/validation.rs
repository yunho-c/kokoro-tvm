//! Validation utilities for comparing Rust pipeline outputs with Python golden tensors.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayD};
use ndarray_npy::{NpzReader, ReadNpyExt, WriteNpyExt};
use std::fs::{self, File};
use std::path::Path;

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
    rust_audio: &[f32],
) -> Result<bool> {
    println!("\n=== Validation Results ===\n");
    
    let mut all_pass = true;
    
    // Compare final audio
    if let Some(ref golden_audio) = golden.audio {
        let stats = ComparisonStats::compute(golden_audio.as_slice().unwrap(), rust_audio);
        let pass = stats.passes(0.01); // 1% tolerance for audio
        all_pass &= pass;
        
        let status = if pass { "✓ PASS" } else { "✗ FAIL" };
        println!("Audio: {} - {}", status, stats);
        
        // Compute correlation
        let corr = compute_correlation(golden_audio.as_slice().unwrap(), rust_audio);
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
