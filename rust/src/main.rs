//! CLI entry point for Kokoro TVM inference.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

use kokoro_tvm::{constants::SAMPLE_RATE, save_wav, KokoroPipeline, Vocab, VoicePack};
use kokoro_tvm::validation::{save_trace_tensors, GoldenTensors, validate_against_golden};

#[derive(Parser, Debug)]
#[command(name = "kokoro-tvm")]
#[command(about = "Kokoro TTS inference using TVM-compiled modules")]
struct Args {
    /// IPA phoneme string to synthesize
    #[arg(short, long)]
    phonemes: String,

    /// Path to voice pack .npy file
    #[arg(short, long)]
    voice: PathBuf,

    /// Path to vocab.json file
    #[arg(long)]
    vocab: PathBuf,

    /// Directory containing compiled TVM modules (.so/.dylib)
    #[arg(short, long, default_value = "tvm_output")]
    lib_dir: PathBuf,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Speech speed multiplier (1.0 = normal speed)
    #[arg(short, long, default_value = "1.0")]
    speed: f32,

    /// Target device: llvm, metal, or cuda
    #[arg(long, default_value = "llvm")]
    device: String,

    /// Directory containing golden tensors from Python for validation
    #[arg(long)]
    golden_dir: Option<PathBuf>,

    /// Save intermediate tensors to this directory for inspection
    #[arg(long)]
    save_tensors: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load vocabulary
    println!("Loading vocabulary from {:?}...", args.vocab);
    let vocab = Vocab::load(&args.vocab).context("Failed to load vocabulary")?;

    // Load voice pack
    println!("Loading voice pack from {:?}...", args.voice);
    let voice_pack = VoicePack::load(&args.voice).context("Failed to load voice pack")?;

    // Encode phonemes to token IDs
    println!("Encoding phonemes: {:?}", args.phonemes);
    let input_ids = vocab.encode(&args.phonemes);
    println!("Token IDs: {:?} (len={})", &input_ids[..input_ids.len().min(20)], input_ids.len());

    // Select style embedding based on phoneme length
    let phoneme_len = args.phonemes.chars().count();
    let ref_s = voice_pack.select_style(phoneme_len);
    println!("Selected style embedding for phoneme_len={}", phoneme_len);

    // Load TVM modules and run inference
    println!("Loading TVM modules from {:?}...", args.lib_dir);
    let pipeline = KokoroPipeline::load(&args.lib_dir, &args.device)
        .context("Failed to load TVM pipeline")?;

    println!("Running inference (speed={})...", args.speed);
    let trace = pipeline
        .forward_trace(&input_ids, ref_s.as_slice().unwrap(), args.speed)
        .context("Inference failed")?;

    // Save audio
    println!("Saving audio to {:?}...", args.output);
    save_wav(&trace.audio, &args.output, SAMPLE_RATE).context("Failed to save WAV")?;

    let duration_secs = trace.audio.len() as f32 / SAMPLE_RATE as f32;
    println!("Done! Generated {:.2}s of audio.", duration_secs);

    // Save tensors if requested
    if let Some(ref tensor_dir) = args.save_tensors {
        println!();
        save_trace_tensors(
            tensor_dir,
            &input_ids,
            ref_s.as_slice().unwrap(),
            args.speed,
            &trace,
        ).context("Failed to save tensors")?;
    }

    // Run validation if golden tensors provided
    if let Some(ref golden_dir) = args.golden_dir {
        println!();
        let golden = GoldenTensors::load(golden_dir)
            .context("Failed to load golden tensors")?;
        
        let passed = validate_against_golden(&golden, &trace)
            .context("Validation failed")?;
        
        if !passed {
            std::process::exit(1);
        }
    }

    Ok(())
}
