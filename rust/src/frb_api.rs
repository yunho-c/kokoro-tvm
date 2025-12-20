//! Flutter Rust Bridge exports.

use crate::runtime;
use crate::SynthesisResult;

#[flutter_rust_bridge::frb]
pub fn frb_init(
    artifacts_dir: String,
    device: String,
    vocab_path: String,
    voice_path: String,
) -> Result<(), String> {
    runtime::init_from_paths(artifacts_dir, device, vocab_path, voice_path)
        .map_err(|err| err.to_string())
}

#[flutter_rust_bridge::frb]
pub fn frb_warmup() -> Result<(), String> {
    runtime::warmup().map_err(|err| err.to_string())
}

#[flutter_rust_bridge::frb]
pub fn frb_synthesize(phonemes: String, speed: f32) -> Result<SynthesisResult, String> {
    runtime::synthesize(&phonemes, speed).map_err(|err| err.to_string())
}

#[flutter_rust_bridge::frb]
pub fn frb_synthesize_with_voice_index(
    phonemes: String,
    speed: f32,
    voice_index: Option<u32>,
) -> Result<SynthesisResult, String> {
    let index = voice_index.map(|value| value as usize);
    runtime::synthesize_with_voice_index(&phonemes, speed, index)
        .map_err(|err| err.to_string())
}

#[flutter_rust_bridge::frb]
pub fn frb_shutdown() -> Result<(), String> {
    runtime::shutdown().map_err(|err| err.to_string())
}
