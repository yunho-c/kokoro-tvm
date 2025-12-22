//! Flutter Rust Bridge exports.

use crate::runtime;
use crate::{AudioChunk, CancelToken, RuntimeStatus, SynthesisResult};
use crate::frb_generated::StreamSink;
use std::env;

#[flutter_rust_bridge::frb]
pub fn frb_init(
    artifacts_dir: String,
    device: String,
    vocab_path: String,
    voice_path: String,
) -> Result<(), String> {
    if env::var_os("RUST_BACKTRACE").is_none() {
        env::set_var("RUST_BACKTRACE", "1");
    }
    if env::var_os("RUST_LIB_BACKTRACE").is_none() {
        env::set_var("RUST_LIB_BACKTRACE", "1");
    }
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
pub fn frb_synthesize_text(
    text: String,
    speed: f32,
    voice_index: Option<u32>,
    language: Option<String>,
) -> Result<SynthesisResult, String> {
    let index = voice_index.map(|value| value as usize);
    runtime::synthesize_text_with_voice_index(&text, speed, index, language.as_deref())
        .map_err(|err| err.to_string())
}

#[flutter_rust_bridge::frb]
pub fn frb_cancel_token_new() -> CancelToken {
    CancelToken::new()
}

#[flutter_rust_bridge::frb]
pub fn frb_cancel_token_cancel(token: CancelToken) {
    token.cancel();
}

#[flutter_rust_bridge::frb]
pub fn frb_synthesize_stream(
    phonemes: String,
    speed: f32,
    voice_index: Option<u32>,
    chunk_size_ms: u32,
    cancel_token: CancelToken,
    sink: StreamSink<AudioChunk>,
) -> Result<(), String> {
    let index = voice_index.map(|value| value as usize);
    runtime::synthesize_stream(
        &phonemes,
        speed,
        index,
        chunk_size_ms,
        cancel_token,
        sink,
    )
    .map_err(|err| err.to_string())
}

#[flutter_rust_bridge::frb]
pub fn frb_shutdown() -> Result<(), String> {
    runtime::shutdown().map_err(|err| err.to_string())
}

#[flutter_rust_bridge::frb]
pub fn frb_status() -> Result<RuntimeStatus, String> {
    runtime::status().map_err(|err| err.to_string())
}
