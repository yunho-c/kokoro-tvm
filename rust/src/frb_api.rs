//! Flutter Rust Bridge exports.

use crate::runtime;
use crate::{
    AudioChunk, CancelToken, RuntimeStatus, SynthesisRequest, SynthesisResult, TtsError, VoiceInfo,
};
use crate::frb_generated::StreamSink;
use std::env;

#[flutter_rust_bridge::frb]
pub fn frb_init(
    artifacts_dir: String,
    device: String,
    vocab_path: String,
    voice_path: String,
) -> Result<(), TtsError> {
    if env::var_os("RUST_BACKTRACE").is_none() {
        env::set_var("RUST_BACKTRACE", "1");
    }
    if env::var_os("RUST_LIB_BACKTRACE").is_none() {
        env::set_var("RUST_LIB_BACKTRACE", "1");
    }
    runtime::init_from_paths(artifacts_dir, device, vocab_path, voice_path).map_err(TtsError::from)
}

#[flutter_rust_bridge::frb]
pub fn frb_warmup() -> Result<(), TtsError> {
    runtime::warmup().map_err(TtsError::from)
}

#[flutter_rust_bridge::frb]
pub fn frb_synthesize(request: SynthesisRequest) -> Result<SynthesisResult, TtsError> {
    runtime::synthesize(request).map_err(TtsError::from)
}

#[flutter_rust_bridge::frb]
pub fn frb_synthesize_stream(
    request: SynthesisRequest,
    cancel_token: CancelToken,
    sink: StreamSink<AudioChunk>,
) -> Result<(), TtsError> {
    runtime::synthesize_stream(request, cancel_token, sink).map_err(TtsError::from)
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
pub fn frb_shutdown() -> Result<(), TtsError> {
    runtime::shutdown().map_err(TtsError::from)
}

#[flutter_rust_bridge::frb]
pub fn frb_status() -> Result<RuntimeStatus, TtsError> {
    runtime::status().map_err(TtsError::from)
}

#[flutter_rust_bridge::frb]
pub fn frb_get_voices() -> Result<Vec<VoiceInfo>, TtsError> {
    runtime::get_voices().map_err(TtsError::from)
}

#[flutter_rust_bridge::frb]
pub fn frb_get_languages() -> Result<Vec<String>, TtsError> {
    runtime::get_languages().map_err(TtsError::from)
}
