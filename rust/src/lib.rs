//! Kokoro TVM Inference Library
//!
//! A Rust library for running Kokoro TTS inference using TVM-compiled modules.

#[cfg(feature = "frb")]
mod frb_generated;

mod audio;
pub mod error;
pub mod g2p;
pub mod pipeline;
pub mod preprocessing;
pub mod runtime;
pub mod validation;
pub mod vocab;
pub mod voice;
#[cfg(feature = "frb")]
pub mod frb_api;

pub use audio::save_wav;
pub use error::TtsError;
pub use pipeline::{KokoroPipeline, PipelineTrace};
pub use preprocessing::{build_alignment, build_alignment_with_pred, create_masks, pad_input_ids, sigmoid};
#[doc(hidden)]
pub use std::path::PathBuf;
pub use runtime::{
    AudioChunk,
    CancelToken,
    get_languages,
    get_voices,
    init,
    init_from_paths,
    shutdown,
    synthesize,
    synthesize_text,
    synthesize_text_with_voice_index,
    synthesize_with_voice_index,
    status,
    warmup,
    RuntimeConfig,
    RuntimeStatus,
    SynthesisResult,
};
#[cfg(feature = "frb")]
pub use runtime::synthesize_stream;
#[cfg(feature = "frb")]
pub use frb_api::*;
pub use vocab::Vocab;
pub use voice::{load_voice_manifest, VoiceInfo, VoiceManifest, VoicePack};

/// Static constants that must match compilation parameters
pub mod constants {
    pub const STATIC_TEXT_LEN: usize = 512;
    pub const STATIC_AUDIO_LEN: usize = 512;
    // pub const STATIC_AUDIO_LEN: usize = 5120;
    pub const STYLE_DIM: usize = 128;
    pub const SAMPLES_PER_FRAME: usize = 600;
    pub const SAMPLE_RATE: u32 = 24000;
}
