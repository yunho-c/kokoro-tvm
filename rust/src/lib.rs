mod frb_generated; /* AUTO INJECTED BY flutter_rust_bridge. This line may not be accurate, and you can change it according to your needs. */
//! Kokoro TVM Inference Library
//!
//! A Rust library for running Kokoro TTS inference using TVM-compiled modules.

mod audio;
pub mod pipeline;
pub mod preprocessing;
pub mod runtime;
pub mod validation;
pub mod vocab;
pub mod voice;
#[cfg(feature = "frb")]
pub mod frb_api;

pub use audio::save_wav;
pub use pipeline::{KokoroPipeline, PipelineTrace};
pub use preprocessing::{build_alignment, build_alignment_with_pred, create_masks, pad_input_ids, sigmoid};
pub use runtime::{
    init,
    init_from_paths,
    shutdown,
    synthesize,
    synthesize_with_voice_index,
    status,
    warmup,
    RuntimeConfig,
    RuntimeStatus,
    SynthesisResult,
};
#[cfg(feature = "frb")]
pub use frb_api::*;
pub use vocab::Vocab;
pub use voice::VoicePack;

/// Static constants that must match compilation parameters
pub mod constants {
    pub const STATIC_TEXT_LEN: usize = 512;
    // pub const STATIC_AUDIO_LEN: usize = 150;
    pub const STATIC_AUDIO_LEN: usize = 5120;
    pub const STYLE_DIM: usize = 128;
    pub const SAMPLES_PER_FRAME: usize = 600;
    pub const SAMPLE_RATE: u32 = 24000;
}
