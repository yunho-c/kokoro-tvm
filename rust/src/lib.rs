//! Kokoro TVM Inference Library
//!
//! A Rust library for running Kokoro TTS inference using TVM-compiled modules.

mod audio;
pub mod pipeline;
pub mod preprocessing;
pub mod validation;
pub mod vocab;
pub mod voice;

pub use audio::save_wav;
pub use pipeline::KokoroPipeline;
pub use preprocessing::{build_alignment, create_masks, pad_input_ids, sigmoid};
pub use vocab::Vocab;
pub use voice::VoicePack;

/// Static constants that must match compilation parameters
pub mod constants {
    pub const STATIC_TEXT_LEN: usize = 512;
    pub const STATIC_AUDIO_LEN: usize = 150;
    pub const STYLE_DIM: usize = 128;
    pub const SAMPLES_PER_FRAME: usize = 600;
    pub const SAMPLE_RATE: u32 = 24000;
}
