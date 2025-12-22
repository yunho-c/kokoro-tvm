//! Structured error type for public API surfaces.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TtsError {
    NotInitialized { message: String },
    InvalidInput { message: String },
    G2p { message: String },
    Voice { message: String },
    Io { message: String },
    Internal { message: String },
}

impl TtsError {
    pub fn from_anyhow(err: anyhow::Error) -> Self {
        let message = err.to_string();
        if message.contains("Runtime is not initialized") {
            return TtsError::NotInitialized { message };
        }
        if message.contains("G2P")
            || message.contains("Unsupported language")
            || message.contains("G2p")
        {
            return TtsError::G2p { message };
        }
        if message.contains("Voice index")
            || message.contains("voice pack")
            || message.contains("voice manifest")
        {
            return TtsError::Voice { message };
        }
        if message.contains("Invalid")
            || message.contains("chunk_size_ms")
            || message.contains("vocab symbols")
        {
            return TtsError::InvalidInput { message };
        }
        if message.contains("Failed to open")
            || message.contains("Failed to load")
            || message.contains("Failed to read")
        {
            return TtsError::Io { message };
        }
        TtsError::Internal { message }
    }
}

impl From<anyhow::Error> for TtsError {
    fn from(err: anyhow::Error) -> Self {
        Self::from_anyhow(err)
    }
}
