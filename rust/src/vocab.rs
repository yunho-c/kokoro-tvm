//! Vocabulary loading and tokenization.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Vocabulary mapping from characters to token IDs.
#[derive(Debug, Deserialize)]
pub struct Vocab {
    #[serde(flatten)]
    map: HashMap<String, i64>,
}

impl Vocab {
    /// Load vocabulary from a JSON file.
    ///
    /// The JSON file should be a flat object mapping characters to token IDs:
    /// ```json
    /// { "a": 1, "b": 2, ... }
    /// ```
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open vocab file")?;
        let reader = BufReader::new(file);
        let vocab: Vocab = serde_json::from_reader(reader).context("Failed to parse vocab JSON")?;
        Ok(vocab)
    }

    /// Encode a phoneme string to token IDs.
    ///
    /// Adds start token (0) at the beginning and end token (0) at the end.
    /// Characters not in the vocabulary are skipped.
    pub fn encode(&self, phonemes: &str) -> Vec<i64> {
        let mut ids = Vec::with_capacity(phonemes.len() + 2);

        // Start token
        ids.push(0);

        // Encode each character
        for ch in phonemes.chars() {
            let key = ch.to_string();
            if let Some(&id) = self.map.get(&key) {
                ids.push(id);
            }
            // Skip characters not in vocabulary
        }

        // End token
        ids.push(0);

        ids
    }

    /// Get the vocabulary size.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_basic() {
        let mut map = HashMap::new();
        map.insert("h".to_string(), 1);
        map.insert("ə".to_string(), 2);
        map.insert("l".to_string(), 3);
        map.insert("ˈ".to_string(), 4);
        map.insert("o".to_string(), 5);
        map.insert("ʊ".to_string(), 6);

        let vocab = Vocab { map };
        let ids = vocab.encode("həlˈoʊ");

        // Start token + encoded chars + end token
        assert_eq!(ids[0], 0); // start
        assert_eq!(ids[ids.len() - 1], 0); // end
        assert!(ids.len() > 2);
    }
}
