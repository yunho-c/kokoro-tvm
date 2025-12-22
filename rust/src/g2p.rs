//! VoiRS G2P integration and Kokoro phoneme normalization.

use anyhow::Result;
use futures::executor::block_on;
use voirs_g2p::rules::EnglishRuleG2p;
use voirs_g2p::{G2p, LanguageCode, Phoneme};

use crate::Vocab;

pub struct G2pEngine {
    inner: EnglishRuleG2p,
}

impl G2pEngine {
    pub fn new() -> Result<Self> {
        let inner =
            EnglishRuleG2p::new().map_err(|err| anyhow::anyhow!("G2P init failed: {err}"))?;
        Ok(Self { inner })
    }

    pub fn text_to_kokoro_phonemes(
        &self,
        text: &str,
        language: LanguageCode,
        vocab: &Vocab,
    ) -> Result<String> {
        let phonemes =
            block_on(self.inner.to_phonemes(text, Some(language))).map_err(|err| {
                anyhow::anyhow!("G2P conversion failed: {err}")
            })?;
        let raw = phonemes_to_kokoro_string(&phonemes);
        let (filtered, _dropped) = vocab.filter_to_vocab(&raw);
        if filtered.is_empty() {
            anyhow::bail!("G2P produced no vocab symbols for input");
        }
        Ok(filtered)
    }
}

pub fn parse_language(code: Option<&str>) -> Result<LanguageCode> {
    let normalized = code.unwrap_or("en-US").trim().to_ascii_lowercase();
    match normalized.as_str() {
        "en-us" | "en_us" | "en" => Ok(LanguageCode::EnUs),
        "en-gb" | "en_gb" | "en-uk" | "en_uk" => Ok(LanguageCode::EnGb),
        "" => Ok(LanguageCode::EnUs),
        other => anyhow::bail!("Unsupported language code: {}", other),
    }
}

fn phonemes_to_kokoro_string(phonemes: &[Phoneme]) -> String {
    let mut output = String::new();
    for phoneme in phonemes {
        let symbol = phoneme.effective_symbol();
        output.push_str(map_symbol(symbol));
    }
    output
}

fn map_symbol(symbol: &str) -> &str {
    match symbol {
        "tʃ" => "ʧ",
        "dʒ" => "ʤ",
        "ts" => "ʦ",
        "dz" => "ʣ",
        "tɕ" => "ʨ",
        "dʑ" => "ʥ",
        _ => symbol,
    }
}
