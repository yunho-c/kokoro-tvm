//! VoiRS G2P integration and Kokoro phoneme normalization.

use anyhow::Result;
use crate::Vocab;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LanguageCode {
    EnUs,
    EnGb,
}

pub trait G2pBackend: Send + Sync {
    fn text_to_kokoro_phonemes(
        &self,
        text: &str,
        language: LanguageCode,
        vocab: &Vocab,
    ) -> Result<String>;
}

pub struct G2pEngine {
    backend: Box<dyn G2pBackend>,
}

impl G2pEngine {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "g2p-voirs")]
        {
            Ok(Self {
                backend: Box::new(VoiRsBackend::new()?),
            })
        }
        #[cfg(not(feature = "g2p-voirs"))]
        {
            Ok(Self {
                backend: Box::new(DisabledG2pBackend),
            })
        }
    }

    pub fn text_to_kokoro_phonemes(
        &self,
        text: &str,
        language: LanguageCode,
        vocab: &Vocab,
    ) -> Result<String> {
        self.backend.text_to_kokoro_phonemes(text, language, vocab)
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

struct DisabledG2pBackend;

impl G2pBackend for DisabledG2pBackend {
    fn text_to_kokoro_phonemes(
        &self,
        _text: &str,
        _language: LanguageCode,
        _vocab: &Vocab,
    ) -> Result<String> {
        anyhow::bail!("G2P backend is disabled; enable the g2p-voirs feature")
    }
}

#[cfg(feature = "g2p-voirs")]
struct VoiRsBackend {
    inner: voirs_g2p::rules::EnglishRuleG2p,
}

#[cfg(feature = "g2p-voirs")]
impl VoiRsBackend {
    fn new() -> Result<Self> {
        let inner = voirs_g2p::rules::EnglishRuleG2p::new()
            .map_err(|err| anyhow::anyhow!("G2P init failed: {err}"))?;
        Ok(Self { inner })
    }
}

#[cfg(feature = "g2p-voirs")]
impl G2pBackend for VoiRsBackend {
    fn text_to_kokoro_phonemes(
        &self,
        text: &str,
        language: LanguageCode,
        vocab: &Vocab,
    ) -> Result<String> {
        let lang = match language {
            LanguageCode::EnUs => voirs_g2p::LanguageCode::EnUs,
            LanguageCode::EnGb => voirs_g2p::LanguageCode::EnGb,
        };
        let phonemes = futures::executor::block_on(
            voirs_g2p::G2p::to_phonemes(&self.inner, text, Some(lang)),
        )
        .map_err(|err| anyhow::anyhow!("G2P conversion failed: {err}"))?;
        let raw = phonemes_to_kokoro_string(&phonemes);
        let (filtered, dropped) = vocab.filter_to_vocab(&raw);
        if dropped > 0 {
            let total = raw.chars().count();
            let ratio = if total > 0 {
                (dropped as f32 / total as f32) * 100.0
            } else {
                0.0
            };
            eprintln!(
                "G2P dropped {} non-vocab symbols ({:.1}% of {}).",
                dropped, ratio, total
            );
        }
        if filtered.is_empty() {
            anyhow::bail!("G2P produced no vocab symbols for input");
        }
        Ok(filtered)
    }
}

#[cfg(feature = "g2p-voirs")]
fn phonemes_to_kokoro_string(phonemes: &[voirs_g2p::Phoneme]) -> String {
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
