//! Voice pack loading and style selection.

use anyhow::{Context, Result};
use ndarray::{s, Array2, Array3, ArrayD};
use ndarray_npy::ReadNpyExt;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceInfo {
    #[serde(default)]
    pub id: String,
    pub name: Option<String>,
    pub language: Option<String>,
    pub gender: Option<String>,
    pub quality: Option<String>,
    pub tags: Option<Vec<String>>,
    pub index: Option<usize>,
}

impl VoiceInfo {
    pub fn with_index(id: String, index: usize) -> Self {
        Self {
            id,
            name: Some(format!("Voice {index}")),
            language: None,
            gender: None,
            quality: None,
            tags: None,
            index: Some(index),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VoiceManifest {
    voices: Vec<VoiceInfo>,
}

impl VoiceManifest {
    pub fn voices(&self) -> &[VoiceInfo] {
        &self.voices
    }

    pub fn languages(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for voice in &self.voices {
            if let Some(language) = voice.language.as_ref() {
                if seen.insert(language.clone()) {
                    out.push(language.clone());
                }
            }
        }
        out
    }

    pub fn load(path: &Path, voice_count: usize) -> Result<Self> {
        let file = File::open(path).context("Failed to open voice manifest")?;
        let value: serde_json::Value =
            serde_json::from_reader(file).context("Failed to parse voice manifest JSON")?;
        let mut voices = parse_manifest_value(value)?;
        assign_indices(&mut voices, voice_count);
        enrich_voice_info(&mut voices);
        Ok(Self { voices })
    }
}

pub fn load_voice_manifest(voice_path: &Path, voice_count: usize) -> Result<Option<VoiceManifest>> {
    let candidates = manifest_candidates(voice_path);
    for path in candidates {
        if path.exists() {
            return Ok(Some(VoiceManifest::load(&path, voice_count)?));
        }
    }
    Ok(None)
}

fn manifest_candidates(voice_path: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(voice_path.with_extension("json"));
    if let Some(parent) = voice_path.parent() {
        candidates.push(parent.join("voices.json"));
        candidates.push(parent.join("voice_manifest.json"));
    }
    candidates
}

fn parse_manifest_value(value: serde_json::Value) -> Result<Vec<VoiceInfo>> {
    match value {
        serde_json::Value::Array(values) => values
            .into_iter()
            .map(|entry| {
                let mut info: VoiceInfo =
                    serde_json::from_value(entry).context("Invalid voice manifest entry")?;
                if info.id.is_empty() {
                    anyhow::bail!("Voice manifest entry is missing id");
                }
                Ok(info)
            })
            .collect(),
        serde_json::Value::Object(map) => {
            if let Some(voices_value) = map.get("voices") {
                return parse_manifest_value(voices_value.clone());
            }
            let mut voices = Vec::new();
            for (id, entry) in map {
                let mut info: VoiceInfo =
                    serde_json::from_value(entry).context("Invalid voice manifest entry")?;
                if info.id.is_empty() {
                    info.id = id;
                }
                voices.push(info);
            }
            Ok(voices)
        }
        _ => anyhow::bail!("Voice manifest must be a list or object"),
    }
}

fn assign_indices(voices: &mut [VoiceInfo], voice_count: usize) {
    let mut used = HashSet::new();
    for voice in voices.iter_mut() {
        if let Some(index) = voice.index {
            if index < voice_count {
                used.insert(index);
            } else {
                voice.index = None;
            }
        }
    }

    let mut next_index = 0usize;
    for voice in voices.iter_mut() {
        if voice.index.is_some() {
            continue;
        }
        while used.contains(&next_index) && next_index < voice_count {
            next_index += 1;
        }
        if next_index >= voice_count {
            break;
        }
        voice.index = Some(next_index);
        used.insert(next_index);
        next_index += 1;
    }
}

fn enrich_voice_info(voices: &mut [VoiceInfo]) {
    for voice in voices.iter_mut() {
        let (language, gender, name) = parse_voice_id(&voice.id);
        if voice.language.is_none() {
            voice.language = language;
        }
        if voice.gender.is_none() {
            voice.gender = gender;
        }
        if voice.name.is_none() {
            voice.name = name;
        }
    }
}

fn parse_voice_id(id: &str) -> (Option<String>, Option<String>, Option<String>) {
    let mut parts = id.splitn(2, '_');
    let prefix = parts.next().unwrap_or("");
    let suffix = parts.next();

    let language = prefix
        .chars()
        .next()
        .and_then(|ch| match ch {
            'a' => Some("en-US"),
            'b' => Some("en-GB"),
            'j' => Some("ja"),
            'z' => Some("zh-CN"),
            'e' => Some("es"),
            'f' => Some("fr-FR"),
            'h' => Some("hi"),
            'i' => Some("it"),
            'p' => Some("pt-BR"),
            _ => None,
        })
        .map(|value| value.to_string());

    let gender = prefix
        .chars()
        .nth(1)
        .and_then(|ch| match ch {
            'f' => Some("female"),
            'm' => Some("male"),
            _ => None,
        })
        .map(|value| value.to_string());

    let name = suffix.map(|value| value.replace('-', " ").replace('_', " ").trim().to_string());

    (language, gender, name)
}

/// Voice pack containing style embeddings.
///
/// Each voice pack is a 2D array of shape `[N, 256]` where N varies by pack.
/// The first 128 dimensions are the style vector, the last 128 are additional style info.
pub struct VoicePack {
    data: Array2<f32>,
}

impl VoicePack {
    /// Load a voice pack from a .npy file.
    ///
    /// Handles both 2D `[N, 256]` and 3D `[N, 1, 256]` arrays (squeezes middle dim).
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open voice pack file")?;

        // Try to load as dynamic array first to handle different shapes
        let data_dyn: ArrayD<f32> =
            ArrayD::read_npy(file).context("Failed to read voice pack .npy")?;

        let shape = data_dyn.shape();

        // Handle different shapes
        let data: Array2<f32> = match shape.len() {
            2 => {
                // Already 2D: [N, 256]
                data_dyn
                    .into_dimensionality()
                    .context("Failed to convert to 2D array")?
            }
            3 => {
                // 3D: [N, 1, 256] - squeeze middle dimension
                if shape[1] != 1 {
                    anyhow::bail!(
                        "3D voice pack should have shape [N, 1, 256], got {:?}",
                        shape
                    );
                }
                let arr3: Array3<f32> = data_dyn
                    .into_dimensionality()
                    .context("Failed to convert to 3D array")?;
                // Remove axis 1 to get [N, 256]
                arr3.index_axis_move(ndarray::Axis(1), 0)
            }
            _ => {
                anyhow::bail!(
                    "Voice pack must be 2D [N, 256] or 3D [N, 1, 256], got shape {:?}",
                    shape
                );
            }
        };

        if data.ncols() != 256 {
            anyhow::bail!("Voice pack must have 256 columns, got {}", data.ncols());
        }

        println!("  Loaded voice pack: {} entries, shape {:?}", data.nrows(), data.shape());

        Ok(Self { data })
    }

    /// Select a style embedding based on phoneme length.
    ///
    /// Returns a `[1, 256]` array containing the selected style embedding.
    /// Selection logic: `pack[min(phoneme_len - 1, pack.nrows() - 1)]`
    pub fn select_style(&self, phoneme_len: usize) -> Array2<f32> {
        let index = if phoneme_len == 0 {
            0
        } else {
            (phoneme_len - 1).min(self.data.nrows() - 1)
        };

        // Get the row and reshape to [1, 256]
        let row = self.data.slice(s![index, ..]);
        row.insert_axis(ndarray::Axis(0)).to_owned()
    }

    /// Select a style embedding by explicit index.
    ///
    /// Returns a `[1, 256]` array containing the selected style embedding.
    pub fn select_style_by_index(&self, index: usize) -> Result<Array2<f32>> {
        if self.data.is_empty() {
            anyhow::bail!("Voice pack is empty");
        }
        if index >= self.data.nrows() {
            anyhow::bail!(
                "Voice index {} out of range (max {})",
                index,
                self.data.nrows().saturating_sub(1)
            );
        }

        let row = self.data.slice(s![index, ..]);
        Ok(row.insert_axis(ndarray::Axis(0)).to_owned())
    }

    /// Get the number of style entries in this voice pack.
    pub fn len(&self) -> usize {
        self.data.nrows()
    }

    /// Check if voice pack is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_select_style() {
        // Create a mock voice pack
        let data = Array2::<f32>::zeros((100, 256));
        let pack = VoicePack { data };

        // Test various phoneme lengths
        let style = pack.select_style(0);
        assert_eq!(style.shape(), &[1, 256]);

        let style = pack.select_style(50);
        assert_eq!(style.shape(), &[1, 256]);

        let style = pack.select_style(1000); // Larger than pack size
        assert_eq!(style.shape(), &[1, 256]);
    }
}
