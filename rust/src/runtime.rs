//! Runtime wrapper for long-lived inference sessions.

use crate::constants::SAMPLE_RATE;
use crate::g2p::{parse_language, G2pEngine};
use crate::{load_voice_manifest, KokoroPipeline, Vocab, VoiceInfo, VoiceManifest, VoicePack};
use anyhow::{Context, Result};
#[cfg(feature = "frb")]
use crate::frb_generated::StreamSink;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc,
    Arc,
    OnceLock,
};
use std::thread;

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub artifacts_dir: PathBuf,
    pub device: String,
    pub vocab_path: PathBuf,
    pub voice_path: PathBuf,
}

#[cfg_attr(feature = "frb", flutter_rust_bridge::frb(opaque))]
#[derive(Clone, Debug)]
pub struct CancelToken {
    cancelled: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Debug)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub is_final: bool,
    pub chunk_index: u32,
    pub start_sample: u64,
}

impl RuntimeConfig {
    pub fn new(
        artifacts_dir: impl Into<PathBuf>,
        device: impl Into<String>,
        vocab_path: impl Into<PathBuf>,
        voice_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            artifacts_dir: artifacts_dir.into(),
            device: device.into(),
            vocab_path: vocab_path.into(),
            voice_path: voice_path.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SynthesisResult {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Clone, Debug)]
pub struct RuntimeStatus {
    pub initialized: bool,
    pub device: Option<String>,
    pub artifacts_dir: Option<PathBuf>,
    pub vocab_path: Option<PathBuf>,
    pub voice_path: Option<PathBuf>,
}

pub const DEFAULT_SPEED: f32 = 1.0;
pub const DEFAULT_CHUNK_SIZE_MS: u32 = 250;

#[derive(Clone, Debug)]
pub enum SynthesisInput {
    Text { text: String, language: Option<String> },
    Phonemes { phonemes: String },
}

#[derive(Clone, Debug)]
pub enum VoiceSelection {
    Auto,
    Id(String),
    Index(u32),
}

#[derive(Clone, Debug)]
pub struct SynthesisRequest {
    pub input: SynthesisInput,
    pub voice: VoiceSelection,
    pub speed: Option<f32>,
    pub chunk_size_ms: Option<u32>,
}

enum RuntimeCommand {
    Init {
        config: RuntimeConfig,
        reply: mpsc::Sender<Result<(), String>>,
    },
    Warmup {
        reply: mpsc::Sender<Result<(), String>>,
    },
    Synthesize {
        request: SynthesisRequest,
        reply: mpsc::Sender<Result<SynthesisResult, String>>,
    },
    GetVoices {
        reply: mpsc::Sender<Result<Vec<VoiceInfo>, String>>,
    },
    GetLanguages {
        reply: mpsc::Sender<Result<Vec<String>, String>>,
    },
    #[cfg(feature = "frb")]
    SynthesizeStream {
        request: SynthesisRequest,
        cancel_token: CancelToken,
        sink: StreamSink<AudioChunk>,
        reply: mpsc::Sender<Result<(), String>>,
    },
    Status {
        reply: mpsc::Sender<Result<RuntimeStatus, String>>,
    },
    Reset {
        reply: mpsc::Sender<Result<(), String>>,
    },
}

struct RuntimeWorker {
    config: Option<RuntimeConfig>,
    pipeline: Option<KokoroPipeline>,
    vocab: Option<Vocab>,
    voice_pack: Option<VoicePack>,
    voice_manifest: Option<VoiceManifest>,
    voice_index_by_id: HashMap<String, usize>,
    g2p: OnceLock<G2pEngine>,
}

impl RuntimeWorker {
    fn new() -> Self {
        Self {
            config: None,
            pipeline: None,
            vocab: None,
            voice_pack: None,
            voice_manifest: None,
            voice_index_by_id: HashMap::new(),
            g2p: OnceLock::new(),
        }
    }

    fn init(&mut self, config: RuntimeConfig) -> Result<()> {
        let vocab = Vocab::load(&config.vocab_path).context("Failed to load vocab")?;
        let voice_pack = VoicePack::load(&config.voice_path).context("Failed to load voice pack")?;
        let pipeline = KokoroPipeline::load_from_artifacts_dir(
            &config.artifacts_dir,
            &config.device,
        )
        .context("Failed to load TVM pipeline")?;

        let voice_manifest = load_voice_manifest(&config.voice_path, voice_pack.len())?;
        let mut voice_index_by_id = HashMap::new();
        if let Some(manifest) = voice_manifest.as_ref() {
            for voice in manifest.voices() {
                if let Some(index) = voice.index {
                    voice_index_by_id.entry(voice.id.clone()).or_insert(index);
                }
            }
        }

        self.config = Some(config);
        self.pipeline = Some(pipeline);
        self.vocab = Some(vocab);
        self.voice_pack = Some(voice_pack);
        self.voice_manifest = voice_manifest;
        self.voice_index_by_id = voice_index_by_id;
        Ok(())
    }

    fn warmup(&mut self) -> Result<()> {
        if self.pipeline.is_none() {
            anyhow::bail!("Runtime is not initialized");
        }

        let vocab = self.vocab.as_ref().context("Runtime is not initialized")?;
        let voice_pack = self
            .voice_pack
            .as_ref()
            .context("Runtime is not initialized")?;
        let pipeline = self
            .pipeline
            .as_ref()
            .context("Runtime is not initialized")?;

        let phonemes = "a";
        let input_ids = vocab.encode(phonemes);
        let ref_s = voice_pack.select_style_by_index(0)?;
        let ref_s_slice = ref_s
            .as_slice()
            .context("Style embedding must be contiguous")?;
        let _ = pipeline.forward(&input_ids, ref_s_slice, 1.0)?;
        Ok(())
    }

    fn synthesize(&mut self, request: &SynthesisRequest) -> Result<SynthesisResult> {
        let pipeline = self
            .pipeline
            .as_ref()
            .context("Runtime is not initialized")?;
        let vocab = self.vocab.as_ref().context("Runtime is not initialized")?;
        let voice_pack = self
            .voice_pack
            .as_ref()
            .context("Runtime is not initialized")?;

        let phonemes = self.phonemes_from_request(&request.input, vocab)?;
        let phoneme_len = phonemes.chars().count();
        let ref_s = self.select_ref_s(voice_pack, &request.voice, phoneme_len)?;
        let ref_s_slice = ref_s
            .as_slice()
            .context("Style embedding must be contiguous")?;
        let speed = request.speed.unwrap_or(DEFAULT_SPEED);
        let input_ids = vocab.encode(&phonemes);

        let audio = pipeline.forward(&input_ids, ref_s_slice, speed)?;
        Ok(SynthesisResult {
            audio,
            sample_rate: SAMPLE_RATE,
        })
    }

    fn g2p(&self) -> Result<&G2pEngine> {
        if self.g2p.get().is_none() {
            let engine = G2pEngine::new()?;
            let _ = self.g2p.set(engine);
        }
        self.g2p
            .get()
            .ok_or_else(|| anyhow::anyhow!("G2P engine must be initialized"))
    }

    fn phonemes_from_request(
        &self,
        input: &SynthesisInput,
        vocab: &Vocab,
    ) -> Result<String> {
        match input {
            SynthesisInput::Text { text, language } => {
                let language = parse_language(language.as_deref())?;
                self.g2p()?
                    .text_to_kokoro_phonemes(text, language, vocab)
            }
            SynthesisInput::Phonemes { phonemes } => Ok(phonemes.clone()),
        }
    }

    fn select_ref_s(
        &self,
        voice_pack: &VoicePack,
        voice: &VoiceSelection,
        phoneme_len: usize,
    ) -> Result<ndarray::Array2<f32>> {
        match voice {
            VoiceSelection::Auto => Ok(voice_pack.select_style(phoneme_len)),
            VoiceSelection::Index(index) => voice_pack.select_style_by_index(*index as usize),
            VoiceSelection::Id(id) => {
                let index = self
                    .voice_index_by_id
                    .get(id)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Unknown voice id: {}", id))?;
                voice_pack.select_style_by_index(index)
            }
        }
    }

    fn get_voices(&self) -> Result<Vec<VoiceInfo>> {
        let pack = self
            .voice_pack
            .as_ref()
            .context("Runtime is not initialized")?;
        if let Some(manifest) = self.voice_manifest.as_ref() {
            Ok(manifest.voices().to_vec())
        } else {
            Ok((0..pack.len())
                .map(|index| VoiceInfo::with_index(index.to_string(), index))
                .collect())
        }
    }

    fn get_languages(&self) -> Result<Vec<String>> {
        if let Some(manifest) = self.voice_manifest.as_ref() {
            Ok(manifest.languages())
        } else {
            Ok(Vec::new())
        }
    }

    #[cfg(feature = "frb")]
    fn synthesize_stream(
        &mut self,
        request: &SynthesisRequest,
        cancel_token: CancelToken,
        sink: StreamSink<AudioChunk>,
    ) -> Result<()> {
        let chunk_size_ms = request.chunk_size_ms.unwrap_or(DEFAULT_CHUNK_SIZE_MS);
        if chunk_size_ms == 0 {
            anyhow::bail!("chunk_size_ms must be greater than zero");
        }

        let pipeline = self
            .pipeline
            .as_ref()
            .context("Runtime is not initialized")?;
        let vocab = self.vocab.as_ref().context("Runtime is not initialized")?;
        let voice_pack = self
            .voice_pack
            .as_ref()
            .context("Runtime is not initialized")?;

        let phonemes = self.phonemes_from_request(&request.input, vocab)?;
        let phoneme_len = phonemes.chars().count();
        let ref_s = self.select_ref_s(voice_pack, &request.voice, phoneme_len)?;
        let ref_s_slice = ref_s
            .as_slice()
            .context("Style embedding must be contiguous")?;
        let speed = request.speed.unwrap_or(DEFAULT_SPEED);

        let chunks = segment_phonemes(&phonemes, chunk_size_ms, SAMPLE_RATE);
        let total_chunks = chunks.len();
        let mut start_sample = 0u64;

        for (index, chunk_phonemes) in chunks.into_iter().enumerate() {
            if cancel_token.is_cancelled() {
                return Ok(());
            }

            let input_ids = vocab.encode(&chunk_phonemes);

            let audio = pipeline.forward(&input_ids, ref_s_slice, speed)?;
            let audio_len = audio.len() as u64;
            let is_final = index + 1 == total_chunks;

            sink.add(AudioChunk {
                samples: audio,
                sample_rate: SAMPLE_RATE,
                is_final,
                chunk_index: index as u32,
                start_sample,
            })
            .map_err(|err| anyhow::anyhow!("{}", err))?;

            start_sample += audio_len;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.config = None;
        self.pipeline = None;
        self.vocab = None;
        self.voice_pack = None;
        self.voice_manifest = None;
        self.voice_index_by_id.clear();
        self.g2p = OnceLock::new();
        Ok(())
    }

    fn status(&self) -> RuntimeStatus {
        let config = self.config.as_ref();
        RuntimeStatus {
            initialized: self.pipeline.is_some(),
            device: config.map(|cfg| cfg.device.clone()),
            artifacts_dir: config.map(|cfg| cfg.artifacts_dir.clone()),
            vocab_path: config.map(|cfg| cfg.vocab_path.clone()),
            voice_path: config.map(|cfg| cfg.voice_path.clone()),
        }
    }
}

struct RuntimeHandle {
    tx: mpsc::Sender<RuntimeCommand>,
}

impl RuntimeHandle {
    fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<RuntimeCommand>();
        thread::spawn(move || {
            let mut worker = RuntimeWorker::new();
            for command in rx {
                match command {
                    RuntimeCommand::Init { config, reply } => {
                        let result = match worker.init(config) {
                            Ok(()) => Ok(()),
                            Err(err) => {
                                eprintln!("Init failed: {:#}", err);
                                Err(err.to_string())
                            }
                        };
                        let _ = reply.send(result);
                    }
                    RuntimeCommand::Warmup { reply } => {
                        let result = match worker.warmup() {
                            Ok(()) => Ok(()),
                            Err(err) => {
                                eprintln!("Warmup failed: {:#}", err);
                                Err(err.to_string())
                            }
                        };
                        let _ = reply.send(result);
                    }
                    RuntimeCommand::Synthesize { request, reply } => {
                        let result = match worker.synthesize(&request) {
                            Ok(audio) => Ok(audio),
                            Err(err) => {
                                eprintln!("Synthesize failed: {:#}", err);
                                Err(err.to_string())
                            }
                        };
                        let _ = reply.send(result);
                    }
                    RuntimeCommand::GetVoices { reply } => {
                        let result = match worker.get_voices() {
                            Ok(voices) => Ok(voices),
                            Err(err) => {
                                eprintln!("Get voices failed: {:#}", err);
                                Err(err.to_string())
                            }
                        };
                        let _ = reply.send(result);
                    }
                    RuntimeCommand::GetLanguages { reply } => {
                        let result = match worker.get_languages() {
                            Ok(languages) => Ok(languages),
                            Err(err) => {
                                eprintln!("Get languages failed: {:#}", err);
                                Err(err.to_string())
                            }
                        };
                        let _ = reply.send(result);
                    }
                    #[cfg(feature = "frb")]
                    RuntimeCommand::SynthesizeStream {
                        request,
                        cancel_token,
                        sink,
                        reply,
                    } => {
                        let result =
                            match worker.synthesize_stream(&request, cancel_token, sink) {
                            Ok(()) => Ok(()),
                            Err(err) => {
                                eprintln!("Synthesize stream failed: {:#}", err);
                                Err(err.to_string())
                            }
                        };
                        let _ = reply.send(result);
                    }
                    RuntimeCommand::Reset { reply } => {
                        let result = match worker.reset() {
                            Ok(()) => Ok(()),
                            Err(err) => {
                                eprintln!("Reset failed: {:#}", err);
                                Err(err.to_string())
                            }
                        };
                        let _ = reply.send(result);
                    }
                    RuntimeCommand::Status { reply } => {
                        let result = Ok(worker.status());
                        let _ = reply.send(result);
                    }
                }
            }
        });
        Self { tx }
    }
}

static RUNTIME: OnceLock<RuntimeHandle> = OnceLock::new();

fn runtime_handle() -> Result<&'static RuntimeHandle> {
    Ok(RUNTIME.get_or_init(RuntimeHandle::spawn))
}

fn recv_result<T>(rx: mpsc::Receiver<Result<T, String>>) -> Result<T> {
    let result = rx
        .recv()
        .context("Runtime worker thread stopped")?;
    result.map_err(|err| anyhow::anyhow!(err))
}

pub fn init(config: RuntimeConfig) -> Result<()> {
    let handle = runtime_handle()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    handle
        .tx
        .send(RuntimeCommand::Init {
            config,
            reply: reply_tx,
        })
        .context("Failed to send init request")?;
    recv_result(reply_rx)
}

pub fn init_from_paths(
    artifacts_dir: String,
    device: String,
    vocab_path: String,
    voice_path: String,
) -> Result<()> {
    let config = RuntimeConfig::new(artifacts_dir, device, vocab_path, voice_path);
    init(config)
}

pub fn warmup() -> Result<()> {
    let handle = runtime_handle()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    handle
        .tx
        .send(RuntimeCommand::Warmup { reply: reply_tx })
        .context("Failed to send warmup request")?;
    recv_result(reply_rx)
}

pub fn synthesize(request: SynthesisRequest) -> Result<SynthesisResult> {
    let handle = runtime_handle()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    handle
        .tx
        .send(RuntimeCommand::Synthesize {
            request,
            reply: reply_tx,
        })
        .context("Failed to send synthesize request")?;
    recv_result(reply_rx)
}

#[cfg(feature = "frb")]
pub fn synthesize_stream(
    request: SynthesisRequest,
    cancel_token: CancelToken,
    sink: StreamSink<AudioChunk>,
) -> Result<()> {
    let handle = runtime_handle()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    handle
        .tx
        .send(RuntimeCommand::SynthesizeStream {
            request,
            cancel_token,
            sink,
            reply: reply_tx,
        })
        .context("Failed to send synthesize stream request")?;
    recv_result(reply_rx)
}

pub fn shutdown() -> Result<()> {
    let handle = runtime_handle()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    handle
        .tx
        .send(RuntimeCommand::Reset { reply: reply_tx })
        .context("Failed to send shutdown request")?;
    recv_result(reply_rx)
}

pub fn status() -> Result<RuntimeStatus> {
    let handle = runtime_handle()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    handle
        .tx
        .send(RuntimeCommand::Status { reply: reply_tx })
        .context("Failed to send status request")?;
    recv_result(reply_rx)
}

pub fn get_voices() -> Result<Vec<VoiceInfo>> {
    let handle = runtime_handle()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    handle
        .tx
        .send(RuntimeCommand::GetVoices { reply: reply_tx })
        .context("Failed to send get voices request")?;
    recv_result(reply_rx)
}

pub fn get_languages() -> Result<Vec<String>> {
    let handle = runtime_handle()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    handle
        .tx
        .send(RuntimeCommand::GetLanguages { reply: reply_tx })
        .context("Failed to send get languages request")?;
    recv_result(reply_rx)
}

#[cfg(feature = "frb")]
fn segment_phonemes(phonemes: &str, chunk_size_ms: u32, sample_rate: u32) -> Vec<String> {
    fn is_hard_boundary(ch: char) -> bool {
        matches!(ch, '.' | '!' | '?' | ',' | ';' | ':' | '|' | '\n' | '\r')
    }

    fn estimate_char_cost(ch: char) -> u64 {
        if ch.is_whitespace() || is_hard_boundary(ch) {
            0
        } else {
            100
        }
    }

    fn estimate_samples_for_str(value: &str) -> u64 {
        value.chars().map(estimate_char_cost).sum()
    }

    fn find_last_boundary_index(value: &str) -> Option<usize> {
        let mut last = None;
        for (index, ch) in value.char_indices() {
            if is_hard_boundary(ch) {
                last = Some(index + ch.len_utf8());
            }
        }
        last
    }

    let target_samples = (chunk_size_ms as u64 * sample_rate as u64) / 1000;
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut approx_samples = 0u64;
    let mut last_boundary_index = None;

    for ch in phonemes.chars() {
        current.push(ch);
        approx_samples = approx_samples.saturating_add(estimate_char_cost(ch));
        if is_hard_boundary(ch) {
            last_boundary_index = Some(current.len());
        }

        if approx_samples >= target_samples && !current.is_empty() {
            let split_at = last_boundary_index.unwrap_or_else(|| current.len());
            let (chunk, remainder) = current.split_at(split_at);
            if !chunk.is_empty() {
                chunks.push(chunk.to_string());
            }
            current = remainder.to_string();
            approx_samples = estimate_samples_for_str(&current);
            last_boundary_index = find_last_boundary_index(&current);
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

#[cfg(all(test, feature = "frb"))]
mod tests {
    use super::segment_phonemes;

    #[test]
    fn segment_phonemes_prefers_hard_boundaries() {
        let chunks = segment_phonemes("aa, bb", 10, 30_000);
        assert_eq!(chunks[0], "aa,");
        assert_eq!(chunks.join(""), "aa, bb");
    }
}
