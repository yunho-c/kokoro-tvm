//! Runtime wrapper for long-lived inference sessions.

use crate::constants::SAMPLE_RATE;
use crate::{KokoroPipeline, Vocab, VoicePack};
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::{mpsc, OnceLock};
use std::thread;

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub artifacts_dir: PathBuf,
    pub device: String,
    pub vocab_path: PathBuf,
    pub voice_path: PathBuf,
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

enum RuntimeCommand {
    Init {
        config: RuntimeConfig,
        reply: mpsc::Sender<Result<(), String>>,
    },
    Warmup {
        reply: mpsc::Sender<Result<(), String>>,
    },
    Synthesize {
        phonemes: String,
        speed: f32,
        voice_index: Option<usize>,
        reply: mpsc::Sender<Result<SynthesisResult, String>>,
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
}

impl RuntimeWorker {
    fn new() -> Self {
        Self {
            config: None,
            pipeline: None,
            vocab: None,
            voice_pack: None,
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

        self.config = Some(config);
        self.pipeline = Some(pipeline);
        self.vocab = Some(vocab);
        self.voice_pack = Some(voice_pack);
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

    fn synthesize(
        &mut self,
        phonemes: &str,
        speed: f32,
        voice_index: Option<usize>,
    ) -> Result<SynthesisResult> {
        let pipeline = self
            .pipeline
            .as_ref()
            .context("Runtime is not initialized")?;
        let vocab = self.vocab.as_ref().context("Runtime is not initialized")?;
        let voice_pack = self
            .voice_pack
            .as_ref()
            .context("Runtime is not initialized")?;

        let input_ids = vocab.encode(phonemes);
        let ref_s = if let Some(index) = voice_index {
            voice_pack.select_style_by_index(index)?
        } else {
            let phoneme_len = phonemes.chars().count();
            voice_pack.select_style(phoneme_len)
        };
        let ref_s_slice = ref_s
            .as_slice()
            .context("Style embedding must be contiguous")?;

        let audio = pipeline.forward(&input_ids, ref_s_slice, speed)?;
        Ok(SynthesisResult {
            audio,
            sample_rate: SAMPLE_RATE,
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.config = None;
        self.pipeline = None;
        self.vocab = None;
        self.voice_pack = None;
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
                        let result = worker.init(config).map_err(|err| err.to_string());
                        let _ = reply.send(result);
                    }
                    RuntimeCommand::Warmup { reply } => {
                        let result = worker.warmup().map_err(|err| err.to_string());
                        let _ = reply.send(result);
                    }
                    RuntimeCommand::Synthesize {
                        phonemes,
                        speed,
                        voice_index,
                        reply,
                    } => {
                        let result = worker
                            .synthesize(&phonemes, speed, voice_index)
                            .map_err(|err| err.to_string());
                        let _ = reply.send(result);
                    }
                    RuntimeCommand::Reset { reply } => {
                        let result = worker.reset().map_err(|err| err.to_string());
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

pub fn synthesize(phonemes: &str, speed: f32) -> Result<SynthesisResult> {
    synthesize_with_voice_index(phonemes, speed, None)
}

pub fn synthesize_with_voice_index(
    phonemes: &str,
    speed: f32,
    voice_index: Option<usize>,
) -> Result<SynthesisResult> {
    let handle = runtime_handle()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    handle
        .tx
        .send(RuntimeCommand::Synthesize {
            phonemes: phonemes.to_string(),
            speed,
            voice_index,
            reply: reply_tx,
        })
        .context("Failed to send synthesize request")?;
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
