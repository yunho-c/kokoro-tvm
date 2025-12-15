# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI for running Kokoro TTS inference with compiled TVM modules."""

import argparse
import json

import soundfile as sf
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from kokoro.pipeline import KPipeline

from kokoro_tvm.pipeline import KokoroPipeline
from kokoro_tvm.config import TARGET_CONFIGS


def load_vocab() -> dict:
    """Load the Kokoro vocabulary from HuggingFace."""
    repo_id = "hexgrad/Kokoro-82M"
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["vocab"]


def load_voice_pack(voice: str) -> torch.Tensor:
    """Load voice pack(s).

    `voice` can be a single name (e.g. `af_bella`) or a comma-separated list
    (e.g. `af_bella,af_jessica`), in which case packs are averaged.
    """
    repo_id = "hexgrad/Kokoro-82M"

    packs: list[torch.Tensor] = []
    for voice_name in voice.split(","):
        voice_name = voice_name.strip()
        if not voice_name:
            continue
        voice_file = f"voices/{voice_name}.pt"
        voice_path = hf_hub_download(repo_id=repo_id, filename=voice_file)
        packs.append(torch.load(voice_path, weights_only=True))
    if not packs:
        msg = f"Invalid voice spec: {voice!r}"
        raise ValueError(msg)
    if len(packs) == 1:
        return packs[0]
    return torch.mean(torch.stack(packs), dim=0)


def select_ref_s(voice_pack: torch.Tensor, phoneme_len: int) -> torch.Tensor:
    """Select a style embedding like reference Kokoro: `pack[len(ps)-1]`.

    Returns a `[1, 256]` tensor.
    """
    if phoneme_len <= 0:
        index = 0
    else:
        index = phoneme_len - 1
    index = max(0, min(index, int(voice_pack.shape[0]) - 1))
    ref_s = voice_pack[index]
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)
    return ref_s


def text_to_ids(text: str, vocab: dict) -> torch.LongTensor:
    """Convert text/phonemes to token IDs.

    Args:
        text: Input text (phoneme string)
        vocab: Vocabulary mapping

    Returns:
        input_ids: [1, seq_len] token tensor
    """
    # Filter to valid tokens, wrap with start/end tokens
    token_ids = [0]  # Start token
    for char in text:
        if char in vocab:
            token_ids.append(vocab[char])
    token_ids.append(0)  # End token

    return torch.LongTensor([token_ids])


def g2p_text_to_phonemes(text: str, lang_code: str = "a") -> list[tuple[str, str]]:
    """Convert graphemes to phonemes using misaki (via kokoro's KPipeline).

    Returns a list of (graphemes_chunk, phonemes_chunk) pairs.
    """
    pipeline = KPipeline(lang_code=lang_code, model=False)
    results: list[tuple[str, str]] = []
    for r in pipeline(text):
        if r.phonemes:
            results.append((r.graphemes, r.phonemes))
    return results


def save_audio(audio: np.ndarray, output_path: str, sample_rate: int = 24000):
    """Save audio to WAV file.

    Args:
        audio: Audio samples as numpy array
        output_path: Output file path
        sample_rate: Sample rate in Hz
    """
    sf.write(output_path, audio, sample_rate)


def main():
    parser = argparse.ArgumentParser(description="Run Kokoro TTS inference with TVM compiled modules")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text or phonemes to synthesize")
    parser.add_argument(
        "--phonemes",
        action="store_true",
        help="Treat `--text` as a phoneme string (skip G2P)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="a",
        help="Language code for G2P (kokoro KPipeline codes, e.g. a=en-us, b=en-gb, e=es, f=fr-fr)",
    )
    parser.add_argument(
        "--lib-dir", "-l", type=str, default="tvm_output", help="Directory containing compiled .so modules"
    )
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--voice", "-v", type=str, default="af_bella", help="Voice name (e.g., af_bella, am_adam)")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument(
        "--target", type=str, default="llvm", choices=list(TARGET_CONFIGS.keys()), help="Target device for inference"
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="Hybrid mode: encoder on CPU, decoder on GPU (for Metal/CUDA targets)"
    )

    args = parser.parse_args()

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = load_vocab()

    # Load voice
    print(f"Loading voice '{args.voice}'...")
    voice_pack = load_voice_pack(args.voice)

    if args.phonemes:
        chunks = [("", args.text)]
    else:
        chunks = g2p_text_to_phonemes(args.text, lang_code=args.lang)
        if not chunks:
            msg = f"G2P produced no phonemes for input: {args.text!r}"
            raise ValueError(msg)

    if len(chunks) > 1:
        print(f"G2P chunked input into {len(chunks)} segments.")

    # Initialize pipeline
    print(f"Loading TVM modules from '{args.lib_dir}'...")
    # resolve_target returns (target, target_host, extension, description)
    # We just need the device type for the pipeline
    device_type = args.target.split("-")[0]  # 'llvm', 'metal', etc.
    pipeline = KokoroPipeline(args.lib_dir, device_type, hybrid=args.hybrid)

    # Run inference (possibly chunked)
    print(f"Running inference (speed={args.speed})...")
    audio_chunks: list[torch.Tensor] = []
    silence = torch.zeros(int(0.05 * 24000), dtype=torch.float32)
    for graphemes, phonemes in chunks:
        if graphemes:
            print(f"  Segment: {graphemes!r}")
        print(f"  Phonemes: {phonemes!r}")
        input_ids = text_to_ids(phonemes, vocab)
        ref_s = select_ref_s(voice_pack, len(phonemes))
        audio = pipeline.forward(input_ids, ref_s, speed=args.speed)
        audio_chunks.append(audio.squeeze().cpu())

    if not audio_chunks:
        msg = "No audio chunks were produced."
        raise RuntimeError(msg)

    if len(audio_chunks) == 1:
        audio_full = audio_chunks[0]
    else:
        joined: list[torch.Tensor] = []
        for i, chunk in enumerate(audio_chunks):
            if i > 0:
                joined.append(silence)
            joined.append(chunk)
        audio_full = torch.cat(joined, dim=0)

    audio_np = audio_full.numpy()

    # Save audio
    print(f"Saving audio to '{args.output}'...")
    save_audio(audio_np, args.output)

    print(f"Done! Generated {len(audio_np) / 24000:.2f}s of audio.")


if __name__ == "__main__":
    main()
