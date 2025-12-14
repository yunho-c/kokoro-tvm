# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI for running Kokoro TTS inference with compiled TVM modules."""

import argparse
import json

import torch
import numpy as np

from kokoro_tvm.pipeline import KokoroPipeline
from kokoro_tvm.config import TARGET_CONFIGS


def load_vocab() -> dict:
    """Load the Kokoro vocabulary from HuggingFace."""
    from huggingface_hub import hf_hub_download

    repo_id = "hexgrad/Kokoro-82M"
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["vocab"]


def load_voice(voice_name: str) -> torch.Tensor:
    """Load a voice style embedding.

    Args:
        voice_name: Voice name (e.g., 'af_bella', 'am_adam')

    Returns:
        ref_s: [1, 256] style tensor
    """
    from huggingface_hub import hf_hub_download

    repo_id = "hexgrad/Kokoro-82M"

    voice_file = f"voices/{voice_name}.pt"
    voice_path = hf_hub_download(repo_id=repo_id, filename=voice_file)
    ref_s = torch.load(voice_path, weights_only=True)

    # Voice files are [N, 1, 256] - select first style and reshape to [1, 256]
    if ref_s.ndim == 3:
        ref_s = ref_s[0].squeeze(0)  # [1, 256] -> [256]
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)  # [256] -> [1, 256]
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


def save_audio(audio: np.ndarray, output_path: str, sample_rate: int = 24000):
    """Save audio to WAV file.

    Args:
        audio: Audio samples as numpy array
        output_path: Output file path
        sample_rate: Sample rate in Hz
    """
    try:
        import soundfile as sf

        sf.write(output_path, audio, sample_rate)
    except ImportError:
        from scipy.io import wavfile

        # Normalize to int16 range
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)


def main():
    parser = argparse.ArgumentParser(description="Run Kokoro TTS inference with TVM compiled modules")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text or phonemes to synthesize")
    parser.add_argument(
        "--lib-dir", "-l", type=str, default="tvm_output", help="Directory containing compiled .so modules"
    )
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--voice", "-v", type=str, default="af_bella", help="Voice name (e.g., af_bella, am_adam)")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument(
        "--target", type=str, default="llvm", choices=list(TARGET_CONFIGS.keys()), help="Target device for inference"
    )

    args = parser.parse_args()

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = load_vocab()

    # Load voice
    print(f"Loading voice '{args.voice}'...")
    try:
        ref_s = load_voice(args.voice)
    except Exception as e:
        print(f"Failed to load voice '{args.voice}': {e}")
        print("Using random style embedding instead.")
        ref_s = torch.randn(1, 256)

    # Convert text to token IDs
    print(f"Converting text: '{args.text}'")
    input_ids = text_to_ids(args.text, vocab)
    print(f"Token IDs shape: {input_ids.shape}")

    # Initialize pipeline
    print(f"Loading TVM modules from '{args.lib_dir}'...")
    # resolve_target returns (target, target_host, extension, description)
    # We just need the device type for the pipeline
    device_type = args.target.split("-")[0]  # 'llvm', 'metal', etc.
    pipeline = KokoroPipeline(args.lib_dir, device_type)

    # Run inference
    print(f"Running inference (speed={args.speed})...")
    audio = pipeline.forward(input_ids, ref_s, speed=args.speed)

    # Convert to numpy and trim silence
    audio_np = audio.squeeze().numpy()

    # Save audio
    print(f"Saving audio to '{args.output}'...")
    save_audio(audio_np, args.output)

    print(f"Done! Generated {len(audio_np) / 24000:.2f}s of audio.")


if __name__ == "__main__":
    main()
