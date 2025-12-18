#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Convert Kokoro voice packs and vocabulary to portable formats.

This script converts:
- Voice packs: .pt (PyTorch) -> .npy (NumPy)
- Vocabulary: config.json -> vocab.json

Usage:
    uv run python scripts/convert_assets.py --voice af_bella --output ./assets/
    uv run python scripts/convert_assets.py --vocab-only --output ./assets/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def convert_voice_pack(voice_name: str, output_dir: Path) -> Path:
    """Convert a voice pack from .pt to .npy format.

    Args:
        voice_name: Voice pack name (e.g., 'af_bella')
        output_dir: Directory to save the .npy file

    Returns:
        Path to the saved .npy file
    """
    repo_id = "hexgrad/Kokoro-82M"
    voice_file = f"voices/{voice_name}.pt"

    print(f"Downloading {voice_file}...")
    voice_path = hf_hub_download(repo_id=repo_id, filename=voice_file)

    print(f"Loading voice pack from {voice_path}...")
    pack = torch.load(voice_path, weights_only=True)

    # Convert to numpy
    pack_np = pack.numpy().astype(np.float32)
    print(f"Voice pack shape: {pack_np.shape}, dtype: {pack_np.dtype}")

    # Save as .npy
    output_path = output_dir / f"{voice_name}.npy"
    np.save(output_path, pack_np)
    print(f"Saved voice pack to {output_path}")

    return output_path


def export_vocab(output_dir: Path) -> Path:
    """Export vocabulary from HuggingFace config.json to vocab.json.

    Args:
        output_dir: Directory to save vocab.json

    Returns:
        Path to the saved vocab.json file
    """
    repo_id = "hexgrad/Kokoro-82M"

    print("Downloading config.json...")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    vocab = config.get("vocab", {})
    print(f"Vocabulary size: {len(vocab)} tokens")

    # Save as standalone vocab.json
    output_path = output_dir / "vocab.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro assets to portable formats for Rust inference")
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        nargs="*",
        help="Voice pack name(s) to convert (e.g., af_bella am_adam)",
    )
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="Only export vocabulary, skip voice packs",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./assets",
        help="Output directory for converted files",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always export vocab
    export_vocab(output_dir)

    # Convert voice packs if specified
    if args.voice and not args.vocab_only:
        for voice_name in args.voice:
            convert_voice_pack(voice_name, output_dir)
    elif not args.vocab_only and not args.voice:
        print("\nNo voice packs specified. Use --voice to convert voice packs.")
        print("Example: --voice af_bella am_adam")


if __name__ == "__main__":
    main()
