# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end validation: TVM pipeline vs reference Kokoro (PyTorch).

This compares the generated waveform against reference Kokoro for the same input
phonemes + voice style. It is intended as a debugging aid when audio sounds wrong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from kokoro.model import KModel
from kokoro.pipeline import KPipeline

from kokoro_tvm.cli.inference import load_voice_pack, select_ref_s, text_to_ids
from kokoro_tvm.pipeline import KokoroPipeline, SAMPLES_PER_FRAME


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x / (peak + 1e-8)


def _best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 2400) -> tuple[float, int]:
    """Return (corr, lag) with lag in samples. Positive lag shifts b right."""
    if a.size == 0 or b.size == 0:
        return 0.0, 0

    a = _normalize(a)
    b = _normalize(b)

    max_lag = max(0, min(max_lag, min(a.size, b.size) - 1))
    best = (-1.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a_seg = a[-lag:]
            b_seg = b[: a_seg.size]
        elif lag > 0:
            b_seg = b[lag:]
            a_seg = a[: b_seg.size]
        else:
            n = min(a.size, b.size)
            a_seg = a[:n]
            b_seg = b[:n]

        n = min(a_seg.size, b_seg.size)
        a_seg = a_seg[:n]
        b_seg = b_seg[:n]

        if a_seg.size < 2048:
            continue
        corr = float(np.corrcoef(a_seg, b_seg)[0, 1])
        if corr > best[0]:
            best = (corr, lag)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TVM audio against reference Kokoro (PyTorch)")
    parser.add_argument("--text", type=str, default="Hello world", help="Input text to synthesize")
    parser.add_argument("--lang", type=str, default="a", help="Kokoro language code for G2P (e.g. a=en-us)")
    parser.add_argument("--voice", type=str, default="af_bella", help="Voice name (e.g. af_bella)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument("--lib-dir", type=str, default="tvm_output", help="Directory containing compiled TVM modules")
    parser.add_argument(
        "--device",
        type=str,
        default="metal",
        choices=["metal", "llvm", "cuda"],
        help="Device type for TVM pipeline",
    )
    parser.add_argument("--hybrid", action="store_true", help="Hybrid mode (encoder on CPU, decoder on device)")
    parser.add_argument("--save-dir", type=str, default=None, help="If set, write wavs for listening")
    parser.add_argument("--threshold", type=float, default=0.2, help="Fail if corr < threshold (default: 0.2)")
    args = parser.parse_args()

    kp = KPipeline(lang_code=args.lang, model=False)
    phoneme_chunks = [(r.graphemes, r.phonemes) for r in kp(args.text) if r.phonemes]
    if not phoneme_chunks:
        raise ValueError(f"G2P produced no phonemes for input: {args.text!r}")

    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).to("cpu").eval()
    voice_pack_ref = torch.load(
        hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{args.voice}.pt"),
        weights_only=True,
    )

    ref_audio_chunks: list[np.ndarray] = []
    ref_frames_total = 0
    for graphemes, phonemes in phoneme_chunks:
        print(f"[REF] {graphemes!r} -> {phonemes!r}")
        ref_s = select_ref_s(voice_pack_ref, len(phonemes))
        ids = list(filter(lambda i: i is not None, map(lambda ch: kmodel.vocab.get(ch), phonemes)))
        input_ids = torch.LongTensor([[0, *ids, 0]])
        with torch.no_grad():
            audio_ref, pred_dur = kmodel.forward_with_tokens(input_ids, ref_s, speed=args.speed)
        ref_audio_chunks.append(audio_ref.squeeze().cpu().numpy())
        ref_frames_total += int(pred_dur.sum().item())

    audio_ref = np.concatenate(ref_audio_chunks) if len(ref_audio_chunks) > 1 else ref_audio_chunks[0]

    vocab = kmodel.vocab
    voice_pack_tvm = load_voice_pack(args.voice)
    pipeline = KokoroPipeline(args.lib_dir, args.device, hybrid=args.hybrid)

    tvm_audio_chunks: list[np.ndarray] = []
    tvm_frames_total = 0
    for graphemes, phonemes in phoneme_chunks:
        print(f"[TVM] {graphemes!r} -> {phonemes!r}")
        input_ids = text_to_ids(phonemes, vocab)
        ref_s = select_ref_s(voice_pack_tvm, len(phonemes))
        audio_tvm = pipeline.forward(input_ids, ref_s, speed=args.speed).squeeze().cpu().numpy()
        tvm_audio_chunks.append(audio_tvm)
        tvm_frames_total += int(round(audio_tvm.size / SAMPLES_PER_FRAME))

    audio_tvm = np.concatenate(tvm_audio_chunks) if len(tvm_audio_chunks) > 1 else tvm_audio_chunks[0]

    ref_seconds = audio_ref.size / 24000
    tvm_seconds = audio_tvm.size / 24000
    print(f"ref_seconds={ref_seconds:.3f}, tvm_seconds={tvm_seconds:.3f}")
    print(f"ref_frames_total={ref_frames_total}, tvm_frames_total≈{tvm_frames_total}")

    corr, lag = _best_lag_corr(audio_ref, audio_tvm, max_lag=2400)
    print(f"corr={corr:.4f}, lag={lag} samples (max_lag=2400)")

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / "e2e_ref.wav", audio_ref, 24000)
        sf.write(out_dir / "e2e_tvm.wav", audio_tvm, 24000)
        print(f"Wrote {out_dir / 'e2e_ref.wav'}")
        print(f"Wrote {out_dir / 'e2e_tvm.wav'}")

    if corr < args.threshold:
        print(f"FAIL: corr {corr:.4f} < threshold {args.threshold:.4f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
