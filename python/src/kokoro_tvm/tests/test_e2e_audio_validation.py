"""End-to-end audio validation against reference Kokoro.

This is an opt-in integration test (requires compiled TVM artifacts).
Set `KOKORO_TVM_E2E=1` to enable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from kokoro.model import KModel
from kokoro.pipeline import KPipeline

from kokoro_tvm.cli.inference import load_voice_pack, select_ref_s, text_to_ids
from kokoro_tvm.pipeline import KokoroPipeline, SAMPLES_PER_FRAME


def _find_compiled_artifact(lib_dir: Path, name: str) -> Path | None:
    for ext in (".dylib", ".so"):
        p = lib_dir / f"{name}_compiled{ext}"
        if p.exists():
            return p
    return None


def _require_compiled_libs(lib_dir: Path) -> None:
    required = ["bert", "duration", "f0n", "text_encoder", "decoder"]
    missing = [n for n in required if _find_compiled_artifact(lib_dir, n) is None]
    if missing:
        pytest.skip(f"Missing compiled artifacts in {lib_dir}: {', '.join(missing)}")


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


def test_e2e_audio_matches_reference():
    if os.environ.get("KOKORO_TVM_E2E") != "1":
        pytest.skip("Set KOKORO_TVM_E2E=1 to enable this integration test.")

    lib_dir = Path(os.environ.get("KOKORO_TVM_LIB_DIR", "tvm_output"))
    _require_compiled_libs(lib_dir)

    device_type = os.environ.get("KOKORO_TVM_DEVICE", "metal")  # 'metal' | 'llvm'
    text = os.environ.get("KOKORO_TVM_TEXT", "Hello world")
    lang_code = os.environ.get("KOKORO_TVM_LANG", "a")
    voice = os.environ.get("KOKORO_TVM_VOICE", "af_bella")
    speed = float(os.environ.get("KOKORO_TVM_SPEED", "1.0"))

    kp = KPipeline(lang_code=lang_code, model=False)
    phoneme_chunks = [(r.graphemes, r.phonemes) for r in kp(text) if r.phonemes]
    assert phoneme_chunks, f"G2P produced no phonemes for input: {text!r}"

    # Reference PyTorch Kokoro (CPU).
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).to("cpu").eval()
    voice_pack_ref = torch.load(
        hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{voice}.pt"),
        weights_only=True,
    )

    ref_audio_chunks: list[np.ndarray] = []
    ref_frames_total = 0
    for _, phonemes in phoneme_chunks:
        ref_s = select_ref_s(voice_pack_ref, len(phonemes))
        ids = list(filter(lambda i: i is not None, map(lambda ch: kmodel.vocab.get(ch), phonemes)))
        input_ids = torch.LongTensor([[0, *ids, 0]])
        with torch.no_grad():
            audio_ref, pred_dur = kmodel.forward_with_tokens(input_ids, ref_s, speed=speed)
        ref_audio_chunks.append(audio_ref.squeeze().cpu().numpy())
        ref_frames_total += int(pred_dur.sum().item())

    audio_ref = np.concatenate(ref_audio_chunks) if len(ref_audio_chunks) > 1 else ref_audio_chunks[0]

    # TVM KokoroPipeline (uses compiled artifacts).
    vocab = kmodel.vocab
    voice_pack_tvm = load_voice_pack(voice)
    pipeline = KokoroPipeline(str(lib_dir), device_type=device_type, hybrid=False)

    tvm_audio_chunks: list[np.ndarray] = []
    tvm_frames_total = 0
    for _, phonemes in phoneme_chunks:
        input_ids = text_to_ids(phonemes, vocab)
        ref_s = select_ref_s(voice_pack_tvm, len(phonemes))
        audio_tvm = pipeline.forward(input_ids, ref_s, speed=speed).squeeze()
        tvm_audio_chunks.append(audio_tvm)
        tvm_frames_total += int(round(audio_tvm.size / SAMPLES_PER_FRAME))

    audio_tvm = np.concatenate(tvm_audio_chunks) if len(tvm_audio_chunks) > 1 else tvm_audio_chunks[0]

    # Basic length sanity: waveform length should roughly track predicted frames.
    ref_seconds = audio_ref.size / 24000
    tvm_seconds = audio_tvm.size / 24000
    print(f"ref_seconds={ref_seconds:.3f}, tvm_seconds={tvm_seconds:.3f}")
    print(f"ref_frames_total={ref_frames_total}, tvm_frames_total≈{tvm_frames_total}")

    # Save wavs to inspect if requested.
    if os.environ.get("KOKORO_TVM_SAVE_E2E_AUDIO") == "1":
        import soundfile as sf

        out_dir = Path(os.environ.get("KOKORO_TVM_SAVE_DIR", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / "e2e_ref.wav", audio_ref, 24000)
        sf.write(out_dir / "e2e_tvm.wav", audio_tvm, 24000)
        print(f"Wrote {out_dir / 'e2e_ref.wav'}")
        print(f"Wrote {out_dir / 'e2e_tvm.wav'}")

    corr, lag = _best_lag_corr(audio_ref, audio_tvm, max_lag=2400)
    print(f"corr={corr:.4f}, lag={lag} samples")

    # This threshold is intentionally loose; it will still catch "pure noise" failures.
    assert corr > 0.2, (
        "TVM audio does not correlate with reference audio (likely incorrect inference). "
        "Set KOKORO_TVM_SAVE_E2E_AUDIO=1 to write wavs for listening."
    )
