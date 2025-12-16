# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Validate intermediate tensors across Kokoro stages.

This tool compares:
- Reference Kokoro (dynamic, as implemented in `external/kokoro`)
- Reference Kokoro with static padding (matches kokoro-tvm pipeline shapes)
- TVM-compiled modules (kokoro-tvm pipeline)

It reports per-stage numerical error and basic waveform similarity metrics.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from kokoro.model import KModel
from kokoro.pipeline import KPipeline

from kokoro_tvm.cli.inference import load_voice_pack, select_ref_s, text_to_ids
from kokoro_tvm.pipeline import KokoroPipeline, SAMPLES_PER_FRAME, STATIC_AUDIO_LEN, STATIC_TEXT_LEN


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x / (peak + 1e-8)


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.size == 0 or b.size == 0:
        return {"mae": float("nan"), "max_abs": float("nan"), "rmse": float("nan")}
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    diff = a - b
    return {
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def _best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 2400) -> tuple[float, int]:
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


def _tail_summary_1d(x: np.ndarray, valid: int, *, preview: int = 8, atol: float = 1e-6) -> dict[str, object]:
    flat = np.asarray(x).reshape(-1).astype(np.float32, copy=False)
    total = int(flat.size)
    valid = int(max(0, min(valid, total)))
    pad = flat[valid:]
    if pad.size == 0:
        return {"valid": valid, "total": total, "pad": 0, "finite_frac": 1.0, "nonzero_frac": 0.0, "head": [], "tail": []}
    finite = np.isfinite(pad)
    finite_frac = float(np.mean(finite))
    nonzero_frac = float(np.mean(np.abs(pad) > atol))
    head = pad[:preview].tolist()
    tail = pad[-preview:].tolist() if pad.size >= preview else pad.tolist()
    return {
        "valid": valid,
        "total": total,
        "pad": int(pad.size),
        "finite_frac": finite_frac,
        "nonzero_frac": nonzero_frac,
        "nonzero_atol": atol,
        "head": head,
        "tail": tail,
    }


def _array_stats(name: str, x: np.ndarray, *, atol: float = 1e-8) -> None:
    arr = np.asarray(x).reshape(-1).astype(np.float32, copy=False)
    if arr.size == 0:
        print(f"{name}: empty")
        return
    finite = np.isfinite(arr)
    finite_frac = float(np.mean(finite))
    arr_f = arr[finite] if np.any(finite) else np.array([], dtype=np.float32)
    min_v = float(np.min(arr_f)) if arr_f.size else float("nan")
    max_v = float(np.max(arr_f)) if arr_f.size else float("nan")
    mean_v = float(np.mean(arr_f)) if arr_f.size else float("nan")
    std_v = float(np.std(arr_f)) if arr_f.size else float("nan")
    nonzero_frac = float(np.mean(np.abs(arr) > atol))
    print(
        f"{name}: n={arr.size} finite_frac={finite_frac:.4f} nonzero_frac={nonzero_frac:.4f} "
        f"min={min_v:.4g} max={max_v:.4g} mean={mean_v:.4g} std={std_v:.4g}"
    )


def _trace_dynamic(kmodel: KModel, phonemes: str, ref_s: torch.Tensor, speed: float) -> dict[str, object]:
    ids = list(filter(lambda i: i is not None, map(lambda ch: kmodel.vocab.get(ch), phonemes)))
    input_ids = torch.LongTensor([[0, *ids, 0]])
    cur_len = int(input_ids.shape[1])
    input_lengths = torch.full((1,), cur_len, dtype=torch.long)
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

    with torch.no_grad():
        bert_dur = kmodel.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        lengths = input_lengths if input_lengths.device == torch.device("cpu") else input_lengths.to("cpu")
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(d, lengths, batch_first=True, enforce_sorted=False)
        kmodel.predictor.lstm.flatten_parameters()
        x_packed, _ = kmodel.predictor.lstm(x_packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True, total_length=cur_len)
        duration_logits = kmodel.predictor.duration_proj(x)
        duration = torch.sigmoid(duration_logits).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
        pred_dur = pred_dur[:cur_len]
        indices = torch.repeat_interleave(torch.arange(cur_len), pred_dur)
        pred_aln_trg = torch.zeros((cur_len, indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = kmodel.predictor.F0Ntrain(en, s)
        t_en = kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = kmodel.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze().cpu()

    frames = int(pred_aln_trg.shape[-1])
    audio_trimmed = audio[: frames * SAMPLES_PER_FRAME].numpy()

    return {
        "cur_len": cur_len,
        "frames": frames,
        "d_en": d_en.numpy(),
        "duration_logits": duration_logits.numpy(),
        "d": d.numpy(),
        "pred_dur": pred_dur.numpy(),
        "t_en": t_en.numpy(),
        "f0": f0_pred.numpy(),
        "n": n_pred.numpy(),
        "audio_trimmed": audio_trimmed,
    }


def _trace_static(kmodel: KModel, phonemes: str, ref_s: torch.Tensor, speed: float) -> dict[str, object]:
    ids = list(filter(lambda i: i is not None, map(lambda ch: kmodel.vocab.get(ch), phonemes)))
    input_ids_dyn = torch.LongTensor([[0, *ids, 0]])
    cur_len = int(input_ids_dyn.shape[1])

    input_ids = torch.zeros((1, STATIC_TEXT_LEN), dtype=torch.long)
    input_ids[0, :cur_len] = input_ids_dyn[0]
    text_mask = torch.zeros((1, STATIC_TEXT_LEN), dtype=torch.bool)
    text_mask[:, cur_len:] = True
    attention_mask = (~text_mask).int()
    input_lengths = torch.tensor([cur_len], dtype=torch.long)

    with torch.no_grad():
        bert_dur = kmodel.bert(input_ids, attention_mask=attention_mask)
        d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        lengths = input_lengths if input_lengths.device == torch.device("cpu") else input_lengths.to("cpu")
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(d, lengths, batch_first=True, enforce_sorted=False)
        kmodel.predictor.lstm.flatten_parameters()
        x_packed, _ = kmodel.predictor.lstm(x_packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True, total_length=STATIC_TEXT_LEN)
        duration_logits = kmodel.predictor.duration_proj(x)
        duration = torch.sigmoid(duration_logits).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
        pred_dur = pred_dur[:cur_len]

        indices = torch.repeat_interleave(torch.arange(cur_len), pred_dur)
        frames = int(indices.numel())
        if frames > STATIC_AUDIO_LEN:
            indices = indices[:STATIC_AUDIO_LEN]
            frames = STATIC_AUDIO_LEN

        pred_aln_trg = torch.zeros((cur_len, STATIC_AUDIO_LEN))
        pred_aln_trg[indices, torch.arange(frames)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        full_aln = torch.zeros((1, STATIC_TEXT_LEN, STATIC_AUDIO_LEN), dtype=torch.float32)
        full_aln[0, :cur_len, :] = pred_aln_trg[0]

        en = d.transpose(-1, -2) @ full_aln
        f0_pred, n_pred = kmodel.predictor.F0Ntrain(en, s)
        t_en = kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ full_aln
        audio = kmodel.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze().cpu()

    audio_trimmed = audio[: frames * SAMPLES_PER_FRAME].numpy()

    return {
        "cur_len": cur_len,
        "frames": frames,
        "d_en": d_en.numpy(),
        "duration_logits": duration_logits.numpy(),
        "d": d.numpy(),
        "pred_dur": pred_dur.numpy(),
        "t_en": t_en.numpy(),
        "f0": f0_pred.numpy(),
        "n": n_pred.numpy(),
        "audio_trimmed": audio_trimmed,
    }


@contextlib.contextmanager
def _mock_packed_sequence():
    """Temporarily disable PackedSequence behavior for LSTMs.

    This mimics the export-time behavior used in kokoro-tvm where packing is
    replaced with a passthrough to avoid dynamic/data-dependent shapes.
    """
    orig_pack = torch.nn.utils.rnn.pack_padded_sequence
    orig_pad = torch.nn.utils.rnn.pad_packed_sequence

    def mock_pack(x, lengths, batch_first=False, enforce_sorted=True):
        return x

    def mock_pad(x, batch_first=False, padding_value=0.0, total_length=None):
        return x, None

    torch.nn.utils.rnn.pack_padded_sequence = mock_pack
    torch.nn.utils.rnn.pad_packed_sequence = mock_pad
    try:
        yield
    finally:
        torch.nn.utils.rnn.pack_padded_sequence = orig_pack
        torch.nn.utils.rnn.pad_packed_sequence = orig_pad


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate intermediate Kokoro stages (PyTorch vs TVM)")
    parser.add_argument("--text", type=str, default="Hello world", help="Input text")
    parser.add_argument("--lang", type=str, default="a", help="G2P language code (KPipeline)")
    parser.add_argument("--voice", type=str, default="af_bella", help="Voice name")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--lib-dir", type=str, default="tvm_output", help="Directory containing compiled TVM modules")
    parser.add_argument("--device", type=str, default="metal", choices=["metal", "llvm", "cuda"], help="TVM device")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid mode (encoder on CPU, decoder on device)")
    parser.add_argument("--save-dir", type=str, default=None, help="If set, write wavs for listening")
    args = parser.parse_args()

    kp = KPipeline(lang_code=args.lang, model=False)
    chunks = [(r.graphemes, r.phonemes) for r in kp(args.text) if r.phonemes]
    if not chunks:
        raise ValueError(f"G2P produced no phonemes for input: {args.text!r}")
    if len(chunks) > 1:
        print(f"Warning: G2P chunked into {len(chunks)} segments; validating the first segment only.")
    graphemes, phonemes = chunks[0]
    print(f"graphemes={graphemes!r}")
    print(f"phonemes={phonemes!r}")

    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).to("cpu").eval()
    voice_pack_ref = torch.load(
        hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{args.voice}.pt"),
        weights_only=True,
    )
    ref_s = select_ref_s(voice_pack_ref, len(phonemes))

    trace_dyn = _trace_dynamic(kmodel, phonemes, ref_s, args.speed)
    trace_static = _trace_static(kmodel, phonemes, ref_s, args.speed)
    with _mock_packed_sequence():
        trace_static_nopack = _trace_static(kmodel, phonemes, ref_s, args.speed)

    vocab = kmodel.vocab
    input_ids_tvm = text_to_ids(phonemes, vocab)
    voice_pack_tvm = load_voice_pack(args.voice)
    ref_s_tvm = select_ref_s(voice_pack_tvm, len(phonemes))
    pipeline = KokoroPipeline(args.lib_dir, args.device, hybrid=args.hybrid)
    trace_tvm = pipeline.trace(input_ids_tvm, ref_s_tvm, speed=args.speed)

    cur_len = int(trace_static["cur_len"])
    print(f"cur_len={cur_len}")
    print(f"frames_dynamic={trace_dyn['frames']}, frames_static={trace_static['frames']}, frames_tvm={trace_tvm['frames']}")

    def show(name: str, a: np.ndarray, b: np.ndarray):
        m = _metrics(a, b)
        print(f"{name}: mae={m['mae']:.3e} max_abs={m['max_abs']:.3e} rmse={m['rmse']:.3e}")

    print("\nStatic PyTorch vs TVM (module fidelity):")
    show(
        "bert.d_en[:cur_len]",
        np.asarray(trace_static["d_en"])[:, :, :cur_len],
        np.asarray(trace_tvm["d_en"])[:, :, :cur_len],
    )
    show(
        "duration.logits[:cur_len]",
        np.asarray(trace_static["duration_logits"])[:, :cur_len, :],
        np.asarray(trace_tvm["duration_logits"])[:, :cur_len, :],
    )
    show(
        "duration.d[:cur_len]",
        np.asarray(trace_static["d"])[:, :cur_len, :],
        np.asarray(trace_tvm["d"])[:, :cur_len, :],
    )
    show(
        "text_encoder.t_en[:cur_len]",
        np.asarray(trace_static["t_en"])[:, :, :cur_len],
        np.asarray(trace_tvm["t_en"])[:, :, :cur_len],
    )

    static_frames = int(trace_static["frames"])
    tvm_f0 = np.asarray(trace_tvm["f0"]).reshape(-1)
    tvm_n = np.asarray(trace_tvm["n"]).reshape(-1)
    static_f0 = np.asarray(trace_static["f0"]).reshape(-1)
    static_n = np.asarray(trace_static["n"]).reshape(-1)
    f_len = min(static_f0.size, tvm_f0.size, static_frames * 2)
    show("f0[:2*frames_static]", static_f0[:f_len], tvm_f0[:f_len])
    n_len = min(static_n.size, tvm_n.size, static_frames * 2)
    show("n[:2*frames_static]", static_n[:n_len], tvm_n[:n_len])

    print("\nF0/N padded tail (what fills beyond valid frames?):")
    valid_f0n = static_frames * 2
    tvm_f0_pad = trace_tvm.get("f0_pad_summary") or _tail_summary_1d(tvm_f0, valid_f0n)
    tvm_n_pad = trace_tvm.get("n_pad_summary") or _tail_summary_1d(tvm_n, valid_f0n)
    static_f0_pad = _tail_summary_1d(static_f0, valid_f0n)
    static_n_pad = _tail_summary_1d(static_n, valid_f0n)
    print(
        "tvm.f0 pad="
        f"{tvm_f0_pad['pad']} finite_frac={tvm_f0_pad['finite_frac']:.3f} nonzero_frac={tvm_f0_pad['nonzero_frac']:.3f} "
        f"head={tvm_f0_pad['head']} tail={tvm_f0_pad['tail']}"
    )
    print(
        "tvm.n  pad="
        f"{tvm_n_pad['pad']} finite_frac={tvm_n_pad['finite_frac']:.3f} nonzero_frac={tvm_n_pad['nonzero_frac']:.3f} "
        f"head={tvm_n_pad['head']} tail={tvm_n_pad['tail']}"
    )
    print(
        "pt.static f0 pad="
        f"{static_f0_pad['pad']} finite_frac={static_f0_pad['finite_frac']:.3f} nonzero_frac={static_f0_pad['nonzero_frac']:.3f} "
        f"head={static_f0_pad['head']} tail={static_f0_pad['tail']}"
    )
    print(
        "pt.static n  pad="
        f"{static_n_pad['pad']} finite_frac={static_n_pad['finite_frac']:.3f} nonzero_frac={static_n_pad['nonzero_frac']:.3f} "
        f"head={static_n_pad['head']} tail={static_n_pad['tail']}"
    )

    audio_static = np.asarray(trace_static["audio_trimmed"]).reshape(-1)
    audio_tvm = np.asarray(trace_tvm["audio_trimmed"]).reshape(-1)
    print("\nDecoder audio stats:")
    _array_stats("pt.static audio_trimmed", audio_static)
    _array_stats("tvm audio_trimmed", audio_tvm)
    if "decoder_input_stats" in trace_tvm:
        stats = trace_tvm["decoder_input_stats"]
        print("Decoder input stats (TVM pipeline):")
        print(f"  asr: {stats['asr']}")
        print(f"  f0:  {stats['f0']}")
        print(f"  n:   {stats['n']}")
        print(f"  s:   {stats['style_128']}")
    corr, lag = _best_lag_corr(audio_static, audio_tvm, max_lag=2400)
    print(f"decoder.audio_trimmed corr={corr:.4f} lag={lag} samples")

    print("\nStatic PyTorch packed vs no-pack (packing semantics impact):")
    show(
        "duration.logits[:cur_len]",
        np.asarray(trace_static["duration_logits"])[:, :cur_len, :],
        np.asarray(trace_static_nopack["duration_logits"])[:, :cur_len, :],
    )
    show(
        "text_encoder.t_en[:cur_len]",
        np.asarray(trace_static["t_en"])[:, :, :cur_len],
        np.asarray(trace_static_nopack["t_en"])[:, :, :cur_len],
    )
    corr_np, lag_np = _best_lag_corr(
        np.asarray(trace_static["audio_trimmed"]).reshape(-1),
        np.asarray(trace_static_nopack["audio_trimmed"]).reshape(-1),
        max_lag=2400,
    )
    print(f"audio_trimmed corr={corr_np:.4f} lag={lag_np} samples")

    print("\nStatic PyTorch no-pack vs TVM (does TVM match no-pack semantics?):")
    show(
        "duration.logits[:cur_len]",
        np.asarray(trace_static_nopack["duration_logits"])[:, :cur_len, :],
        np.asarray(trace_tvm["duration_logits"])[:, :cur_len, :],
    )
    show(
        "text_encoder.t_en[:cur_len]",
        np.asarray(trace_static_nopack["t_en"])[:, :, :cur_len],
        np.asarray(trace_tvm["t_en"])[:, :, :cur_len],
    )
    corr_np2, lag_np2 = _best_lag_corr(np.asarray(trace_static_nopack["audio_trimmed"]).reshape(-1), audio_tvm, 2400)
    print(f"decoder.audio_trimmed corr={corr_np2:.4f} lag={lag_np2} samples")

    print("\nDynamic PyTorch vs Static PyTorch (shape/static padding impact):")
    show(
        "bert.d_en[:cur_len]",
        np.asarray(trace_dyn["d_en"]),
        np.asarray(trace_static["d_en"])[:, :, :cur_len],
    )
    show(
        "duration.logits",
        np.asarray(trace_dyn["duration_logits"]),
        np.asarray(trace_static["duration_logits"])[:, :cur_len, :],
    )
    show("duration.d", np.asarray(trace_dyn["d"]), np.asarray(trace_static["d"])[:, :cur_len, :])
    show("text_encoder.t_en", np.asarray(trace_dyn["t_en"]), np.asarray(trace_static["t_en"])[:, :, :cur_len])
    corr2, lag2 = _best_lag_corr(np.asarray(trace_dyn["audio_trimmed"]), audio_static, max_lag=2400)
    print(f"audio_trimmed corr={corr2:.4f} lag={lag2} samples")

    print("\nDynamic PyTorch vs TVM (aligned-length F0/N):")
    dyn_frames = int(trace_dyn["frames"])
    tvm_frames = int(trace_tvm["frames"])
    prefix_frames = min(dyn_frames, tvm_frames)
    if dyn_frames != tvm_frames:
        print(f"Warning: frames differ (dynamic={dyn_frames}, tvm={tvm_frames}); comparing prefix={prefix_frames}.")

    dyn_f0 = np.asarray(trace_dyn["f0"]).reshape(-1)
    dyn_n = np.asarray(trace_dyn["n"]).reshape(-1)
    f_len_dyn = min(dyn_f0.size, tvm_f0.size, prefix_frames * 2)
    n_len_dyn = min(dyn_n.size, tvm_n.size, prefix_frames * 2)
    show("f0[:2*prefix_frames]", dyn_f0[:f_len_dyn], tvm_f0[:f_len_dyn])
    show("n[:2*prefix_frames]", dyn_n[:n_len_dyn], tvm_n[:n_len_dyn])

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / "trace_dynamic.wav", np.asarray(trace_dyn["audio_trimmed"]), 24000)
        sf.write(out_dir / "trace_static.wav", audio_static, 24000)
        sf.write(out_dir / "trace_static_nopack.wav", np.asarray(trace_static_nopack["audio_trimmed"]), 24000)
        sf.write(out_dir / "trace_tvm.wav", audio_tvm, 24000)
        print(f"Wrote {out_dir / 'trace_dynamic.wav'}")
        print(f"Wrote {out_dir / 'trace_static.wav'}")
        print(f"Wrote {out_dir / 'trace_static_nopack.wav'}")
        print(f"Wrote {out_dir / 'trace_tvm.wav'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
