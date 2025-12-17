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
import tvm
from huggingface_hub import hf_hub_download
from kokoro.model import KModel
from kokoro.pipeline import KPipeline

from kokoro_tvm.cli.inference import load_voice_pack, select_ref_s, text_to_ids
from kokoro_tvm.pipeline import KokoroPipeline, SAMPLES_PER_FRAME, STATIC_AUDIO_LEN, STATIC_TEXT_LEN

_plain_print = print

try:
    from rich.console import Console
    from rich.pretty import Pretty
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
    console = Console()
except ImportError:
    _RICH_AVAILABLE = False
    console = None


_USE_RICH = _RICH_AVAILABLE


def _set_use_rich(use_rich: bool) -> None:
    global _USE_RICH, print  # noqa: A001 - intentional rebind for pretty output
    _USE_RICH = bool(use_rich and _RICH_AVAILABLE)
    if _USE_RICH:
        print = console.print  # type: ignore[assignment]
    else:
        print = _plain_print  # type: ignore[assignment]


def _rule(title: str) -> None:
    if _USE_RICH:
        console.rule(title)
    else:
        print(f"\n{title}\n" + "-" * len(title))


def _fmt_sci(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3e}"


def _err_text(x: float, *, green_max: float = 1e-3, yellow_max: float = 1.0) -> object:
    if not np.isfinite(x):
        return Text("nan", style="red") if _USE_RICH else "nan"
    s = f"{float(x):.3e}"
    if not _USE_RICH:
        return s
    if x < green_max:
        return Text(s, style="green")
    if x < yellow_max:
        return Text(s, style="yellow")
    return Text(s, style="red")


def _print_metrics_table(title: str, rows: list[tuple[str, dict[str, float]]]) -> None:
    if not rows:
        return
    if _USE_RICH:
        table = Table(title=title, show_lines=False)
        table.add_column("name", overflow="fold")
        table.add_column("mae", justify="right")
        table.add_column("max_abs", justify="right")
        table.add_column("rmse", justify="right")
        for name, m in rows:
            table.add_row(name, _err_text(m["mae"]), _err_text(m["max_abs"]), _err_text(m["rmse"]))
        console.print(table)
        return

    print(title)
    for name, m in rows:
        print(f"{name}: mae={_fmt_sci(m['mae'])} max_abs={_fmt_sci(m['max_abs'])} rmse={_fmt_sci(m['rmse'])}")


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
        return {
            "valid": valid,
            "total": total,
            "pad": 0,
            "finite_frac": 1.0,
            "nonzero_frac": 0.0,
            "head": [],
            "tail": [],
        }
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


def _array_percentiles(
    name: str,
    x: np.ndarray,
    *,
    q: tuple[float, ...] = (0.0, 1.0, 5.0, 50.0, 95.0, 99.0, 100.0),
    max_samples: int = 200_000,
    seed: int = 0,
) -> None:
    arr = np.asarray(x).reshape(-1).astype(np.float32, copy=False)
    if arr.size == 0:
        print(f"{name}: empty")
        return
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size == 0:
        print(f"{name}: all non-finite")
        return
    if arr.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(arr.size, size=max_samples, replace=False)
        arr = arr[idx]

    qs = np.array(q, dtype=np.float32) / 100.0
    vals = np.quantile(arr, qs).astype(np.float32, copy=False)
    abs_vals = np.quantile(np.abs(arr), qs).astype(np.float32, copy=False)

    q_str = ",".join([f"p{int(v):02d}" if v.is_integer() else f"p{v:g}" for v in q])
    vals_str = ", ".join([f"{v:.4g}" for v in vals.tolist()])
    abs_str = ", ".join([f"{v:.4g}" for v in abs_vals.tolist()])
    print(f"{name}: {q_str}=[{vals_str}] abs=[{abs_str}] (sample_n={arr.size})")


def _summary_stats_1d(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x).reshape(-1).astype(np.float32, copy=False)
    if arr.size == 0:
        return {"n": 0.0, "finite_frac": 1.0, "max_abs": 0.0, "std": 0.0}
    finite = np.isfinite(arr)
    finite_frac = float(np.mean(finite))
    arr_f = arr[finite] if np.any(finite) else np.array([], dtype=np.float32)
    max_abs = float(np.max(np.abs(arr_f))) if arr_f.size else float("nan")
    std = float(np.std(arr_f)) if arr_f.size else float("nan")
    return {"n": float(arr.size), "finite_frac": finite_frac, "max_abs": max_abs, "std": std}


def _audio_stats_1d(x: np.ndarray, *, atol: float = 1e-8) -> dict[str, float]:
    arr = np.asarray(x).reshape(-1).astype(np.float32, copy=False)
    if arr.size == 0:
        return {
            "n": 0.0,
            "finite_frac": 1.0,
            "nonzero_frac": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
        }
    finite = np.isfinite(arr)
    finite_frac = float(np.mean(finite))
    arr_f = arr[finite] if np.any(finite) else np.array([], dtype=np.float32)
    min_v = float(np.min(arr_f)) if arr_f.size else float("nan")
    max_v = float(np.max(arr_f)) if arr_f.size else float("nan")
    mean_v = float(np.mean(arr_f)) if arr_f.size else float("nan")
    std_v = float(np.std(arr_f)) if arr_f.size else float("nan")
    max_abs = float(np.max(np.abs(arr_f))) if arr_f.size else float("nan")
    nonzero_frac = float(np.mean(np.abs(arr) > float(atol)))
    return {
        "n": float(arr.size),
        "finite_frac": finite_frac,
        "nonzero_frac": nonzero_frac,
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
        "max_abs": max_abs,
    }


def _fmt_bool_flag(ok: bool, text: str) -> str:
    if _USE_RICH:
        return f"[green]{text}[/green]" if ok else f"[red]{text}[/red]"
    return text if ok else f"!! {text}"


def _fmt_float_flag(
    value: float, *, ok_min: float | None = None, ok_max: float | None = None, fmt: str = "{:.4f}"
) -> str:
    ok = True
    if ok_min is not None:
        ok = ok and (value >= ok_min)
    if ok_max is not None:
        ok = ok and (value <= ok_max)
    s = "nan" if not np.isfinite(value) else fmt.format(value)
    return _fmt_bool_flag(ok, s)


def _table_float_flag(
    value: float, *, ok_min: float | None = None, ok_max: float | None = None, fmt: str = "{:.3f}"
) -> object:
    ok = True
    if ok_min is not None:
        ok = ok and (value >= ok_min)
    if ok_max is not None:
        ok = ok and (value <= ok_max)
    s = "nan" if not np.isfinite(value) else fmt.format(float(value))
    if not _USE_RICH:
        return s if ok else f"!! {s}"
    return Text(s, style="green" if ok else "red")


def _table_corr(corr: float, *, good_min: float = 0.7, warn_min: float = 0.2) -> object:
    if not np.isfinite(corr):
        return Text("nan", style="red") if _USE_RICH else "nan"
    s = f"{float(corr):.4f}"
    if not _USE_RICH:
        return s
    if corr >= good_min:
        return Text(s, style="green")
    if corr >= warn_min:
        return Text(s, style="yellow")
    return Text(s, style="red")


def _parse_int_list(s: str | None) -> list[int]:
    if not s:
        return []
    items: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _print_summary_table(title: str, rows: list[tuple[str, str, str, str]]) -> None:
    if not rows:
        return
    if _USE_RICH:
        table = Table(title=title, show_lines=False)
        table.add_column("check", overflow="fold")
        table.add_column("value", overflow="fold")
        table.add_column("finite", overflow="fold")
        table.add_column("notes", overflow="fold")
        for check, value, finite, notes in rows:
            table.add_row(check, value, finite, notes)
        console.print(table)
        return

    print(title)
    for check, value, finite, notes in rows:
        print(f"{check}: {value} | {finite} | {notes}")


def _print_decoder_input_stats(stats: dict[str, object]) -> None:
    if not stats:
        return

    def _cell(v: float, *, kind: str) -> str:
        if kind == "finite_frac":
            return _fmt_float_flag(float(v), ok_min=1.0, fmt="{:.3f}")
        if kind == "nonzero_frac":
            return _fmt_float_flag(float(v), ok_min=0.0, ok_max=1.0, fmt="{:.3f}")
        if kind in {"min", "max", "mean", "std"}:
            return "nan" if not np.isfinite(v) else f"{float(v):.4g}"
        return str(v)

    rows = []
    for key in ["asr", "f0", "n", "style_128"]:
        s = stats.get(key)
        if s is None:
            continue
        if not isinstance(s, dict):
            continue
        rows.append(
            (
                key,
                _cell(float(s.get("finite_frac", float("nan"))), kind="finite_frac"),
                _cell(float(s.get("nonzero_frac", float("nan"))), kind="nonzero_frac"),
                _cell(float(s.get("min", float("nan"))), kind="min"),
                _cell(float(s.get("max", float("nan"))), kind="max"),
                _cell(float(s.get("mean", float("nan"))), kind="mean"),
                _cell(float(s.get("std", float("nan"))), kind="std"),
            )
        )

    if _USE_RICH:
        table = Table(title="Decoder input stats", show_lines=False)
        table.add_column("tensor")
        table.add_column("finite", justify="right")
        table.add_column("nonzero", justify="right")
        table.add_column("min", justify="right")
        table.add_column("max", justify="right")
        table.add_column("mean", justify="right")
        table.add_column("std", justify="right")
        for r in rows:
            table.add_row(*r)
        console.print(table)
    else:
        print("Decoder input stats:")
        for r in rows:
            tensor, finite, nonzero, min_v, max_v, mean_v, std_v = r
            print(f"  {tensor}: finite={finite} nonzero={nonzero} min={min_v} max={max_v} mean={mean_v} std={std_v}")


def _fmt_corr(value: float) -> str:
    return _fmt_float_flag(value, ok_min=0.7, fmt="{:.4f}")


def _fmt_corr_lag(corr: float, lag: int) -> str:
    return f"corr={_fmt_corr(corr)} lag={lag}"


def _build_full_aln_from_pred_dur(pred_dur: np.ndarray, *, cur_len: int) -> tuple[torch.Tensor, int]:
    return _build_aln_from_pred_dur(pred_dur, token_len=STATIC_TEXT_LEN, cur_len=cur_len, total_frames=STATIC_AUDIO_LEN)


def _build_aln_from_pred_dur(
    pred_dur: np.ndarray,
    *,
    token_len: int,
    cur_len: int,
    total_frames: int | None,
) -> tuple[torch.Tensor, int]:
    pred = torch.as_tensor(pred_dur, dtype=torch.long).reshape(-1)
    if pred.numel() == 0 or cur_len <= 0 or token_len <= 0:
        frames = 0 if total_frames is None else 0
        width = frames if total_frames is None else int(total_frames)
        return torch.zeros((1, int(token_len), width), dtype=torch.float32), 0

    cur_len = int(max(0, min(int(cur_len), int(pred.numel()), int(token_len))))
    pred = pred[:cur_len]
    indices = torch.repeat_interleave(torch.arange(cur_len), pred)
    frames = int(indices.numel())

    if total_frames is not None:
        total_frames = int(total_frames)
        frames = min(frames, total_frames)
        indices = indices[:frames]
        width = total_frames
    else:
        width = frames

    aln = torch.zeros((1, int(token_len), int(width)), dtype=torch.float32)
    if frames:
        aln[0, indices, torch.arange(frames)] = 1.0
    return aln, frames


def _decode_pytorch(
    kmodel: KModel,
    *,
    asr: np.ndarray,
    f0: np.ndarray,
    n: np.ndarray,
    s128: np.ndarray,
    frames: int,
) -> np.ndarray:
    with torch.no_grad():
        asr_t = torch.as_tensor(asr, dtype=torch.float32)
        f0_t = torch.as_tensor(f0, dtype=torch.float32)
        n_t = torch.as_tensor(n, dtype=torch.float32)
        s_t = torch.as_tensor(s128, dtype=torch.float32)
        audio = kmodel.decoder(asr_t, f0_t, n_t, s_t).squeeze().cpu().numpy().astype(np.float32, copy=False)
    target = max(0, min(int(audio.size), int(frames) * SAMPLES_PER_FRAME))
    return audio[:target]


def _decode_tvm(
    pipeline: KokoroPipeline,
    *,
    asr: np.ndarray,
    f0: np.ndarray,
    n: np.ndarray,
    s128: np.ndarray,
    frames: int,
    bucket_len: int | None = None,
) -> np.ndarray:
    dev = pipeline.gpu_dev
    asr_np = np.asarray(asr, dtype=np.float32)
    f0_np = np.asarray(f0, dtype=np.float32)
    n_np = np.asarray(n, dtype=np.float32)
    s_np = np.asarray(s128, dtype=np.float32)

    if bucket_len is not None:
        b = int(bucket_len)
        asr_np = _pad_3d_time(asr_np, target_t=b)
        f0_np = _pad_2d_time(f0_np, target_t=b * 2)
        n_np = _pad_2d_time(n_np, target_t=b * 2)

        fns = getattr(pipeline, "_decoder_fns", None)
        if isinstance(fns, dict) and b in fns:
            f_decoder = fns[b]
        else:
            f_decoder = pipeline.f_decoder
    else:
        f_decoder = pipeline.f_decoder

    out = pipeline._unwrap(
        f_decoder(
            tvm.runtime.tensor(asr_np, device=dev),
            tvm.runtime.tensor(f0_np, device=dev),
            tvm.runtime.tensor(n_np, device=dev),
            tvm.runtime.tensor(s_np, device=dev),
        )
    )
    audio = out.numpy().squeeze().astype(np.float32, copy=False)
    target = max(0, min(int(audio.size), int(frames) * SAMPLES_PER_FRAME))
    return audio[:target]


def _pad_3d_time(x: np.ndarray, *, target_t: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape={arr.shape}")
    t = int(arr.shape[2])
    target_t = int(target_t)
    if t == target_t:
        return arr
    if t > target_t:
        return arr[:, :, :target_t]
    out = np.zeros((arr.shape[0], arr.shape[1], target_t), dtype=np.float32)
    out[:, :, :t] = arr
    return out


def _pad_2d_time(x: np.ndarray, *, target_t: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape={arr.shape}")
    t = int(arr.shape[1])
    target_t = int(target_t)
    if t == target_t:
        return arr
    if t > target_t:
        return arr[:, :target_t]
    out = np.zeros((arr.shape[0], target_t), dtype=np.float32)
    out[:, :t] = arr
    return out


def _pad_last_dim(x: np.ndarray, *, target_t: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim < 1:
        raise ValueError(f"Expected >=1D tensor, got shape={arr.shape}")
    t = int(arr.shape[-1])
    target_t = int(target_t)
    if t == target_t:
        return arr
    if t > target_t:
        return arr[..., :target_t]
    out = np.zeros((*arr.shape[:-1], target_t), dtype=np.float32)
    out[..., :t] = arr
    return out


def _compute_asr_from_trace(trace: dict[str, object], *, cur_len: int) -> tuple[np.ndarray, int]:
    return _compute_asr_from_trace_with_target(trace, cur_len=cur_len, total_frames=STATIC_AUDIO_LEN)


def _compute_asr_from_trace_with_target(
    trace: dict[str, object],
    *,
    cur_len: int,
    total_frames: int | None,
) -> tuple[np.ndarray, int]:
    pred_dur = np.asarray(trace["pred_dur"]).reshape(-1)
    t_en = torch.as_tensor(np.asarray(trace["t_en"]), dtype=torch.float32)
    token_len = int(t_en.shape[-1])
    aln, frames = _build_aln_from_pred_dur(pred_dur, token_len=token_len, cur_len=cur_len, total_frames=total_frames)
    asr = (t_en @ aln).numpy().astype(np.float32, copy=False)
    return asr, frames


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

        pred_aln_trg_prefix = pred_aln_trg[:, :, :frames]
        d_prefix = d[:, :cur_len, :]
        en_prefix = d_prefix.transpose(-1, -2) @ pred_aln_trg_prefix
        f0_prefix, n_prefix = kmodel.predictor.F0Ntrain(en_prefix, s)
        f0_pred = torch.as_tensor(_pad_last_dim(f0_prefix.numpy(), target_t=STATIC_AUDIO_LEN * 2), dtype=torch.float32)
        n_pred = torch.as_tensor(_pad_last_dim(n_prefix.numpy(), target_t=STATIC_AUDIO_LEN * 2), dtype=torch.float32)
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
    parser.add_argument("--no-rich", action="store_true", help="Disable Rich formatting (plain text output)")
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output (distributions, percentiles, extra tables)"
    )
    parser.add_argument(
        "--cross-decoder",
        action="store_true",
        help="Run crossed decoder tests (TVM encoder -> PyTorch decoder, and PyTorch encoder -> TVM decoder)",
    )
    parser.add_argument(
        "--decoder-bucket-sweep",
        action="store_true",
        help="Sweep across available TVM decoder bucket lengths and report audio correlation vs references",
    )
    parser.add_argument(
        "--decoder-buckets",
        type=str,
        default=None,
        help="Comma-separated decoder bucket lengths to evaluate (defaults to all buckets present in lib-dir)",
    )
    parser.add_argument(
        "--decoder-bucket-sweep-input",
        type=str,
        default="pt.dynamic",
        choices=["pt.dynamic", "pt.static", "tvm"],
        help="Which inputs to feed the TVM decoder during bucket sweep",
    )
    parser.add_argument(
        "--tvm-ref-s",
        type=str,
        default="inference",
        choices=["inference", "hf"],
        help="Select style source for TVM pipeline: inference voice pack (default) or HF voice pack (matches PyTorch)",
    )
    args = parser.parse_args()

    _set_use_rich(not args.no_rich)

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
    ref_s_hf = select_ref_s(voice_pack_ref, len(phonemes))
    ref_s = ref_s_hf

    trace_dyn = _trace_dynamic(kmodel, phonemes, ref_s, args.speed)
    trace_static = _trace_static(kmodel, phonemes, ref_s, args.speed)
    with _mock_packed_sequence():
        trace_static_nopack = _trace_static(kmodel, phonemes, ref_s, args.speed)

    vocab = kmodel.vocab
    input_ids_tvm = text_to_ids(phonemes, vocab)
    voice_pack_tvm = load_voice_pack(args.voice)
    ref_s_inference = select_ref_s(voice_pack_tvm, len(phonemes))

    ref_s_tvm = ref_s_inference if args.tvm_ref_s == "inference" else ref_s_hf

    pipeline = KokoroPipeline(args.lib_dir, args.device, hybrid=args.hybrid)
    trace_tvm = pipeline.trace(input_ids_tvm, ref_s_tvm, speed=args.speed)

    cur_len = int(trace_static["cur_len"])
    _rule("Inputs")
    print(
        f"text={args.text!r} voice={args.voice!r} device={args.device!r} lib_dir={args.lib_dir!r} tvm_ref_s={args.tvm_ref_s!r}"
    )
    print(f"cur_len={cur_len}")
    print(
        f"frames_dynamic={trace_dyn['frames']}, frames_static={trace_static['frames']}, frames_tvm={trace_tvm['frames']}"
    )

    _print_metrics_table(
        "Style vectors",
        [
            ("ref_s(hf) vs ref_s(inference)", _metrics(ref_s_hf.numpy(), ref_s_inference.numpy())),
            ("ref_s(hf) vs ref_s(tvm used)", _metrics(ref_s_hf.numpy(), ref_s_tvm.numpy())),
        ],
    )

    _rule("Static PyTorch vs TVM (module fidelity)")
    module_rows: list[tuple[str, dict[str, float]]] = []
    module_rows.append(
        (
            "bert.d_en[:cur_len]",
            _metrics(np.asarray(trace_static["d_en"])[:, :, :cur_len], np.asarray(trace_tvm["d_en"])[:, :, :cur_len]),
        )
    )
    module_rows.append(
        (
            "duration.logits[:cur_len]",
            _metrics(
                np.asarray(trace_static["duration_logits"])[:, :cur_len, :],
                np.asarray(trace_tvm["duration_logits"])[:, :cur_len, :],
            ),
        )
    )
    module_rows.append(
        (
            "duration.d[:cur_len]",
            _metrics(np.asarray(trace_static["d"])[:, :cur_len, :], np.asarray(trace_tvm["d"])[:, :cur_len, :]),
        )
    )
    module_rows.append(
        (
            "text_encoder.t_en[:cur_len]",
            _metrics(np.asarray(trace_static["t_en"])[:, :, :cur_len], np.asarray(trace_tvm["t_en"])[:, :, :cur_len]),
        )
    )

    static_frames = int(trace_static["frames"])
    tvm_f0 = np.asarray(trace_tvm["f0"]).reshape(-1)
    tvm_n = np.asarray(trace_tvm["n"]).reshape(-1)
    static_f0 = np.asarray(trace_static["f0"]).reshape(-1)
    static_n = np.asarray(trace_static["n"]).reshape(-1)
    f_len = min(static_f0.size, tvm_f0.size, static_frames * 2)
    module_rows.append(("f0[:2*frames_static]", _metrics(static_f0[:f_len], tvm_f0[:f_len])))
    n_len = min(static_n.size, tvm_n.size, static_frames * 2)
    module_rows.append(("n[:2*frames_static]", _metrics(static_n[:n_len], tvm_n[:n_len])))
    _print_metrics_table("Module fidelity", module_rows)

    _rule("F0/N padded tail (what fills beyond valid frames?)")
    valid_f0n = static_frames * 2
    print(f"frames_static={static_frames} valid_f0n=2*frames_static={valid_f0n}")
    tvm_f0_pad = trace_tvm.get("f0_pad_summary") or _tail_summary_1d(tvm_f0, valid_f0n)
    tvm_n_pad = trace_tvm.get("n_pad_summary") or _tail_summary_1d(tvm_n, valid_f0n)
    static_f0_pad = _tail_summary_1d(static_f0, valid_f0n)
    static_n_pad = _tail_summary_1d(static_n, valid_f0n)
    if _USE_RICH:
        tail_table = Table(show_lines=False)
        tail_table.add_column("tensor")
        tail_table.add_column("valid", justify="right")
        tail_table.add_column("total", justify="right")
        tail_table.add_column("pad", justify="right")
        tail_table.add_column("finite_frac", justify="right")
        tail_table.add_column("nonzero_frac", justify="right")
        tail_table.add_column("head")
        tail_table.add_column("tail")
        for label, s in [
            ("tvm.f0", tvm_f0_pad),
            ("tvm.n", tvm_n_pad),
            ("pt.static f0", static_f0_pad),
            ("pt.static n", static_n_pad),
        ]:
            tail_table.add_row(
                label,
                str(s["valid"]),
                str(s["total"]),
                str(s["pad"]),
                f"{float(s['finite_frac']):.3f}",
                f"{float(s['nonzero_frac']):.3f}",
                str(s["head"]),
                str(s["tail"]),
            )
        console.print(tail_table)
    else:
        for label, s in [
            ("tvm.f0", tvm_f0_pad),
            ("tvm.n", tvm_n_pad),
            ("pt.static f0", static_f0_pad),
            ("pt.static n", static_n_pad),
        ]:
            print(
                f"{label} valid={s['valid']} total={s['total']} pad={s['pad']} "
                f"finite_frac={float(s['finite_frac']):.3f} nonzero_frac={float(s['nonzero_frac']):.3f} "
                f"head={s['head']} tail={s['tail']}"
            )

    if args.verbose:
        _rule("F0/N distribution (valid + padded)")
        dyn_f0 = np.asarray(trace_dyn["f0"]).reshape(-1)
        dyn_n = np.asarray(trace_dyn["n"]).reshape(-1)
        _array_percentiles("pt.dynamic f0", dyn_f0)
        _array_percentiles("pt.dynamic n", dyn_n)
        _array_percentiles("pt.static f0[:2*frames_static]", static_f0[:valid_f0n])
        _array_percentiles("pt.static n[:2*frames_static]", static_n[:valid_f0n])
        _array_percentiles("tvm f0[:2*frames_static]", tvm_f0[:valid_f0n])
        _array_percentiles("tvm n[:2*frames_static]", tvm_n[:valid_f0n])
        _array_percentiles("pt.static f0[pad]", static_f0[valid_f0n:])
        _array_percentiles("pt.static n[pad]", static_n[valid_f0n:])
        _array_percentiles("tvm f0[pad]", tvm_f0[valid_f0n:])
        _array_percentiles("tvm n[pad]", tvm_n[valid_f0n:])

    audio_static = np.asarray(trace_static["audio_trimmed"]).reshape(-1)
    audio_tvm = np.asarray(trace_tvm["audio_trimmed"]).reshape(-1)
    _rule("Decoder audio stats")
    _array_stats("pt.static audio_trimmed", audio_static)
    _array_stats("tvm audio_trimmed", audio_tvm)
    if "decoder_input_stats" in trace_tvm:
        stats = trace_tvm["decoder_input_stats"]
        _print_decoder_input_stats(stats)
    corr, lag = _best_lag_corr(audio_static, audio_tvm, max_lag=2400)
    print(f"decoder.audio_trimmed corr={corr:.4f} lag={lag} samples")

    _rule("Alignment/ASR (decoder conditioning) fidelity")
    tvm_pred_dur = np.asarray(trace_tvm["pred_dur"]).reshape(-1)
    static_pred_dur = np.asarray(trace_static["pred_dur"]).reshape(-1)
    static_np_pred_dur = np.asarray(trace_static_nopack["pred_dur"]).reshape(-1)
    dyn_pred_dur = np.asarray(trace_dyn["pred_dur"]).reshape(-1)

    _print_metrics_table(
        "pred_dur",
        [
            (
                "pred_dur(pt.static)[:cur_len]",
                _metrics(static_pred_dur[:cur_len].astype(np.float32), tvm_pred_dur[:cur_len].astype(np.float32)),
            ),
            (
                "pred_dur(pt.no-pack)[:cur_len]",
                _metrics(static_np_pred_dur[:cur_len].astype(np.float32), tvm_pred_dur[:cur_len].astype(np.float32)),
            ),
        ],
    )

    asr_tvm, frames_tvm_recon = _compute_asr_from_trace_with_target(
        trace_tvm, cur_len=cur_len, total_frames=STATIC_AUDIO_LEN
    )
    asr_static, frames_static_recon = _compute_asr_from_trace_with_target(
        trace_static, cur_len=cur_len, total_frames=STATIC_AUDIO_LEN
    )
    asr_static_np, frames_static_np_recon = _compute_asr_from_trace_with_target(
        trace_static_nopack, cur_len=cur_len, total_frames=STATIC_AUDIO_LEN
    )
    asr_dyn, frames_dyn_recon = _compute_asr_from_trace_with_target(trace_dyn, cur_len=cur_len, total_frames=None)

    frames_static = int(trace_static["frames"])
    frames_tvm = int(trace_tvm["frames"])
    frames_dyn = int(trace_dyn["frames"])
    if frames_static != frames_static_recon:
        print(f"Warning: pt.static frames mismatch (trace={frames_static}, recon={frames_static_recon})")
    if frames_tvm != frames_tvm_recon:
        print(f"Warning: tvm frames mismatch (trace={frames_tvm}, recon={frames_tvm_recon})")
    if frames_dyn != frames_dyn_recon:
        print(f"Warning: pt.dynamic frames mismatch (trace={frames_dyn}, recon={frames_dyn_recon})")

    prefix_frames = min(frames_static_recon, frames_tvm_recon)
    _print_metrics_table(
        "asr",
        [
            (
                "asr(pt.static)[:prefix_frames]",
                _metrics(asr_static[:, :, :prefix_frames], asr_tvm[:, :, :prefix_frames]),
            ),
            (
                "asr(pt.no-pack)[:prefix_frames]",
                _metrics(asr_static_np[:, :, :prefix_frames], asr_tvm[:, :, :prefix_frames]),
            ),
        ],
    )
    if args.verbose:
        _array_percentiles("asr(pt.static)[:prefix_frames]", asr_static[:, :, :prefix_frames])
        _array_percentiles("asr(tvm)[:prefix_frames]", asr_tvm[:, :, :prefix_frames])

    _rule("PyTorch dynamic vs static conditioning (prefix)")
    prefix_ds = min(frames_dyn_recon, frames_static_recon)
    _print_metrics_table(
        "dynamic vs static (conditioning)",
        [
            (
                "pred_dur(pt.dynamic)[:cur_len]",
                _metrics(dyn_pred_dur[:cur_len].astype(np.float32), static_pred_dur[:cur_len].astype(np.float32)),
            ),
            ("asr(pt.dynamic)[:prefix]", _metrics(asr_dyn[:, :, :prefix_ds], asr_static[:, :, :prefix_ds])),
        ],
    )
    dyn_f0_full = np.asarray(trace_dyn["f0"]).astype(np.float32, copy=False)
    dyn_n_full = np.asarray(trace_dyn["n"]).astype(np.float32, copy=False)
    static_f0_full_2d = np.asarray(trace_static["f0"]).astype(np.float32, copy=False)
    static_n_full_2d = np.asarray(trace_static["n"]).astype(np.float32, copy=False)
    f0_prefix = min(dyn_f0_full.shape[1], static_f0_full_2d.shape[1], prefix_ds * 2)
    n_prefix = min(dyn_n_full.shape[1], static_n_full_2d.shape[1], prefix_ds * 2)
    _print_metrics_table(
        "dynamic vs static (F0/N prefix)",
        [
            ("f0[:2*prefix_frames]", _metrics(dyn_f0_full[:, :f0_prefix], static_f0_full_2d[:, :f0_prefix])),
            ("n[:2*prefix_frames]", _metrics(dyn_n_full[:, :n_prefix], static_n_full_2d[:, :n_prefix])),
        ],
    )

    _rule("Static PyTorch packed vs no-pack (packing semantics impact)")
    _print_metrics_table(
        "packed vs no-pack",
        [
            (
                "duration.logits[:cur_len]",
                _metrics(
                    np.asarray(trace_static["duration_logits"])[:, :cur_len, :],
                    np.asarray(trace_static_nopack["duration_logits"])[:, :cur_len, :],
                ),
            ),
            (
                "text_encoder.t_en[:cur_len]",
                _metrics(
                    np.asarray(trace_static["t_en"])[:, :, :cur_len],
                    np.asarray(trace_static_nopack["t_en"])[:, :, :cur_len],
                ),
            ),
        ],
    )
    corr_np, lag_np = _best_lag_corr(
        np.asarray(trace_static["audio_trimmed"]).reshape(-1),
        np.asarray(trace_static_nopack["audio_trimmed"]).reshape(-1),
        max_lag=2400,
    )
    print(f"audio_trimmed corr={corr_np:.4f} lag={lag_np} samples")

    _rule("Static PyTorch no-pack vs TVM (does TVM match no-pack semantics?)")
    _print_metrics_table(
        "no-pack vs tvm",
        [
            (
                "duration.logits[:cur_len]",
                _metrics(
                    np.asarray(trace_static_nopack["duration_logits"])[:, :cur_len, :],
                    np.asarray(trace_tvm["duration_logits"])[:, :cur_len, :],
                ),
            ),
            (
                "text_encoder.t_en[:cur_len]",
                _metrics(
                    np.asarray(trace_static_nopack["t_en"])[:, :, :cur_len],
                    np.asarray(trace_tvm["t_en"])[:, :, :cur_len],
                ),
            ),
        ],
    )
    corr_np2, lag_np2 = _best_lag_corr(np.asarray(trace_static_nopack["audio_trimmed"]).reshape(-1), audio_tvm, 2400)
    print(f"decoder.audio_trimmed corr={corr_np2:.4f} lag={lag_np2} samples")

    if args.cross_decoder:
        _rule("Crossed decoder inputs (encoder->decoder matrix)")

        tvm_full_aln, tvm_frames = _build_full_aln_from_pred_dur(tvm_pred_dur, cur_len=cur_len)
        tvm_t_en = torch.as_tensor(np.asarray(trace_tvm["t_en"]), dtype=torch.float32)
        asr_tvm = (tvm_t_en @ tvm_full_aln).numpy()

        tvm_f0_full = np.asarray(trace_tvm["f0"])
        tvm_n_full = np.asarray(trace_tvm["n"])
        tvm_s128 = ref_s_tvm[:, :128].cpu().numpy().astype(np.float32, copy=False)

        audio_pt_from_tvm = _decode_pytorch(
            kmodel,
            asr=asr_tvm,
            f0=tvm_f0_full,
            n=tvm_n_full,
            s128=tvm_s128,
            frames=tvm_frames,
        )
        corr_x1, lag_x1 = _best_lag_corr(audio_static, audio_pt_from_tvm, max_lag=2400)
        corr_x1d, lag_x1d = _best_lag_corr(
            np.asarray(trace_dyn["audio_trimmed"]).reshape(-1), audio_pt_from_tvm, max_lag=2400
        )
        corr_x1b, lag_x1b = _best_lag_corr(audio_tvm, audio_pt_from_tvm, max_lag=2400)

        pt_pred_dur = np.asarray(trace_static["pred_dur"]).reshape(-1)
        pt_full_aln, pt_frames = _build_full_aln_from_pred_dur(pt_pred_dur, cur_len=cur_len)
        pt_t_en = torch.as_tensor(np.asarray(trace_static["t_en"]), dtype=torch.float32)
        asr_pt = (pt_t_en @ pt_full_aln).numpy()
        pt_f0_full = np.asarray(trace_static["f0"])
        pt_n_full = np.asarray(trace_static["n"])
        pt_s128 = ref_s[:, :128].cpu().numpy().astype(np.float32, copy=False)

        audio_tvm_from_pt = _decode_tvm(
            pipeline,
            asr=asr_pt,
            f0=pt_f0_full,
            n=pt_n_full,
            s128=pt_s128,
            frames=pt_frames,
        )
        corr_x2, lag_x2 = _best_lag_corr(audio_static, audio_tvm_from_pt, max_lag=2400)
        corr_x2b2, lag_x2b2 = _best_lag_corr(
            np.asarray(trace_dyn["audio_trimmed"]).reshape(-1), audio_tvm_from_pt, max_lag=2400
        )
        corr_x2b, lag_x2b = _best_lag_corr(audio_tvm, audio_tvm_from_pt, max_lag=2400)

        pt_np_pred_dur = np.asarray(trace_static_nopack["pred_dur"]).reshape(-1)
        pt_np_full_aln, pt_np_frames = _build_full_aln_from_pred_dur(pt_np_pred_dur, cur_len=cur_len)
        pt_np_t_en = torch.as_tensor(np.asarray(trace_static_nopack["t_en"]), dtype=torch.float32)
        asr_pt_np = (pt_np_t_en @ pt_np_full_aln).numpy()
        pt_np_f0_full = np.asarray(trace_static_nopack["f0"])
        pt_np_n_full = np.asarray(trace_static_nopack["n"])

        audio_tvm_from_pt_np = _decode_tvm(
            pipeline,
            asr=asr_pt_np,
            f0=pt_np_f0_full,
            n=pt_np_n_full,
            s128=pt_s128,
            frames=pt_np_frames,
        )
        corr_x3, lag_x3 = _best_lag_corr(audio_static, audio_tvm_from_pt_np, max_lag=2400)
        corr_x3b, lag_x3b = _best_lag_corr(
            np.asarray(trace_static_nopack["audio_trimmed"]).reshape(-1), audio_tvm_from_pt_np, max_lag=2400
        )
        corr_x3d, lag_x3d = _best_lag_corr(
            np.asarray(trace_dyn["audio_trimmed"]).reshape(-1), audio_tvm_from_pt_np, max_lag=2400
        )
        corr_x3t, lag_x3t = _best_lag_corr(audio_tvm, audio_tvm_from_pt_np, max_lag=2400)

        asr_dyn_pad = _pad_3d_time(asr_dyn, target_t=STATIC_AUDIO_LEN)
        f0_dyn_pad = _pad_2d_time(dyn_f0_full, target_t=STATIC_AUDIO_LEN * 2)
        n_dyn_pad = _pad_2d_time(dyn_n_full, target_t=STATIC_AUDIO_LEN * 2)
        frames_dyn_clamped = min(int(frames_dyn_recon), STATIC_AUDIO_LEN)
        audio_tvm_from_dyn = _decode_tvm(
            pipeline,
            asr=asr_dyn_pad,
            f0=f0_dyn_pad,
            n=n_dyn_pad,
            s128=pt_s128,
            frames=frames_dyn_clamped,
        )
        corr_x4, lag_x4 = _best_lag_corr(
            np.asarray(trace_dyn["audio_trimmed"]).reshape(-1), audio_tvm_from_dyn, max_lag=2400
        )
        corr_x4s, lag_x4s = _best_lag_corr(audio_static, audio_tvm_from_dyn, max_lag=2400)
        corr_x4t, lag_x4t = _best_lag_corr(audio_tvm, audio_tvm_from_dyn, max_lag=2400)

        if _USE_RICH:
            stats_table = Table(title="Crossed decoder: audio stats", show_lines=False)
            stats_table.add_column("case", overflow="fold")
            stats_table.add_column("frames", justify="right")
            stats_table.add_column("finite", justify="right")
            stats_table.add_column("nonzero", justify="right")
            stats_table.add_column("max_abs", justify="right")
            stats_table.add_column("std", justify="right")

            corr_table = Table(title="Crossed decoder: correlations", show_lines=False)
            corr_table.add_column("case", overflow="fold")
            corr_table.add_column("vs pt.dynamic", justify="right")
            corr_table.add_column("lag", justify="right")
            corr_table.add_column("vs pt.static", justify="right")
            corr_table.add_column("lag", justify="right")
            corr_table.add_column("vs tvm", justify="right")
            corr_table.add_column("lag", justify="right")
            corr_table.add_column("vs pt.no-pack", justify="right")
            corr_table.add_column("lag", justify="right")

            cases = [
                (
                    "pt(dec<-tvm)",
                    audio_pt_from_tvm,
                    int(tvm_frames),
                    (corr_x1d, lag_x1d),
                    (corr_x1, lag_x1),
                    (corr_x1b, lag_x1b),
                    (0.0, 0),
                ),
                (
                    "tvm(dec<-pt.static)",
                    audio_tvm_from_pt,
                    int(pt_frames),
                    (corr_x2b2, lag_x2b2),
                    (corr_x2, lag_x2),
                    (corr_x2b, lag_x2b),
                    (0.0, 0),
                ),
                (
                    "tvm(dec<-pt.no-pack)",
                    audio_tvm_from_pt_np,
                    int(pt_np_frames),
                    (corr_x3d, lag_x3d),
                    (corr_x3, lag_x3),
                    (corr_x3t, lag_x3t),
                    (corr_x3b, lag_x3b),
                ),
                (
                    "tvm(dec<-pt.dynamic)",
                    audio_tvm_from_dyn,
                    int(frames_dyn_clamped),
                    (corr_x4, lag_x4),
                    (corr_x4s, lag_x4s),
                    (corr_x4t, lag_x4t),
                    (0.0, 0),
                ),
            ]

            def _fmt_sci_or_nan(v: float) -> str:
                return "nan" if not np.isfinite(v) else f"{float(v):.3e}"

            for name, audio_case, frames_case, (cd, ld), (cs, ls), (ct, lt), (cnp, lnp) in cases:
                st = _audio_stats_1d(audio_case)
                stats_table.add_row(
                    name,
                    str(frames_case),
                    _table_float_flag(st["finite_frac"], ok_min=1.0),
                    _table_float_flag(st["nonzero_frac"], ok_min=0.0, ok_max=1.0),
                    _fmt_sci_or_nan(st["max_abs"]),
                    _fmt_sci_or_nan(st["std"]),
                )
                corr_table.add_row(
                    name,
                    _table_corr(cd),
                    str(int(ld)),
                    _table_corr(cs),
                    str(int(ls)),
                    _table_corr(ct),
                    str(int(lt)),
                    _table_corr(cnp) if name == "tvm(dec<-pt.no-pack)" else "-",
                    str(int(lnp)) if name == "tvm(dec<-pt.no-pack)" else "-",
                )

            console.print(stats_table)
            console.print(corr_table)
        else:
            _array_stats("pt(dec<-tvm).audio_trimmed", audio_pt_from_tvm)
            if args.verbose:
                _array_percentiles("pt(dec<-tvm).audio_trimmed", audio_pt_from_tvm, max_samples=100_000)
            print(f"pt(dec<-tvm) corr(vs pt.static)={corr_x1:.4f} lag={lag_x1} samples")
            print(f"pt(dec<-tvm) corr(vs pt.dynamic)={corr_x1d:.4f} lag={lag_x1d} samples")
            print(f"pt(dec<-tvm) corr(vs tvm)={corr_x1b:.4f} lag={lag_x1b} samples")

            _array_stats("tvm(dec<-pt.static).audio_trimmed", audio_tvm_from_pt)
            if args.verbose:
                _array_percentiles("tvm(dec<-pt.static).audio_trimmed", audio_tvm_from_pt, max_samples=100_000)
            print(f"tvm(dec<-pt.static) corr(vs pt.static)={corr_x2:.4f} lag={lag_x2} samples")
            print(f"tvm(dec<-pt.static) corr(vs pt.dynamic)={corr_x2b2:.4f} lag={lag_x2b2} samples")
            print(f"tvm(dec<-pt.static) corr(vs tvm)={corr_x2b:.4f} lag={lag_x2b} samples")

            _array_stats("tvm(dec<-pt.no-pack).audio_trimmed", audio_tvm_from_pt_np)
            if args.verbose:
                _array_percentiles("tvm(dec<-pt.no-pack).audio_trimmed", audio_tvm_from_pt_np, max_samples=100_000)
            print(f"tvm(dec<-pt.no-pack) corr(vs pt.static)={corr_x3:.4f} lag={lag_x3} samples")
            print(f"tvm(dec<-pt.no-pack) corr(vs pt.no-pack)={corr_x3b:.4f} lag={lag_x3b} samples")

            _array_stats("tvm(dec<-pt.dynamic).audio_trimmed", audio_tvm_from_dyn)
            if args.verbose:
                _array_percentiles("tvm(dec<-pt.dynamic).audio_trimmed", audio_tvm_from_dyn, max_samples=100_000)
            print(f"tvm(dec<-pt.dynamic) corr(vs pt.dynamic)={corr_x4:.4f} lag={lag_x4} samples")
            print(f"tvm(dec<-pt.dynamic) corr(vs pt.static)={corr_x4s:.4f} lag={lag_x4s} samples")

    if args.decoder_bucket_sweep:
        _rule("Decoder bucket sweep")
        available = getattr(pipeline, "_decoder_bucket_lens", None)
        if not isinstance(available, list) or not available:
            available = [STATIC_AUDIO_LEN]
        req = _parse_int_list(args.decoder_buckets)
        bucket_lens = req if req else list(available)
        bucket_lens = sorted(set(int(x) for x in bucket_lens))

        if req:
            missing = [b for b in bucket_lens if b not in available]
            if missing:
                print(f"Warning: requested buckets not found in lib-dir: {missing}")
            bucket_lens = [b for b in bucket_lens if b in available]
            if not bucket_lens:
                print("No requested decoder buckets are available; skipping sweep.")
                bucket_lens = []

        if args.decoder_bucket_sweep_input == "pt.dynamic":
            ref_name = "pt.dynamic"
            ref_audio = np.asarray(trace_dyn["audio_trimmed"]).reshape(-1)
            asr_in = asr_dyn
            f0_in = np.asarray(trace_dyn["f0"]).astype(np.float32, copy=False)
            n_in = np.asarray(trace_dyn["n"]).astype(np.float32, copy=False)
            frames_in = int(frames_dyn_recon)
            s128_in = ref_s[:, :128].cpu().numpy().astype(np.float32, copy=False)
        elif args.decoder_bucket_sweep_input == "pt.static":
            ref_name = "pt.static"
            ref_audio = audio_static
            asr_in = asr_static
            f0_in = np.asarray(trace_static["f0"]).astype(np.float32, copy=False)
            n_in = np.asarray(trace_static["n"]).astype(np.float32, copy=False)
            frames_in = int(frames_static_recon)
            s128_in = ref_s[:, :128].cpu().numpy().astype(np.float32, copy=False)
        else:
            ref_name = "tvm"
            ref_audio = audio_tvm
            asr_in = asr_tvm
            f0_in = np.asarray(trace_tvm["f0"]).astype(np.float32, copy=False)
            n_in = np.asarray(trace_tvm["n"]).astype(np.float32, copy=False)
            frames_in = int(frames_tvm_recon)
            s128_in = ref_s_tvm[:, :128].cpu().numpy().astype(np.float32, copy=False)

        frames_in = max(0, min(int(frames_in), STATIC_AUDIO_LEN))

        f0_in_pad = _pad_2d_time(f0_in, target_t=STATIC_AUDIO_LEN * 2)
        n_in_pad = _pad_2d_time(n_in, target_t=STATIC_AUDIO_LEN * 2)
        stats_by_bucket: dict[int, dict[str, float]] = {}
        corr_rows: list[tuple[int, float, int, float, int, float, int, float, int]] = []

        for b in bucket_lens:
            b = int(b)
            if b < frames_in:
                print(f"bucket={b}: skipped (bucket < frames={frames_in})")
                continue
            try:
                audio_b = _decode_tvm(
                    pipeline,
                    asr=asr_in,
                    f0=f0_in_pad,
                    n=n_in_pad,
                    s128=s128_in,
                    frames=frames_in,
                    bucket_len=b,
                )
            except Exception as e:
                print(f"bucket={b}: error: {e}")
                continue

            st = _audio_stats_1d(audio_b)
            stats_by_bucket[b] = st

            corr_ref, lag_ref = _best_lag_corr(ref_audio, audio_b, max_lag=2400)
            corr_dyn, lag_dyn = _best_lag_corr(
                np.asarray(trace_dyn["audio_trimmed"]).reshape(-1), audio_b, max_lag=2400
            )
            corr_static, lag_static = _best_lag_corr(audio_static, audio_b, max_lag=2400)
            corr_tvm2, lag_tvm2 = _best_lag_corr(audio_tvm, audio_b, max_lag=2400)
            corr_rows.append((b, corr_ref, lag_ref, corr_dyn, lag_dyn, corr_static, lag_static, corr_tvm2, lag_tvm2))

            if args.save_dir:
                out_dir = Path(args.save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                sf.write(
                    out_dir / f"trace_tvm_dec_bucket{b}_from_{args.decoder_bucket_sweep_input}.wav", audio_b, 24000
                )

        if _USE_RICH:
            stats_table = Table(
                title=f"Decoder bucket sweep stats (inputs={args.decoder_bucket_sweep_input})", show_lines=False
            )
            stats_table.add_column("bucket", justify="right")
            stats_table.add_column("finite", justify="right")
            stats_table.add_column("nonzero", justify="right")
            stats_table.add_column("max_abs", justify="right")
            stats_table.add_column("std", justify="right")

            for b, _, _, _, _, _, _, _, _ in corr_rows:
                s = stats_by_bucket.get(int(b))
                if s is None:
                    continue
                stats_table.add_row(
                    str(b),
                    _table_float_flag(float(s["finite_frac"]), ok_min=1.0),
                    _table_float_flag(float(s["nonzero_frac"]), ok_min=0.0, ok_max=1.0),
                    "nan" if not np.isfinite(s["max_abs"]) else f"{float(s['max_abs']):.3e}",
                    "nan" if not np.isfinite(s["std"]) else f"{float(s['std']):.3e}",
                )
            console.print(stats_table)

            corr_table = Table(title=f"Decoder bucket sweep correlations (ref={ref_name})", show_lines=False)
            corr_table.add_column("bucket", justify="right")
            corr_table.add_column(f"vs {ref_name}", justify="right")
            corr_table.add_column("lag", justify="right")
            if ref_name != "pt.dynamic":
                corr_table.add_column("vs pt.dynamic", justify="right")
                corr_table.add_column("lag", justify="right")
            corr_table.add_column("vs pt.static", justify="right")
            corr_table.add_column("lag", justify="right")
            corr_table.add_column("vs tvm", justify="right")
            corr_table.add_column("lag", justify="right")
            for b, cr, lr, cd, ld, cs, ls, ct, lt in corr_rows:
                row = [str(b), _table_corr(cr), str(lr)]
                if ref_name != "pt.dynamic":
                    row.extend([_table_corr(cd), str(ld)])
                row.extend([_table_corr(cs), str(ls), _table_corr(ct), str(lt)])
                corr_table.add_row(*row)
            console.print(corr_table)
        else:
            print(f"Decoder bucket sweep (inputs={args.decoder_bucket_sweep_input} ref={ref_name})")
            for b, cr, lr, cd, ld, cs, ls, ct, lt in corr_rows:
                parts = [f"bucket={b}: corr(vs {ref_name})={cr:.4f} lag={lr}"]
                if ref_name != "pt.dynamic":
                    parts.append(f"corr(vs pt.dynamic)={cd:.4f} lag={ld}")
                parts.append(f"corr(vs pt.static)={cs:.4f} lag={ls}")
                parts.append(f"corr(vs tvm)={ct:.4f} lag={lt}")
                print(" ".join(parts))

    _rule("Dynamic PyTorch vs Static PyTorch (shape/static padding impact)")
    _print_metrics_table(
        "dynamic vs static",
        [
            (
                "bert.d_en[:cur_len]",
                _metrics(np.asarray(trace_dyn["d_en"]), np.asarray(trace_static["d_en"])[:, :, :cur_len]),
            ),
            (
                "duration.logits",
                _metrics(
                    np.asarray(trace_dyn["duration_logits"]),
                    np.asarray(trace_static["duration_logits"])[:, :cur_len, :],
                ),
            ),
            ("duration.d", _metrics(np.asarray(trace_dyn["d"]), np.asarray(trace_static["d"])[:, :cur_len, :])),
            (
                "text_encoder.t_en",
                _metrics(np.asarray(trace_dyn["t_en"]), np.asarray(trace_static["t_en"])[:, :, :cur_len]),
            ),
        ],
    )
    corr2, lag2 = _best_lag_corr(np.asarray(trace_dyn["audio_trimmed"]), audio_static, max_lag=2400)
    print(f"audio_trimmed corr={corr2:.4f} lag={lag2} samples")

    _rule("Dynamic PyTorch vs TVM (aligned-length F0/N)")
    dyn_frames = int(trace_dyn["frames"])
    tvm_frames = int(trace_tvm["frames"])
    prefix_frames = min(dyn_frames, tvm_frames)
    if dyn_frames != tvm_frames:
        print(f"Warning: frames differ (dynamic={dyn_frames}, tvm={tvm_frames}); comparing prefix={prefix_frames}.")

    dyn_f0 = np.asarray(trace_dyn["f0"]).reshape(-1)
    dyn_n = np.asarray(trace_dyn["n"]).reshape(-1)
    f_len_dyn = min(dyn_f0.size, tvm_f0.size, prefix_frames * 2)
    n_len_dyn = min(dyn_n.size, tvm_n.size, prefix_frames * 2)
    _print_metrics_table(
        "aligned-length F0/N",
        [
            ("f0[:2*prefix_frames]", _metrics(dyn_f0[:f_len_dyn], tvm_f0[:f_len_dyn])),
            ("n[:2*prefix_frames]", _metrics(dyn_n[:n_len_dyn], tvm_n[:n_len_dyn])),
        ],
    )

    _rule("Summary")
    tvm_audio_stats = _summary_stats_1d(audio_tvm)
    pt_audio_stats = _summary_stats_1d(audio_static)
    summary_rows: list[tuple[str, str, str, str]] = []
    summary_rows.append(
        (
            "pt.static vs tvm (audio corr/lag)",
            _fmt_corr_lag(corr, lag),
            _fmt_float_flag(tvm_audio_stats["finite_frac"], ok_min=1.0, fmt="{:.3f}"),
            f"tvm max_abs={_fmt_sci(tvm_audio_stats['max_abs'])} std={_fmt_sci(tvm_audio_stats['std'])}",
        )
    )
    summary_rows.append(
        (
            "pt.static audio",
            f"n={int(pt_audio_stats['n'])}",
            _fmt_float_flag(pt_audio_stats["finite_frac"], ok_min=1.0, fmt="{:.3f}"),
            f"max_abs={_fmt_sci(pt_audio_stats['max_abs'])} std={_fmt_sci(pt_audio_stats['std'])}",
        )
    )
    if args.cross_decoder:
        audio_pt_from_tvm_stats = _summary_stats_1d(audio_pt_from_tvm)
        summary_rows.append(
            (
                "pt(dec<-tvm) corr(vs pt.static)",
                _fmt_corr_lag(corr_x1, lag_x1),
                _fmt_float_flag(audio_pt_from_tvm_stats["finite_frac"], ok_min=1.0, fmt="{:.3f}"),
                f"max_abs={_fmt_sci(audio_pt_from_tvm_stats['max_abs'])}",
            )
        )
        audio_tvm_from_pt_stats = _summary_stats_1d(audio_tvm_from_pt)
        summary_rows.append(
            (
                "tvm(dec<-pt.static) corr(vs pt.static)",
                _fmt_corr_lag(corr_x2, lag_x2),
                _fmt_float_flag(audio_tvm_from_pt_stats["finite_frac"], ok_min=1.0, fmt="{:.3f}"),
                f"max_abs={_fmt_sci(audio_tvm_from_pt_stats['max_abs'])}",
            )
        )
        audio_tvm_from_dyn_stats = _summary_stats_1d(audio_tvm_from_dyn)
        summary_rows.append(
            (
                "tvm(dec<-pt.dynamic) corr(vs pt.dynamic)",
                _fmt_corr_lag(corr_x4, lag_x4),
                _fmt_float_flag(audio_tvm_from_dyn_stats["finite_frac"], ok_min=1.0, fmt="{:.3f}"),
                f"max_abs={_fmt_sci(audio_tvm_from_dyn_stats['max_abs'])}",
            )
        )
    _print_summary_table("Run summary", summary_rows)

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / "trace_dynamic.wav", np.asarray(trace_dyn["audio_trimmed"]), 24000)
        sf.write(out_dir / "trace_static.wav", audio_static, 24000)
        sf.write(out_dir / "trace_static_nopack.wav", np.asarray(trace_static_nopack["audio_trimmed"]), 24000)
        sf.write(out_dir / "trace_tvm.wav", audio_tvm, 24000)
        if args.cross_decoder:
            sf.write(out_dir / "trace_pt_dec_from_tvm.wav", audio_pt_from_tvm, 24000)
            sf.write(out_dir / "trace_tvm_dec_from_pt_static.wav", audio_tvm_from_pt, 24000)
            sf.write(out_dir / "trace_tvm_dec_from_pt_nopack.wav", audio_tvm_from_pt_np, 24000)
            sf.write(out_dir / "trace_tvm_dec_from_pt_dynamic.wav", audio_tvm_from_dyn, 24000)
        print(f"Wrote {out_dir / 'trace_dynamic.wav'}")
        print(f"Wrote {out_dir / 'trace_static.wav'}")
        print(f"Wrote {out_dir / 'trace_static_nopack.wav'}")
        print(f"Wrote {out_dir / 'trace_tvm.wav'}")
        if args.cross_decoder:
            print(f"Wrote {out_dir / 'trace_pt_dec_from_tvm.wav'}")
            print(f"Wrote {out_dir / 'trace_tvm_dec_from_pt_static.wav'}")
            print(f"Wrote {out_dir / 'trace_tvm_dec_from_pt_nopack.wav'}")
            print(f"Wrote {out_dir / 'trace_tvm_dec_from_pt_dynamic.wav'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
