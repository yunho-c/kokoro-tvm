# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Debug utilities for pipeline tracing and validation."""

from __future__ import annotations

import numpy as np


def tail_summary(x: np.ndarray, valid: int, *, preview: int = 8, atol: float = 0.0) -> dict[str, object]:
    """Summarize the padded tail of an array beyond the valid region.

    Args:
        x: Input array (any shape, will be flattened).
        valid: Number of valid elements from the front.
        preview: Number of elements to show in head/tail previews.
        atol: Absolute tolerance for nonzero fraction calculation.

    Returns:
        Dictionary with statistics about the padded region.
    """
    flat = np.asarray(x).reshape(-1)
    total = int(flat.size)
    valid = int(max(0, min(valid, total)))
    pad = flat[valid:]

    if pad.size == 0:
        return {
            "valid": valid,
            "total": total,
            "pad": 0,
            "finite_frac": 1.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "nonzero_frac": 0.0,
            "nonzero_atol": atol,
            "head": [],
            "tail": [],
        }

    pad_f32 = pad.astype(np.float32, copy=False)
    finite = np.isfinite(pad_f32)
    finite_frac = float(np.mean(finite)) if pad_f32.size else 1.0

    abs_pad = np.abs(pad_f32[finite]) if np.any(finite) else np.array([], dtype=np.float32)
    max_abs = float(np.max(abs_pad)) if abs_pad.size else float("nan")
    mean_abs = float(np.mean(abs_pad)) if abs_pad.size else float("nan")
    nonzero_frac = float(np.mean(np.abs(pad_f32) > atol)) if pad_f32.size else 0.0

    head = pad_f32[:preview].tolist()
    tail = pad_f32[-preview:].tolist() if pad_f32.size >= preview else pad_f32.tolist()

    return {
        "valid": valid,
        "total": total,
        "pad": int(pad_f32.size),
        "finite_frac": finite_frac,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "nonzero_frac": nonzero_frac,
        "nonzero_atol": atol,
        "head": head,
        "tail": tail,
    }


def stats_summary(x: np.ndarray, *, atol: float = 1e-8) -> dict[str, float]:
    """Compute basic statistics for an array.

    Args:
        x: Input array (any shape, will be flattened).
        atol: Absolute tolerance for nonzero fraction calculation.

    Returns:
        Dictionary with min, max, mean, std, and fraction statistics.
    """
    arr = np.asarray(x).reshape(-1).astype(np.float32, copy=False)
    if arr.size == 0:
        return {"finite_frac": 1.0, "nonzero_frac": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    finite = np.isfinite(arr)
    finite_frac = float(np.mean(finite))
    arr_f = arr[finite] if np.any(finite) else np.array([], dtype=np.float32)
    min_v = float(np.min(arr_f)) if arr_f.size else float("nan")
    max_v = float(np.max(arr_f)) if arr_f.size else float("nan")
    mean_v = float(np.mean(arr_f)) if arr_f.size else float("nan")
    std_v = float(np.std(arr_f)) if arr_f.size else float("nan")
    nonzero_frac = float(np.mean(np.abs(arr) > atol))
    return {
        "finite_frac": finite_frac,
        "nonzero_frac": nonzero_frac,
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
    }
