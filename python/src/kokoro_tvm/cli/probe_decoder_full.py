# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Compile and run a decoder probe that returns full intermediate tensors."""

from __future__ import annotations

import argparse

import numpy as np
import tvm
from tvm import relax

from kokoro_tvm.config import TARGET_CONFIGS, resolve_target
from kokoro_tvm.models.decoder_probe import create_decoder_probe_full_module


def _stats(name: str, x: np.ndarray) -> None:
    arr = np.asarray(x).astype(np.float32, copy=False).reshape(-1)
    finite = np.isfinite(arr)
    finite_frac = float(np.mean(finite)) if arr.size else 1.0
    arr_f = arr[finite] if np.any(finite) else np.array([], dtype=np.float32)
    min_v = float(np.min(arr_f)) if arr_f.size else float("nan")
    max_v = float(np.max(arr_f)) if arr_f.size else float("nan")
    print(f"{name}: shape={tuple(np.asarray(x).shape)} n={arr.size} finite_frac={finite_frac:.6f} min={min_v:.6g} max={max_v:.6g}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run full decoder probe")
    parser.add_argument("--seq-len", type=int, default=150, help="Sequence length for compilation/inference")
    parser.add_argument("--target", type=str, default="llvm", choices=list(TARGET_CONFIGS.keys()))
    parser.add_argument("--output", type=str, default=None, help="Output library path")
    parser.add_argument("--no-weights", action="store_true", help="Skip loading pretrained weights")
    parser.add_argument("--no-fuse-ops", action="store_true", help="Skip relax.transform.FuseOps")
    parser.add_argument("--no-fuse-tir", action="store_true", help="Skip relax.transform.FuseTIR")
    parser.add_argument("--no-dlight", action="store_true", help="Skip DLight scheduling (Metal targets only)")
    args = parser.parse_args()

    target, _, ext, desc = resolve_target(args.target)
    if args.output is None:
        args.output = f"decoder_probe_full{ext}"
    print(f"Target: {desc}")

    mod = create_decoder_probe_full_module(seq_len=args.seq_len, load_weights=not args.no_weights)

    is_metal = "metal" in str(target).lower()
    with target:
        mod = relax.transform.DecomposeOpsForInference()(mod)
        mod = relax.transform.LegalizeOps()(mod)
        if not args.no_fuse_ops or not args.no_fuse_tir:
            mod = relax.transform.AnnotateTIROpPattern()(mod)
        if not args.no_fuse_ops:
            mod = relax.transform.FuseOps()(mod)
        if not args.no_fuse_tir:
            mod = relax.transform.FuseTIR()(mod)
        if is_metal and not args.no_dlight:
            from tvm import dlight as dl

            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)

    with tvm.transform.PassContext(opt_level=3, config={"tir.enable_debug": False}):
        ex = relax.build(mod, target)
    ex.export_library(args.output)
    print(f"Wrote {args.output}")

    dev = tvm.metal(0) if is_metal else tvm.cpu(0)
    vm = relax.VirtualMachine(ex, dev)

    hidden_dim = 512
    style_dim = 128
    asr = np.zeros((1, hidden_dim, args.seq_len), dtype="float32")
    f0 = np.zeros((1, args.seq_len * 2), dtype="float32")
    n = np.zeros((1, args.seq_len * 2), dtype="float32")
    s = np.zeros((1, style_dim), dtype="float32")

    out = vm["decoder_probe_full_forward"](
        tvm.runtime.tensor(asr, device=dev),
        tvm.runtime.tensor(f0, device=dev),
        tvm.runtime.tensor(n, device=dev),
        tvm.runtime.tensor(s, device=dev),
    )

    items = list(out) if hasattr(out, "__len__") and not hasattr(out, "numpy") else [out]
    names = ["x_source0_conv", "nr0_n1", "nr0_snake1", "nr0_conv1_w", "nr0_conv1_b", "nr0_conv1"]
    for name, item in zip(names, items, strict=False):
        arr = item.numpy() if hasattr(item, "numpy") else np.asarray(item)
        _stats(name, arr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

