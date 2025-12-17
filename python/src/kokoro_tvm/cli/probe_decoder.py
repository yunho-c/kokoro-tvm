# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Compile and run a decoder probe that reports per-stage stats."""

from __future__ import annotations

import argparse

import numpy as np
import tvm
from tvm import relax

from kokoro_tvm.config import TARGET_CONFIGS, resolve_target
from kokoro_tvm.models.decoder_probe import create_decoder_probe_module


def _print_probe(probe: np.ndarray) -> None:
    names = [
        "f0_input",
        "n_input",
        "x_cat",
        "x_after_decode",
        "har_stft",
        "x_source0_conv",
        "nr0_n1",
        "nr0_snake1",
        "nr0_conv1_w",
        "nr0_conv1_b",
        "nr0_conv1",
        "x_source0",
        "x_up0",
        "xg0_pre_res",
        "xg0_post_res",
        "xg1_pre_res",
        "xg1_post_res",
        "x_post_conv",
        "spec_exp",
        "audio_out",
    ]
    for i, name in enumerate(names):
        row = np.asarray(probe[i]).reshape(-1)
        finite = np.isfinite(row)
        finite_frac = float(np.mean(finite)) if row.size else 1.0
        row_f = row[finite] if np.any(finite) else np.array([], dtype=np.float32)
        min_v = float(np.min(row_f)) if row_f.size else float("nan")
        max_v = float(np.max(row_f)) if row_f.size else float("nan")
        print(
            f"{name}: n={row.size} finite_frac={finite_frac:.3f} min={min_v:.6g} max={max_v:.6g} head={row[:8].tolist()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run decoder probe")
    parser.add_argument("--seq-len", type=int, default=150, help="Sequence length for compilation/inference")
    parser.add_argument("--target", type=str, default="metal-macos", choices=list(TARGET_CONFIGS.keys()))
    parser.add_argument("--output", type=str, default=None, help="Output library path")
    parser.add_argument("--no-weights", action="store_true", help="Skip loading pretrained weights")
    parser.add_argument("--no-fuse-ops", action="store_true", help="Skip relax.transform.FuseOps")
    parser.add_argument("--no-fuse-tir", action="store_true", help="Skip relax.transform.FuseTIR")
    parser.add_argument("--no-dlight", action="store_true", help="Skip DLight scheduling (Metal targets only)")
    parser.add_argument("--dump-ir", type=str, default=None, help="Write final Relax IRModule script to path")
    args = parser.parse_args()

    target, target_host, ext, desc = resolve_target(args.target)
    if args.output is None:
        args.output = f"decoder_probe{ext}"
    print(f"Target: {desc}")

    mod = create_decoder_probe_module(seq_len=args.seq_len, load_weights=not args.no_weights)

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

    if args.dump_ir:
        with open(args.dump_ir, "w", encoding="utf-8") as f:
            f.write(mod.script())

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

    out = vm["decoder_probe_forward"](
        tvm.runtime.tensor(asr, device=dev),
        tvm.runtime.tensor(f0, device=dev),
        tvm.runtime.tensor(n, device=dev),
        tvm.runtime.tensor(s, device=dev),
    )

    probe = (out.numpy() if hasattr(out, "numpy") else out[0].numpy()).astype(np.float32, copy=False)
    print("Probe stats (min/max/mean) per stage:")
    _print_probe(probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
