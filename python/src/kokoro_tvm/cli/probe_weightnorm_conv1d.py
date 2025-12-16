# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Compile and run a minimal Conv1d(+weight_norm) probe."""

from __future__ import annotations

import argparse

import numpy as np
import tvm
from tvm import relax

from kokoro_tvm.config import TARGET_CONFIGS, resolve_target
from kokoro_tvm.models.weightnorm_conv1d_probe import create_weightnorm_conv1d_probe_module


def _print_probe(probe: np.ndarray) -> None:
    names = ["x_input", "wn_g", "wn_v", "wn_w", "y_output"]
    for i, name in enumerate(names):
        row = np.asarray(probe[i]).reshape(-1)
        finite = np.isfinite(row)
        finite_frac = float(np.mean(finite)) if row.size else 1.0
        row_f = row[finite] if np.any(finite) else np.array([], dtype=np.float32)
        min_v = float(np.min(row_f)) if row_f.size else float("nan")
        max_v = float(np.max(row_f)) if row_f.size else float("nan")
        print(f"{name}: n={row.size} finite_frac={finite_frac:.3f} min={min_v:.6g} max={max_v:.6g} head={row[:8].tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run Conv1d(+weight_norm) probe")
    parser.add_argument("--length", type=int, default=32, help="Input length for compilation/inference")
    parser.add_argument("--target", type=str, default="llvm", choices=list(TARGET_CONFIGS.keys()))
    parser.add_argument("--output", type=str, default=None, help="Output library path")
    parser.add_argument("--plain", action="store_true", help="Use plain Conv1d (no weight_norm)")
    parser.add_argument("--no-weights", action="store_true", help="Do not load decoder checkpoint weights")
    parser.add_argument("--dump-ir", type=str, default=None, help="Write final Relax IRModule script to path")
    args = parser.parse_args()

    target, _, ext, desc = resolve_target(args.target)
    if args.output is None:
        args.output = f"weightnorm_conv1d_probe{ext}"
    print(f"Target: {desc}")

    mod = create_weightnorm_conv1d_probe_module(
        length=args.length,
        use_weight_norm=not args.plain,
        load_decoder_weights=not args.no_weights,
    )

    with target:
        mod = relax.transform.DecomposeOpsForInference()(mod)
        mod = relax.transform.LegalizeOps()(mod)

    if args.dump_ir:
        with open(args.dump_ir, "w", encoding="utf-8") as f:
            f.write(mod.script())

    with tvm.transform.PassContext(opt_level=3, config={"tir.enable_debug": False}):
        ex = relax.build(mod, target)
    ex.export_library(args.output)
    print(f"Wrote {args.output}")

    dev = tvm.cpu(0) if "llvm" in str(target).lower() else tvm.metal(0)
    vm = relax.VirtualMachine(ex, dev)

    channels = 256
    x = np.zeros((1, channels, args.length), dtype="float32")
    out = vm["weightnorm_conv1d_probe_forward"](tvm.runtime.tensor(x, device=dev))

    probe = (out.numpy() if hasattr(out, "numpy") else out[0].numpy()).astype(np.float32, copy=False)
    _print_probe(probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

