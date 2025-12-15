# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for compiling Kokoro Encoder components to TVM."""

import argparse
import os

import numpy as np
import torch
import tvm
from tvm import relax

from kokoro_tvm import tvm_extensions
from kokoro_tvm.config import resolve_target


def main():
    parser = argparse.ArgumentParser(description="Compile Kokoro Encoder Components to TVM")
    parser.add_argument(
        "--component",
        type=str,
        required=True,
        choices=["bert", "duration", "f0n", "text_encoder", "all"],
        help="Component to compile",
    )
    parser.add_argument("--seq-len", type=int, default=512, help="Static text sequence length")
    parser.add_argument("--aligned-len", type=int, default=5120, help="Static aligned (audio) length for f0n")
    parser.add_argument("--output-dir", type=str, default="tvm_output", help="Directory for output libraries")
    parser.add_argument(
        "--target", type=str, default="llvm", choices=["llvm", "metal-macos", "metal-ios"], help="Compilation target"
    )
    parser.add_argument("--validate", action="store_true", help="Run basic execution check after compilation")
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip loading pretrained weights (use random weights for faster iteration)",
    )
    parser.add_argument(
        "--lstm-method",
        type=str,
        default="topi",
        choices=["mps", "tir", "relax", "topi"],
        help="LSTM implementation: mps (Metal/MPS extern), tir (non-unrolled, O(1) IR), relax (unrolled), topi (original)",
    )
    parser.add_argument(
        "--lstm-semantics",
        type=str,
        default="padded",
        choices=["padded", "packed"],
        help="LSTM variable-length semantics: padded (current, ignores packing) or packed (preserves lengths via a custom op)",
    )

    args = parser.parse_args()

    # Configure LSTM method
    if args.lstm_method == "mps":
        tvm_extensions.USE_MPS_LSTM = True
        tvm_extensions.USE_TIR_LSTM = False
        tvm_extensions.USE_RELAX_LSTM = False
        print("Using MPS LSTM (tvm.contrib.mps.lstm extern)")
    elif args.lstm_method == "tir":
        tvm_extensions.USE_MPS_LSTM = False
        tvm_extensions.USE_TIR_LSTM = True
        tvm_extensions.USE_RELAX_LSTM = False
        print("Using TIR LSTM (non-unrolled, O(1) IR size)")
    elif args.lstm_method == "relax":
        tvm_extensions.USE_MPS_LSTM = False
        tvm_extensions.USE_TIR_LSTM = False
        tvm_extensions.USE_RELAX_LSTM = True
        print("Using Relax LSTM (unrolled)")
    else:
        tvm_extensions.USE_MPS_LSTM = False
        tvm_extensions.USE_TIR_LSTM = False
        tvm_extensions.USE_RELAX_LSTM = False
        print("Using TOPI LSTM (original)")

    os.makedirs(args.output_dir, exist_ok=True)

    target, _, ext, desc = resolve_target(args.target)
    print(f"Target: {desc}")

    if args.component == "all":
        components = ["bert", "duration", "f0n", "text_encoder"]
    else:
        components = [args.component]

    for comp in components:
        print(f"\nProcessing component: {comp}")
        compile_component(comp, args, target, ext)


def compile_component(name, args, target, ext):
    from kokoro_tvm.models.encoder import (
        create_bert_module,
        create_duration_module,
        create_f0n_module,
        create_text_encoder_module,
    )

    mod = None

    if name == "bert":
        mod = create_bert_module(load_weights=not args.no_weights, dump_ir=f"{args.output_dir}/bert_ir.py")

    elif name == "duration":
        mod = create_duration_module(
            seq_len=args.seq_len,
            load_weights=not args.no_weights,
            dump_ir=f"{args.output_dir}/duration_ir.py",
            lstm_semantics=args.lstm_semantics,
        )

    elif name == "f0n":
        mod = create_f0n_module(
            aligned_len=args.aligned_len,
            load_weights=not args.no_weights,
            dump_ir=f"{args.output_dir}/f0n_ir.py",
            lstm_semantics=args.lstm_semantics,
        )

    elif name == "text_encoder":
        mod = create_text_encoder_module(
            seq_len=args.seq_len,
            load_weights=not args.no_weights,
            dump_ir=f"{args.output_dir}/text_encoder_ir.py",
            lstm_semantics=args.lstm_semantics,
        )

    if mod is None:
        print(f"Error creating module for {name}")
        return

    print(f"Compiling {name}...")
    output_path = os.path.join(args.output_dir, f"{name}_compiled{ext}")

    with target:
        mod = relax.transform.DecomposeOpsForInference()(mod)
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.AnnotateTIROpPattern()(mod)
        mod = relax.transform.DeadCodeElimination()(mod)
        mod = relax.transform.FuseOps()(mod)
        try:
            mod = relax.transform.FuseTIR()(mod)
        except Exception as e:
            print(f"Warning: FuseTIR failed: {e}")
            print("Continuing without FuseTIR...")

        # Apply GPU scheduling for Metal targets
        is_metal = "metal" in str(target).lower()
        if is_metal:
            print("Applying DLight GPU scheduling for Metal...")
            from tvm import dlight as dl

            try:
                mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                )(mod)
                print("DLight scheduling applied (Matmul + Reduction + Fallback).")
            except Exception as e:
                print(f"Warning: DLight scheduling failed: {e}")
                print("Retrying with Fallback only...")
                try:
                    mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
                    print("DLight Fallback scheduling applied.")
                except Exception as e2:
                    print(f"Warning: DLight Fallback also failed: {e2}")
                    print("Continuing without DLight scheduling...")

    print("Building...")
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target)

    ex.export_library(output_path)
    print(f"Saved {name} to {output_path}")

    if args.validate:
        validate_component(name, ex, args, target)


def validate_component(name, ex, args, target):
    print(f"Validating {name} execution...")
    dev = tvm.metal() if "metal" in str(target).lower() else tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)

    inputs = []
    func_name = ""

    if name == "bert":
        from kokoro_tvm.models.encoder import get_kokoro_config

        cfg = get_kokoro_config()
        func_name = "bert_forward"
        inputs.append(
            tvm.runtime.tensor(torch.randint(0, cfg["n_token"], (1, 512), dtype=torch.long).numpy(), device=dev)
        )
        inputs.append(tvm.runtime.tensor(torch.ones((1, 512), dtype=torch.long).numpy(), device=dev))

    elif name == "duration":
        func_name = "duration_forward"
        inputs.append(tvm.runtime.tensor(torch.randn(1, 512, args.seq_len).numpy().astype("float32"), device=dev))
        inputs.append(tvm.runtime.tensor(torch.randn(1, 128).numpy().astype("float32"), device=dev))
        inputs.append(tvm.runtime.tensor(np.array([args.seq_len], dtype=np.int64), device=dev))
        inputs.append(tvm.runtime.tensor(np.zeros((1, args.seq_len), dtype=bool), device=dev))

    elif name == "f0n":
        func_name = "f0n_forward"
        # F0N expects `en` with channel dim 640 (text hidden + style features).
        inputs.append(tvm.runtime.tensor(torch.randn(1, 640, args.aligned_len).numpy().astype("float32"), device=dev))
        inputs.append(tvm.runtime.tensor(torch.randn(1, 128).numpy().astype("float32"), device=dev))
        inputs.append(tvm.runtime.tensor(np.array([args.aligned_len], dtype=np.int64), device=dev))

    elif name == "text_encoder":
        func_name = "text_encoder_forward"
        inputs.append(
            tvm.runtime.tensor(torch.randint(0, 100, (1, args.seq_len), dtype=torch.long).numpy(), device=dev)
        )
        inputs.append(tvm.runtime.tensor(np.array([args.seq_len], dtype=np.int64), device=dev))
        inputs.append(tvm.runtime.tensor(np.zeros((1, args.seq_len), dtype=bool), device=dev))

    try:
        _ = vm[func_name](*inputs)
        print(f"Validation SUCCESS for {name}")
    except Exception as e:
        print(f"Validation FAILED for {name}: {e}")


if __name__ == "__main__":
    main()
