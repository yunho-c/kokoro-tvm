# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for compiling Kokoro Decoder to TVM with static shapes."""

import argparse
import json
import os
from pathlib import Path

import torch
import tvm
from tvm import relax

from kokoro_tvm import tvm_extensions  # applies TVM patches on import
from kokoro_tvm.config import TARGET_CONFIGS, resolve_target
from kokoro_tvm.patches.adain import apply_adain_patch
from kokoro_tvm.patches.sinegen import apply_sinegen_patch


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Compile Kokoro Decoder to TVM with static shapes")
    parser.add_argument("--seq-len", type=int, default=150, help="Static sequence length for compilation (default: 150)")
    parser.add_argument(
        "--seq-lens",
        type=str,
        default=None,
        help="Comma-separated list of bucket lengths to compile (e.g. 256,512,1024). If set, compiles all buckets.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path for compiled library (single build only)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for compiled outputs (used for multi-bucket builds).",
    )
    parser.add_argument("--target", type=str, default="llvm",
                        choices=list(TARGET_CONFIGS.keys()),
                        help=f"Compilation target: {', '.join(TARGET_CONFIGS.keys())} (default: llvm)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output from TVM extensions")
    parser.add_argument("--no-weights", action="store_true",
                        help="Skip loading pretrained weights (use random weights for faster iteration)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate TVM output against PyTorch using real encoder data")
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Dump intermediate decoder IR (before_opt/decomposed/legalized). Off by default.",
    )
    parser.add_argument("--no-dlight", action="store_true",
                        help="Skip DLight GPU scheduling (useful to debug numerical issues on Metal)")
    parser.add_argument("--no-fuse", action="store_true",
                        help="Skip FuseOps/FuseTIR (useful to debug fusion-related issues)")
    parser.add_argument("--dlight-fallback-only", action="store_true",
                        help="Apply only DLight Fallback scheduling (can help debug incorrect schedules)")

    # Tuning arguments
    parser.add_argument("--tune", action="store_true",
                        help="Auto-tune with MetaSchedule before building")
    parser.add_argument("--tune-trials", type=int, default=2000,
                        help="Maximum tuning trials (default: 2000)")
    parser.add_argument("--tune-dir", type=str, default="tuning_logs",
                        help="Directory for tuning artifacts (default: tuning_logs)")

    args = parser.parse_args()

    # Configure debug output in extensions
    if args.debug:
        tvm_extensions.DEBUG_ENABLED = True

    # Resolve target
    target, target_host, ext, desc = resolve_target(args.target)

    seq_lens: list[int]
    if args.seq_lens:
        seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    else:
        seq_lens = [int(args.seq_len)]
    if not seq_lens:
        raise ValueError("No --seq-len/--seq-lens provided.")
    seq_lens = sorted(set(seq_lens))

    print(f"Target: {desc}")

    # Apply patches
    apply_sinegen_patch()
    apply_adain_patch()

    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(seq_lens) == 1:
        seq_len = seq_lens[0]
        if args.output is None:
            args.output = str(output_dir / f"decoder_compiled{ext}")
        compile_decoder(args, target, seq_len=seq_len, output_path=args.output, dump_ir=args.dump_ir)
    else:
        if args.output is not None:
            raise ValueError("Use --output-dir for multi-bucket builds; --output only supports single build.")
        for seq_len in seq_lens:
            out_path = str(output_dir / f"decoder_compiled_seq{seq_len}{ext}")
            print(f"\n=== Compiling decoder bucket seq_len={seq_len} -> {out_path} ===")
            compile_decoder(args, target, seq_len=seq_len, output_path=out_path, dump_ir=args.dump_ir)

    # Run validation if requested
    if args.validate:
        import platform
        is_macos_host = platform.system() == "Darwin"

        # Validation supported for: LLVM (CPU) or metal-macos on macOS host
        from kokoro_tvm.validation import validate_decoder_against_pytorch

        device = "metal" if args.target == "metal-macos" and is_macos_host else "cpu"
        if args.target != "llvm" and not (args.target == "metal-macos" and is_macos_host):
            print(f"Warning: Validation is not supported for target '{args.target}' on this platform. Skipping.")
            return

        for seq_len in seq_lens:
            lib_path = (
                args.output
                if len(seq_lens) == 1
                else str(output_dir / f"decoder_compiled_seq{seq_len}{ext}")
            )
            validate_decoder_against_pytorch(lib_path, seq_len, device=device)


def compile_decoder(args, target, *, seq_len: int, output_path: str, dump_ir: bool):
    """Compile the Decoder module to TVM.

    Args:
        args: CLI arguments
        target: TVM target object
        seq_len: Static decoder length for this build
        output_path: Output library path (.so/.dylib)
        dump_ir: Whether to dump intermediate IR scripts
    """
    from kokoro_tvm.models import create_decoder_module
    from kokoro_tvm.models.decoder import get_decoder_config

    # Create Relax module using shared function
    mod = create_decoder_module(
        seq_len=seq_len,
        load_weights=not args.no_weights,
        func_name="decoder_forward",
        dump_ir=(f"decoder_before_opt_seq{seq_len}.py" if dump_ir else None),
    )

    # Get config for verification later
    config = get_decoder_config()
    hidden_dim = config["hidden_dim"]
    style_dim = config["style_dim"]

    # Compile
    print(f"Compiling for target: {target}")
    is_metal = "metal" in str(target).lower()

    with target:
        print("Running DecomposeOpsForInference...")
        mod = relax.transform.DecomposeOpsForInference()(mod)
        if dump_ir:
            path = f"decoder_decomposed_seq{seq_len}.py"
            with open(path, "w") as f:
                f.write(mod.script())
            print(f"Dumped {path}")

        print("Running LegalizeOps...")
        mod = relax.transform.LegalizeOps()(mod)
        if dump_ir:
            path = f"decoder_legalized_seq{seq_len}.py"
            with open(path, "w") as f:
                f.write(mod.script())
            print(f"Dumped {path}")

        # Auto-tune with MetaSchedule if requested
        if args.tune:
            print(f"\nStarting MetaSchedule tuning (max_trials={args.tune_trials})...")
            print(f"Tuning logs will be saved to: {args.tune_dir}")
            mod = relax.transform.MetaScheduleTuneTIR(
                work_dir=args.tune_dir,
                max_trials_global=args.tune_trials,
            )(mod)
            print("Tuning complete! Applying best schedules...")
            mod = relax.transform.MetaScheduleApplyDatabase(
                work_dir=args.tune_dir,
                enable_warning=True,
            )(mod)
            print("Best schedules applied.")

        if not args.no_fuse:
            print("Running AnnotateTIROpPattern...")
            mod = relax.transform.AnnotateTIROpPattern()(mod)

        print("Running FoldConstant...")
        # mod = relax.transform.FoldConstant()(mod)

        if not args.no_fuse:
            print("Running FuseOps...")
            mod = relax.transform.FuseOps()(mod)

            print("Running FuseTIR...")
            mod = relax.transform.FuseTIR()(mod)

        # Apply GPU scheduling for Metal targets
        if is_metal:
            if args.no_dlight:
                print("Skipping DLight GPU scheduling (--no-dlight).")
            elif args.no_fuse:
                print("Skipping DLight GPU scheduling because fusion is disabled (--no-fuse).")
            else:
                print("Applying DLight GPU scheduling for Metal...")
                from tvm import dlight as dl
                try:
                    if args.dlight_fallback_only:
                        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
                        print("DLight scheduling applied (Fallback only).")
                    else:
                        # Use specialized rules for better performance
                        # Note: GEMV is excluded due to crash on compound iteration patterns
                        # (KeyError in normalize() for complex spatial indexing like v_yy * 5 + v_ry)
                        mod = dl.ApplyDefaultSchedule(
                            dl.gpu.Matmul(),           # Optimized tiling + shared memory for matmuls
                            dl.gpu.Reduction(),        # Parallel reduction trees
                            dl.gpu.GeneralReduction(), # Multi-axis reductions
                            dl.gpu.Fallback(),         # Basic parallelization for everything else
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

    # Build with debug info disabled to avoid LLVM verification bug
    print("Building with standard Relax pipeline (debug info disabled)...")
    with tvm.transform.PassContext(opt_level=3, config={"tir.enable_debug": False}):
        ex = relax.build(mod, target)
    print("Compilation successful!")

    # Save compiled library
    ex.export_library(output_path)
    print(f"Saved compiled library to: {output_path}")

    # Quick verification
    # For Metal targets: only run on macOS host for metal-macos, skip iOS
    import platform
    is_macos_host = platform.system() == "Darwin"
    is_ios_target = "ios" in str(target).lower()

    if is_metal and is_ios_target:
        print("Skipping verification for iOS target (requires iOS device).")
        print("Compilation Successful!")
        return

    if is_metal and not is_macos_host:
        print("Skipping verification for Metal target (requires macOS host).")
        print("Compilation Successful!")
        return

    # Select device: Metal GPU for metal-macos on macOS, else CPU
    if is_metal and is_macos_host:
        # Check if Metal runtime is available
        try:
            dev = tvm.metal()
            # Try to check if the device is actually usable
            _ = dev.exist
            print("Using Metal device for verification...")
        except Exception as e:
            print(f"Metal runtime not available: {e}")
            print("Skipping verification (Metal runtime not enabled in TVM build).")
            print("Compilation Successful!")
            return
    else:
        dev = tvm.cpu()

    vm = relax.VirtualMachine(ex, dev)

    test_len = int(seq_len)
    asr_in = torch.randn(1, hidden_dim, test_len).numpy()
    f0_in = torch.randn(1, test_len * 2).numpy()
    n_in = torch.randn(1, test_len * 2).numpy()
    s_in = torch.randn(1, style_dim).numpy()

    inputs = [
        tvm.runtime.tensor(asr_in, device=dev),
        tvm.runtime.tensor(f0_in, device=dev),
        tvm.runtime.tensor(n_in, device=dev),
        tvm.runtime.tensor(s_in, device=dev)
    ]

    print(f"Running inference with test_len={test_len}...")
    output = vm["decoder_forward"](*inputs)

    if hasattr(output, "shape"):
        print("Output shape:", output.shape)
    else:
        print(f"Output is an Array with {len(output)} elements:")
        for i, out in enumerate(output):
            print(f"  Output[{i}] shape: {out.shape}")
    print("Verification Successful!")


if __name__ == "__main__":
    main()
