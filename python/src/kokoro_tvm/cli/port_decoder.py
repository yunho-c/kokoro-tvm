# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for compiling Kokoro Decoder to TVM with static shapes."""

import argparse
import json
import os

import torch
import tvm
from tvm import relax


# Import extensions (applies TVM patches on import)
from kokoro_tvm import tvm_extensions  # noqa: F401
from kokoro_tvm.patches.sinegen import apply_sinegen_patch


# Target configurations for different platforms
TARGET_CONFIGS = {
    "llvm": {
        "target": "llvm -opt-level=3",
        "extension": ".so",
        "description": "CPU (LLVM)",
    },
    "metal-macos": {
        "target_host": "llvm -mtriple=arm64-apple-macos",
        "target": "metal",
        "extension": ".dylib",
        "description": "macOS (Metal GPU + ARM64 CPU)",
    },
    "metal-ios": {
        "target_host": "llvm -mtriple=arm64-apple-ios",
        "target": "metal",
        "extension": ".dylib",
        "description": "iOS (Metal GPU + ARM64 CPU)",
    },
}


def resolve_target(target_name: str) -> tuple:
    """Resolve target name to TVM target objects.
    
    Returns:
        Tuple of (target, target_host, extension, description)
    """
    if target_name not in TARGET_CONFIGS:
        raise ValueError(f"Unknown target: {target_name}. Available: {list(TARGET_CONFIGS.keys())}")
    
    config = TARGET_CONFIGS[target_name]
    
    if "target_host" in config:
        # Metal targets have separate host target
        target_host = tvm.target.Target(config["target_host"])
        target = tvm.target.Target(config["target"], host=target_host)
    else:
        # CPU-only targets
        target = tvm.target.Target(config["target"])
        target_host = None
    
    return target, target_host, config["extension"], config["description"]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Compile Kokoro Decoder to TVM with static shapes")
    parser.add_argument("--seq-len", type=int, default=150, 
                        help="Static sequence length for compilation (default: 150)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for compiled library (default: decoder_compiled.<ext>)")
    parser.add_argument("--target", type=str, default="llvm",
                        choices=list(TARGET_CONFIGS.keys()),
                        help=f"Compilation target: {', '.join(TARGET_CONFIGS.keys())} (default: llvm)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output from TVM extensions")
    parser.add_argument("--no-weights", action="store_true",
                        help="Skip loading pretrained weights (use random weights for faster iteration)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate TVM output against PyTorch using real encoder data")
    
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
    
    # Set default output path based on target extension
    if args.output is None:
        args.output = f"decoder_compiled{ext}"
    
    print(f"Target: {desc}")
    
    # Apply patches
    apply_sinegen_patch()
    
    # Compile decoder
    compile_decoder(args, target)
    
    # Run validation if requested
    if args.validate:
        import platform
        is_macos_host = platform.system() == "Darwin"
        
        # Validation supported for: LLVM (CPU) or metal-macos on macOS host
        if args.target == "llvm":
            from kokoro_tvm.validation import validate_decoder_against_pytorch
            validate_decoder_against_pytorch(args.output, args.seq_len)
        elif args.target == "metal-macos" and is_macos_host:
            from kokoro_tvm.validation import validate_decoder_against_pytorch
            validate_decoder_against_pytorch(args.output, args.seq_len, device="metal")
        else:
            print(f"Warning: Validation is not supported for target '{args.target}' on this platform. Skipping.")


def compile_decoder(args, target):
    """Compile the Decoder module to TVM.
    
    Args:
        args: CLI arguments
        target: TVM target object
    """
    from kokoro_tvm.models import create_decoder_module
    from kokoro_tvm.models.decoder import get_decoder_config
    
    # Create Relax module using shared function
    mod = create_decoder_module(
        seq_len=args.seq_len,
        load_weights=not args.no_weights,
        func_name="decoder_forward",
        dump_ir="decoder_before_opt.py",
    )
    
    # Get config for verification later
    config = get_decoder_config()
    hidden_dim = config['hidden_dim']
    style_dim = config['style_dim']

    # Compile
    print(f"Compiling for target: {target}")
    is_metal = "metal" in str(target).lower()
    
    with target:
        print("Running DecomposeOpsForInference...")
        mod = relax.transform.DecomposeOpsForInference()(mod)
        with open("decoder_decomposed.py", "w") as f: 
            f.write(mod.script())
        print("Dumped decoder_decomposed.py")

        print("Running LegalizeOps...")
        mod = relax.transform.LegalizeOps()(mod)
        with open("decoder_legalized.py", "w") as f: 
            f.write(mod.script())
        print("Dumped decoder_legalized.py")
        
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

        print("Running AnnotateTIROpPattern...")
        mod = relax.transform.AnnotateTIROpPattern()(mod)

        print("Running FoldConstant...")
        # mod = relax.transform.FoldConstant()(mod)
        
        print("Running FuseOps...")
        mod = relax.transform.FuseOps()(mod)
        
        print("Running FuseTIR...")
        mod = relax.transform.FuseTIR()(mod)

        # Apply GPU scheduling for Metal targets
        if is_metal:
            print("Applying DLight GPU scheduling for Metal...")
            from tvm import dlight as dl
            try:
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
    output_path = args.output
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
    
    test_len = args.seq_len
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
    
    if hasattr(output, 'shape'):
        print("Output shape:", output.shape)
    else:
        print(f"Output is an Array with {len(output)} elements:")
        for i, out in enumerate(output):
            print(f"  Output[{i}] shape: {out.shape}")
    print("Verification Successful!")


if __name__ == "__main__":
    main()
