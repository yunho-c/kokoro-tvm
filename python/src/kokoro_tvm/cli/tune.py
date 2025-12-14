# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for MetaSchedule auto-tuning."""

import argparse

import tvm
from tvm import relax

from kokoro_tvm.models import create_decoder_module
from kokoro_tvm.tuning import tune_module, estimate_tuning_time
from kokoro_tvm.cli.port_decoder import TARGET_CONFIGS, resolve_target


def main():
    """Main CLI entry point for tuning."""
    parser = argparse.ArgumentParser(description="Auto-tune Kokoro Decoder with MetaSchedule")
    parser.add_argument("--seq-len", type=int, default=150,
                        help="Static sequence length (default: 150)")
    parser.add_argument("--target", type=str, default="llvm",
                        choices=list(TARGET_CONFIGS.keys()),
                        help=f"Target: {', '.join(TARGET_CONFIGS.keys())} (default: llvm)")
    parser.add_argument("--work-dir", type=str, default="tuning_logs",
                        help="Directory for tuning artifacts (default: tuning_logs)")
    parser.add_argument("--max-trials", type=int, default=2000,
                        help="Maximum tuning trials (default: 2000)")
    parser.add_argument("--no-weights", action="store_true",
                        help="Skip loading pretrained weights (faster for testing)")
    parser.add_argument("--estimate", action="store_true",
                        help="Only estimate tuning time, don't run tuning")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for tuned module")
    args = parser.parse_args()
    
    # Resolve target
    target, _, ext, desc = resolve_target(args.target)
    
    # Set default output
    if args.output is None:
        args.output = f"decoder_tuned{ext}"
    
    print(f"Target: {desc}")
    print(f"Work directory: {args.work_dir}")
    print(f"Max trials: {args.max_trials}")
    
    # Create Relax module using shared function
    mod = create_decoder_module(
        seq_len=args.seq_len,
        load_weights=not args.no_weights,
    )
    
    # Estimate tuning time
    estimate = estimate_tuning_time(mod, target, args.max_trials)
    
    print(f"\nTuning estimation:")
    print(f"  TIR functions to tune: {estimate['num_tir_functions']}")
    print(f"  Estimated time: {estimate['estimated_minutes']:.1f} minutes")
    
    if args.estimate:
        print("\n(--estimate flag set, skipping actual tuning)")
        return
    
    # Run tuning
    print("\nStarting tuning...")
    tuned_mod = tune_module(
        mod, 
        target, 
        work_dir=args.work_dir,
        max_trials=args.max_trials,
    )
    
    # Build and save
    print("\nBuilding tuned module...")
    with tvm.transform.PassContext(opt_level=3, config={"tir.enable_debug": False}):
        ex = relax.build(tuned_mod, target)
    
    ex.export_library(args.output)
    print(f"Saved tuned module to: {args.output}")


if __name__ == "__main__":
    main()
