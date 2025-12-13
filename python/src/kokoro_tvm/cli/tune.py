# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for MetaSchedule auto-tuning."""

import argparse
import json
from pathlib import Path

import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter
from huggingface_hub import hf_hub_download
from kokoro.istftnet import Decoder

from kokoro_tvm import tvm_extensions  # noqa: F401
from kokoro_tvm.patches.sinegen import apply_sinegen_patch
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
    parser.add_argument("--estimate", action="store_true",
                        help="Only estimate tuning time, don't run tuning")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for tuned module")
    args = parser.parse_args()
    
    # Apply patches
    apply_sinegen_patch()
    
    # Resolve target
    target, _, ext, desc = resolve_target(args.target)
    
    # Set default output
    if args.output is None:
        args.output = f"decoder_tuned{ext}"
    
    print(f"Target: {desc}")
    print(f"Work directory: {args.work_dir}")
    print(f"Max trials: {args.max_trials}")
    
    # Create Relax module (same as port_decoder but stop before build)
    mod = create_decoder_module(args.seq_len)
    
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


def create_decoder_module(seq_len: int):
    """Create the Relax module for the decoder (before building)."""
    # Load config
    repo_id = 'hexgrad/Kokoro-82M'
    config_path = hf_hub_download(repo_id=repo_id, filename='config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    hidden_dim = config['hidden_dim']
    style_dim = config['style_dim']
    n_mels = config['n_mels']
    istftnet_params = config['istftnet']
    
    decoder = Decoder(
        dim_in=hidden_dim,
        style_dim=style_dim,
        dim_out=n_mels,
        disable_complex=True,
        **istftnet_params
    )
    decoder.eval()
    
    # Create inputs
    asr = torch.randn(1, hidden_dim, seq_len)
    f0 = torch.randn(1, seq_len * 2)
    n = torch.randn(1, seq_len * 2)
    s = torch.randn(1, style_dim)
    
    # Export
    print(f"Exporting decoder with seq_len={seq_len}...")
    exported_program = torch.export.export(decoder, (asr, f0, n, s))
    
    try:
        exported_program = exported_program.run_decompositions()
    except Exception as e:
        print(f"Warning: run_decompositions failed: {e}")
    
    # Import to Relax
    importer = ExportedProgramImporter()
    mod = importer.from_exported_program(
        exported_program,
        keep_params_as_input=False,
        unwrap_unit_return_tuple=False,
        no_bind_return_tuple=False
    )
    
    # Rename main to decoder_forward
    new_funcs = {}
    for gv, func in mod.functions.items():
        if gv.name_hint == "main":
            new_gv = tvm.ir.GlobalVar("decoder_forward")
            if hasattr(func, "attrs") and func.attrs and "global_symbol" in func.attrs:
                new_attrs = dict(func.attrs)
                new_attrs["global_symbol"] = "decoder_forward"
                func = func.with_attrs(new_attrs)
            new_funcs[new_gv] = func
        else:
            new_funcs[gv] = func
    
    return tvm.IRModule(new_funcs, attrs=mod.attrs)


if __name__ == "__main__":
    main()
