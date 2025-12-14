# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for compiling Kokoro Encoder components to TVM."""

import argparse
import os
import tvm
from tvm import relax

from kokoro_tvm import tvm_extensions  # noqa: F401
from kokoro_tvm.config import resolve_target

def main():
    parser = argparse.ArgumentParser(description="Compile Kokoro Encoder Components to TVM")
    parser.add_argument("--component", type=str, required=True,
                        choices=["bert", "prosody", "text_encoder", "all"],
                        help="Component to compile")
    parser.add_argument("--seq-len", type=int, default=50,
                        help="Static sequence length for compilation (default: 50)")
    parser.add_argument("--output-dir", type=str, default="tvm_output",
                        help="Directory for output libraries")
    parser.add_argument("--target", type=str, default="llvm",
                        choices=["llvm", "metal-macos", "metal-ios"],
                        help="Compilation target path")
    parser.add_argument("--validate", action="store_true",
                        help="Run basic successful execution check")
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resolve target
    target, target_host, ext, desc = resolve_target(args.target)
    print(f"Target: {desc}")
    
    components = []
    if args.component == "all":
        components = ["bert", "prosody", "text_encoder"]
    else:
        components = [args.component]
        
    for comp in components:
        print(f"\nProcessing component: {comp}")
        compile_component(comp, args, target, ext)

def compile_component(name, args, target, ext):
    from kokoro_tvm.models.encoder import (
        create_bert_module, 
        create_predictor_module, 
        create_text_encoder_module
    )
    
    mod = None
    func_name = ""
    
    if name == "bert":
        # Bert (CustomAlbert)
        func_name = "bert_forward"
        # We use default config values for now or could expose via CLI
        mod = create_bert_module(dump_ir=f"{args.output_dir}/bert_before_opt.py")
        
    elif name == "prosody":
        func_name = "prosody_forward"
        mod = create_predictor_module(seq_len=args.seq_len, dump_ir=f"{args.output_dir}/prosody_before_opt.py")
        
    elif name == "text_encoder":
        func_name = "text_encoder_forward"
        mod = create_text_encoder_module(seq_len=args.seq_len, dump_ir=f"{args.output_dir}/text_encoder_before_opt.py")
        
    if mod is None:
        print(f"Error creating module for {name}")
        return

    # Compile
    print(f"Compiling {name}...")
    output_filename = f"{name}_compiled{ext}"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Standard pipeline
    # Note: DLight or Tuning could be added here similar to port_decoder.py
    # For now, using standard build
    
    with target:
        mod = relax.transform.DecomposeOpsForInference()(mod)
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.AnnotateTIROpPattern()(mod)
        mod = relax.transform.FuseOps()(mod)
        mod = relax.transform.FuseTIR()(mod)
        
    print("Building...")
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target)
        
    ex.export_library(output_path)
    print(f"Saved {name} to {output_path}")
    
    if args.validate:
        validate_component(name, ex, args, target)

def validate_component(name, ex, args, target):
    print(f"Validating {name} execution...")
    # Determine device
    if "metal" in str(target).lower():
         dev = tvm.metal()
    else:
         dev = tvm.cpu()
         
    vm = relax.VirtualMachine(ex, dev)
    
    # Create dummy inputs matching the factory defaults
    # This involves duplicating some logic from models/encoder.py or shared util
    # For quick validation, we iterate.
    import torch
    import numpy as np
    
    inputs = []
    
    if name == "bert":
        # (input_ids, attention_mask)
        inputs.append(tvm.runtime.tensor(torch.randint(0, 100, (1, 50), dtype=torch.long).numpy(), device=dev))
        inputs.append(tvm.runtime.tensor(torch.ones((1, 50), dtype=torch.long).numpy(), device=dev))
    
    elif name == "prosody":
        # (texts, style, lengths, alignment, m)
        inputs.append(tvm.runtime.tensor(torch.randn(1, 512, args.seq_len).numpy().astype("float32"), device=dev))
        inputs.append(tvm.runtime.tensor(torch.randn(1, 64).numpy().astype("float32"), device=dev))
        inputs.append(tvm.runtime.tensor(np.array([args.seq_len], dtype=np.int64), device=dev))
        inputs.append(tvm.runtime.tensor(torch.randn(1, args.seq_len, args.seq_len*2).numpy().astype("float32"), device=dev))
        inputs.append(tvm.runtime.tensor(np.zeros((1, args.seq_len), dtype=bool), device=dev))

    elif name == "text_encoder":
        # (x, lengths, m)
        inputs.append(tvm.runtime.tensor(torch.randint(0, 100, (1, args.seq_len), dtype=torch.long).numpy(), device=dev))
        inputs.append(tvm.runtime.tensor(np.array([args.seq_len], dtype=np.int64), device=dev))
        inputs.append(tvm.runtime.tensor(np.zeros((1, args.seq_len), dtype=bool), device=dev))
        
    # Run
    try:
        if name == "bert":
             _ = vm["bert_forward"](*inputs)
        elif name == "prosody":
             _ = vm["prosody_forward"](*inputs)
        elif name == "text_encoder":
             _ = vm["text_encoder_forward"](*inputs)
        print(f"Validation SUCCESS for {name}")
    except Exception as e:
        print(f"Validation FAILED for {name}: {e}")

if __name__ == "__main__":
    main()
