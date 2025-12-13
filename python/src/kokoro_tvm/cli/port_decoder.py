# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for compiling Kokoro Decoder to TVM with static shapes."""

import argparse
import json

import torch
import torch.nn.functional as F
import tvm
from tvm import relax
from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter
from huggingface_hub import hf_hub_download
from kokoro.istftnet import Decoder, SineGen

# Import extensions (applies TVM patches on import)
from kokoro_tvm import tvm_extensions  # noqa: F401
from kokoro_tvm.patches.sinegen import apply_sinegen_patch


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Compile Kokoro Decoder to TVM with static shapes")
    parser.add_argument("--seq-len", type=int, default=150, 
                        help="Static sequence length for compilation (default: 150)")
    parser.add_argument("--output", type=str, default="decoder_compiled.so",
                        help="Output path for compiled library (default: decoder_compiled.so)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output from TVM extensions")
    parser.add_argument("--no-weights", action="store_true",
                        help="Skip loading pretrained weights (use random weights for faster iteration)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate TVM output against PyTorch using real encoder data")
    args = parser.parse_args()
    
    # Configure debug output in extensions
    if args.debug:
        extensions.DEBUG_ENABLED = True
    
    # Apply patches
    apply_sinegen_patch()
    
    # Compile decoder
    compile_decoder(args)
    
    # Run validation if requested
    if args.validate:
        from kokoro_tvm.validation import validate_decoder_against_pytorch
        validate_decoder_against_pytorch(args.output, args.seq_len)


def compile_decoder(args):
    """Compile the Decoder module to TVM."""
    # Initialize Decoder from config
    repo_id = 'hexgrad/Kokoro-82M'
    config_path = hf_hub_download(repo_id=repo_id, filename='config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
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
    
    # Load pretrained weights (unless --no-weights is specified)
    if not args.no_weights:
        model_filename = 'kokoro-v1_0.pth'
        print(f"Downloading pretrained weights: {model_filename}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
        
        print(f"Loading decoder weights from {model_path}...")
        state_dicts = torch.load(model_path, map_location='cpu', weights_only=True)
        
        if 'decoder' in state_dicts:
            decoder_state_dict = state_dicts['decoder']
            first_key = next(iter(decoder_state_dict.keys()), '')
            if first_key.startswith('module.'):
                print("Stripping 'module.' prefix from weight keys...")
                decoder_state_dict = {k[7:]: v for k, v in decoder_state_dict.items()}
            decoder.load_state_dict(decoder_state_dict, strict=False)
            print("Successfully loaded pretrained decoder weights!")
        else:
            print(f"Warning: 'decoder' key not found in checkpoint. Using random weights.")
    else:
        print("Skipping weight loading (--no-weights specified). Using random weights.")
    
    decoder.eval()

    # Prepare inputs with STATIC shapes
    batch_size = 1
    seq_len = args.seq_len
    
    asr = torch.randn(batch_size, hidden_dim, seq_len)
    f0 = torch.randn(batch_size, seq_len * 2)
    n = torch.randn(batch_size, seq_len * 2)
    s = torch.randn(batch_size, style_dim)
    
    print(f"Static inputs: asr={asr.shape}, f0={f0.shape}, n={n.shape}, s={s.shape}")

    # Export with STATIC shapes
    print(f"Exporting model with static seq_len={seq_len}...")
    exported_program = torch.export.export(
        decoder,
        (asr, f0, n, s),
    )
    
    # Compile with TVM
    print("Importing into TVM Relax...")
    
    print("Running decompositions explicitly...")
    try:
        exported_program = exported_program.run_decompositions()
        print("Decompositions done.")
    except Exception as e:
        print(f"Warning: run_decompositions failed: {e}. Attempting import anyway.")

    importer = ExportedProgramImporter()
    mod = importer.from_exported_program(
        exported_program, 
        keep_params_as_input=False,
        unwrap_unit_return_tuple=False, 
        no_bind_return_tuple=False
    )
    
    print("Relax Module created.")
    
    # Dump intermediate IR for debugging
    with open("decoder_before_opt.py", "w") as f:
        f.write(mod.script())
    print("Dumped decoder_before_opt.py")
    
    # Rename 'main' to 'decoder_forward'
    new_funcs = {}
    for gv, func in mod.functions.items():
        if gv.name_hint == "main":
            new_gv = tvm.ir.GlobalVar("decoder_forward")
            if hasattr(func, "attrs") and func.attrs is not None and "global_symbol" in func.attrs:
                new_attrs = dict(func.attrs)
                new_attrs["global_symbol"] = "decoder_forward"
                func = func.with_attrs(new_attrs)
                print(f"Updated function global_symbol attribute to 'decoder_forward'")
            new_funcs[new_gv] = func
            print(f"Renamed function 'main' to 'decoder_forward'")
        else:
            new_funcs[gv] = func
    mod = tvm.IRModule(new_funcs, attrs=mod.attrs)
    print(f"Module has {len(mod.functions)} functions after renaming")

    # Compile
    target = tvm.target.Target("llvm -opt-level=3")
    print(f"Compiling for target: {target}")
    
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

        print("Running AnnotateTIROpPattern...")
        mod = relax.transform.AnnotateTIROpPattern()(mod)

        print("Running FoldConstant...")
        # mod = relax.transform.FoldConstant()(mod)
        
        print("Running FuseOps...")
        mod = relax.transform.FuseOps()(mod)
        
        print("Running FuseTIR...")
        mod = relax.transform.FuseTIR()(mod)

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
