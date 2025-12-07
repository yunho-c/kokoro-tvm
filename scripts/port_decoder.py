
import torch
import sys
import os
import json
import argparse
from huggingface_hub import hf_hub_download
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Add external/kokoro to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'kokoro'))
from kokoro.istftnet import Decoder

from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter
from tvm import relax
import operator
import tvm.relax.op as relax_op

# Import TVM extensions which monkeypatches ExportedProgramImporter
try:
    import scripts.tvm_extensions
    print("Loaded scripts.tvm_extensions")
except ImportError:
    try:
        import tvm_extensions
        print("Loaded tvm_extensions (local)")
    except ImportError:
        print("Warning: Could not load tvm_extensions. Using default importer (might fail).")

# Monkeypatch SineGen to be friendly to symbolic shapes (Same as in export_decoder.py)
from kokoro.istftnet import SineGen
import torch.nn.functional as F

def _f02sine_friendly(self, f0_values):
    rad_values = (f0_values / self.sampling_rate) % 1
    rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    if not self.flag_for_pulse:
        tgt_len = f0_values.shape[1] // self.upsample_scale
        rad_values = F.interpolate(rad_values.transpose(1, 2), size=tgt_len, mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, size=f0_values.shape[1], mode="linear").transpose(1, 2)
        sines = torch.sin(phase)
    else:
        # Original logic fallback (not reached for Kokoro usually)
        return SineGen._old_f02sine(self, f0_values) 
    return sines

if not hasattr(SineGen, '_old_f02sine'):
    SineGen._old_f02sine = SineGen._f02sine
    SineGen._f02sine = _f02sine_friendly

def main(args):
    # 1. Initialize Decoder
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
    decoder.eval()

    # 2. Prepare Inputs with STATIC shapes
    batch_size = 1
    seq_len = args.seq_len  # Static sequence length from CLI
    
    asr = torch.randn(batch_size, hidden_dim, seq_len)
    f0 = torch.randn(batch_size, seq_len * 2)
    n = torch.randn(batch_size, seq_len * 2)
    s = torch.randn(batch_size, style_dim)
    
    print(f"Static inputs: asr={asr.shape}, f0={f0.shape}, n={n.shape}, s={s.shape}")

    # 3. Export with STATIC shapes (no dynamic_shapes)
    print(f"Exporting model with static seq_len={seq_len}...")
    exported_program = torch.export.export(
        decoder,
        (asr, f0, n, s),
        # No dynamic_shapes - using static shapes for reliable TVM compilation
    )
    
    # 4. Compile with TVM
    print("Importing into TVM Relax...")
    # NOTE: We skip run_decompositions() here because we are using the importer class directly.
    # If run_decompositions is not run, we are importing the ATen/Core Aten graph directly?
    # Torch export produces core aten ops by default (mostly). 
    # If we need decomposition, we can run it explicitly. 
    # But earlier debug showed explicit run_decompositions() worked in export script.
    
    print("Running decompositions explicitly (if safe)...")
    try:
        exported_program = exported_program.run_decompositions()
        print("Decompositions done.")
    except Exception as e:
        print(f"Warning: run_decompositions failed: {e}. Attempting import anyway (might fail if high level ops present).")

    importer = ExportedProgramImporter()
    mod = importer.from_exported_program(
        exported_program, 
        keep_params_as_input=True,
        unwrap_unit_return_tuple=False, 
        no_bind_return_tuple=False
    )
    
    print("Relax Module created.")
    
    with open("decoder_before_opt.py", "w") as f:
        f.write(mod.script())
    print("Dumped decoder_before_opt.py")
    # mod.show()
    
    # Rename 'main' to avoid "duplicate global symbol" error during build
    # We must preserve ALL functions (including private TIR functions from legalization)
    # Also update the function's 'global_symbol' attribute if present
    new_funcs = {}
    for gv, func in mod.functions.items():
        if gv.name_hint == "main":
            # Create new GlobalVar with different name
            new_gv = tvm.ir.GlobalVar("decoder_forward")
            # If the function has a global_symbol attribute, update it too
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

    # Use target that avoids debug info issues
    target = tvm.target.Target("llvm -opt-level=3")
    print(f"Compiling for target: {target}")
    
    # seq = tvm.transform.Sequential([
    #     relax.transform.DecomposeOpsForInference(),
    #     relax.transform.LegalizeOps(),
    #     relax.transform.AnnotateTIROpPattern(),
    #     relax.transform.FoldConstant(),
    #     relax.transform.FuseOps(),
    #     relax.transform.FuseTIR(),
    # ])
    
    with target:
        # mod = seq(mod)
        print("Running DecomposeOpsForInference...")
        mod = relax.transform.DecomposeOpsForInference()(mod)
        with open("decoder_decomposed.py", "w") as f: f.write(mod.script())
        print("Dumped decoder_decomposed.py")

        print("Running LegalizeOps...")
        mod = relax.transform.LegalizeOps()(mod)
        with open("decoder_legalized.py", "w") as f: f.write(mod.script())
        print("Dumped decoder_legalized.py")

        print("Running AnnotateTIROpPattern...")
        mod = relax.transform.AnnotateTIROpPattern()(mod)

        print("Running FoldConstant...")
        # mod = relax.transform.FoldConstant()(mod)
        
        print("Running FuseOps...")
        mod = relax.transform.FuseOps()(mod)
        
        print("Running FuseTIR...")
        mod = relax.transform.FuseTIR()(mod)

    # For static shapes, we can use the standard relax.build() pipeline
    # Wrap in PassContext with debug info disabled to avoid LLVM verification bug
    print("Building with standard Relax pipeline (debug info disabled)...")
    with tvm.transform.PassContext(opt_level=3, config={"tir.enable_debug": False}):
        ex = relax.build(mod, target)
    print("Compilation successful!")
    
    # Save compiled library
    output_path = args.output
    ex.export_library(output_path)
    print(f"Saved compiled library to: {output_path}")
    
    # 5. Verify
    dev = tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)
    
    test_len = args.seq_len  # Use same sequence length as compilation
    asr_in = torch.randn(1, hidden_dim, test_len).numpy()
    f0_in = torch.randn(1, test_len * 2).numpy()
    n_in = torch.randn(1, test_len * 2).numpy()
    s_in = torch.randn(1, style_dim).numpy()
    
    inputs = [
        tvm.nd.array(asr_in, dev),
        tvm.nd.array(f0_in, dev),
        tvm.nd.array(n_in, dev),
        tvm.nd.array(s_in, dev)
    ]
    
    # We kept params as input. 
    # However, from_exported_program behavior on params depends on keep_params_as_input.
    # If True, params are inputs. If False, they are bound to module.
    # Let's check how many args 'main' expects.
    # We can inspect input_info if available, or just try running.
    # If keep_params_as_input=True, needed args = inputs + params.
    # If keep_params_as_input=False (default), params are constants/bound.
    
    # Let's re-import with keep_params_as_input=False (default) to make inference call easier
    # Or just use the one we have?
    # If True, we need to pass params.
    # Let's use False for simplicity of verification now.
    
    print("Re-importing with keep_params_as_input=False for easy inference...")
    importer = ExportedProgramImporter()
    mod = importer.from_exported_program(
        exported_program, 
        keep_params_as_input=False,
        unwrap_unit_return_tuple=False, 
        no_bind_return_tuple=False
    )
    with target:
        mod = seq(mod)
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    
    print(f"Running inference with test_len={test_len}...")
    output = vm["main"](*inputs)
    print("Output shape:", output.shape)
    print("Verification Successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Kokoro Decoder to TVM with static shapes")
    parser.add_argument("--seq-len", type=int, default=150, 
                        help="Static sequence length for compilation (default: 150)")
    parser.add_argument("--output", type=str, default="decoder_compiled.so",
                        help="Output path for compiled library (default: decoder_compiled.so)")
    args = parser.parse_args()
    main(args)
