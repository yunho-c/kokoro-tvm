import argparse
import json
import os
import sys

import numpy as np

import operator
import torch
import torch.nn.functional as F
import tvm
import tvm.relax.op as relax_op
from huggingface_hub import hf_hub_download
from kokoro.istftnet import Decoder, SineGen
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter

import tvm_extensions  # monkeypatches ExportedProgramImporter

# Monkeypatch SineGen to be friendly to symbolic shapes (Same as in export_decoder.py)
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
    # Initialize Decoder
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
    
    # Load pretrained weights from HuggingFace (unless --no-weights is specified)
    if not args.no_weights:
        model_filename = 'kokoro-v1_0.pth'
        print(f"Downloading pretrained weights: {model_filename}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
        
        print(f"Loading decoder weights from {model_path}...")
        state_dicts = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # The checkpoint contains weights for all components keyed by name
        # We only need the 'decoder' component
        if 'decoder' in state_dicts:
            decoder_state_dict = state_dicts['decoder']
            # Handle 'module.' prefix from DataParallel training (same as kokoro/model.py)
            # Check if keys have 'module.' prefix
            first_key = next(iter(decoder_state_dict.keys()), '')
            if first_key.startswith('module.'):
                print("Stripping 'module.' prefix from weight keys...")
                decoder_state_dict = {k[7:]: v for k, v in decoder_state_dict.items()}
            decoder.load_state_dict(decoder_state_dict, strict=False)
            print("Successfully loaded pretrained decoder weights!")
        else:
            print(f"Warning: 'decoder' key not found in checkpoint. Available keys: {list(state_dicts.keys())}")
            print("Proceeding with random weights...")
    else:
        print("Skipping weight loading (--no-weights specified). Using random weights.")
    
    decoder.eval()

    # Prepare Inputs with STATIC shapes
    batch_size = 1
    seq_len = args.seq_len  # Static sequence length from CLI
    
    asr = torch.randn(batch_size, hidden_dim, seq_len)
    f0 = torch.randn(batch_size, seq_len * 2)
    n = torch.randn(batch_size, seq_len * 2)
    s = torch.randn(batch_size, style_dim)
    
    print(f"Static inputs: asr={asr.shape}, f0={f0.shape}, n={n.shape}, s={s.shape}")

    # Export with STATIC shapes (no dynamic_shapes)
    print(f"Exporting model with static seq_len={seq_len}...")
    exported_program = torch.export.export(
        decoder,
        (asr, f0, n, s),
        # No dynamic_shapes - using static shapes for reliable TVM compilation
    )
    
    # Compile with TVM
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
        keep_params_as_input=False,  # Embed weights in the compiled module
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
    
    seq = tvm.transform.Sequential([
        relax.transform.DecomposeOpsForInference(),
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])
    
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
    
    # Verify
    dev = tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)
    
    test_len = args.seq_len  # Use same sequence length as compilation
    asr_in = torch.randn(1, hidden_dim, test_len).numpy()
    f0_in = torch.randn(1, test_len * 2).numpy()
    n_in = torch.randn(1, test_len * 2).numpy()
    s_in = torch.randn(1, style_dim).numpy()
    
    # Convert numpy arrays to TVM tensors using tvm.runtime.tensor
    inputs = [
        tvm.runtime.tensor(asr_in, device=dev),
        tvm.runtime.tensor(f0_in, device=dev),
        tvm.runtime.tensor(n_in, device=dev),
        tvm.runtime.tensor(s_in, device=dev)
    ]
    
    
    # Verify using the already-built VM (weights are now embedded via keep_params_as_input=False)
    print(f"Running inference with test_len={test_len}...")
    output = vm["decoder_forward"](*inputs)
    
    # Handle output - could be a single tensor or an Array of tensors
    if hasattr(output, 'shape'):
        print("Output shape:", output.shape)
    else:
        # Output is an Array (tuple of tensors)
        print(f"Output is an Array with {len(output)} elements:")
        for i, out in enumerate(output):
            print(f"  Output[{i}] shape: {out.shape}")
    print("Verification Successful!")


def validate_against_pytorch(args, target):
    """
    Validate TVM decoder output against PyTorch decoder using real encoder data.
    
    This function:
    1. Loads the full KModel (encoder + decoder)
    2. Runs encoder on a test phrase to get real intermediate tensors
    3. Runs both PyTorch and TVM decoders on same inputs
    4. Compares outputs numerically
    """
    from kokoro.model import KModel
    
    print("\n" + "="*60)
    print("VALIDATION: Comparing TVM vs PyTorch decoder")
    print("="*60)
    
    # Load full Kokoro model
    repo_id = 'hexgrad/Kokoro-82M'
    print(f"Loading full KModel from {repo_id}...")
    kmodel = KModel(repo_id=repo_id, disable_complex=True)
    kmodel.eval()
    
    # Load a voice pack
    voice_name = 'af_heart'
    print(f"Loading voice pack: {voice_name}...")
    voice_path = hf_hub_download(repo_id=repo_id, filename=f'voices/{voice_name}.pt')
    voice_pack = torch.load(voice_path, weights_only=True)
    
    # Test phrase - phonemes for "Hello world"
    test_phonemes = "helloworld"  # Simple ASCII for vocab lookup
    print(f"Test phonemes: '{test_phonemes}'")
    
    # Convert phonemes to input_ids first to get actual phoneme count
    input_ids = list(filter(lambda i: i is not None, 
                           map(lambda p: kmodel.vocab.get(p), test_phonemes)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    print(f"Input IDs shape: {input_ids.shape}")
    
    # Get ref_s from voice pack (indexed by phoneme length - similar to KPipeline.infer)
    # Voice pack is indexed by (phoneme_count - 1)
    phoneme_count = len(test_phonemes)
    ref_s = voice_pack[phoneme_count - 1]  # This gives (256,) or (1, 256)
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)  # Make it (1, 256)
    print(f"Voice embedding ref_s shape: {ref_s.shape}")
    
    # Run encoder to extract intermediate tensors
    with torch.no_grad():
        input_lengths = torch.full((1,), input_ids.shape[-1], dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        
        bert_dur = kmodel.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = kmodel.predictor.lstm(d)
        duration = kmodel.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = kmodel.predictor.F0Ntrain(en, s)
        t_en = kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        style_embed = ref_s[:, :128]
    
    print(f"\nExtracted decoder inputs:")
    print(f"  asr shape: {asr.shape}")
    print(f"  F0_pred shape: {F0_pred.shape}")
    print(f"  N_pred shape: {N_pred.shape}")
    print(f"  style_embed shape: {style_embed.shape}")
    
    seq_len = asr.shape[2]
    print(f"\nActual sequence length from encoder: {seq_len}")
    
    # Check if we need to recompile with correct seq_len
    if seq_len != args.seq_len:
        print(f"WARNING: Encoder produced seq_len={seq_len}, but TVM compiled with seq_len={args.seq_len}")
        print(f"For accurate validation, rerun with --seq-len {seq_len}")
        return
    
    # Run PyTorch decoder
    print("\nRunning PyTorch decoder...")
    with torch.no_grad():
        pytorch_output = kmodel.decoder(asr, F0_pred, N_pred, style_embed)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Run TVM decoder
    print("\nLoading TVM compiled module...")
    lib = tvm.runtime.load_module(args.output)
    dev = tvm.cpu()
    vm = relax.VirtualMachine(lib, dev)
    
    # Prepare TVM inputs
    tvm_inputs = [
        tvm.runtime.tensor(asr.numpy(), device=dev),
        tvm.runtime.tensor(F0_pred.numpy(), device=dev),
        tvm.runtime.tensor(N_pred.numpy(), device=dev),
        tvm.runtime.tensor(style_embed.numpy(), device=dev)
    ]
    
    print("Running TVM decoder...")
    tvm_output = vm["decoder_forward"](*tvm_inputs)
    
    # Handle output format
    if hasattr(tvm_output, 'shape'):
        tvm_result = tvm_output.numpy()
    else:
        tvm_result = tvm_output[0].numpy()
    print(f"TVM output shape: {tvm_result.shape}")
    
    # Compare outputs
    pytorch_np = pytorch_output.numpy()
    
    # Compute metrics
    abs_diff = np.abs(pytorch_np - tvm_result)
    mae = np.mean(abs_diff)
    max_error = np.max(abs_diff)
    rel_error = mae / (np.mean(np.abs(pytorch_np)) + 1e-8)
    
    # Correlation
    pytorch_flat = pytorch_np.flatten()
    tvm_flat = tvm_result.flatten()
    correlation = np.corrcoef(pytorch_flat, tvm_flat)[0, 1]
    
    print("\n" + "-"*40)
    print("COMPARISON RESULTS:")
    print("-"*40)
    print(f"  Mean Absolute Error:  {mae:.6e}")
    print(f"  Max Absolute Error:   {max_error:.6e}")
    print(f"  Relative Error:       {rel_error:.2%}")
    print(f"  Correlation:          {correlation:.6f}")
    
    # Pass/Fail threshold
    tolerance = 1e-3
    if mae < tolerance and correlation > 0.999:
        print(f"\n✓ VALIDATION PASSED (MAE < {tolerance}, correlation > 0.999)")
    else:
        print(f"\n✗ VALIDATION FAILED (MAE={mae:.2e}, correlation={correlation:.4f})")
    
    print("="*60 + "\n")


if __name__ == "__main__":
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
    
    # Configure debug output in tvm_extensions
    if args.debug:
        tvm_extensions.DEBUG_ENABLED = True
    
    # Store target for validation
    target = tvm.target.Target("llvm -opt-level=3")
    
    main(args)
    
    # Run validation if requested
    if args.validate:
        validate_against_pytorch(args, target)
