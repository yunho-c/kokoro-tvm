# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Validation utilities for comparing TVM and PyTorch outputs."""

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download


def validate_decoder_against_pytorch(
    tvm_lib_path: str,
    seq_len: int,
    sample_rate: int = 24000,
) -> dict:
    """Validate TVM decoder output against PyTorch decoder using real encoder data.
    
    This function:
    1. Loads the full KModel (encoder + decoder)
    2. Runs encoder on a test phrase to get real intermediate tensors
    3. Runs both PyTorch and TVM decoders on same inputs
    4. Compares outputs numerically
    
    Args:
        tvm_lib_path: Path to compiled TVM library (.so file)
        seq_len: Expected sequence length (must match compilation)
        sample_rate: Audio sample rate for output files
        
    Returns:
        Dictionary with validation metrics
    """
    import tvm
    from tvm import relax
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
    test_phonemes = "helloworld"
    print(f"Test phonemes: '{test_phonemes}'")
    
    # Convert phonemes to input_ids
    input_ids = list(filter(lambda i: i is not None, 
                           map(lambda p: kmodel.vocab.get(p), test_phonemes)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    print(f"Input IDs shape: {input_ids.shape}")
    
    # Get ref_s from voice pack
    phoneme_count = len(test_phonemes)
    ref_s = voice_pack[phoneme_count - 1]
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
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
    
    actual_seq_len = asr.shape[2]
    print(f"\nActual sequence length from encoder: {actual_seq_len}")
    
    if actual_seq_len != seq_len:
        print(f"WARNING: Encoder produced seq_len={actual_seq_len}, but TVM compiled with seq_len={seq_len}")
        print(f"For accurate validation, rerun with --seq-len {actual_seq_len}")
        return {"error": "seq_len_mismatch", "expected": seq_len, "actual": actual_seq_len}
    
    # Run PyTorch decoder
    print("\nRunning PyTorch decoder...")
    with torch.no_grad():
        pytorch_output = kmodel.decoder(asr, F0_pred, N_pred, style_embed)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Run TVM decoder
    print("\nLoading TVM compiled module...")
    lib = tvm.runtime.load_module(tvm_lib_path)
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
    passed = mae < tolerance and correlation > 0.999
    if passed:
        print(f"\n✓ VALIDATION PASSED (MAE < {tolerance}, correlation > 0.999)")
    else:
        print(f"\n✗ VALIDATION FAILED (MAE={mae:.2e}, correlation={correlation:.4f})")
    
    # Save audio files for qualitative comparison
    pytorch_audio = pytorch_np.squeeze()
    tvm_audio = tvm_result.squeeze()
    
    pytorch_wav = "validation_pytorch.wav"
    tvm_wav = "validation_tvm.wav"
    
    sf.write(pytorch_wav, pytorch_audio, sample_rate)
    sf.write(tvm_wav, tvm_audio, sample_rate)
    
    print(f"\n📁 Audio files saved for qualitative comparison:")
    print(f"   - {pytorch_wav} (PyTorch decoder)")
    print(f"   - {tvm_wav} (TVM decoder)")
    print(f"   Sample rate: {sample_rate} Hz, Duration: {len(pytorch_audio)/sample_rate:.2f}s")
    
    print("="*60 + "\n")
    
    return {
        "passed": passed,
        "mae": float(mae),
        "max_error": float(max_error),
        "rel_error": float(rel_error),
        "correlation": float(correlation),
        "pytorch_wav": pytorch_wav,
        "tvm_wav": tvm_wav,
    }
