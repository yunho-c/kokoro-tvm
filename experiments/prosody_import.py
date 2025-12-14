
import torch
import torch.nn as nn
import tvm
import tvm.runtime
from tvm.runtime import tensor
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import numpy as np
import sys
import os

# Add external/kokoro to path to import modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KOKORO_PATH = os.path.join(PROJECT_ROOT, "external", "kokoro")
sys.path.append(KOKORO_PATH)

# Import TVM extensions 
PYTHON_SRC = os.path.join(PROJECT_ROOT, "python", "src")
sys.path.append(PYTHON_SRC)
import kokoro_tvm.tvm_extensions

from kokoro.modules import ProsodyPredictor

# Monkeypatching for PackedSequence
original_pack = nn.utils.rnn.pack_padded_sequence
original_pad = nn.utils.rnn.pad_packed_sequence

def mock_pack(input, lengths, batch_first=False, enforce_sorted=True):
    return input

def mock_pad(sequence, batch_first=False, padding_value=0.0, total_length=None):
    return sequence, None

nn.utils.rnn.pack_padded_sequence = mock_pack
nn.utils.rnn.pad_packed_sequence = mock_pad

def main():
    # Parameters for ProsodyPredictor
    # style_dim, d_hid, nlayers, max_dur=50, dropout=0.1
    BATCH_SIZE = 1
    STYLE_DIM = 64
    D_HID = 64
    NLAYERS = 2
    MAX_DUR = 50
    DROPOUT = 0.0
    SEQ_LEN = 10
    ALIGNED_LEN = 20 # Length after alignment
    
    print("Initializing ProsodyPredictor...")
    model = ProsodyPredictor(style_dim=STYLE_DIM, d_hid=D_HID, nlayers=NLAYERS, max_dur=MAX_DUR, dropout=DROPOUT)
    model.eval()

    # Create dummy inputs
    # forward(self, texts, style, text_lengths, alignment, m)
    # texts: [B, T, D_HID] (input to text_encoder (DurationEncoder)). 
    # Wait, DurationEncoder expects [B, C, T] ? in my previous fix I used [B, C, T].
    # Let's check DurationEncoder usage in ProsodyPredictor.
    # self.text_encoder = DurationEncoder(...)
    # d = self.text_encoder(texts, style, text_lengths, m)
    # If DurationEncoder expects [B, C, T], then texts here should be [B, D_HID, T].
    input_texts = torch.randn(BATCH_SIZE, D_HID, SEQ_LEN)
    
    input_style = torch.randn(BATCH_SIZE, STYLE_DIM)
    input_text_lengths = torch.tensor([SEQ_LEN] * BATCH_SIZE, dtype=torch.long)
    
    # alignment: [B, T, AlignedT] ?
    # en = (d.transpose(-1, -2) @ alignment)
    # d comes from text_encoder output. DurationEncoder returns x.transpose(-1, -2).
    # Inside DurationEncoder: x=[B,T,C]. Returns [B,C,T].
    # So d is [B, C, T] ? 
    # d.transpose(-1, -2) -> [B, T, C].
    # alignment must be compatible with matmul.
    # [B, T, C] @ [B, T, AlignedT] ??? No.
    # Usually alignment matches T to AlignedT.
    # If d is [B, C, T] (channels=d_model), transpose -> [B, T, C].
    # Matrix mul: [B, T, C] @ alignment -> [B, T, AlignedT]? No dimensions mismatch for result.
    # We want result `en` to be aligned features.
    # Usually: [B, C, T] @ [B, T, AlignedT] -> [B, C, AlignedT].
    # So d must be [B, C, T]. 
    # d.transpose(-1, -2) is likely [B, C, T].transpose -> [B, T, C] ??
    # Let's check DurationEncoder return: "return x.transpose(-1, -2)".
    # Inside DurationEncoder, last step: x = x.transpose(-1, -2).
    # If x was [B, T, C], return is [B, C, T].
    # So d is [B, C, T].
    # d.transpose(-1, -2) is [B, T, C].
    # To get [B, C, AlignedT], we need:
    # d @ alignment ? [B, C, T] @ [B, T, AlignedT] -> [B, C, AlignedT].
    # But code says: (d.transpose(-1, -2) @ alignment).
    # [B, T, C] @ alignment.
    # For this to work, alignment must be [B, C, Something]? Unlikely.
    # Unless C is the contracting dimension? 
    # Wait, looking at `kokoro/model.py`:
    # pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0])) -> [T_text, T_audio]
    # en = d.transpose(-1, -2) @ pred_aln_trg
    # d is [B, C, T_text]. d.transpose -> [B, T_text, C].
    # [B, T_text, C] @ [T_text, T_audio] -> NO.
    # [B, C, T_text] @ [T_text, T_audio] -> [B, C, T_audio].
    # So d (before transpose) @ alignment works.
    # But code has d.transpose(-1, -2).
    # Maybe DurationEncoder returns [B, T, C]?
    # Let's checking DurationEncoder again.
    # x (after LSTM) = [B, T, C].
    # x.transpose(-1, -2) -> [B, C, T].
    # Returns [B, C, T].
    # So d = [B, C, T].
    # d.transpose(-1, -2) = [B, T, C].
    # If alignment is [B, C, aligned]? No alignment is usually text-to-audio map.
    
    # RE-READ `kokoro/model.py` carefully:
    # d = self.predictor.text_encoder(...)
    # ...
    # en = d.transpose(-1, -2) @ pred_aln_trg
    # pred_aln_trg is constructed as [T_text, T_audio] (unsqueezed to [1, T_in, T_out]).
    # So d must be [B, C, T] -> transpose -> [B, T, C] ??
    # [B, T, C] @ [B, T, A] -> This is valid batch matmul only if C == T ??? No.
    # [1, 2, 3] @ [1, 3, 4] -> [1, 2, 4].
    # [B, T, C] @ [B, C, A] -> [B, T, A].
    # [B, C, T] @ [B, T, A] -> [B, C, A].
    
    # If d is [B, C, T], then d @ alignment gives [B, C, A].
    # Why is there a transpose in `model.py`?
    # "en = d.transpose(-1, -2) @ pred_aln_trg"
    # This implies d is [B, T, C] coming out of text_encoder?
    # But ProsodyPredictor.text_encoder returns `x.transpose(-1, -2)`.
    # If DurationEncoder x is [B, T, C], return is [B, C, T].
    # Then `d` is [B, C, T].
    # `d.transpose` is [B, T, C].
    # `pred_aln_trg` is [B, T, A].
    # [B, T, C] @ [B, T, A] ??
    # Matmul over last dim of first and first dim of second (after batch).
    # C must equal T? No.
    
    # Wait, maybe `alignment` is NOT [B, T, A].
    # In `model.py`:
    # pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0])) -> [T, A].
    # pred_aln_trg = pred_aln_trg.unsqueeze(0) -> [1, T, A].
    
    # Maybe the line in modules.py is different?
    # modules.py:
    # en = (d.transpose(-1, -2) @ alignment)
    
    # Assuming d=[B,C,T], d^T=[B,T,C].
    # Alignment=[B, T, A].
    # This multiplication [B,T,C] @ [B,T,A] makes no sense unless T=T or T=C.
    
    # Let's consider `d` is just [B, C, T]. And the code intends [B, C, T] @ [B, T, A] -> [B, C, A].
    # Then `d.transpose(-1, -2)` is WRONG unless d was [B, T, C] initially?
    # Or maybe alignment is [B, A, T]? 
    # If alignment is [B, A, T], then [B, T, C] @ [B, A, T]^T ? No.
    
    # Let's TRUST THE CODE but be careful with shapes.
    # If I use d=[B, T, C] output from DurationEncoder?
    # DurationEncoder.forward returns `x.transpose(-1, -2)`.
    # If x is [B, T, C] (batch_first LSTM output), then return is [B, C, T].
    
    # CHECK ProsodyPredictor again.
    # d = self.text_encoder(...)
    # en = (d.transpose(-1, -2) @ alignment)
    
    # If I assume `en` should be [B, C, A] (content aligned to audio frames).
    # Then we need [B, C, T] @ [B, T, A].
    # If d=[B, C, T], we just do d @ alignment.
    # Why transpose?
    # Maybe d is [B, T, C]?
    # If d=[B, T, C], d.transpose = [B, C, T].
    # Then [B, C, T] @ [B, T, A] works!
    
    # So `d` MUST be [B, T, C].
    # But `DurationEncoder` returns [B, C, T]!
    # Contradiction?
    # Or maybe DurationEncoder Logic I analyzed earlier was wrong.
    # `x = x.transpose(-1, -2)` at end of DurationEncoder.
    
    # Implementation detail: I will provide inputs and see what happens.
    # I'll construct alignment as [B, T, A] which is standard alignment matrix.
    # I'll inputs texts as [B, D_HID, T] assuming DurationEncoder permutes it to [B, T, D_HID] internally?
    # DurationEncoder: `x = x.permute(2, 0, 1)`. 
    # If x is [B, C, T]: permute -> [T, B, C].
    # Then cat with style...
    # Then LSTM...
    # Then x (from lstm) is [B, T, C].
    # Returns `x.transpose(-1, -2)` -> [B, C, T].
    
    # So `d` in ProsodyPredictor is [B, C, T].
    # Then `d.transpose` is [B, T, C].
    # Match! 
    # So d.transpose @ alignment ([B, T, A]) is [B, T, C] @ [B, T, A] ??? 
    # This is [B, T, C] x [B, T, A]. 
    # Inner dims C and T don't match.
    # Unless C=T? No.
    
    # Wait, `matmul` ( @ ) logic:
    # (..., M, K) @ (..., K, N) -> (..., M, N).
    # Here: (..., T, C) @ (..., T, A) -> C must equal T? No.
    # Inputs: [B, T, C] and [B, T, A].
    # Inner dims are C and T. Mismatch.
    
    # MAYBE alignment is [B, C, A]? No, alignment is time-based.
    
    # Let's look at `modules.py` ProsodyPredictor line 121 again.
    # `en = (d.transpose(-1, -2) @ alignment)`
    
    # Is it possible d is [B, A, T]? No.
    # Is it possible alignment is [B, C, A]? No.
    
    # Maybe I should just set up the test and let PyTorch error tell me the shape mismatch?
    # Best way to resolve this confusion.
    
    input_alignment = torch.randn(BATCH_SIZE, SEQ_LEN, ALIGNED_LEN) # [B, T, A]
    
    # m: [B, T] booleans
    input_m = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
    
    args = (input_texts, input_style, input_text_lengths, input_alignment, input_m)

    # Export using torch.export
    print("Exporting model with torch.export...")
    exported_program = torch.export.export(model, args)

    # Import into TVM
    print("Importing into TVM using from_exported_program...")
    mod = from_exported_program(exported_program, keep_params_as_input=False)
    
    # Compile and Run
    print("Compiling with TVM...")
    target = tvm.target.Target("llvm")
    ex = tvm.compile(mod, target)

    print("Running compiled module...")
    dev = tvm.device("cpu")
    
    # Prepare inputs
    tvm_args = [tensor(x.numpy(), dev) for x in args]
    
    # Run
    try:
        if not callable(ex):
            vm = tvm.relax.VirtualMachine(ex, dev)
            output_tvm = vm["main"](*tvm_args)
        else:
            output_tvm = ex(*tvm_args)
    except Exception as e:
        print(f"Direct execution failed: {e}. Trying VirtualMachine explicit instantiation...")
        vm = tvm.relax.VirtualMachine(ex, dev)
        output_tvm = vm["main"](*tvm_args)

    # Unwrap output (tuple)
    # returns duration, en
    # So we expect Tuple with 2 elements.
    
    # Verify correctness
    print("Verifying correctness...")
    with torch.no_grad():
        output_torch = model(*args) # tuple (duration, en)

    # Convert TVM output to list of numpy
    output_tvm_np = []
    
    # Check if Array or tuple/list
    if isinstance(output_tvm, (list, tuple)) or "Array" in str(type(output_tvm)):
        for i in range(len(output_tvm)):
            x = output_tvm[i]
            if hasattr(x, "numpy"): output_tvm_np.append(x.numpy())
            else: output_tvm_np.append(x.asnumpy())
    else:
        # Unexpected scalar?
        if hasattr(output_tvm, "numpy"): output_tvm_np = [output_tvm.numpy()]
        else: output_tvm_np = [output_tvm.asnumpy()]

    print(f"Output count: {len(output_tvm_np)}")

    for i, (expect, actual) in enumerate(zip(output_torch, output_tvm_np)):
        print(f"Verifying output {i} shape {expect.shape} vs {actual.shape}")
        np.testing.assert_allclose(expect.numpy(), actual, rtol=1e-4, atol=1e-4)

    print("SUCCESS: TVM output matches PyTorch output!")

if __name__ == "__main__":
    main()
