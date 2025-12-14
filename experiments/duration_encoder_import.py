
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

from kokoro.modules import DurationEncoder

# Import TVM extensions for better operator support (e.g. expand, lstm)
# We assume the python package structure allows this, or we adaptsys parameters.
# The `kokoro-tvm` package is in `python/src`.
PYTHON_SRC = os.path.join(PROJECT_ROOT, "python", "src")
sys.path.append(PYTHON_SRC)
import kokoro_tvm.tvm_extensions

# Monkeypatching to avoid packed sequences which fail torch.export
# and simplify the graph for TVM.
original_pack = nn.utils.rnn.pack_padded_sequence
original_pad = nn.utils.rnn.pad_packed_sequence

def mock_pack(input, lengths, batch_first=False, enforce_sorted=True):
    return input

def mock_pad(sequence, batch_first=False, padding_value=0.0, total_length=None):
    # Returns (output, lengths_int64)
    # We don't have lengths readily available here if we just returned input in mock_pack?
    # Actually, pad_packed_sequence returns (output, lengths).
    # We can return dummy lengths or valid ones if we kept them.
    # But usually we just need the output tensor.
    # Dimensions might need checking if batch_first was changed?
    # In DurationEncoder, both are called with batch_first=True.
    return sequence, None

nn.utils.rnn.pack_padded_sequence = mock_pack
nn.utils.rnn.pad_packed_sequence = mock_pad

def main():
    # Parameters for DurationEncoder
    # init(self, sty_dim, d_model, nlayers, dropout=0.1)
    BATCH_SIZE = 1
    STY_DIM = 64
    D_MODEL = 64
    NLAYERS = 2
    DROPOUT = 0.0 # Deterministic
    SEQ_LEN = 10
    
    # Initialize model
    print("Initializing DurationEncoder...")
    model = DurationEncoder(sty_dim=STY_DIM, d_model=D_MODEL, nlayers=NLAYERS, dropout=DROPOUT)
    model.eval()

    # Create dummy inputs
    # forward(self, x, style, text_lengths, m)
    # x: [batch, length, d_model] ? No, source says: `x = x.permute(2, 0, 1)` implying input is [B, C, T] or [B, T, C]?
    # Let's check modules.py source again.
    # TextEncoder output is x produced by LSTM.flatten_parameters() -> output. 
    # Usually [B, T, C] from LSTM with batch_first=True.
    
    # In DurationEncoder.forward:
    # x = x.permute(2, 0, 1) -> input x is likely [B, T, C] (batch_first) or [B, C, T]?
    # If input is [B, T, C], permute(2, 0, 1) -> [C, B, T].
    # Then `x = torch.cat([x, s], axis=-1)` where s is expanded style [B, sty, T] (after expansion).
    # Wait, `s = style.expand(x.shape[0], x.shape[1], -1)` ??
    # If x is [C, B, T], then x.shape[0]=C, x.shape[1]=B.
    # style is [B, sty_dim].
    # style.expand(C, B, -1) -> [C, B, sty_dim].
    # cat([x, s], axis=-1) -> [C, B, T+sty_dim].
    
    # Wait, the code says:
    # x = x.permute(2, 0, 1)
    # s = style.expand(x.shape[0], x.shape[1], -1)
    # x = torch.cat([x, s], axis=-1)
    
    # This implies x and s are concatenated along the LAST dimension.
    # So if x is [C, B, T], s is [C, B, sty_dim]. 
    # That means concatenation dimension is T + sty_dim? 
    # That seems like channel concat?
    
    # But then:
    # x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    
    # Let's assume input x is [B, T, C] aka [Batch, Length, Channels/ModelDim].
    # Then permute(2, 0, 1) makes it [C, B, T].
    # This matches typical "time-last" or "channel-first" manipulation for some purpose? 
    # Or maybe it's just mixing dimensions.
    
    # Let's use [B, C, T] as input x, based on error analysis involving masking.
    input_x = torch.randn(BATCH_SIZE, D_MODEL, SEQ_LEN)
    input_style = torch.randn(BATCH_SIZE, STY_DIM)
    
    # text_lengths: [batch]
    input_lengths = torch.tensor([SEQ_LEN] * BATCH_SIZE, dtype=torch.long)
    
    # m (mask): [batch, length] ?
    # In TextEncoder: m = m.unsqueeze(1) -> [B, 1, T] usually.
    # Here inputs are `m`.
    # Code: `masks = m`. `x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0) `
    # If masks is [B, T]. unsqueeze(-1) -> [B, T, 1]. transpose(0, 1) -> [T, B, 1].
    # If x is [C, B, T]. masking with [T, B, 1] ?? 
    # That doesn't broadcast well onto [C, B, T] unless it's [T, B, C] or similar.
    
    # Let's look at `x = torch.cat([x, s], axis=-1)`.
    # Only possible if dim -1 matches or if we are appending features?
    # If x is [C, B, T], s is [C, B, sty].
    # They concat to [C, B, T+sty] ? No, axis=-1.
    # So the resulting tensor has size T + sty in the last dim.
    
    # It seems x is expected to be [B, T, calc_dim] or something?
    # Actually, usually x coming into DurationEncoder comes from TextEncoder which outputs [B, T, C].
    
    # Let's assume:
    # x: [B, T, D_MODEL]
    # m: [B, T] boolean mask (True for padding?)
    input_m = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool) # No masking for simplicity
    
    args = (input_x, input_style, input_lengths, input_m)

    # Export using torch.export
    print("Exporting model with torch.export...")
    exported_program = torch.export.export(model, args)

    # Import into TVM
    print("Importing into TVM using from_exported_program...")
    mod = from_exported_program(exported_program, keep_params_as_input=False)
    
    # Print the Relax IRModule
    print("Relax IRModule:")
    print(mod.script())

    # Print function names
    print("Module functions:", [x.name_hint for x in mod.get_global_vars()])

    # Compile and Run
    print("Compiling with TVM...")
    target = tvm.target.Target("llvm")
    ex = tvm.compile(mod, target)

    print("Running compiled module...")
    dev = tvm.device("cpu")
    
    # Prepare inputs
    tvm_x = tensor(input_x.numpy(), dev)
    tvm_s = tensor(input_style.numpy(), dev)
    tvm_l = tensor(input_lengths.numpy(), dev)
    tvm_m = tensor(input_m.numpy(), dev)
    
    # Run
    try:
        # Try calling it directly (if it's a packed func or similar wrapper)
        if not callable(ex):
            vm = tvm.relax.VirtualMachine(ex, dev)
            output_tvm = vm["main"](tvm_x, tvm_s, tvm_l, tvm_m)
        else:
            output_tvm = ex(tvm_x, tvm_s, tvm_l, tvm_m)
    except Exception as e:
        print(f"Direct execution failed: {e}. Trying VirtualMachine explicit instantiation...")
        vm = tvm.relax.VirtualMachine(ex, dev)
        output_tvm = vm["main"](tvm_x, tvm_s, tvm_l, tvm_m)

    # Unwrap output if it is a tuple/list/Array
    if isinstance(output_tvm, (list, tuple)) or "Array" in str(type(output_tvm)):
        if len(output_tvm) == 1:
            output_tvm = output_tvm[0]
        else:
            print(f"Warning: Output has len {len(output_tvm)}, taking index 0")
            output_tvm = output_tvm[0]
            
    # Verify correctness
    print("Verifying correctness...")
    with torch.no_grad():
        output_torch = model(*args).numpy()
    
    if hasattr(output_tvm, "numpy"):
        output_tvm_np = output_tvm.numpy()
    else:
        output_tvm_np = output_tvm.asnumpy()
    
    print("TVM output (first 10 elements):", output_tvm_np.flatten()[:10])
    
    np.testing.assert_allclose(output_torch, output_tvm_np, rtol=1e-4, atol=1e-4)
    print("SUCCESS: TVM output matches PyTorch output!")

if __name__ == "__main__":
    main()
