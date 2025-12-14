
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

from kokoro.modules import TextEncoder

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
    # Parameters for TextEncoder
    # channels, kernel_size, depth, n_symbols
    BATCH_SIZE = 1
    CHANNELS = 64
    KERNEL_SIZE = 3
    DEPTH = 2
    N_SYMBOLS = 100 # Vocab size
    SEQ_LEN = 15
    
    print("Initializing TextEncoder...")
    model = TextEncoder(channels=CHANNELS, kernel_size=KERNEL_SIZE, depth=DEPTH, n_symbols=N_SYMBOLS)
    model.eval()

    # Create dummy inputs
    # forward(self, x, input_lengths, m)
    # x: [B, T] indices
    input_x = torch.randint(0, N_SYMBOLS, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    
    # input_lengths: [B]
    input_lengths = torch.tensor([SEQ_LEN] * BATCH_SIZE, dtype=torch.long)
    
    # m: [B, T] booleans (True = masked/padding?)
    # "x.masked_fill_(m, 0.0)" implies m=True is masked out.
    input_m = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
    
    args = (input_x, input_lengths, input_m)

    # Export using torch.export
    print("Exporting model with torch.export...")
    exported_program = torch.export.export(model, args)

    # Import into TVM
    print("Importing into TVM using from_exported_program...")
    mod = from_exported_program(exported_program, keep_params_as_input=False)
    
    # Print function names
    print("Module functions:", [x.name_hint for x in mod.get_global_vars()])

    if "main" not in [x.name_hint for x in mod.get_global_vars()]:
        print("Error: No 'main' function found. Dumping script...")
        print(mod.script())
        return

    # Compile and Run
    print("Compiling with TVM...")
    target = tvm.target.Target("llvm")
    ex = tvm.compile(mod, target)

    print("Running compiled module...")
    dev = tvm.device("cpu")
    
    # Prepare inputs
    tvm_x = tensor(input_x.numpy(), dev)
    tvm_l = tensor(input_lengths.numpy(), dev)
    tvm_m = tensor(input_m.numpy(), dev)
    
    # Run
    try:
        if not callable(ex):
            vm = tvm.relax.VirtualMachine(ex, dev)
            output_tvm = vm["main"](tvm_x, tvm_l, tvm_m)
        else:
            output_tvm = ex(tvm_x, tvm_l, tvm_m)
    except Exception as e:
        print(f"Direct execution failed: {e}. Trying VirtualMachine explicit instantiation...")
        vm = tvm.relax.VirtualMachine(ex, dev)
        output_tvm = vm["main"](tvm_x, tvm_l, tvm_m)

    # Unwrap output
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
