
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

from kokoro.istftnet import AdainResBlk1d

def main():
    # Parameters for AdainResBlk1d
    BATCH_SIZE = 1
    DIM_IN = 16
    DIM_OUT = 16 # Keep same for simplicity unless testing upsample
    STYLE_DIM = 64
    SEQ_LEN = 20
    
    # 1. Initialize model
    print("Initializing AdainResBlk1d...")
    # Using defaults: upsample='none', dropout_p=0.0 (important for deterministic export)
    model = AdainResBlk1d(DIM_IN, DIM_OUT, style_dim=STYLE_DIM, upsample='none', dropout_p=0.0)
    model.eval()

    # Create dummy inputs
    # Input x: [batch, dim_in, length] (Conv1d expects [B, C, L])
    input_x = torch.randn(BATCH_SIZE, DIM_IN, SEQ_LEN)
    # Input s: [batch, style_dim] (Style vector)
    input_s = torch.randn(BATCH_SIZE, STYLE_DIM)

    # Export using torch.export
    print("Exporting model with torch.export...")
    exported_program = torch.export.export(model, (input_x, input_s))

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
    tvm_s = tensor(input_s.numpy(), dev)
    
    # Run
    try:
        # Try calling it directly (if it's a packed func or similar wrapper)
        if not callable(ex):
            vm = tvm.relax.VirtualMachine(ex, dev)
            output_tvm = vm["main"](tvm_x, tvm_s)
        else:
            output_tvm = ex(tvm_x, tvm_s)
    except Exception as e:
        print(f"Direct execution failed: {e}. Trying VirtualMachine explicit instantiation...")
        vm = tvm.relax.VirtualMachine(ex, dev)
        output_tvm = vm["main"](tvm_x, tvm_s)

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
        output_torch = model(input_x, input_s).numpy()
    
    if hasattr(output_tvm, "numpy"):
        output_tvm_np = output_tvm.numpy()
    else:
        # Fallback for some wrappers
        output_tvm_np = output_tvm.asnumpy()
    
    print("TVM output (first 10 elements):", output_tvm_np.flatten()[:10])
    
    np.testing.assert_allclose(output_torch, output_tvm_np, rtol=1e-4, atol=1e-4)
    print("SUCCESS: TVM output matches PyTorch output!")

if __name__ == "__main__":
    main()
