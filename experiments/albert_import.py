
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
from transformers import AlbertConfig

# Add external/kokoro to path to import modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KOKORO_PATH = os.path.join(PROJECT_ROOT, "external", "kokoro")
sys.path.append(KOKORO_PATH)

# Import TVM extensions 
PYTHON_SRC = os.path.join(PROJECT_ROOT, "python", "src")
sys.path.append(PYTHON_SRC)
import kokoro_tvm.tvm_extensions

from kokoro.modules import CustomAlbert

def main():
    # Parameters for CustomAlbert
    # It takes AlbertConfig.
    # We should create a tiny config for testing to avoid huge model and memory usage.
    vocab_size = 100
    hidden_size = 64
    num_hidden_layers = 2
    num_attention_heads = 4
    intermediate_size = 128
    max_position_embeddings = 50
    
    config = AlbertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    
    print("Initializing CustomAlbert...")
    model = CustomAlbert(config)
    model.eval()

    # Create dummy inputs
    # forward(self, input_ids=None, attention_mask=None, ...)
    # input_ids: [Batch, Seq]
    BATCH_SIZE = 1
    SEQ_LEN = 10
    
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    
    args = (input_ids, attention_mask)

    # Export using torch.export
    print("Exporting model with torch.export...")
    # Transformers models often use sophisticated control flow or optional args.
    # torch.export might struggle unless we restrict args.
    # But CustomAlbert just returns last_hidden_state.
    
    try:
        exported_program = torch.export.export(model, args)
    except Exception as e:
        print(f"Export failed: {e}")
        # Maybe use strict=False or different tracing?
        # But we want strict export for TVM.
        return

    # Import into TVM
    print("Importing into TVM using from_exported_program...")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=False)
    except Exception as e:
         print(f"Import failed: {e}")
         return
    
    # Print function names
    # print("Module functions:", [x.name_hint for x in mod.get_global_vars()])
    
    # Dump script to debug boolean issues
    # print("Dumping Relax IRModule script...")
    # with open("debug_albert.py", "w") as f:
    #     f.write(mod.script())
    # print("Script dumped to debug_albert.py. Exiting for inspection.")
    # return
    
    # Compile and Run
    print("Compiling with TVM...")
    target = tvm.target.Target("llvm")
    try:
        ex = tvm.compile(mod, target)
    except Exception as e:
        print(f"Compilation failed: {e}")
        return

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

    # Unwrap output
    if isinstance(output_tvm, (list, tuple)) or "Array" in str(type(output_tvm)):
        if len(output_tvm) == 1:
            output_tvm = output_tvm[0]
        else:
             print(f"Warning output len: {len(output_tvm)}")
             output_tvm = output_tvm[0]

    # Verify correctness
    print("Verifying correctness...")
    with torch.no_grad():
        output_torch = model(*args) # CustomAlbert returns last_hidden_state (Tensor) directly.
        # But wait, original AlbertModel returns BaseModelOutput... 
        # CustomAlbert overrides forward to return `outputs.last_hidden_state`. So it returns a Tensor.

    if hasattr(output_tvm, "numpy"):
        output_tvm_np = output_tvm.numpy()
    else:
        output_tvm_np = output_tvm.asnumpy()
    
    print(f"Output shape: {output_tvm_np.shape}")
    print("TVM output (first 10 elements):", output_tvm_np.flatten()[:10])
    
    np.testing.assert_allclose(output_torch.numpy(), output_tvm_np, rtol=1e-4, atol=1e-4)
    print("SUCCESS: TVM output matches PyTorch output!")

if __name__ == "__main__":
    main()
