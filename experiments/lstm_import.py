
import torch
import torch.nn as nn
import tvm
import tvm.runtime
# from tvm.runtime import ndarray # This was wrong
from tvm.runtime import tensor
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import numpy as np

# Define a simple network using a bidirectional LSTM with static shape
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        # Bidirectional means output features will be 2 * hidden_size
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # LSTM output shape: (batch_size, sequence_length, 2 * hidden_size)
        out, _ = self.lstm(x)
        
        # We usually take the last time step output for classification
        # But for this simple network, let's just project the last output
        # Use explicit index to avoid potential TVM issue with negative indexing in R.take
        seq_len = x.shape[1]
        last_out = out[:, seq_len - 1, :] 
        return self.fc(last_out)

def main():
    # Parameters
    BATCH_SIZE = 1
    SEQ_LEN = 10
    INPUT_SIZE = 5
    HIDDEN_SIZE = 8
    NUM_LAYERS = 1
    NUM_CLASSES = 3

    print("Initializing PyTorch model...")
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model.eval()

    # Create dummy input with static shape
    input_data = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)

    # Export using torch.export
    print("Exporting model with torch.export...")
    # We pass args as a tuple
    exported_program = torch.export.export(model, (input_data,))

    # Import into TVM
    print("Importing into TVM using from_exported_program...")
    mod = from_exported_program(exported_program, keep_params_as_input=True)
    
    # Print the Relax IRModule
    print("Relax IRModule:")
    print(mod.script())

    # Compile and Run
    print("Compiling with TVM...")
    target = tvm.target.Target("llvm")
    # Use tvm.compile (unified entry point) or relax.build? 
    # The user asked to use `ExportedProgram` interface which we did.
    # For execution, we can use the unified tvm.compile which returns an Executor.
    
    # Note: tvm.compile returns a tvm.runtime.Module or Executor depending on usage.
    # But usually for Relax, we might strictly want to use the virtual machine if it contains dynamic behavior,
    # though here it is static.
    
    # Let's use tvm.compile as it is modern.
    ex = tvm.compile(mod, target)

    print("Running compiled module...")
    # Prepare inputs
    # tvm.compile with relax usually returns an object that we can call with device
    dev = tvm.device("cpu")
    
    # If ex is a simpler Module, we might need a VM. 
    # However, tvm.compile usually wraps it. 
    
    # If the returned object is a JitModule (from tvm.compile of Relax), we might treat it as a function?
    # Actually, let's check what ex is.
    print(f"Compiled object type: {type(ex)}")
    
    # Depending on TVM version, tvm.compile might return a standard runtime Module 
    # that needs to be wrapped in a VirtualMachine or it might return a wrapper.
    # Let's try to use it as a callable first or inspect.
    
    # Actually, a common pattern with recent TVM:
    # vm_exec = tvm.compile(mod, target)
    # output = vm_exec(input_tvm, params...)
    
    # We kept params as input, so we need to pass them.
    # When keep_params_as_input=True, the parameters are explicit arguments to the main function.
    # But wait, from_exported_program might bind params if we didn't say keep_params_as_input=True.
    # I set keep_params_as_input=True in the call above.
    # This means the main function will surely take params as arguments.
    
    # Let's try WITHOUT keep_params_as_input=True to simplify execution (params bound to constants)
    # Re-importing for simplicity in this demo.
    print("Re-importing with params bound to ease execution...")
    mod = from_exported_program(exported_program, keep_params_as_input=False)
    ex = tvm.compile(mod, target)
    
    # Run
    input_tvm = tensor(input_data.numpy(), dev)
    
    # Run the model
    # If ex is a runtime.Module, we might need to load it into a VM if it's Relax.
    # But modern tvm.compile on a Relax module often returns a callable wrapper if possible.
    # If it fails, we will fall back to relax.VirtualMachine.
    
    try:
        # Try calling it directly (if it's a packed func or similar wrapper)
        # Note: tvm.compile returns a module. If it's a Relax module, we typically need a VM.
        # But maybe tvm.compile handles that?
        # Let's assume we need to instantiate a VM if it's not callable.
        if not callable(ex):
            vm = tvm.relax.VirtualMachine(ex, dev)
            output_tvm = vm["main"](input_tvm)
        else:
             output_tvm = ex(input_tvm)
             
    except Exception as e:
        print(f"Direct execution failed: {e}. Trying VirtualMachine explicit instantiation...")
        vm = tvm.relax.VirtualMachine(ex, dev)
        output_tvm = vm["main"](input_tvm)

    # Verify correctness
    print("Verifying correctness...")
    with torch.no_grad():
        output_torch = model(input_data).numpy()
    
    # Inspect output
    print(f"TVM output type: {type(output_tvm)}")
    if isinstance(output_tvm, (list, tuple)) or hasattr(output_tvm, "__getitem__"):
        # If it's a sequence, take the first element (assuming single output model)
        if hasattr(output_tvm, "numpy"): # It might be a tensor directly
             pass
        elif len(output_tvm) > 0:
             print("Output is a sequence, taking first element.")
             output_tvm = output_tvm[0]

    output_tvm_np = output_tvm.numpy()
    
    print("TVM output (first 10 elements):", output_tvm_np.flatten()[:10])
    
    # Compare
    np.testing.assert_allclose(output_torch, output_tvm_np, rtol=1e-4, atol=1e-4)
    print("SUCCESS: TVM output matches PyTorch output!")

if __name__ == "__main__":
    main()
