import torch
import torch.nn as nn
import tvm
import tvm.runtime
from tvm.runtime import tensor
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import numpy as np

# Define a simple network using a bidirectional LSTM
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
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        
        # Take last time step
        # Note: With dynamic shape, x.shape[1] is symbolic.
        # Indexing with symbolic shape on tensor might be tricky if not handled well,
        # but standard slicing usually works.
        last_out = out[:, -1, :] 
        return self.fc(last_out)

def main():
    # Parameters
    INPUT_SIZE = 5
    HIDDEN_SIZE = 8
    NUM_LAYERS = 1
    NUM_CLASSES = 3

    print("Initializing PyTorch model...")
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model.eval()

    # Create dummy input for export tracing
    BATCH_SIZE = 2
    SEQ_LEN = 10
    example_args = (torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE),)

    # Define dynamic shapes
    # Dim 0: batch, Dim 1: time. Dim 2: static input size.
    # The error report suggested using Dim.AUTO for better inference.
    
    dynamic_shapes = {
        "x": {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO}
    }

    print(f"Exporting model with dynamic shapes: {dynamic_shapes}...")
    try:
        # Use draft_export to see errors
        exported_program = torch.export.export(
            model, 
            example_args, 
            dynamic_shapes=dynamic_shapes
        )
    except Exception as e:
        print(f"Export failed (Standard): {e}")
        print("Trying draft_export to get more insights...")
        try:
             # draft_export might return a result even with errors?
             # Actually torch.export.draft_export is likely what I want.
             # Note: draft_export is available in recent pytorch.
             if hasattr(torch.export, "draft_export"):
                  ep = torch.export.draft_export(model, example_args, dynamic_shapes=dynamic_shapes)
                  print("Draft export succeeded. Diagnostics:")
                  print(ep._report)
                  return # Stop here as graph is likely invalid for TVM import
        except Exception as e2:
             print(f"Draft export also failed: {e2}")
        return

    # Import into TVM
    print("Importing into TVM using from_exported_program...")
    try:
        mod = from_exported_program(exported_program, keep_params_as_input=False)
    except Exception as e:
        print(f"Import failed: {e}")
        return
    
    # Print the Relax IRModule script to verify symbolic shapes
    print("Relax IRModule script (excert):")
    script = mod.script()
    print(script[:2000] + "..." if len(script) > 2000 else script)
    
    # Check if shapes are dynamic (look for SymInt/Var) in main args

    # Compile and Run
    print("Compiling with TVM...", flush=True)
    target = tvm.target.Target("llvm")
    
    # Debugging: Break down compilation
    try:
        # 1. Legalize and lower to TIR
        print("Step 1: Legalizing to TIR...", flush=True)
        # Apply default pipeline to get TIR
        # We can use tvm.relax.get_pipeline("default")(mod) or similar, 
        # but tvm.compile handles this.
        # Let's try to just build the executable to see if we can separate code gen.
        # tvm.lower(mod) might work if it's mixed? No, relax uses relax.build or compile.
        
        # Let's try to lower to execution strategy first (TIR generation)
        # This is roughly what happens inside.
        
        # Just printing specific lowering info if we can. 
        # But to test isolation, we can assume tvm.compile crashes in LLVM.
        # Let's try to print the mod *after* some passes if possible?
        
        # Or simpler: Isolate LLVM codegen.
        # If we can generate the TIR module without crashing, it's LLVM.
        pass 
        
        # We'll use a sequence of manual passes to see where it dies if we can,
        # but tvm.compile is a black box.
        # Let's try `relax.build` which is the older API, sometimes clearer stack trace?
        # Or better, just try to lower to TIR first.
        
        # Create an IRModule with legal ops
        import tvm.relax.transform as R_opt
        
        # Simple pipeline to get to TIR (this might crash if it's a lowering bug)
        print("  Applying LegalizeOps...", flush=True)
        mod_legal = R_opt.LegalizeOps()(mod)
        print("  LegalizeOps done. Printing first 20 lines of TIR...", flush=True)
        tir_script = mod_legal.script()
        print(tir_script[:1000] + "...", flush=True)
        
        # Convert to TIR (Decompose, etc typically handled by build)
        # But we can try to inspect.
        
        print("Step 2: Full Compilation (LLVM Codegen)...", flush=True)
        ex = tvm.compile(mod, target)
        print("Compilation finished.", flush=True)
    except Exception as e:
        print(f"Compilation failed with exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    print("Running compiled module with DIFFERENT shapes...", flush=True)
    dev = tvm.device("cpu")
    
    # Test 1: Shape smaller than trace example
    b1, s1 = 1, 5
    input1 = torch.randn(b1, s1, INPUT_SIZE)
    
    # Test 2: Shape larger than trace example
    b2, s2 = 4, 15
    input2 = torch.randn(b2, s2, INPUT_SIZE)
    
    vm = tvm.relax.VirtualMachine(ex, dev)
    
    for i, inp in enumerate([input1, input2]):
        print(f"Test case {i+1}: Input shape {inp.shape}")
        
        # Run TVM
        tvm_in = tensor(inp.numpy(), dev)
        tvm_out = vm["main"](tvm_in)
        
        # Unwrap
        if isinstance(tvm_out, (list, tuple)): tvm_out = tvm_out[0]
        tvm_res = tvm_out.numpy()
        
        # Run PyTorch
        with torch.no_grad():
            torch_res = model(inp).numpy()
            
        print(f"  TVM Output shape: {tvm_res.shape}")
        np.testing.assert_allclose(torch_res, tvm_res, rtol=1e-4, atol=1e-4)
        print("  SUCCESS: Matches PyTorch.")

if __name__ == "__main__":
    main()
