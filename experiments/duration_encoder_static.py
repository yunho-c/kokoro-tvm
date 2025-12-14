
import torch
import torch.nn as nn
import tvm
import tvm.runtime
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import sys
import os

# Add external/kokoro to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KOKORO_PATH = os.path.join(PROJECT_ROOT, "external", "kokoro")
sys.path.append(KOKORO_PATH)
PYTHON_SRC = os.path.join(PROJECT_ROOT, "python", "src")
sys.path.append(PYTHON_SRC)

# Import original module (monkeypatch removed)
from kokoro.modules import DurationEncoder
import kokoro_tvm.tvm_extensions # Still need expand fix

class StaticLengthsWrapper(nn.Module):
    def __init__(self, model, static_lengths):
        super().__init__()
        self.model = model
        # Register as buffer so it's part of the module but not a parameter to optimize
        self.register_buffer("static_lengths", static_lengths)

    def forward(self, x, style, m):
        # We ignore 'm' for lengths or assume it matches
        # Call model with our static lengths
        return self.model(x, style, self.static_lengths, m)

def main():
    BATCH_SIZE = 1
    STY_DIM = 64
    D_MODEL = 64
    NLAYERS = 2
    DROPOUT = 0.0
    SEQ_LEN = 10
    
    print("Initializing DurationEncoder...")
    base_model = DurationEncoder(sty_dim=STY_DIM, d_model=D_MODEL, nlayers=NLAYERS, dropout=DROPOUT)
    base_model.eval()
    
    # Static lengths tensor
    lengths = torch.tensor([SEQ_LEN] * BATCH_SIZE, dtype=torch.long)
    
    # Wrap it
    model = StaticLengthsWrapper(base_model, lengths)

    # Inputs (excluding lengths)
    input_x = torch.randn(BATCH_SIZE, D_MODEL, SEQ_LEN)
    input_style = torch.randn(BATCH_SIZE, STY_DIM)
    input_m = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
    
    args = (input_x, input_style, input_m)

    print("Exporting model with CONSTANT lengths...")
    try:
        exported_program = torch.export.export(model, args)
        print("Export SUCCESS!")
        
        # Try import to TVM to be sure
        print("Importing to TVM...")
        mod = from_exported_program(exported_program, keep_params_as_input=False)
        print("Import SUCCESS!")
        
    except Exception as e:
        print(f"Export/Import FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
