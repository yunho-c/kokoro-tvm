"""End-to-end numerical accuracy test for TVM-compiled duration encoder.

Compares TVM TIR LSTM output against PyTorch reference using the SAME model weights.
"""

import copy
import numpy as np
import torch
import tvm

# Enable TIR LSTM before importing encoder
from kokoro_tvm.ops.lstm_custom_op import patch_lstm_modules
from kokoro_tvm.patches.lstm import apply_lstm_patch
from kokoro_tvm import tvm_extensions

# Import Kokoro's ProsodyPredictor
from kokoro.model import ProsodyPredictor


class DurationWrapper(torch.nn.Module):
    """Wraps the duration path of ProsodyPredictor (text_encoder -> lstm -> duration_proj)."""

    def __init__(self, predictor, seq_len):
        super().__init__()
        self.text_encoder = predictor.text_encoder
        self.lstm = predictor.lstm
        self.duration_proj = predictor.duration_proj
        self.seq_len = seq_len

    def forward(self, d_en, style, text_lengths, m):
        d = self.text_encoder(d_en, style, text_lengths, m)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(d)
        x_pad = torch.zeros([x.shape[0], self.seq_len, x.shape[-1]], device=x.device)
        x_pad[:, : x.shape[1], :] = x
        x = x_pad
        duration = self.duration_proj(x)
        return duration.squeeze(-1), d


def test_duration_encoder_accuracy():
    """Compare TVM-compiled duration encoder against PyTorch reference."""
    print("=" * 60)
    print("Duration Encoder: TVM vs PyTorch Numerical Comparison")
    print("=" * 60)

    # Model parameters (Kokoro-82M defaults)
    style_dim = 128
    d_hid = 512
    nlayers = 3
    max_dur = 50
    seq_len = 32
    batch_size = 1

    # Create test inputs FIRST (before any model creation)
    print("\n1. Creating test inputs...")
    torch.manual_seed(42)
    np.random.seed(42)

    d_en = torch.randn(batch_size, d_hid, seq_len)
    style = torch.randn(batch_size, style_dim)
    text_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.long)
    m = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Create model with fixed seed
    print("\n2. Creating model (same weights for PyTorch and TVM)...")
    torch.manual_seed(123)  # Fixed seed for reproducible weights
    apply_lstm_patch()
    predictor = ProsodyPredictor(style_dim=style_dim, d_hid=d_hid, nlayers=nlayers, max_dur=max_dur, dropout=0.0)
    predictor.eval()

    # Save original model weights for PyTorch reference
    model_pt = DurationWrapper(predictor, seq_len)
    model_pt.eval()

    # Run PyTorch reference (before patching with custom ops)
    print("3. Running PyTorch reference...")
    with torch.no_grad():
        duration_pt, d_pt = model_pt(d_en, style, text_lengths, m)

    print(f"   duration shape: {duration_pt.shape}")
    print(f"   duration[0,0,:5]: {duration_pt[0, 0, :5].numpy()}")
    print(f"   d shape: {d_pt.shape}")

    # Now patch with custom ops for TVM export
    print("\n4. Patching model with custom LSTM ops for TVM...")
    patch_lstm_modules(predictor)

    # Create TVM module by exporting the patched model
    print("5. Exporting to TVM...")
    tvm_extensions.USE_TIR_LSTM = True

    model_tvm = DurationWrapper(predictor, seq_len)
    model_tvm.eval()

    example_inputs = (d_en, style, text_lengths, m)

    ep = torch.export.export(model_tvm, example_inputs, strict=False)
    tvm_mod = tvm.relax.frontend.torch.from_exported_program(
        ep, keep_params_as_input=False, unwrap_unit_return_tuple=True
    )

    # Apply dead code elimination
    tvm_mod = tvm.relax.transform.DeadCodeElimination()(tvm_mod)

    # Compile TVM module
    print("6. Compiling TVM module...")
    target = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relax.build(tvm_mod, target=target)

    # Create TVM runtime
    dev = tvm.cpu()
    vm = tvm.relax.VirtualMachine(lib, dev)

    # Convert inputs to TVM format
    def make_tvm_arr(tensor):
        arr = tvm.runtime.empty(tensor.shape, str(tensor.dtype).replace("torch.", ""), dev)
        arr.copyfrom(tensor.numpy())
        return arr

    d_en_tvm = make_tvm_arr(d_en)
    style_tvm = make_tvm_arr(style)
    text_lengths_tvm = make_tvm_arr(text_lengths)
    m_tvm = make_tvm_arr(m)

    # Run TVM
    print("7. Running TVM inference...")
    tvm_result = vm["main"](d_en_tvm, style_tvm, text_lengths_tvm, m_tvm)

    # Extract outputs
    duration_tvm = tvm_result[0].numpy()
    d_tvm = tvm_result[1].numpy()

    print(f"   duration shape: {duration_tvm.shape}")
    print(f"   duration[0,0,:5]: {duration_tvm[0, 0, :5]}")
    print(f"   d shape: {d_tvm.shape}")

    # Compare
    print("\n" + "=" * 60)
    print("Comparison:")

    duration_diff = np.abs(duration_pt.numpy() - duration_tvm).max()
    d_diff = np.abs(d_pt.numpy() - d_tvm).max()

    print(f"   Max duration diff: {duration_diff:.2e}")
    print(f"   Max d diff: {d_diff:.2e}")

    # Check if within tolerance
    tolerance = 1e-4
    if duration_diff < tolerance and d_diff < tolerance:
        print(f"\n✅ TVM encoder matches PyTorch! (tolerance: {tolerance})")
        return True
    else:
        print(f"\n❌ TVM encoder does NOT match PyTorch (tolerance: {tolerance})")
        return False


if __name__ == "__main__":
    test_duration_encoder_accuracy()
