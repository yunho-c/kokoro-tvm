"""Test MPS LSTM numerical accuracy against PyTorch reference.

This test exercises the tvm.contrib.mps.lstm extern path on the Metal runtime.
"""

import numpy as np
import torch
import tvm
from tvm import te
from tvm.contrib import mps


def test_mps_lstm_vs_pytorch():
    """Compare MPS LSTM output against PyTorch nn.LSTM."""
    if not tvm.get_global_func("tvm.contrib.mps.lstm", True):
        print("skip because tvm.contrib.mps.lstm is not available")
        return False

    if not tvm.get_global_func("device_api.metal", True):
        print("skip because Metal runtime is not available")
        return False

    print("=" * 60)
    print("MPS LSTM vs PyTorch Numerical Comparison")
    print("=" * 60)

    seq_len, batch_size, input_size, hidden_size = 4, 1, 16, 8

    torch.manual_seed(42)
    lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=False)
    lstm.eval()

    Wi_pt = lstm.weight_ih_l0.detach().numpy().astype("float32")
    Wh_pt = lstm.weight_hh_l0.detach().numpy().astype("float32")
    bi_pt = lstm.bias_ih_l0.detach().numpy().astype("float32")
    bh_pt = lstm.bias_hh_l0.detach().numpy().astype("float32")

    np.random.seed(123)
    x_np = np.random.randn(seq_len, batch_size, input_size).astype("float32")
    h0_np = np.zeros((1, batch_size, hidden_size), dtype="float32")
    c0_np = np.zeros((1, batch_size, hidden_size), dtype="float32")

    with torch.no_grad():
        out_pt, (hn_pt, cn_pt) = lstm(torch.from_numpy(x_np), (torch.from_numpy(h0_np), torch.from_numpy(c0_np)))

    out_pt_np = out_pt.numpy()
    hn_pt_np = hn_pt.numpy()
    cn_pt_np = cn_pt.numpy()

    print("\nBuilding MPS LSTM...")
    X = te.placeholder((seq_len, batch_size, input_size), name="X", dtype="float32")
    Wi = te.placeholder((4 * hidden_size, input_size), name="Wi", dtype="float32")
    Wh = te.placeholder((4 * hidden_size, hidden_size), name="Wh", dtype="float32")
    Bi = te.placeholder((4 * hidden_size,), name="Bi", dtype="float32")
    Bh = te.placeholder((4 * hidden_size,), name="Bh", dtype="float32")
    H0 = te.placeholder((1, batch_size, hidden_size), name="H0", dtype="float32")
    C0 = te.placeholder((1, batch_size, hidden_size), name="C0", dtype="float32")

    Y, HN, CN = mps.lstm(X, Wi, Wh, Bi, Bh, H0, C0, hidden_size, num_layers=1, batch_first=False, bidirectional=False)

    prim = te.create_prim_func([X, Wi, Wh, Bi, Bh, H0, C0, Y, HN, CN])
    ex = tvm.compile(prim, target="metal")

    dev = tvm.metal(0)
    x_tvm = tvm.runtime.tensor(x_np, dev)
    wi_tvm = tvm.runtime.tensor(Wi_pt, dev)
    wh_tvm = tvm.runtime.tensor(Wh_pt, dev)
    bi_tvm = tvm.runtime.tensor(bi_pt, dev)
    bh_tvm = tvm.runtime.tensor(bh_pt, dev)
    h0_tvm = tvm.runtime.tensor(h0_np, dev)
    c0_tvm = tvm.runtime.tensor(c0_np, dev)

    y_tvm = tvm.runtime.tensor(np.zeros((seq_len, batch_size, hidden_size), dtype="float32"), dev)
    hn_tvm = tvm.runtime.tensor(np.zeros((1, batch_size, hidden_size), dtype="float32"), dev)
    cn_tvm = tvm.runtime.tensor(np.zeros((1, batch_size, hidden_size), dtype="float32"), dev)

    print("Running MPS LSTM...")
    ex(x_tvm, wi_tvm, wh_tvm, bi_tvm, bh_tvm, h0_tvm, c0_tvm, y_tvm, hn_tvm, cn_tvm)

    y_np = y_tvm.numpy()
    hn_np = hn_tvm.numpy()
    cn_np = cn_tvm.numpy()

    max_diff_out = np.abs(out_pt_np - y_np).max()
    max_diff_h = np.abs(hn_pt_np - hn_np).max()
    max_diff_c = np.abs(cn_pt_np - cn_np).max()

    print("\nComparison:")
    print(f"  Max output diff: {max_diff_out:.2e}")
    print(f"  Max h_n diff:    {max_diff_h:.2e}")
    print(f"  Max c_n diff:    {max_diff_c:.2e}")

    if max_diff_out < 5e-4 and max_diff_h < 5e-4 and max_diff_c < 5e-4:
        print("\n✅ MPS LSTM matches PyTorch within tolerance")
        return True

    print("\n❌ MPS LSTM does NOT match PyTorch within tolerance")
    print("Debug:")
    print("  PyTorch out[0,0,:4] =", out_pt_np[0, 0, :4])
    print("  MPS    out[0,0,:4] =", y_np[0, 0, :4])
    return False


if __name__ == "__main__":
    test_mps_lstm_vs_pytorch()

