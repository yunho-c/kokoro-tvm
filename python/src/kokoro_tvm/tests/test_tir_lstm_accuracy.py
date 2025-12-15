"""Test TIR LSTM numerical accuracy against PyTorch reference."""

import numpy as np
import torch
import tvm

from kokoro_tvm.ops.tir_lstm import create_tir_lstm_primfunc


def test_tir_lstm_vs_pytorch():
    """Compare TIR LSTM output against PyTorch nn.LSTM."""
    print("=" * 60)
    print("TIR LSTM vs PyTorch Numerical Comparison")
    print("=" * 60)

    seq_len, batch_size, input_size, hidden_size = 4, 1, 16, 8

    # Create PyTorch LSTM reference
    torch.manual_seed(42)
    lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=False)
    lstm.eval()

    # Get weights from PyTorch
    Wi_pt = lstm.weight_ih_l0.detach().numpy()
    Wh_pt = lstm.weight_hh_l0.detach().numpy()
    bi_pt = lstm.bias_ih_l0.detach().numpy()
    bh_pt = lstm.bias_hh_l0.detach().numpy()

    print(f"Shapes: Wi={Wi_pt.shape}, Wh={Wh_pt.shape}")

    # Random input
    np.random.seed(123)
    x_np = np.random.randn(seq_len, batch_size, input_size).astype("float32")
    h0_np = np.zeros((batch_size, hidden_size), dtype="float32")
    c0_np = np.zeros((batch_size, hidden_size), dtype="float32")

    # Run PyTorch
    with torch.no_grad():
        out_pt, (hn_pt, cn_pt) = lstm(
            torch.from_numpy(x_np),
            (torch.from_numpy(h0_np[None]), torch.from_numpy(c0_np[None])),
        )

    out_pt_np = out_pt.numpy()
    hn_pt_np = hn_pt.numpy()
    cn_pt_np = cn_pt.numpy()

    print("\nPyTorch output:")
    print(f"  out[0,0,:4] = {out_pt_np[0, 0, :4]}")
    print(f"  hn[0,0,:4]  = {hn_pt_np[0, 0, :4]}")
    print(f"  Non-zero elements: {np.sum(out_pt_np != 0)}")

    # Build TIR LSTM
    print("\nBuilding TIR LSTM...")
    func = create_tir_lstm_primfunc(seq_len, batch_size, input_size, hidden_size)
    mod = tvm.IRModule({"tir_lstm_forward": func})
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.build(mod, target="llvm")

    # Helper to create TVM arrays
    dev = tvm.cpu()

    def make_tvm_arr(data):
        arr = tvm.runtime.empty(data.shape, str(data.dtype), dev)
        arr.copyfrom(data)
        return arr

    # Create input arrays
    x_tvm = make_tvm_arr(x_np)
    h_init_tvm = make_tvm_arr(h0_np)
    c_init_tvm = make_tvm_arr(c0_np)
    Wi_tvm = make_tvm_arr(Wi_pt)
    Wh_tvm = make_tvm_arr(Wh_pt)
    bi_tvm = make_tvm_arr(bi_pt)
    bh_tvm = make_tvm_arr(bh_pt)

    # Output arrays
    out_tvm = tvm.runtime.empty((seq_len, batch_size, hidden_size), "float32", dev)
    h_final_tvm = tvm.runtime.empty((batch_size, hidden_size), "float32", dev)
    c_final_tvm = tvm.runtime.empty((batch_size, hidden_size), "float32", dev)

    # Run TIR LSTM
    print("Running TIR LSTM...")
    lib["tir_lstm_forward"](
        x_tvm,
        h_init_tvm,
        c_init_tvm,
        Wi_tvm,
        Wh_tvm,
        bi_tvm,
        bh_tvm,
        out_tvm,
        h_final_tvm,
        c_final_tvm,
    )

    out_tir = out_tvm.numpy()
    h_final_tir = h_final_tvm.numpy()
    c_final_tir = c_final_tvm.numpy()

    print("\nTIR LSTM output:")
    print(f"  out[0,0,:4] = {out_tir[0, 0, :4]}")
    print(f"  h_final[0,:4] = {h_final_tir[0, :4]}")
    print(f"  Non-zero elements: {np.sum(out_tir != 0)}")

    # Compare
    print("\n" + "=" * 60)
    print("Comparison:")
    max_diff_out = np.abs(out_pt_np - out_tir).max()
    max_diff_h = np.abs(hn_pt_np[0] - h_final_tir).max()
    max_diff_c = np.abs(cn_pt_np[0] - c_final_tir).max()

    print(f"  Max output diff: {max_diff_out:.2e}")
    print(f"  Max h_final diff: {max_diff_h:.2e}")
    print(f"  Max c_final diff: {max_diff_c:.2e}")

    if max_diff_out < 1e-4:
        print("\n✅ TIR LSTM matches PyTorch!")
        return True
    else:
        print("\n❌ TIR LSTM does NOT match PyTorch")
        print("\nDebug - comparing first timestep outputs:")
        print(f"  PyTorch: {out_pt_np[0, 0, :]}")
        print(f"  TIR:     {out_tir[0, 0, :]}")
        return False


if __name__ == "__main__":
    test_tir_lstm_vs_pytorch()
