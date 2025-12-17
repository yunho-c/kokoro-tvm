"""Test MPS LSTM numerical accuracy against PyTorch reference.

This test exercises the tvm.contrib.mps.lstm extern path on the Metal runtime.
"""

from __future__ import annotations

import numpy as np
import torch
import tvm
from tvm import te
from tvm.contrib import mps


def _has_mps_lstm() -> bool:
    return bool(tvm.get_global_func("tvm.contrib.mps.lstm", True)) and bool(
        tvm.get_global_func("device_api.metal", True)
    )


def _has_mps_lstm_packed() -> bool:
    return bool(tvm.get_global_func("tvm.contrib.mps.lstm_packed", True)) and bool(
        tvm.get_global_func("device_api.metal", True)
    )


def _reverse_time(x: np.ndarray, batch_first: bool) -> np.ndarray:
    axis = 1 if batch_first else 0
    return np.flip(x, axis=axis).copy()


def _pytorch_lstm_reference(
    x_np: np.ndarray,
    lstm: torch.nn.LSTM,
    batch_first: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_len = x_np.shape[1] if batch_first else x_np.shape[0]
    batch_size = x_np.shape[0] if batch_first else x_np.shape[1]
    hidden_size = lstm.hidden_size
    num_directions = 2 if lstm.bidirectional else 1

    h0_np = np.zeros((lstm.num_layers * num_directions, batch_size, hidden_size), dtype="float32")
    c0_np = np.zeros((lstm.num_layers * num_directions, batch_size, hidden_size), dtype="float32")
    with torch.no_grad():
        out_pt, (hn_pt, cn_pt) = lstm(
            torch.from_numpy(x_np),
            (torch.from_numpy(h0_np), torch.from_numpy(c0_np)),
        )
    return out_pt.numpy(), hn_pt.numpy(), cn_pt.numpy()


def _mps_lstm_run_unidirectional(
    x_np: np.ndarray,
    wi_np: np.ndarray,
    wh_np: np.ndarray,
    bi_np: np.ndarray,
    bh_np: np.ndarray,
    hidden_size: int,
    batch_first: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if batch_first:
        batch_size, seq_len, input_size = x_np.shape
        y_shape = (batch_size, seq_len, hidden_size)
    else:
        seq_len, batch_size, input_size = x_np.shape
        y_shape = (seq_len, batch_size, hidden_size)

    X = te.placeholder(x_np.shape, name="X", dtype="float32")
    Wi = te.placeholder((4 * hidden_size, input_size), name="Wi", dtype="float32")
    Wh = te.placeholder((4 * hidden_size, hidden_size), name="Wh", dtype="float32")
    Bi = te.placeholder((4 * hidden_size,), name="Bi", dtype="float32")
    Bh = te.placeholder((4 * hidden_size,), name="Bh", dtype="float32")
    H0 = te.placeholder((1, batch_size, hidden_size), name="H0", dtype="float32")
    C0 = te.placeholder((1, batch_size, hidden_size), name="C0", dtype="float32")

    Y, HN, CN = mps.lstm(
        X, Wi, Wh, Bi, Bh, H0, C0, hidden_size, num_layers=1, batch_first=batch_first, bidirectional=False
    )

    prim = te.create_prim_func([X, Wi, Wh, Bi, Bh, H0, C0, Y, HN, CN])
    ex = tvm.compile(prim, target="metal")
    dev = tvm.metal(0)

    h0_np = np.zeros((1, batch_size, hidden_size), dtype="float32")
    c0_np = np.zeros((1, batch_size, hidden_size), dtype="float32")

    y_tvm = tvm.runtime.tensor(np.zeros(y_shape, dtype="float32"), dev)
    hn_tvm = tvm.runtime.tensor(np.zeros((1, batch_size, hidden_size), dtype="float32"), dev)
    cn_tvm = tvm.runtime.tensor(np.zeros((1, batch_size, hidden_size), dtype="float32"), dev)

    ex(
        tvm.runtime.tensor(x_np, dev),
        tvm.runtime.tensor(wi_np, dev),
        tvm.runtime.tensor(wh_np, dev),
        tvm.runtime.tensor(bi_np, dev),
        tvm.runtime.tensor(bh_np, dev),
        tvm.runtime.tensor(h0_np, dev),
        tvm.runtime.tensor(c0_np, dev),
        y_tvm,
        hn_tvm,
        cn_tvm,
    )

    return y_tvm.numpy(), hn_tvm.numpy(), cn_tvm.numpy()


def _run_case(
    *,
    seq_len: int,
    batch_size: int,
    input_size: int,
    hidden_size: int,
    batch_first: bool,
    bidirectional: bool,
    seed: int,
    tol: float,
) -> bool:
    torch.manual_seed(seed)
    np.random.seed(seed + 1000)

    lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=batch_first, bidirectional=bidirectional)
    lstm.eval()

    x_shape = (batch_size, seq_len, input_size) if batch_first else (seq_len, batch_size, input_size)
    x_np = np.random.randn(*x_shape).astype("float32")

    out_pt, hn_pt, cn_pt = _pytorch_lstm_reference(x_np, lstm, batch_first=batch_first)

    if bidirectional:
        wi_f = lstm.weight_ih_l0.detach().numpy().astype("float32")
        wh_f = lstm.weight_hh_l0.detach().numpy().astype("float32")
        bi_f = lstm.bias_ih_l0.detach().numpy().astype("float32")
        bh_f = lstm.bias_hh_l0.detach().numpy().astype("float32")

        wi_b = lstm.weight_ih_l0_reverse.detach().numpy().astype("float32")
        wh_b = lstm.weight_hh_l0_reverse.detach().numpy().astype("float32")
        bi_b = lstm.bias_ih_l0_reverse.detach().numpy().astype("float32")
        bh_b = lstm.bias_hh_l0_reverse.detach().numpy().astype("float32")

        y_f, hn_f, cn_f = _mps_lstm_run_unidirectional(x_np, wi_f, wh_f, bi_f, bh_f, hidden_size, batch_first)
        x_rev = _reverse_time(x_np, batch_first=batch_first)
        y_b_rev, hn_b, cn_b = _mps_lstm_run_unidirectional(x_rev, wi_b, wh_b, bi_b, bh_b, hidden_size, batch_first)

        y_b = _reverse_time(y_b_rev, batch_first=batch_first)
        out_mps = np.concatenate([y_f, y_b], axis=2)
        hn_mps = np.concatenate([hn_f, hn_b], axis=0)
        cn_mps = np.concatenate([cn_f, cn_b], axis=0)
    else:
        wi = lstm.weight_ih_l0.detach().numpy().astype("float32")
        wh = lstm.weight_hh_l0.detach().numpy().astype("float32")
        bi = lstm.bias_ih_l0.detach().numpy().astype("float32")
        bh = lstm.bias_hh_l0.detach().numpy().astype("float32")
        out_mps, hn_mps, cn_mps = _mps_lstm_run_unidirectional(x_np, wi, wh, bi, bh, hidden_size, batch_first)

    max_diff_out = float(np.abs(out_pt - out_mps).max())
    max_diff_h = float(np.abs(hn_pt - hn_mps).max())
    max_diff_c = float(np.abs(cn_pt - cn_mps).max())

    print(
        f"case(batch_first={batch_first}, bidirectional={bidirectional}, "
        f"seq={seq_len}, batch={batch_size}, in={input_size}, hid={hidden_size}): "
        f"max_out={max_diff_out:.2e} max_h={max_diff_h:.2e} max_c={max_diff_c:.2e}"
    )

    ok = max_diff_out < tol and max_diff_h < tol and max_diff_c < tol
    if not ok:
        if batch_first:
            pt_snip = out_pt[0, 0, :4]
            mps_snip = out_mps[0, 0, :4]
        else:
            pt_snip = out_pt[0, 0, :4]
            mps_snip = out_mps[0, 0, :4]
        print("Debug:")
        print("  PyTorch out[:4] =", pt_snip)
        print("  MPS     out[:4] =", mps_snip)
    return ok


def test_mps_lstm_accuracy_matrix():
    """Scaffold accuracy checks across common LSTM settings.

    Notes:
    - The underlying `tvm.contrib.mps.lstm` op is unidirectional only, but we
      can still validate bidirectional LSTMs by running two unidirectional passes
      (forward + reversed) and concatenating outputs like PyTorch.
    """
    if not _has_mps_lstm():
        print("skip because tvm.contrib.mps.lstm / Metal runtime is not available")
        return False

    print("=" * 60)
    print("MPS LSTM accuracy matrix (vs PyTorch)")
    print("=" * 60)

    tol = 5e-4
    seed = 42

    cases: list[dict[str, object]] = [
        {
            "seq_len": 4,
            "batch_size": 1,
            "input_size": 16,
            "hidden_size": 8,
            "batch_first": False,
            "bidirectional": False,
        },
        {
            "seq_len": 8,
            "batch_size": 2,
            "input_size": 32,
            "hidden_size": 16,
            "batch_first": False,
            "bidirectional": False,
        },
        {
            "seq_len": 8,
            "batch_size": 2,
            "input_size": 32,
            "hidden_size": 16,
            "batch_first": True,
            "bidirectional": False,
        },
        {
            "seq_len": 6,
            "batch_size": 1,
            "input_size": 16,
            "hidden_size": 8,
            "batch_first": False,
            "bidirectional": True,
        },
        {"seq_len": 6, "batch_size": 1, "input_size": 16, "hidden_size": 8, "batch_first": True, "bidirectional": True},
    ]

    all_ok = True
    for cfg in cases:
        all_ok &= _run_case(**cfg, seed=seed, tol=tol)

    if all_ok:
        print("\n✅ All MPS LSTM cases match PyTorch within tolerance")
    else:
        print("\n❌ Some MPS LSTM cases do NOT match PyTorch within tolerance")
    return all_ok


def test_mps_lstm_packed_batch1():
    if not _has_mps_lstm_packed():
        print("skip because tvm.contrib.mps.lstm_packed / Metal runtime is not available")
        return False

    torch.manual_seed(0)
    np.random.seed(1)

    seq_len = 16
    length0 = 7
    batch_size = 1
    input_size = 32
    hidden_size = 16
    batch_first = False

    lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=batch_first, bidirectional=False)
    lstm.eval()

    x_np = np.random.randn(seq_len, batch_size, input_size).astype("float32")

    with torch.no_grad():
        lengths_pt = torch.tensor([length0], dtype=torch.long)
        x_pt = torch.from_numpy(x_np)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x_pt, lengths_pt, batch_first=False, enforce_sorted=False)
        packed_out, (hn_pt, cn_pt) = lstm(packed)
        out_pt, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=False, total_length=seq_len)

    wi_np = lstm.weight_ih_l0.detach().numpy().astype("float32")
    wh_np = lstm.weight_hh_l0.detach().numpy().astype("float32")
    bi_np = lstm.bias_ih_l0.detach().numpy().astype("float32")
    bh_np = lstm.bias_hh_l0.detach().numpy().astype("float32")

    X = te.placeholder(x_np.shape, name="X", dtype="float32")
    L = te.placeholder((batch_size,), name="L", dtype="int64")
    Wi = te.placeholder((4 * hidden_size, input_size), name="Wi", dtype="float32")
    Wh = te.placeholder((4 * hidden_size, hidden_size), name="Wh", dtype="float32")
    Bi = te.placeholder((4 * hidden_size,), name="Bi", dtype="float32")
    Bh = te.placeholder((4 * hidden_size,), name="Bh", dtype="float32")
    H0 = te.placeholder((1, batch_size, hidden_size), name="H0", dtype="float32")
    C0 = te.placeholder((1, batch_size, hidden_size), name="C0", dtype="float32")

    Y, HN, CN = mps.lstm_packed(
        X, L, Wi, Wh, Bi, Bh, H0, C0, hidden_size, num_layers=1, batch_first=batch_first, bidirectional=False
    )

    prim = te.create_prim_func([X, L, Wi, Wh, Bi, Bh, H0, C0, Y, HN, CN])
    ex = tvm.compile(prim, target="metal")
    dev = tvm.metal(0)

    h0_np = np.zeros((1, batch_size, hidden_size), dtype="float32")
    c0_np = np.zeros((1, batch_size, hidden_size), dtype="float32")

    y_tvm = tvm.runtime.tensor(np.zeros((seq_len, batch_size, hidden_size), dtype="float32"), dev)
    hn_tvm = tvm.runtime.tensor(np.zeros((1, batch_size, hidden_size), dtype="float32"), dev)
    cn_tvm = tvm.runtime.tensor(np.zeros((1, batch_size, hidden_size), dtype="float32"), dev)

    ex(
        tvm.runtime.tensor(x_np, dev),
        tvm.runtime.tensor(np.array([length0], dtype=np.int64), dev),
        tvm.runtime.tensor(wi_np, dev),
        tvm.runtime.tensor(wh_np, dev),
        tvm.runtime.tensor(bi_np, dev),
        tvm.runtime.tensor(bh_np, dev),
        tvm.runtime.tensor(h0_np, dev),
        tvm.runtime.tensor(c0_np, dev),
        y_tvm,
        hn_tvm,
        cn_tvm,
    )

    max_diff_out = float(np.abs(out_pt.numpy() - y_tvm.numpy()).max())
    max_diff_h = float(np.abs(hn_pt.numpy() - hn_tvm.numpy()).max())
    max_diff_c = float(np.abs(cn_pt.numpy() - cn_tvm.numpy()).max())
    print(
        f"packed(batch=1, len={length0}/{seq_len}): max_out={max_diff_out:.2e} max_h={max_diff_h:.2e} max_c={max_diff_c:.2e}"
    )

    tol = 5e-4
    return max_diff_out < tol and max_diff_h < tol and max_diff_c < tol


if __name__ == "__main__":
    ok_matrix = test_mps_lstm_accuracy_matrix()
    ok_packed = test_mps_lstm_packed_batch1()
    raise SystemExit(0 if (ok_matrix and ok_packed) else 1)
