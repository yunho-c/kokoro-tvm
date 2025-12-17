# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Custom LSTM op for torch.export preservation.

This module defines custom LSTM ops using @torch.library.custom_op that remain
as single opaque nodes during torch.export, preventing decomposition into
primitive ops.

Supports both unidirectional and bidirectional single-layer LSTMs.
The custom ops are then converted to our TIR LSTM implementation in TVM.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Define custom op namespaces
LSTM_OP_NAME = "kokoro::lstm_forward"
LSTM_BIDIR_OP_NAME = "kokoro::lstm_forward_bidirectional"
LSTM_PACKED_OP_NAME = "kokoro::lstm_forward_packed"
LSTM_PACKED_BIDIR_OP_NAME = "kokoro::lstm_forward_packed_bidirectional"


@torch.library.custom_op(LSTM_OP_NAME, mutates_args=())
def lstm_forward(
    input: Tensor,
    h0: Tensor,
    c0: Tensor,
    weight_ih_l0: Tensor,
    weight_hh_l0: Tensor,
    bias_ih_l0: Tensor | None,
    bias_hh_l0: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Custom unidirectional LSTM forward op that stays opaque during torch.export.

    Args:
        input: Input tensor of shape (seq_len, batch, input_size)
        h0: Initial hidden state of shape (1, batch, hidden_size)
        c0: Initial cell state of shape (1, batch, hidden_size)
        weight_ih_l0: Input-hidden weights of shape (4*hidden_size, input_size)
        weight_hh_l0: Hidden-hidden weights of shape (4*hidden_size, hidden_size)
        bias_ih_l0: Input-hidden bias of shape (4*hidden_size,) or None
        bias_hh_l0: Hidden-hidden bias of shape (4*hidden_size,) or None

    Returns:
        Tuple of (output, h_n, c_n):
            - output: shape (seq_len, batch, hidden_size)
            - h_n: shape (1, batch, hidden_size)
            - c_n: shape (1, batch, hidden_size)
    """
    seq_len, batch, input_size = input.shape
    hidden_size = weight_hh_l0.shape[1]

    if bias_ih_l0 is not None and bias_hh_l0 is not None:
        weights = [weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0]
        has_biases = True
    else:
        weights = [weight_ih_l0, weight_hh_l0]
        has_biases = False

    output, h_n, c_n = torch._VF.lstm(
        input,
        (h0, c0),
        weights,
        has_biases,
        1,  # num_layers
        0.0,  # dropout
        False,  # training
        False,  # bidirectional
        False,  # batch_first
    )

    return output, h_n, c_n


@lstm_forward.register_fake
def _(
    input: Tensor,
    h0: Tensor,
    c0: Tensor,
    weight_ih_l0: Tensor,
    weight_hh_l0: Tensor,
    bias_ih_l0: Tensor | None,
    bias_hh_l0: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fake kernel for symbolic tracing (unidirectional)."""
    seq_len, batch, input_size = input.shape
    hidden_size = weight_hh_l0.shape[1]

    output = input.new_empty(seq_len, batch, hidden_size)
    h_n = h0.new_empty(h0.shape)
    c_n = c0.new_empty(c0.shape)

    return output, h_n, c_n


# ============================================================================
# Bidirectional LSTM Custom Op
# ============================================================================


@torch.library.custom_op(LSTM_BIDIR_OP_NAME, mutates_args=())
def lstm_forward_bidirectional(
    input: Tensor,
    h0: Tensor,
    c0: Tensor,
    weight_ih_l0: Tensor,
    weight_hh_l0: Tensor,
    bias_ih_l0: Tensor | None,
    bias_hh_l0: Tensor | None,
    weight_ih_l0_reverse: Tensor,
    weight_hh_l0_reverse: Tensor,
    bias_ih_l0_reverse: Tensor | None,
    bias_hh_l0_reverse: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Custom bidirectional LSTM forward op that stays opaque during torch.export.

    Args:
        input: Input tensor of shape (seq_len, batch, input_size)
        h0: Initial hidden state of shape (2, batch, hidden_size)
        c0: Initial cell state of shape (2, batch, hidden_size)
        weight_ih_l0: Forward input-hidden weights (4*hidden_size, input_size)
        weight_hh_l0: Forward hidden-hidden weights (4*hidden_size, hidden_size)
        bias_ih_l0: Forward input-hidden bias (4*hidden_size,) or None
        bias_hh_l0: Forward hidden-hidden bias (4*hidden_size,) or None
        weight_ih_l0_reverse: Reverse input-hidden weights
        weight_hh_l0_reverse: Reverse hidden-hidden weights
        bias_ih_l0_reverse: Reverse input-hidden bias or None
        bias_hh_l0_reverse: Reverse hidden-hidden bias or None

    Returns:
        Tuple of (output, h_n, c_n):
            - output: shape (seq_len, batch, 2*hidden_size)
            - h_n: shape (2, batch, hidden_size)
            - c_n: shape (2, batch, hidden_size)
    """
    seq_len, batch, input_size = input.shape
    hidden_size = weight_hh_l0.shape[1]

    # Build weights list for bidirectional LSTM
    if bias_ih_l0 is not None and bias_hh_l0 is not None:
        weights = [
            weight_ih_l0,
            weight_hh_l0,
            bias_ih_l0,
            bias_hh_l0,
            weight_ih_l0_reverse,
            weight_hh_l0_reverse,
            bias_ih_l0_reverse,
            bias_hh_l0_reverse,
        ]
        has_biases = True
    else:
        weights = [
            weight_ih_l0,
            weight_hh_l0,
            weight_ih_l0_reverse,
            weight_hh_l0_reverse,
        ]
        has_biases = False

    output, h_n, c_n = torch._VF.lstm(
        input,
        (h0, c0),
        weights,
        has_biases,
        1,  # num_layers
        0.0,  # dropout
        False,  # training
        True,  # bidirectional
        False,  # batch_first
    )

    return output, h_n, c_n


@lstm_forward_bidirectional.register_fake
def _(
    input: Tensor,
    h0: Tensor,
    c0: Tensor,
    weight_ih_l0: Tensor,
    weight_hh_l0: Tensor,
    bias_ih_l0: Tensor | None,
    bias_hh_l0: Tensor | None,
    weight_ih_l0_reverse: Tensor,
    weight_hh_l0_reverse: Tensor,
    bias_ih_l0_reverse: Tensor | None,
    bias_hh_l0_reverse: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fake kernel for symbolic tracing (bidirectional)."""
    seq_len, batch, input_size = input.shape
    hidden_size = weight_hh_l0.shape[1]

    # Bidirectional output has 2*hidden_size
    output = input.new_empty(seq_len, batch, 2 * hidden_size)
    h_n = h0.new_empty(h0.shape)
    c_n = c0.new_empty(c0.shape)

    return output, h_n, c_n


# ============================================================================
# Wrappers
# ============================================================================


class LSTMWrapper(torch.nn.Module):
    """Drop-in replacement for unidirectional nn.LSTM that uses our custom op."""

    def __init__(self, lstm: torch.nn.LSTM):
        super().__init__()

        if lstm.num_layers != 1:
            raise ValueError("LSTMWrapper only supports num_layers=1")
        if lstm.bidirectional:
            raise ValueError("LSTMWrapper only supports unidirectional LSTM")

        self.input_size = lstm.input_size
        self.hidden_size = lstm.hidden_size
        self.batch_first = lstm.batch_first

        self.weight_ih_l0 = lstm.weight_ih_l0
        self.weight_hh_l0 = lstm.weight_hh_l0
        self.bias_ih_l0 = lstm.bias_ih_l0 if lstm.bias else None
        self.bias_hh_l0 = lstm.bias_hh_l0 if lstm.bias else None

    def flatten_parameters(self) -> None:
        """No-op for compatibility with nn.LSTM API."""
        pass

    def forward(
        self,
        input: Tensor,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass using our custom op."""

        if self.batch_first:
            input = input.transpose(0, 1)

        batch = input.shape[1]

        if hx is None:
            h0 = input.new_zeros(1, batch, self.hidden_size)
            c0 = input.new_zeros(1, batch, self.hidden_size)
        else:
            h0, c0 = hx

        output, h_n, c_n = lstm_forward(
            input,
            h0,
            c0,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        )

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_n, c_n)


class BidirectionalLSTMWrapper(torch.nn.Module):
    """Drop-in replacement for bidirectional nn.LSTM that uses our custom op."""

    def __init__(self, lstm: torch.nn.LSTM):
        super().__init__()

        if lstm.num_layers != 1:
            raise ValueError("BidirectionalLSTMWrapper only supports num_layers=1")
        if not lstm.bidirectional:
            raise ValueError("BidirectionalLSTMWrapper only supports bidirectional LSTM")

        self.input_size = lstm.input_size
        self.hidden_size = lstm.hidden_size
        self.batch_first = lstm.batch_first

        # Forward direction weights
        self.weight_ih_l0 = lstm.weight_ih_l0
        self.weight_hh_l0 = lstm.weight_hh_l0
        self.bias_ih_l0 = lstm.bias_ih_l0 if lstm.bias else None
        self.bias_hh_l0 = lstm.bias_hh_l0 if lstm.bias else None

        # Reverse direction weights (PyTorch uses _reverse suffix)
        self.weight_ih_l0_reverse = lstm.weight_ih_l0_reverse
        self.weight_hh_l0_reverse = lstm.weight_hh_l0_reverse
        self.bias_ih_l0_reverse = lstm.bias_ih_l0_reverse if lstm.bias else None
        self.bias_hh_l0_reverse = lstm.bias_hh_l0_reverse if lstm.bias else None

    def flatten_parameters(self) -> None:
        """No-op for compatibility with nn.LSTM API."""
        pass

    def forward(
        self,
        input: Tensor,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass using our custom bidirectional op."""

        if self.batch_first:
            input = input.transpose(0, 1)

        batch = input.shape[1]

        if hx is None:
            h0 = input.new_zeros(2, batch, self.hidden_size)
            c0 = input.new_zeros(2, batch, self.hidden_size)
        else:
            h0, c0 = hx

        output, h_n, c_n = lstm_forward_bidirectional(
            input,
            h0,
            c0,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
            self.weight_ih_l0_reverse,
            self.weight_hh_l0_reverse,
            self.bias_ih_l0_reverse,
            self.bias_hh_l0_reverse,
        )

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_n, c_n)


def patch_lstm_modules(model: torch.nn.Module) -> None:
    """Recursively replace all nn.LSTM modules with our custom wrappers.

    Args:
        model: The model to patch
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.LSTM):
            try:
                if module.bidirectional:
                    wrapper = BidirectionalLSTMWrapper(module)
                    setattr(model, name, wrapper)
                    print(f"Patched {name}: nn.LSTM -> BidirectionalLSTMWrapper")
                else:
                    wrapper = LSTMWrapper(module)
                    setattr(model, name, wrapper)
                    print(f"Patched {name}: nn.LSTM -> LSTMWrapper")
            except ValueError as e:
                print(f"Skipping {name}: {e}")
        else:
            patch_lstm_modules(module)


def _to_cpu_lengths(lengths: Tensor) -> Tensor:
    if lengths.device.type != "cpu":
        return lengths.to("cpu")
    return lengths


@torch.library.custom_op(LSTM_PACKED_OP_NAME, mutates_args=())
def lstm_forward_packed(
    input: Tensor,
    lengths: Tensor,
    h0: Tensor,
    c0: Tensor,
    weight_ih_l0: Tensor,
    weight_hh_l0: Tensor,
    bias_ih_l0: Tensor | None,
    bias_hh_l0: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Packed-semantics unidirectional LSTM.

    This op is export-safe: it carries `lengths` explicitly, so TVM can later lower it
    to a packed/variable-length backend (e.g. MPS ragged-row encoding) without relying
    on PyTorch PackedSequence objects.

    Input layout matches our other custom ops: (seq_len, batch, input_size).
    The output is padded back to `seq_len` (total_length=seq_len).
    """
    seq_len, batch, _ = input.shape
    hidden_size = weight_hh_l0.shape[1]

    has_bias = bias_ih_l0 is not None and bias_hh_l0 is not None
    lstm = torch.nn.LSTM(
        input_size=input.shape[2],
        hidden_size=hidden_size,
        num_layers=1,
        bias=has_bias,
        batch_first=False,
        bidirectional=False,
    )
    with torch.no_grad():
        lstm.weight_ih_l0.copy_(weight_ih_l0)
        lstm.weight_hh_l0.copy_(weight_hh_l0)
        if has_bias:
            lstm.bias_ih_l0.copy_(bias_ih_l0)
            lstm.bias_hh_l0.copy_(bias_hh_l0)

    packed = torch.nn.utils.rnn.pack_padded_sequence(
        input, _to_cpu_lengths(lengths), batch_first=False, enforce_sorted=False
    )
    packed_out, (h_n, c_n) = lstm(packed, (h0, c0))
    out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=False, total_length=seq_len)
    return out, h_n, c_n


@lstm_forward_packed.register_fake
def _(
    input: Tensor,
    lengths: Tensor,
    h0: Tensor,
    c0: Tensor,
    weight_ih_l0: Tensor,
    weight_hh_l0: Tensor,
    bias_ih_l0: Tensor | None,
    bias_hh_l0: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    seq_len, batch, _ = input.shape
    hidden_size = weight_hh_l0.shape[1]
    return (
        input.new_empty(seq_len, batch, hidden_size),
        h0.new_empty(h0.shape),
        c0.new_empty(c0.shape),
    )


@torch.library.custom_op(LSTM_PACKED_BIDIR_OP_NAME, mutates_args=())
def lstm_forward_packed_bidirectional(
    input: Tensor,
    lengths: Tensor,
    h0: Tensor,
    c0: Tensor,
    weight_ih_l0: Tensor,
    weight_hh_l0: Tensor,
    bias_ih_l0: Tensor | None,
    bias_hh_l0: Tensor | None,
    weight_ih_l0_reverse: Tensor,
    weight_hh_l0_reverse: Tensor,
    bias_ih_l0_reverse: Tensor | None,
    bias_hh_l0_reverse: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Packed-semantics bidirectional LSTM (single layer)."""
    seq_len, batch, _ = input.shape
    hidden_size = weight_hh_l0.shape[1]

    has_bias = bias_ih_l0 is not None and bias_hh_l0 is not None
    lstm = torch.nn.LSTM(
        input_size=input.shape[2],
        hidden_size=hidden_size,
        num_layers=1,
        bias=has_bias,
        batch_first=False,
        bidirectional=True,
    )
    with torch.no_grad():
        lstm.weight_ih_l0.copy_(weight_ih_l0)
        lstm.weight_hh_l0.copy_(weight_hh_l0)
        lstm.weight_ih_l0_reverse.copy_(weight_ih_l0_reverse)
        lstm.weight_hh_l0_reverse.copy_(weight_hh_l0_reverse)
        if has_bias:
            lstm.bias_ih_l0.copy_(bias_ih_l0)
            lstm.bias_hh_l0.copy_(bias_hh_l0)
            lstm.bias_ih_l0_reverse.copy_(bias_ih_l0_reverse)
            lstm.bias_hh_l0_reverse.copy_(bias_hh_l0_reverse)

    packed = torch.nn.utils.rnn.pack_padded_sequence(
        input, _to_cpu_lengths(lengths), batch_first=False, enforce_sorted=False
    )
    packed_out, (h_n, c_n) = lstm(packed, (h0, c0))
    out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=False, total_length=seq_len)
    return out, h_n, c_n


@lstm_forward_packed_bidirectional.register_fake
def _(
    input: Tensor,
    lengths: Tensor,
    h0: Tensor,
    c0: Tensor,
    weight_ih_l0: Tensor,
    weight_hh_l0: Tensor,
    bias_ih_l0: Tensor | None,
    bias_hh_l0: Tensor | None,
    weight_ih_l0_reverse: Tensor,
    weight_hh_l0_reverse: Tensor,
    bias_ih_l0_reverse: Tensor | None,
    bias_hh_l0_reverse: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    seq_len, batch, _ = input.shape
    hidden_size = weight_hh_l0.shape[1]
    return (
        input.new_empty(seq_len, batch, 2 * hidden_size),
        h0.new_empty(h0.shape),
        c0.new_empty(c0.shape),
    )


def test_lstm_custom_op():
    """Test the custom LSTM ops with numerical validation."""
    import numpy as np

    print("=" * 60)
    print("LSTM Custom Op Tests with Numerical Validation")
    print("=" * 60)

    seq_len, batch, input_size, hidden_size = 8, 2, 64, 32
    rtol, atol = 1e-5, 1e-5

    # ==========================================================================
    # Test 1: Unidirectional LSTM numerical validation
    # ==========================================================================
    print("\n[Test 1] Unidirectional LSTM numerical validation")

    torch.manual_seed(42)
    lstm_uni = torch.nn.LSTM(input_size, hidden_size, batch_first=False, bidirectional=False)
    lstm_uni.eval()

    x = torch.randn(seq_len, batch, input_size)
    h0 = torch.randn(1, batch, hidden_size)
    c0 = torch.randn(1, batch, hidden_size)

    # Reference: PyTorch built-in
    with torch.no_grad():
        ref_out, (ref_hn, ref_cn) = lstm_uni(x, (h0, c0))

    # Custom op
    with torch.no_grad():
        custom_out, custom_hn, custom_cn = lstm_forward(
            x, h0, c0, lstm_uni.weight_ih_l0, lstm_uni.weight_hh_l0, lstm_uni.bias_ih_l0, lstm_uni.bias_hh_l0
        )

    # Compare
    out_match = torch.allclose(ref_out, custom_out, rtol=rtol, atol=atol)
    hn_match = torch.allclose(ref_hn, custom_hn, rtol=rtol, atol=atol)
    cn_match = torch.allclose(ref_cn, custom_cn, rtol=rtol, atol=atol)

    if out_match and hn_match and cn_match:
        print("  ✅ Unidirectional LSTM: outputs match PyTorch reference!")
    else:
        print("  ❌ Unidirectional LSTM: outputs do NOT match!")
        print(f"     output match: {out_match}, max diff: {(ref_out - custom_out).abs().max():.2e}")
        print(f"     h_n match: {hn_match}, max diff: {(ref_hn - custom_hn).abs().max():.2e}")
        print(f"     c_n match: {cn_match}, max diff: {(ref_cn - custom_cn).abs().max():.2e}")

    # ==========================================================================
    # Test 2: Bidirectional LSTM numerical validation
    # ==========================================================================
    print("\n[Test 2] Bidirectional LSTM numerical validation")

    torch.manual_seed(42)
    lstm_bi = torch.nn.LSTM(input_size, hidden_size, batch_first=False, bidirectional=True)
    lstm_bi.eval()

    x = torch.randn(seq_len, batch, input_size)
    h0 = torch.randn(2, batch, hidden_size)
    c0 = torch.randn(2, batch, hidden_size)

    # Reference: PyTorch built-in
    with torch.no_grad():
        ref_out, (ref_hn, ref_cn) = lstm_bi(x, (h0, c0))

    # Custom op
    with torch.no_grad():
        custom_out, custom_hn, custom_cn = lstm_forward_bidirectional(
            x,
            h0,
            c0,
            lstm_bi.weight_ih_l0,
            lstm_bi.weight_hh_l0,
            lstm_bi.bias_ih_l0,
            lstm_bi.bias_hh_l0,
            lstm_bi.weight_ih_l0_reverse,
            lstm_bi.weight_hh_l0_reverse,
            lstm_bi.bias_ih_l0_reverse,
            lstm_bi.bias_hh_l0_reverse,
        )

    # Compare
    out_match = torch.allclose(ref_out, custom_out, rtol=rtol, atol=atol)
    hn_match = torch.allclose(ref_hn, custom_hn, rtol=rtol, atol=atol)
    cn_match = torch.allclose(ref_cn, custom_cn, rtol=rtol, atol=atol)

    if out_match and hn_match and cn_match:
        print("  ✅ Bidirectional LSTM: outputs match PyTorch reference!")
    else:
        print("  ❌ Bidirectional LSTM: outputs do NOT match!")
        print(f"     output match: {out_match}, max diff: {(ref_out - custom_out).abs().max():.2e}")
        print(f"     h_n match: {hn_match}, max diff: {(ref_hn - custom_hn).abs().max():.2e}")
        print(f"     c_n match: {cn_match}, max diff: {(ref_cn - custom_cn).abs().max():.2e}")

    # ==========================================================================
    # Test 3: LSTMWrapper numerical validation
    # ==========================================================================
    print("\n[Test 3] LSTMWrapper numerical validation")

    torch.manual_seed(42)
    lstm_raw = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
    lstm_raw.eval()
    wrapper = LSTMWrapper(lstm_raw)

    x = torch.randn(batch, seq_len, input_size)  # batch_first

    with torch.no_grad():
        ref_out, (ref_hn, ref_cn) = lstm_raw(x)
        wrap_out, (wrap_hn, wrap_cn) = wrapper(x)

    out_match = torch.allclose(ref_out, wrap_out, rtol=rtol, atol=atol)
    if out_match:
        print("  ✅ LSTMWrapper: outputs match nn.LSTM!")
    else:
        print(f"  ❌ LSTMWrapper: outputs do NOT match! Max diff: {(ref_out - wrap_out).abs().max():.2e}")

    # ==========================================================================
    # Test 4: BidirectionalLSTMWrapper numerical validation
    # ==========================================================================
    print("\n[Test 4] BidirectionalLSTMWrapper numerical validation")

    torch.manual_seed(42)
    lstm_raw_bi = torch.nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
    lstm_raw_bi.eval()
    wrapper_bi = BidirectionalLSTMWrapper(lstm_raw_bi)

    x = torch.randn(batch, seq_len, input_size)

    with torch.no_grad():
        ref_out, (ref_hn, ref_cn) = lstm_raw_bi(x)
        wrap_out, (wrap_hn, wrap_cn) = wrapper_bi(x)

    out_match = torch.allclose(ref_out, wrap_out, rtol=rtol, atol=atol)
    if out_match:
        print("  ✅ BidirectionalLSTMWrapper: outputs match nn.LSTM!")
    else:
        print(f"  ❌ BidirectionalLSTMWrapper: outputs do NOT match! Max diff: {(ref_out - wrap_out).abs().max():.2e}")

    # ==========================================================================
    # Test 5: torch.export preservation
    # ==========================================================================
    print("\n[Test 5] torch.export preservation")

    class TestModelUni(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=False)

        def forward(self, x):
            output, _ = self.lstm(x)
            return output

    class TestModelBi(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=False, bidirectional=True)

        def forward(self, x):
            output, _ = self.lstm(x)
            return output

    # Test unidirectional
    model_uni = TestModelUni()
    patch_lstm_modules(model_uni)
    ep_uni = torch.export.export(model_uni, (torch.randn(seq_len, batch, input_size),))
    graph_uni = str(ep_uni.graph)
    if "kokoro.lstm_forward" in graph_uni and "bidirectional" not in graph_uni:
        print("  ✅ Unidirectional: kokoro.lstm_forward preserved in export!")
    else:
        print("  ❌ Unidirectional: custom op not preserved")

    # Test bidirectional
    model_bi = TestModelBi()
    patch_lstm_modules(model_bi)
    ep_bi = torch.export.export(model_bi, (torch.randn(seq_len, batch, input_size),))
    graph_bi = str(ep_bi.graph)
    if "kokoro.lstm_forward_bidirectional" in graph_bi:
        print("  ✅ Bidirectional: kokoro.lstm_forward_bidirectional preserved in export!")
    else:
        print("  ❌ Bidirectional: custom op not preserved")

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_lstm_custom_op()
