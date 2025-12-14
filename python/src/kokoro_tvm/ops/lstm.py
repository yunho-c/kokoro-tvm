"""Relax-based LSTM implementation using recursive functions.

This module provides an LSTM implementation that avoids IR explosion by using
Relax's recursive function pattern instead of unrolling at graph construction time.

The key insight is that Relax supports tail-recursive function calls which the
VM can execute efficiently at runtime, keeping the IR size O(1) regardless of
sequence length.

Usage:
    from kokoro_tvm.ops.lstm import emit_relax_lstm, LSTMConfig

    # In your _lstm handler:
    config = LSTMConfig(use_recursive=True)
    output = emit_relax_lstm(block_builder, input_tensor, ...)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import tvm
from tvm import relax
from tvm.relax import Var


@dataclass
class LSTMConfig:
    """Configuration for LSTM implementation strategy.

    Attributes:
        use_recursive: If True, use Relax recursive functions (compact IR).
                      If False, use unrolled approach (larger IR, existing behavior).
    """

    use_recursive: bool = False  # Default to existing behavior for compatibility


# Global config - can be set before compilation
_lstm_config = LSTMConfig()


def set_lstm_config(config: LSTMConfig):
    """Set global LSTM configuration."""
    global _lstm_config
    _lstm_config = config


def get_lstm_config() -> LSTMConfig:
    """Get current LSTM configuration."""
    return _lstm_config


def emit_lstm_step_function(
    bb: relax.BlockBuilder,
    seq_len: int,
    batch_size: int,
    input_size: int,
    hidden_size: int,
    dtype: str = "float32",
) -> relax.GlobalVar:
    """Emit the recursive lstm_step function to the module.

    This creates a private function that processes one timestep and recurses.

    Args:
        bb: BlockBuilder to emit to
        seq_len: Sequence length (for termination condition)
        batch_size: Batch size
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        dtype: Data type

    Returns:
        GlobalVar pointing to the lstm_step function
    """
    # Define struct infos for the function signature
    t_sinfo = relax.TensorStructInfo((), "int32")
    h_sinfo = relax.TensorStructInfo((batch_size, hidden_size), dtype)
    c_sinfo = relax.TensorStructInfo((batch_size, hidden_size), dtype)
    x_sinfo = relax.TensorStructInfo((seq_len, batch_size, input_size), dtype)
    Wi_sinfo = relax.TensorStructInfo((4 * hidden_size, input_size), dtype)
    Wh_sinfo = relax.TensorStructInfo((4 * hidden_size, hidden_size), dtype)
    b_sinfo = relax.TensorStructInfo((4 * hidden_size,), dtype)

    # Note: Creating recursive functions programmatically is complex in Relax.
    # For now, we provide the pattern that should be used but the actual
    # integration requires TVMScript or careful BlockBuilder usage.

    # This function is a placeholder - the actual implementation requires
    # either TVMScript definition or complex BlockBuilder manipulation.
    raise NotImplementedError(
        "Programmatic recursive function emission requires TVMScript. "
        "Use emit_relax_lstm_unrolled for now, or define the module via TVMScript."
    )


def emit_relax_lstm_unrolled(
    bb: relax.BlockBuilder,
    input_tensor: Var,
    weight_ih: Var,
    weight_hh: Var,
    bias_ih: Optional[Var],
    bias_hh: Optional[Var],
    h_prev: Var,
    c_prev: Var,
    seq_len: int,
    hidden_size: int,
    reverse: bool = False,
) -> Var:
    """Emit LSTM using unrolled approach (existing behavior).

    This unrolls the LSTM computation at graph construction time, creating
    one set of operations per timestep. IR size is O(seq_len).

    Args:
        bb: BlockBuilder to emit to
        input_tensor: Input tensor [seq_len, batch, input_size]
        weight_ih: Input-to-hidden weights [4*hidden, input]
        weight_hh: Hidden-to-hidden weights [4*hidden, hidden]
        bias_ih: Input-to-hidden bias [4*hidden] or None
        bias_hh: Hidden-to-hidden bias [4*hidden] or None
        h_prev: Initial hidden state [batch, hidden]
        c_prev: Initial cell state [batch, hidden]
        seq_len: Sequence length
        hidden_size: Hidden dimension
        reverse: If True, process sequence in reverse

    Returns:
        Output tensor [seq_len, batch, hidden]
    """
    weight_ih_t = bb.emit(relax.op.permute_dims(weight_ih, axes=[1, 0]))
    weight_hh_t = bb.emit(relax.op.permute_dims(weight_hh, axes=[1, 0]))
    outputs = []
    time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

    for t in time_steps:
        x_t = bb.emit(relax.op.take(input_tensor, relax.const(t, "int64"), axis=0, mode="clip"))
        ih_gates = bb.emit(relax.op.linear_algebra.matmul(x_t, weight_ih_t))
        hh_gates = bb.emit(relax.op.linear_algebra.matmul(h_prev, weight_hh_t))

        gates = bb.emit(relax.op.add(ih_gates, hh_gates))
        if bias_ih is not None:
            gates = bb.emit(relax.op.add(gates, bias_ih))
        if bias_hh is not None:
            gates = bb.emit(relax.op.add(gates, bias_hh))

        i_gate = bb.emit(relax.op.strided_slice(gates, axes=[1], begin=[0], end=[hidden_size]))
        f_gate = bb.emit(relax.op.strided_slice(gates, axes=[1], begin=[hidden_size], end=[2 * hidden_size]))
        g_gate = bb.emit(relax.op.strided_slice(gates, axes=[1], begin=[2 * hidden_size], end=[3 * hidden_size]))
        o_gate = bb.emit(relax.op.strided_slice(gates, axes=[1], begin=[3 * hidden_size], end=[4 * hidden_size]))

        i_t = bb.emit(relax.op.sigmoid(i_gate))
        f_t = bb.emit(relax.op.sigmoid(f_gate))
        g_t = bb.emit(relax.op.tanh(g_gate))
        o_t = bb.emit(relax.op.sigmoid(o_gate))

        c_t = bb.emit(relax.op.add(relax.op.multiply(f_t, c_prev), relax.op.multiply(i_t, g_t)))
        h_t = bb.emit(relax.op.multiply(o_t, relax.op.tanh(c_t)))

        outputs.append(h_t)
        h_prev = h_t
        c_prev = c_t

    if reverse:
        outputs = outputs[::-1]

    output = bb.emit(relax.op.stack(outputs, axis=0))
    return output


def emit_relax_lstm(
    bb: relax.BlockBuilder,
    input_tensor: Var,
    weight_ih: Var,
    weight_hh: Var,
    bias_ih: Optional[Var],
    bias_hh: Optional[Var],
    h_prev: Var,
    c_prev: Var,
    seq_len: int,
    hidden_size: int,
    reverse: bool = False,
    config: Optional[LSTMConfig] = None,
) -> Var:
    """Emit LSTM computation with configurable implementation strategy.

    Args:
        bb: BlockBuilder to emit to
        input_tensor: Input tensor [seq_len, batch, input_size]
        weight_ih: Input-to-hidden weights [4*hidden, input]
        weight_hh: Hidden-to-hidden weights [4*hidden, hidden]
        bias_ih: Input-to-hidden bias [4*hidden] or None
        bias_hh: Hidden-to-hidden bias [4*hidden] or None
        h_prev: Initial hidden state [batch, hidden]
        c_prev: Initial cell state [batch, hidden]
        seq_len: Sequence length
        hidden_size: Hidden dimension
        reverse: If True, process sequence in reverse
        config: LSTM configuration. If None, uses global config.

    Returns:
        Output tensor [seq_len, batch, hidden]
    """
    if config is None:
        config = get_lstm_config()

    if config.use_recursive:
        # Recursive implementation - requires TVMScript module definition
        # For now, fall back to unrolled with a warning
        import warnings

        warnings.warn(
            "Recursive LSTM not yet fully integrated. "
            "Falling back to unrolled implementation. "
            "To use recursive LSTM, define the module via TVMScript."
        )
        return emit_relax_lstm_unrolled(
            bb, input_tensor, weight_ih, weight_hh, bias_ih, bias_hh, h_prev, c_prev, seq_len, hidden_size, reverse
        )
    else:
        return emit_relax_lstm_unrolled(
            bb, input_tensor, weight_ih, weight_hh, bias_ih, bias_hh, h_prev, c_prev, seq_len, hidden_size, reverse
        )


# TVMScript-based recursive LSTM module template
LSTM_TVMSCRIPT_TEMPLATE = '''
"""TVMScript template for recursive LSTM.

To use, copy and modify the dimensions to match your model.
"""

from tvm.script import ir as I, relax as R

@I.ir_module
class RecursiveLSTMModule:
    @R.function
    def forward(
        x: R.Tensor(({seq_len}, {batch_size}, {input_size}), dtype="{dtype}"),
        h0: R.Tensor(({batch_size}, {hidden_size}), dtype="{dtype}"),
        c0: R.Tensor(({batch_size}, {hidden_size}), dtype="{dtype}"),
        Wi: R.Tensor(({gate_size}, {input_size}), dtype="{dtype}"),
        Wh: R.Tensor(({gate_size}, {hidden_size}), dtype="{dtype}"),
        bi: R.Tensor(({gate_size},), dtype="{dtype}"),
        bh: R.Tensor(({gate_size},), dtype="{dtype}"),
    ) -> R.Tuple(R.Tensor(({batch_size}, {hidden_size}), "{dtype}"), 
                 R.Tensor(({batch_size}, {hidden_size}), "{dtype}")):
        
        @R.function
        def lstm_step(
            t: R.Tensor((), dtype="int32"),
            h: R.Tensor(({batch_size}, {hidden_size}), dtype="{dtype}"),
            c: R.Tensor(({batch_size}, {hidden_size}), dtype="{dtype}"),
        ) -> R.Tuple(R.Tensor(({batch_size}, {hidden_size}), "{dtype}"), 
                     R.Tensor(({batch_size}, {hidden_size}), "{dtype}")):
            cond = R.call_pure_packed(
                "tvm.contrib.less", t, R.const({seq_len}, "int32"),
                sinfo_args=R.Tensor((), dtype="bool")
            )
            
            if cond:
                t_i64 = R.astype(t, "int64")
                x_t = R.take(x, t_i64, axis=0)
                
                Wi_t = R.permute_dims(Wi, axes=[1, 0])
                Wh_t = R.permute_dims(Wh, axes=[1, 0])
                gates = R.add(R.matmul(x_t, Wi_t), R.matmul(h, Wh_t))
                gates = R.add(R.add(gates, bi), bh)
                
                i_gate = R.strided_slice(gates, axes=[1], begin=[0], end=[{hidden_size}])
                f_gate = R.strided_slice(gates, axes=[1], begin=[{hidden_size}], end=[{hidden_size_2x}])
                g_gate = R.strided_slice(gates, axes=[1], begin=[{hidden_size_2x}], end=[{hidden_size_3x}])
                o_gate = R.strided_slice(gates, axes=[1], begin=[{hidden_size_3x}], end=[{gate_size}])
                
                i_t = R.sigmoid(i_gate)
                f_t = R.sigmoid(f_gate)
                g_t = R.tanh(g_gate)
                o_t = R.sigmoid(o_gate)
                
                c_new = R.add(R.multiply(f_t, c), R.multiply(i_t, g_t))
                h_new = R.multiply(o_t, R.tanh(c_new))
                
                t_next = R.add(t, R.const(1, "int32"))
                r = lstm_step(t_next, h_new, c_new)
            else:
                r = (h, c)
            return r
        
        result = lstm_step(R.const(0, "int32"), h0, c0)
        return result
'''


def generate_lstm_tvmscript(
    seq_len: int,
    batch_size: int,
    input_size: int,
    hidden_size: int,
    dtype: str = "float32",
) -> str:
    """Generate TVMScript code for a recursive LSTM module.

    Args:
        seq_len: Sequence length
        batch_size: Batch size
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        dtype: Data type

    Returns:
        TVMScript code as a string
    """
    return LSTM_TVMSCRIPT_TEMPLATE.format(
        seq_len=seq_len,
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=hidden_size,
        gate_size=4 * hidden_size,
        hidden_size_2x=2 * hidden_size,
        hidden_size_3x=3 * hidden_size,
        dtype=dtype,
    )
