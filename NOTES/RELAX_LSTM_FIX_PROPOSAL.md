# Proposal: Fixing LSTM IR Explosion in TVM Relax

## Problem Summary

When compiling Kokoro's ProsodyPredictor LSTM through `torch.export` → `from_exported_program`:
- PyTorch exports LSTM as **171 nodes** (single `aten::lstm.input` op)
- TVM's `emit_te(topi.nn.lstm)` expands this into **11,533 lines** (1.1MB) for seq_len=64

The explosion occurs because `emit_te()` materializes the `te.scan` loop as explicit tensor operations per timestep.

## To Your Question: Does Code Size Hurt CPU Performance?

**Yes, absolutely.** Here's why:

| Issue | Impact |
|-------|--------|
| **I-cache pressure** | Modern CPU L1 I-cache is ~32KB. 1MB+ code causes constant cache misses |
| **Branch prediction** | Unrolled code = many unique branch targets, thrashing the BTB |
| **Compilation artifacts** | LLVM optimizes each unrolled block separately, losing loop optimizations |
| **Binary size** | Final `.so` file bloats, affecting mobile deployment |

For LSTM on CPU, a **tight loop** over timesteps is actually faster than explicit unrolling because:
- The same small kernel fits in L1 I-cache
- Branch predictor learns the loop pattern
- Better CPU prefetching

## How Relax Handles Loops

Based on my research of TVM source:

### Approach 1: Recursive Functions (Current Relax Pattern)

Relax doesn't have an explicit `while_loop` operator. Instead, it uses **tail-recursive function calls**:

```python
@I.ir_module
class LSTMModule:
    @R.function
    def main(x: R.Tensor, h0: R.Tensor, c0: R.Tensor):
        n = x.shape[0]  # sequence length
        
        @R.function
        def lstm_step(t: R.Tensor, h: R.Tensor, c: R.Tensor, out_list):
            """Recursive LSTM step."""
            cond = R.call_pure_packed("test.less", t, n, sinfo_args=R.Tensor((), "bool"))
            if cond:
                # Compute one LSTM cell
                x_t = R.strided_slice(x, [t, 0], [t+1, x.shape[1]])
                h_new, c_new = lstm_cell(x_t, h, c)  # defined elsewhere
                new_out = R.concat([out_list, R.expand_dims(h_new, 0)])
                return lstm_step(t + 1, h_new, c_new, new_out)
            else:
                return out_list, h, c
        
        return lstm_step(R.const(0), h0, c0, R.empty([0, h0.shape[1]]))
```

**Pros**: Native to Relax, well-supported by VM
**Cons**: Complex to construct, closure handling tricky

### Approach 2: TIR PrimFunc with `While`

Use TIR-level loop and call it from Relax via `call_tir`:

```python
@T.prim_func
def lstm_forward(
    x: T.Buffer((seq_len, batch, input_dim), "float32"),
    h0: T.Buffer((batch, hidden_dim), "float32"),
    c0: T.Buffer((batch, hidden_dim), "float32"),
    Wi: T.Buffer((4 * hidden_dim, input_dim), "float32"),
    Wh: T.Buffer((4 * hidden_dim, hidden_dim), "float32"),
    output: T.Buffer((seq_len, batch, hidden_dim), "float32"),
):
    # Allocate hidden/cell state buffers
    h = T.alloc_buffer((batch, hidden_dim), "float32")
    c = T.alloc_buffer((batch, hidden_dim), "float32")
    gates = T.alloc_buffer((batch, 4 * hidden_dim), "float32")
    
    # Initialize from h0, c0
    for i, j in T.grid(batch, hidden_dim):
        h[i, j] = h0[i, j]
        c[i, j] = c0[i, j]
    
    # Loop over sequence
    for t in range(seq_len):
        # Input-to-hidden: gates = x[t] @ Wi.T
        for b, g in T.grid(batch, 4 * hidden_dim):
            gates[b, g] = T.float32(0)
            for k in range(input_dim):
                gates[b, g] += x[t, b, k] * Wi[g, k]
        
        # Hidden-to-hidden: gates += h @ Wh.T
        for b, g in T.grid(batch, 4 * hidden_dim):
            for k in range(hidden_dim):
                gates[b, g] += h[b, k] * Wh[g, k]
        
        # Apply gate activations and update states
        for b, j in T.grid(batch, hidden_dim):
            i_gate = T.sigmoid(gates[b, j])
            f_gate = T.sigmoid(gates[b, hidden_dim + j])
            g_gate = T.tanh(gates[b, 2 * hidden_dim + j])
            o_gate = T.sigmoid(gates[b, 3 * hidden_dim + j])
            
            c[b, j] = f_gate * c[b, j] + i_gate * g_gate
            h[b, j] = o_gate * T.tanh(c[b, j])
            output[t, b, j] = h[b, j]
```

**Pros**: Explicit loop stays as loop, no unrolling
**Cons**: Loses high-level optimizations, manual implementation

### Approach 3: External Library Call (call_packed)

Defer to an optimized LSTM implementation:

```python
@R.function
def lstm_forward(x, h0, c0, weights):
    output = R.call_packed(
        "kokoro.lstm.optimized",  # External C++ function
        x, h0, c0, weights,
        sinfo_args=R.Tensor((seq_len, batch, hidden), "float32")
    )
    return output
```

**Pros**: Maximum performance, reuse cuDNN/oneDNN
**Cons**: Requires external implementation, cross-platform complexity

## Recommended Fix: Hybrid Approach

### Short-term: Use TIR PrimFunc LSTM

1. Implement LSTM as a TIR `@T.prim_func` with explicit `for` loops
2. Call it from Relax via `R.call_tir`
3. The loop stays as a loop—no unrolling

### Medium-term: Improve `_lstm` Handler

Modify `tvm_extensions.py` to:
1. **Not use `emit_te(topi.nn.lstm)`** - this causes the explosion
2. Instead, emit a `call_tir` to a pre-defined LSTM PrimFunc
3. Store the PrimFunc in the module once, reuse for all LSTM calls

### Long-term: Contribute Relax-native LSTM Op

1. Add `R.nn.lstm()` operator to Relax operator library
2. Implement it using the recursive function pattern internally
3. The VM can execute the recursive calls efficiently without unrolling

## Prototype: TIR-based LSTM

See `experiments/tir_lstm_prototype.py` for a working implementation.

## References

- TVM TIR `while_loop`: `python/tvm/tir/ir_builder.py:286`
- Relax recursive loop pattern: `tests/python/relax/test_transform_dead_code_elimination.py:740`
- TOPI LSTM with `te.scan`: `python/tvm/topi/nn/lstm.py:223`
