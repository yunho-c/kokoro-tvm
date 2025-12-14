# LSTM IR Explosion Investigation Report

## Executive Summary

The duration/f0 predictor in Kokoro TTS generates **11,533 lines of IR (1.1MB)** for `seq_len=64`, causing slow compilation. This investigation traced the root cause and identified solutions.

## Root Cause

**Static unrolling during graph construction**, not during compilation.

### Where It Happens

| Stage | Output Size | Time |
|-------|-------------|------|
| `torch.export` | 171 nodes | 1.8s |
| `from_exported_program` | **11,533 lines** | 7.7s |
| After all transforms | 10,888 lines | +5s |

The explosion occurs in `from_exported_program`, specifically in the `_lstm` handler.

### How It Happens

Both upstream TVM (PR #18346) and our `tvm_extensions.py` implement LSTM like this:

```python
def _lstm_cell_unroll(self, ..., seq_len, ...):
    outputs = []
    for t in time_steps:  # Python for loop at import time!
        x_t = self.block_builder.emit(relax.op.take(...))
        # ... emit gate computations ...
        outputs.append(h_t)
    
    output = self.block_builder.emit(relax.op.stack(outputs, axis=0))
```

This Python `for` loop runs during the import process, calling `emit()` once per timestep. For `seq_len=64`, it creates 64 explicit copies of all LSTM gate computations.

## Performance Impact

### Compilation Time
- Scales linearly with sequence length
- `seq_len=128` → ~10+ minutes to compile

### Runtime Performance
- **1MB+ code hurts CPU performance**:
  - L1 I-cache is ~32KB → constant cache thrashing
  - Branch predictor can't learn patterns across 1M+ instructions
  - LLVM loses loop optimization opportunities
- A tight loop is actually **faster** than unrolled code for sequential ops

## Solutions

### Option 1: TIR PrimFunc with Explicit Loops (Recommended)

Write LSTM as a TIR `@T.prim_func` with `for` loops → loops stay as loops.

```
IR Size: 18 lines (vs 11,533)
Compiled: 49.5 KB (vs multi-MB)
```

See: [tir_lstm_prototype.py](file:///Users/yunhocho/GitHub/kokoro-tvm/experiments/tir_lstm_prototype.py)

### Option 2: Relax Recursive Functions

Relax supports iteration via tail-recursive function calls:

```python
@R.function
def lstm_step(t, h, c, x):
    if t < seq_len:
        h_new, c_new = lstm_cell(...)
        return lstm_step(t + 1, h_new, c_new, x)
    else:
        return h, c
```

```
IR Size: 24 lines (constant regardless of seq_len)
```

See: [relax_loop_prototype.py](file:///Users/yunhocho/GitHub/kokoro-tvm/experiments/relax_loop_prototype.py)

### Option 3: External Library (call_packed)

Defer to optimized LSTM implementation (e.g., Apple Accelerate, oneDNN).

**Caveat**: No good cross-platform LSTM library exists for iOS/Android.

## Trade-offs

| Approach | IR Size | Ease | Performance |
|----------|---------|------|-------------|
| TIR PrimFunc | ✅ Smallest | ⚠️ Low-level | ✅ Best |
| Relax Recursive | ✅ Small | ✅ High-level | ⚠️ VM call overhead |
| External Library | ✅ Minimal | ⚠️ Platform-specific | ✅ Best |
| Current (unrolled) | ❌ Huge | ✅ Simple | ❌ Poor |

## Recommendation

For **Kokoro TVM**:

1. **Short-term**: Live with compile time, or reduce `seq_len` for dev iterations
2. **Medium-term**: Replace `_lstm` handler to use TIR PrimFunc or Relax recursive
3. **Long-term**: Consider transformer distillation (see [LSTM_TO_TRANSFORMER_DISTILLATION.md](file:///Users/yunhocho/GitHub/kokoro-tvm/NOTES/LSTM_TO_TRANSFORMER_DISTILLATION.md))

## Files Created

| File | Purpose |
|------|---------|
| `experiments/ir_explosion_diagnosis.py` | Measures IR size at each compilation stage |
| `experiments/tir_lstm_prototype.py` | Demonstrates TIR loop preservation |
| `experiments/relax_loop_prototype.py` | Demonstrates Relax recursive functions |
| `NOTES/RELAX_LSTM_FIX_PROPOSAL.md` | Technical proposal for fixes |

## Key Insight

> **TVM's LSTM support unrolls at graph construction time, not runtime.**
> 
> The `for t in time_steps:` Python loop in the importer creates explicit IR nodes per timestep. To avoid this, implement iteration at the IR level (TIR loops or Relax recursion), not at the Python meta-level.
