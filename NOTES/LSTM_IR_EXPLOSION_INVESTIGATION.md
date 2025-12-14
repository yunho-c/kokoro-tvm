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

---

## Implementation Results

### Failed Approaches

#### 1. TVMScript @T.prim_func with T.serial loop ❌
**Error**: `Well-formedness check failed: Loop iterator var t is defined outside of any block, but is used inside the non-opaque current block`

```python
@T.prim_func
def lstm_forward(...):
    for t in T.serial(seq_len):
        for b, g in T.grid(batch, gates):
            with T.block("gates"):
                # Error: can't use `t` here!
                gates[vb, vg] = x[t, vb, k] * Wi[vg, k]
```

**Why it failed**: TVM's block well-formedness rules forbid using outer loop variables inside non-opaque blocks.

#### 2. TVMScript IRBuilder with T.While ❌
**Error**: `'Buffer' object does not support item assignment`

```python
with IRBuilder() as ib:
    h = T.alloc_buffer(...)
    with T.While(t < seq_len):
        h[b, j] = h_init[b, j]  # Error!
```

**Why it failed**: IRBuilder context uses different syntax - requires `T.buffer_store()` instead of direct indexing.

#### 3. TIR with decl_buffer and custom data variable ❌
**Error**: `Variable h_data is not a pointer`

```python
h_data = tir.Var("h_data", "handle")  # Wrong type!
h_buf = tir.decl_buffer(..., data=h_data)
```

**Why it failed**: Buffer data variables need proper pointer type annotation.

### TIR LSTM with While Loop ✅

Successfully implemented a non-unrolled LSTM using `tir.ir_builder.while_loop()`:

```python
# Generated TIR (excerpt):
t_1[0] = 0
while t_1[0] < 8:
    # ... LSTM gate computation ...
    # ... State updates ...
    t_1[0] = t_1[0] + 1
```

**Results:**
| Metric | Value |
|--------|-------|
| IR Lines | **42** (O(1) regardless of seq_len) |
| Build | ✅ LLVM successful |
| Execution | ✅ Works |

Compare: Unrolled TOPI version generates **11,500+ lines** for seq_len=64.

### Files Implemented

| File | Purpose |
|------|---------|
| `python/src/kokoro_tvm/ops/__init__.py` | Ops module |
| `python/src/kokoro_tvm/ops/lstm.py` | Configurable LSTM handler |
| `python/src/kokoro_tvm/ops/tir_lstm.py` | **TIR LSTM with While loop** |
| `experiments/lstm/relax_lstm_full.py` | Relax recursive prototype |

### Key Technical Learnings

1. **TVMScript block well-formedness**: Blocks can't access outer loop variables in non-opaque mode
2. **`tir.ir_builder`** bypasses block restrictions and supports `while_loop()`
3. **Integration pattern**: Use `bb.add_func()` + `relax.call_tir(gvar, args, out_sinfo)`

### Usage

```python
from kokoro_tvm.ops.tir_lstm import create_tir_lstm_primfunc

# Create TIR function
tir_func = create_tir_lstm_primfunc(seq_len, batch, input_size, hidden_size)

# Add to module and call via call_tir
tir_gv = block_builder.add_func(tir_func, "tir_lstm")
output = block_builder.emit(
    relax.call_tir(tir_gv, [x, h0, c0, Wi, Wh, bi, bh], out_sinfo=[...])
)
```

