# TVM Relax Dynamic Shape Limitations

This document describes why TVM Relax fails to compile models with complex dynamic shapes, specifically the Kokoro TTS encoder and decoder.

## Executive Summary

TVM Relax was designed for dynamic shapes but its implementation is incomplete. LLM workloads (Transformers) work because they use **simple, shape-preserving operations**. Audio models like Kokoro fail because they use **strided convolutions and upsampling** that produce complex derived symbolic expressions.

## The Two Critical Blockers

### 1. `emit_te` Requires `ShapeExpr` (Not `PrimExpr`)

**What it is**: `emit_te` is TVM's bridge from Relax (high-level IR) to TensorIR (low-level loop-based IR). Most legalization passes use it internally.

**The crash**:
```
AssertionError: emit_te now only supports Tensor that has ShapeExpr shape
```

**Root cause**: TVM has two shape representations:
- `ShapeExpr`: A tuple of dimensions, e.g., `(batch, seq_len, hidden)`
- `PrimExpr`: A single scalar symbolic value, e.g., `m` or `2048 * m`

When tensor shapes contain raw `PrimExpr` (not wrapped in `ShapeExpr`), `emit_te` crashes. This happens after operations that produce derived dimensions.

**Assessment**: This is a **bug**, not a fundamental limitation. The converters need to ensure shapes are always wrapped properly. Fixable with moderate effort.

### 2. `VMShapeLower` Cannot Handle Derived Expressions

**What it is**: `VMShapeLower` converts symbolic shape expressions to runtime VM bytecode. It's required for any execution.

**The crash**:
```
PrimExpr has not been computed
```

**Root cause**: Complex derived expressions fail:
- `2048 * m` (from upsampling)
- `m // 5 - 1` (from strided convolutions)
- `(m + padding - kernel) // stride + 1` (general conv shapes)

The pass maintains a registry of "known" symbolic variables. When it encounters a compound expression involving operations on symbolic vars that haven't been previously registered, it fails.

**Assessment**: This is a **fundamental architectural gap**. Fixing it would require:
- Bidirectional shape inference (solving `m` from `2048 * m = output_len`)
- Integration with a proper symbolic algebra system
- Months of work + architectural changes

## Optional Passes That Also Crash

These can be skipped, but skipping them doesn't help if the necessary passes already fail:

| Pass | Purpose | Failure Mode |
|------|---------|--------------|
| `FoldConstant` | Evaluate constants at compile time | Crashes on binary ops with symbolic shapes |
| `RewriteDataflowReshape` | Optimize reshape patterns | Can't prove divisibility for symbolic expressions |

## Why MLC-LLM Works But Kokoro Doesn't

### LLM Architecture (Works)
```
input: (batch, seq_len, hidden)
  → attention → (batch, seq_len, hidden)
  → FFN → (batch, seq_len, hidden)
  → output: (batch, seq_len, hidden)
```
- Shapes are **preserved** through operations
- Only simple symbolic vars: `batch`, `seq_len`
- No derived expressions like `2048 * seq_len`

### Kokoro Decoder Architecture (Fails)
```
input: (1, 512, m)  # m = number of acoustic frames
  → Conv1D stride=1 → (1, C, m)
  → Upsample 2048× → (1, C, 2048*m)  ❌ Derived expression
  → Conv1D stride=5 → (1, C, m//5-1)  ❌ Derived expression
  → output: (1, 1, ~2048*m)
```

## TVM's Symbolic Algebra Architecture

### SymPy at Import Only
TVM uses SymPy only at the import boundary (when parsing PyTorch's `torch.export` shapes). Once inside TVM, it converts to its own `PrimExpr` representation.

### Internal Arith Analyzer Is Limited
TVM's `tvm.arith.Analyzer` has fewer capabilities than SymPy:

| Capability | SymPy | TVM Arith Analyzer |
|------------|-------|-------------------|
| Simplify `2048 * m` | ✅ | ✅ |
| Track derived vars (`x = 2048*m`, use `x` later) | ✅ | ❌ |
| Solve `m` from `2048 * m = output_len` | ✅ | ❌ |
| Prove `m // 5 * 5 <= m` | ✅ | ❌ (limited) |

TVM parses SymPy but then discards its power by converting to a weaker internal representation.

## Current Workarounds

### For Decoder: Static Shape Bucketing
```python
# Compile multiple sequence length buckets
for seq_len in [50, 150, 300, 500]:
    compile_decoder(seq_len=seq_len, output=f"decoder_m{seq_len}.so")
```
At runtime, pad inputs to the nearest bucket.

### For Encoder: Length-Aware LSTM Ops
The encoder issue is compounded by bidirectional LSTM semantics (see `LSTM_PACKED_SEMANTICS_ISSUE.md`). Even with static shapes, padding corrupts the backward recurrence.

Workaround: Carry `lengths` explicitly through the graph and use packed-sequence semantics at runtime.

## Upstream Status (as of late 2024)

| Question | Answer |
|----------|--------|
| Is there active work on symbolic analysis? | Incremental only |
| Is there an RFC for better `VMShapeLower`? | No |
| Will it get fixed soon? | Unlikely without a champion |
| Primary TVM focus | LLM workloads (Transformers) |

Audio/signal processing models with strided convolutions are edge cases that few production users hit.

## Potential Fix Paths

### Short Term (Weeks)
- Fix `emit_te` shape wrapping bug
- Skip optional passes that crash
- Use static shape bucketing

### Medium Term (Months)
- Extend `arith.Analyzer` to track derived expressions
- Improve `VMShapeLower` to handle compound expressions
- Requires TVM upstream contribution

### Long Term (Redesign)
- Propagate SymPy expressions through the compiler pipeline
- Implement bidirectional shape inference
- Major architectural change

## Related Notes

- `DECODER_PORTING_INSIGHTS.md` - Detailed decoder porting experience
- `LSTM_PACKED_SEMANTICS_ISSUE.md` - Encoder LSTM issues
- `MPS_LSTM_FEASIBILITY.md` - GPU LSTM alternatives
