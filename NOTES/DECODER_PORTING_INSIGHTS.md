# Kokoro Decoder TVM Porting Insights

This document captures lessons learned during the process of porting the Kokoro TTS Decoder to TVM Relax.

## Table of Contents
1. [PyTorch Export Challenges](#pytorch-export-challenges)
2. [TVM Import Issues](#tvm-import-issues)
3. [Dynamic vs Static Shapes](#dynamic-vs-static-shapes)
4. [Custom Operator Handling](#custom-operator-handling)
5. [Build Pipeline Issues](#build-pipeline-issues)
6. [LLVM Compatibility Issues](#llvm-compatibility-issues)
7. [Files to Commit vs Temporary](#files-to-commit-vs-temporary)

---

## PyTorch Export Challenges

### Symbolic Shape Constraints
When using `torch.export` with `dynamic_shapes`, you must define explicit `Dim` constraints:
```python
from torch.export import Dim
dim_time = Dim("m", min=1)
dynamic_shapes = {
    "asr": {2: dim_time},
    "f0": {1: 2 * dim_time},  # Derived dimensions supported
}
```

### Model Modifications for Export
Some operations are not export-friendly. For the Kokoro model:
- **SineGen._f02sine**: Had to patch to avoid using `item()` on tensors and use `F.interpolate` with explicit size instead of symbolic division
- **Weight normalization**: Used `parametrizations` which exports cleanly

### Decompositions Caveat
`run_decompositions()` after loading a serialized `ExportedProgram` can fail with:
```
AssertionError: Unsupported function types ['wrap_with_set_grad_enabled', 'upsample_nearest1d.vec', '_weight_norm.default']
```
Solution: Run decompositions immediately after `torch.export.export()` before saving, or wrap in try-except.

---

## TVM Import Issues

### Missing Operator Converters
TVM's `ExportedProgramImporter` lacks handlers for several ATen ops. We monkeypatched:

| Op | Solution |
|---|---|
| `aten.atan2` | Use `relax.op.atan2` |
| `aten.floordiv` | Use `relax.op.floor_divide` with explicit int64 casting |
| `aten.sym_float` | Return value directly (already numeric PrimExpr) |
| `aten.clamp` | Use `relax.op.clip` |
| `aten.index_put_` | Use `relax.op.where` for boolean mask case |
| `aten.leaky_relu` | Fix operator name from `leaky_relu_` to `leaky_relu` |
| `aten._convolution` | Add debug logging for shape inference |

### Shape Inference Failures
Multiple converters failed due to improper `struct_info` handling:
- **`_mul`**: When multiplying tensors by symbolic scalars, must create proper scalar tensors with valid `struct_info`
- **`_cat`**: Output shape inference failed; added explicit `R.match_cast` for validation
- **`_create_scalar_tensor`**: Must emit ops through `block_builder` to populate `struct_info` - using `relax.op.shape_to_tensor`, `reshape`, and `astype`

### PrimExpr vs ShapeExpr
TVM distinguishes between `PrimExpr` (scalar symbolic values) and `ShapeExpr` (shape tuples). Many internal functions expect `ShapeExpr` but receive `PrimExpr`, causing crashes:
```
AssertionError: emit_te now only supports Tensor that has ShapeExpr shape
```

---

## Dynamic vs Static Shapes

### Dynamic Shape Limitations in TVM Relax
TVM Relax has significant limitations with dynamic shapes:

1. **`emit_te` requires ShapeExpr**: Many legalization passes use `emit_te` which crashes on symbolic `PrimExpr` shapes
2. **`VMShapeLower` failures**: Complex symbolic expressions (e.g., `2048 * m`, `m // 5 - 1`) fail with "PrimExpr has not been computed"
3. **Iterator analysis**: `RewriteDataflowReshape` can't prove divisibility for symbolic expressions
4. **`FoldConstant`**: Crashes on binary ops with symbolic shapes

### Recommended Approach: Static Shape Bucketing
For production TTS deployment:
```python
# Compile multiple sequence length buckets
for seq_len in [50, 150, 300, 500]:
    compile_decoder(seq_len=seq_len, output=f"decoder_m{seq_len}.so")
```

At runtime, pad inputs to nearest bucket and select appropriate compiled module.

---

## Custom Operator Handling

### Custom FLegalize for slice_scatter
The default legalization for `slice_scatter` uses `emit_te` which fails on symbolic shapes. Solution:

```python
# Counter for unique function names
_slice_scatter_counter = [0]

@tvm.ir.register_op_attr("relax.slice_scatter", "FLegalize", level=11)
def _custom_slice_scatter(bb, call):
    _slice_scatter_counter[0] += 1
    func_id = _slice_scatter_counter[0]
    
    # Use topi.slice_scatter and call_tir directly
    prim_func = te.create_prim_func(te_args)
    # CRITICAL: Set unique global_symbol to avoid collisions
    prim_func = prim_func.with_attr("global_symbol", f"slice_scatter_tir_{func_id}")
    gvar = bb.add_func(prim_func, f"slice_scatter_custom_{func_id}")
    return bb.emit(relax.call_tir(gvar, call_args, out_sinfo=call.struct_info))
```

### Key Lessons:
1. **Use high level** (11+) to override default registrations at level 10
2. **Set unique `global_symbol`** - each TIR function needs unique naming or you get "duplicate global symbol" errors
3. **Use `bb.emit(relax.call_tir(...))` not `bb.call_tir()`** - BlockBuilder has no direct `call_tir` method
4. **`te.create_prim_func` defaults to `global_symbol="main"`** - always override this

---

## Build Pipeline Issues

### Duplicate Global Symbol: main
When importing from `from_exported_program`, the main function is named "main". This conflicts with TVM's internal `NormalizeGlobalVar` pass:

```python
# Solution: Rename main function AND its global_symbol attribute
for gv, func in mod.functions.items():
    if gv.name_hint == "main":
        new_gv = tvm.ir.GlobalVar("decoder_forward")
        # CRITICAL: Must also update the function's global_symbol attribute
        if hasattr(func, "attrs") and func.attrs is not None and "global_symbol" in func.attrs:
            new_attrs = dict(func.attrs)
            new_attrs["global_symbol"] = "decoder_forward"
            func = func.with_attrs(new_attrs)
        new_funcs[new_gv] = func
```

### Optimization Pass Ordering
The standard optimization sequence:
1. `DecomposeOpsForInference` - handles batch norm inference mode, etc.
2. `LegalizeOps` - converts Relax ops to TIR (uses FLegalize)
3. `AnnotateTIROpPattern` - marks ops for fusion
4. `FoldConstant` - evaluates constant expressions (may crash with symbolic shapes)
5. `FuseOps` - combines operators
6. `FuseTIR` - fuses TIR primitives

### PassContext Configuration
```python
# Attempt to disable debug info (doesn't fully work with LLVM 21)
with tvm.transform.PassContext(opt_level=3, config={"tir.enable_debug": False}):
    ex = relax.build(mod, target)
```

---

## LLVM Compatibility Issues

### LLVM Debug Info Verification Bug
**Error**: `LLVM module verification failed: location of #dbg_declare must be a pointer or int`

**Your TVM Configuration**:
```
LLVM_VERSION: 21.1.6
```

This is a **very recent LLVM version** (LLVM 21 is cutting edge). TVM's CI primarily tests with LLVM 10 and 17.

**Cause**: TVM's codegen emits `#dbg_declare` for float reduction temporaries in `instance_norm` fused kernels. LLVM 15+ has stricter verification that rejects this.

**Function names affected**:
- `fused_add*_instance_norm*_multiply*_leaky_relu*_compute_`

**Resolution Options**:
1. **Rebuild TVM with older LLVM** (10 or 17 are well-tested in TVM CI)
2. **Disable debug info at TVM build time**: `cmake -DCMAKE_BUILD_TYPE=Release` without debug flags
3. **Patch TVM source** in `src/target/llvm/codegen_llvm.cc` to skip debug info for scalar temporaries
4. **Wait for TVM fix** - this is a known LLVM compatibility issue

### Checking Your LLVM Version
```python
import tvm
print(tvm.support.libinfo()["LLVM_VERSION"])  # Returns "21.1.6"
```

---


## Recommended TVM Upstream Action Items

Based on issues encountered during this porting effort, the following improvements would benefit the TVM community:

### 🔴 Critical Priority

#### 1. LLVM 15+ Debug Info Compatibility
**File:** `src/target/llvm/codegen_llvm.cc`

TVM's LLVM codegen emits `#dbg_declare` for floating-point reduction temporaries (e.g., in `instance_norm` fused kernels). LLVM 15+ has stricter verification requiring `#dbg_declare` locations to be pointers or integers.

**Proposed Fix:**
- Skip debug info emission for scalar float temporaries in reduction loops
- Or update to use `#dbg_value` instead of `#dbg_declare` for non-pointer values
- Add LLVM version checks to adjust behavior for 15+

**Related Functions:** `fused_*_instance_norm_*_compute_`

#### 2. `te.create_prim_func` Default `global_symbol`
**File:** `python/tvm/te/operation.py` (or equivalent)

`te.create_prim_func()` defaults to `global_symbol="main"`, causing "duplicate global symbol" errors when multiple TIR functions are generated (e.g., in custom FLegalize implementations).

**Proposed Fix:**
- Generate unique default names (e.g., `prim_func_0`, `prim_func_1`)
- Or require explicit `global_symbol` parameter
- Document this behavior clearly in docstrings

### 🟠 High Priority

#### 3. Missing ATen Operator Converters
**File:** `python/tvm/relax/frontend/torch/exported_program_translator.py`

The following operators are missing or have bugs in `ExportedProgramImporter`:

| Operator | Issue | Suggested Implementation |
|----------|-------|-------------------------|
| `aten.atan2` | Missing | `relax.op.atan2(lhs, rhs)` |
| `aten.clamp` | Missing | `relax.op.clip(x, min, max)` |
| `aten.index_put_` | Missing | `relax.op.where` for boolean mask case |
| `aten.leaky_relu` | Wrong name | Uses `leaky_relu_` instead of `leaky_relu` |
| `aten.sym_float` | Missing | Return PrimExpr directly |
| `aten.floordiv` | Missing | `relax.op.floor_divide` with int64 cast |

#### 4. `emit_te` Symbolic Shape Support
**File:** `python/tvm/relax/block_builder.py`

`emit_te` fails with symbolic `PrimExpr` shapes:
```
AssertionError: emit_te now only supports Tensor that has ShapeExpr shape
```

**Proposed Fix:**
- Add handling for `PrimExpr` shapes by wrapping in `ShapeExpr`
- Or provide clear error message with workaround guidance

#### 5. PassContext `tir.enable_debug` Ineffective for LLVM
**Issue:** Setting `{"tir.enable_debug": False}` in `PassContext` does not prevent LLVM debug info emission.

**Proposed Fix:**
- Wire this config option to LLVM codegen's debug info logic
- Or add a separate `llvm.emit_debug_info` config option

### 🟡 Medium Priority

#### 6. `VMShapeLower` Symbolic Expression Handling
**File:** `src/relax/transform/vm_shape_lower.cc`

Complex symbolic expressions like `2048 * m`, `m // 5 - 1` fail with "PrimExpr has not been computed" in `VMShapeLower`.

**Proposed Fix:**
- Improve symbolic expression simplification
- Add fallback for unresolved symbolic expressions
- Better error messages indicating which expression failed

#### 7. `RewriteDataflowReshape` Divisibility Analysis
Cannot prove divisibility for symbolic expressions, causing reshape failures.

**Proposed Fix:**
- Leverage symbolic constraints from `Dim` specifications
- Add optional "trust me" mode that skips divisibility proofs

#### 8. `FoldConstant` with Symbolic Shapes
Crashes on binary ops involving symbolic shape expressions.

**Proposed Fix:**
- Skip folding for ops with symbolic inputs
- Or add graceful fallback instead of crash

### 🟢 Nice to Have

#### 9. `from_exported_program` Default Function Naming
The imported main function is named "main", which can conflict with user TIR functions that also default to "main".

**Proposed Fix:**
- Use a more unique default name (e.g., `exported_main`, `torch_forward`)
- Or auto-detect conflicts and rename

#### 10. Documentation for Custom FLegalize
Current documentation is minimal for implementing custom `FLegalize` handlers. 

**Proposed Additions:**
- Document `level` parameter behavior (11+ to override defaults at 10)
- Clarify `bb.emit(relax.call_tir(...))` pattern (not `bb.call_tir()`)
- Add examples for unique TIR function naming

---

## Summary of Required Changes

### Essential Files
- [`scripts/tvm_extensions.py`](file:///Users/yunhocho/GitHub/kokoro-tvm/scripts/tvm_extensions.py): Custom converters and FLegalize
- [`scripts/port_decoder.py`](file:///Users/yunhocho/GitHub/kokoro-tvm/scripts/port_decoder.py): Static shape export, main renaming, build config

### Critical Patches Applied
1. SineGen._f02sine patch for symbolic shapes
2. Custom converters for 6+ missing ATen ops
3. Custom FLegalize for slice_scatter with unique naming
4. Main function renaming with global_symbol attribute update
5. Unique TIR function naming with counters

### Remaining Blocker
LLVM 21.1.6 has stricter `#dbg_declare` verification than TVM's codegen supports. Requires TVM rebuild with older LLVM or source patch.

