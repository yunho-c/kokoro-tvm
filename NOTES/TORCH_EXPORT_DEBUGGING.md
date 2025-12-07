# Torch Export Debugging Learnings

## Context
We are attempting to export the Kokoro model using `torch.export` for compilation with TVM. The model uses `nn.LSTM` and has dynamic input shapes (specifically sequence length).

## Key Issues & Solutions

### 1. `nn.LSTM` Decomposition and Dynamic Shapes
**Problem:** `torch.export` decomposes high-level operators like `nn.LSTM` into their constituent ATen operations (e.g., matrix multiplications, element-wise ops). When `nn.LSTM` is used with dynamic shapes (symbolic dimensions), this decomposition can lead to `GuardOnDataDependentSymNode` errors. This happens because the unrolled loop structure of the decomposed LSTM might depend on the sequence length, which `torch.export` tries to specialize on, failing when it encounters a symbolic variable.

**Solution:** Prevent decomposition by wrapping `aten.lstm.input` in a custom operator.
- We defined a custom op `kokoro::lstm` using `torch.library.custom_op`.
- This op wraps `torch.ops.aten.lstm.input`.
- We registered a "fake" kernel for it to handle meta-tensor propagation (shape/dtype inference) without running the actual computation during tracing.
- We monkeypatched `nn.LSTM.forward` to use this `kokoro::lstm` op instead of the standard implementation.
- This forces `torch.export` to treat the LSTM as a single, opaque leaf node in the graph, preserving it for TVM to handle (TVM has an LSTM relay/relax op).

### 2. `AdaIN1d` Dynamic Shape Guards
**Problem:** The `AdaIN1d` module (in `istftnet.py`) caused a guard error on the sequence length dimension. `torch.export` couldn't prove that the sequence length was non-zero or valid for certain operations.

**Solution:** Explicit hints.
- We monkeypatched `AdaIN1d.forward` to include `torch._check(x.size(2) > 1)`.
- This informs the symbolic shape system that the sequence length is strictly greater than 1, satisfying the requirements of downstream operations and removing the need for a guard.

### 3. `istftnet.py` and `torch.cumsum`
**Problem:** Errors in `istftnet.py` within the `SineGen` module, specifically involving `torch.cumsum` and `F.interpolate` with symbolic shapes.

**Solution:**
- Monkeypatched `SineGen._f02sine` to use `size` argument in `F.interpolate` instead of `scale_factor`. `scale_factor` with symbolic shapes can be problematic for export.

### 4. TVM Converter Registration
**Problem:** TVM's `from_exported_program` failed with `AssertionError: Unsupported function types` for several operators, including our custom `kokoro::lstm`, `atan2`, `rand`, `full`, `arange`, etc. TVM's `ExportedProgramImporter` does not expose a simple `register_converter` API.

**Solution:** Monkeypatching `ExportedProgramImporter`.
- We created `scripts/tvm_extensions.py` to monkeypatch `ExportedProgramImporter.create_convert_map` to inject our custom converters.
- We implemented converters for missing ops:
    - `kokoro.lstm.default`: Mapped to `self._lstm` (reusing TVM's existing LSTM implementation).
    - `atan2`: Implemented using `emit_te(tvm.tir.atan2, ...)`.
    - `rand`, `randn_like`: Implemented using `relax.op.random`.
    - `full`: Monkeypatched to handle `PrimExpr` (symbolic/dynamic) fill values using `emit_te`.
    - `arange`: Monkeypatched to handle `relax.Var` arguments that wrap `PrimStructInfo` (symbolic scalars) by using `match_cast` to extract the underlying `PrimExpr`.

### 5. Handling Symbolic Values in TVM Converters
**Problem:** `torch.export` produces symbolic integers (`SymInt`) which are translated to `relax.Var` with `PrimStructInfo` or `TensorStructInfo` (scalar tensors) in TVM.
- `relax.op.full` and `relax.op.arange` (via python wrapper) expect `PrimValue` or `PrimExpr` for certain arguments, but receive `relax.Var` (DataflowVar).
- `relax.const` fails on `PrimExpr` or `relax.Var`.

**Solution:**
- For `full`: If the fill value is a `PrimExpr`, use `emit_te` to create a tensor filled with that symbolic value.
- For `arange`: If arguments are `relax.Var` with `PrimStructInfo`, use `block_builder.match_cast` to extract the symbolic `PrimExpr` variable and pass that to `relax.op.arange`.
- For `binary_op`: If arguments are `PrimExpr`, convert them to scalar tensors using a helper `_create_scalar_tensor` (via `emit_te`) before passing to Relax ops that expect Tensors.

## General Strategy
1.  **Isolate:** Create a minimal reproduction script (`scripts/reproduce_error.py`) that calls `torch.export` on the model with dummy inputs and dynamic shapes.
2.  **Simplify:** Monkeypatch problematic modules (`TextEncoder`, `ProsodyPredictor`, `nn.LSTM`) to remove complex control flow (like `pack_padded_sequence`) or force specific behaviors.
3.  **Preserve:** Use custom ops to prevent `torch.export` from decomposing complex operators that TVM can handle natively.
4.  **Hint:** Use `torch._check()` to provide constraints on symbolic dimensions to satisfy guards.
5.  **Extend TVM:** Monkeypatch `ExportedProgramImporter` to add missing converters and handle symbolic values correctly.
