# Kokoro Component Porting Analysis

## DurationEncoder Deep Dive

During the porting of `DurationEncoder`, two significant issues were encountered that required workarounds. This document analyzes the root causes of these failures.

### 1. `torch.export` & `PackedSequence` Limitations

**Symptom:**
`TypeError: zeros(): argument 'size' failed to unpack the object at pos 2 with error "type must be tuple of ints,but got FakeTensor"`

**Root Cause:**
*   The `DurationEncoder` uses `nn.LSTM`, which internally utilizes `pack_padded_sequence` and `pad_packed_sequence` for variable-length sequence handling.
*   `PackedSequence` objects hold data in a flattened format and manage batch sizes dynamically via a tensor (`batch_sizes`).
*   When tracing with `torch.export`, the dynamic nature of `batch_sizes` inside the `PackedSequence` logic creates `FakeTensor` objects for shapes or control flow conditions where static integers are strictly expected by downstream operations (like `torch.zeros` initialization of hidden states).
*   **Update:** Even when `lengths` (and thus `batch_sizes`) are provided as **static constants**, `torch.export` still fails. This indicates that the tracing mechanism treats the `PackedSequence` construction or the `batch_sizes` tensor derived from it as inherently symbolic/dynamic data, and refuses to evaluate it to static integers during the trace. It sees "Tensor derived from operation -> Argument to torch.zeros", and since `torch.zeros` is a factory requiring static ints in this context, it errors out.

**Solution/Workaround:**
*   **Monkeypatching:** We replaced `pack_padded_sequence` and `pad_packed_sequence` with mock functions that essentially pass through the padded tensor (ignoring packing).
*   **Implication:** The verified model in TVM assumes padded input processing (which is standard for many inference compilers) rather than optimized packed processing. This simplifies the graph and works correctly for static batch inference.

### 2. TVM Relax Frontend `expand` Bug

**Symptom:**
`IndexError: ShapeExpr index out of range` in `_expand` (`tvm/relax/frontend/torch/base_fx_graph_translator.py`).

**Root Cause:**
The standard implementation of `_expand` in TVM's Relax frontend is flawed when handling implicit dimension prepending (broadcasting where new dimensions are added at the front).

The failing code structure (approximate):
```python
# tvm/relax/frontend/torch/base_fx_graph_translator.py
def _expand(self, node: fx.Node) -> relax.Var:
    # ...
    sizes = args[1] # Target sizes, e.g., [64, 1, 64] (Rank 3)
    in_shape = self.shape_of(args[0]) # Input shape, e.g., [1, 64] (Rank 2)
    
    broadcast_shape = []
    for idx, i in enumerate(sizes):
        if isinstance(i, int) and i == -1:
             # ERROR HERE: idx goes from 0 to 2. in_shape has length 2.
             # When idx=2, in_shape[2] throws IndexError.
            broadcast_shape.append(in_shape[idx]) 
        else:
            broadcast_shape.append(i)
```

The logic naively assumes a 1:1 mapping from target dimensions to input dimensions using the *target* index. It fails to account for:
1.  **Rank Promotion:** If `len(sizes) > len(in_shape)`, PyTorch implicitly prepends dimensions. Using `idx` from `sizes` to access `in_shape` is incorrect without an offset (e.g., `in_shape[idx - (len(sizes) - len(in_shape))]`).
2.  **Broadcasting rules:** `expand` relies on broadcasting semantics for the existing dimensions.

**Solution/Workaround:**
*   **`tvm_extensions.py`:** We utilized a custom `_expand` implementation that robustly handles rank differences by calculating the correct offset and correctly identifying which input dimension corresponds to which output dimension (or if it's a new prepended dimension).
*   **Fix Verification:** The workaround in `tvm_extensions` successfully resolves the `IndexError` and produces the correct graph.

## Recommendations
1.  **Submit Upstream Fix:** The `_expand` bug in TVM should be reported or fixed upstream.
2.  **Avoid Packing in Inference:** For TVM/compilation targets, avoiding `PackedSequence` is generally preferred to enable static graph optimizations. The monkeypatch approach is a valid strategy for export.
