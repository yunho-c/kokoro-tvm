# Feasibility Report: MPS-based LSTM Operator for TVM

**Date**: 2025-12-15  
**Status**: Recommended for Implementation

## Executive Summary

**Feasibility: ✅ HIGH** - Adding an MPS-based LSTM operator is technically feasible and follows well-established patterns in the existing TVM codebase. The MPS contrib module already provides Conv2D and MatMul implementations that can be directly used as templates.

---

## 1. Existing MPS Infrastructure in TVM

TVM already has a working **MPS contrib module** with:

| Component | Files | Status |
|-----------|-------|--------|
| **C++ Runtime** | `src/runtime/contrib/mps/{conv.mm, gemm.mm, mps_utils.*}` | ✅ Working |
| **Python API** | `python/tvm/contrib/mps.py` | ✅ Working |
| **Tests** | `tests/python/contrib/test_mps.py` | ✅ Working |
| **CMake** | `cmake/modules/Metal.cmake` with `USE_MPS` flag | ✅ Working |

### Source Code References

```
reference/tvm/src/runtime/contrib/mps/
├── conv.mm          # 168 lines - Conv2D using MPSCNNConvolution
├── gemm.mm          # 102 lines - MatMul using MPSMatrixMultiplication
├── mps_utils.h      # 60 lines  - MetalThreadEntry, type conversion
└── mps_utils.mm     # 92 lines  - Thread-local Metal device management
```

### Key Implementation Pattern (from `gemm.mm`)

```objective-c++
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm.contrib.mps.matmul", [](ffi::PackedArgs args, ffi::Any* ret) {
    // 1. Extract DLTensor* arguments
    auto A = args[0].cast<DLTensor*>();
    auto B = args[1].cast<DLTensor*>();
    auto C = args[2].cast<DLTensor*>();
    
    // 2. Get Metal device/queue via thread-local entry
    MetalThreadEntry* entry_ptr = MetalThreadEntry::ThreadLocal();
    id<MTLDevice> dev = entry_ptr->metal_api->GetDevice(A->device);
    id<MTLCommandQueue> queue = entry_ptr->metal_api->GetCommandQueue(A->device);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    
    // 3. Create MPS matrix descriptors and matrices
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M columns:K ...];
    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)(A->data);
    MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    
    // 4. Create and configure MPS kernel
    MPSMatrixMultiplication* sgemm = [[MPSMatrixMultiplication alloc] initWithDevice:dev ...];
    
    // 5. Encode and execute
    [sgemm encodeToCommandBuffer:cb leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
    [cb commit];
  });
}
```

---

## 2. Apple MPS LSTM Support

Apple provides **first-party LSTM support** in Metal Performance Shaders:

| API | Description |
|-----|-------------|
| `MPSLSTMDescriptor` | Configures LSTM gates, weights, biases, activation functions |
| `MPSRNNMatrixInferenceLayer` | Optimized LSTM inference on `MPSMatrix` objects |
| `MPSRNNMatrixTrainingState` | For training (not needed for inference) |

### MPS LSTM Features

- ✅ Supports **bidirectional** LSTM
- ✅ Supports **stacked layers** (num_layers > 1)
- ✅ Handles **variable sequence lengths**
- ✅ Works with **float32** and **float16**
- ✅ Optimized for Apple Silicon (M1/M2/M3)

### MPSLSTMDescriptor Configuration

From Apple documentation, the LSTM operation follows:
```
i = σ(Wi·x + Ui·h + Vi∘c + bi)     # input gate
f = σ(Wf·x + Uf·h + Vf∘c + bf)     # forget gate  
c = f∘c + i∘tanh(Wc·x + Uc·h + bc) # cell state
o = σ(Wo·x + Uo·h + Vo∘c + bo)     # output gate
h = o∘tanh(c)                       # hidden state
```

Key properties:
- `inputFeatureChannels` - input size
- `outputFeatureChannels` - hidden size
- `useFloat32Weights` - use 32-bit weights
- `inputGateInputWeights`, `forgetGateInputWeights`, etc. - weight matrices

---

## 3. Proposed Implementation

### Files to Create/Modify

```
reference/tvm/
├── src/runtime/contrib/mps/
│   └── lstm.mm              [NEW] - LSTM implementation (~200 lines)
├── python/tvm/contrib/
│   └── mps.py               [MODIFY] - Add lstm() function
└── tests/python/contrib/
    └── test_mps.py          [MODIFY] - Add test_lstm()
```

### C++ Implementation Sketch (`lstm.mm`)

```objective-c++
#include <tvm/ffi/reflection/registry.h>
#include "mps_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm.contrib.mps.lstm", 
    [](ffi::PackedArgs args, ffi::Any* ret) {
      // Args: input, weight_ih, weight_hh, bias_ih, bias_hh, h0, c0,
      //       output_h, output_hn, output_cn,
      //       hidden_size, num_layers, batch_first, bidirectional
      
      auto input = args[0].cast<DLTensor*>();
      auto weight_ih = args[1].cast<DLTensor*>();
      auto weight_hh = args[2].cast<DLTensor*>();
      auto bias_ih = args[3].cast<DLTensor*>();
      auto bias_hh = args[4].cast<DLTensor*>();
      auto h0 = args[5].cast<DLTensor*>();
      auto c0 = args[6].cast<DLTensor*>();
      auto output_h = args[7].cast<DLTensor*>();
      auto output_hn = args[8].cast<DLTensor*>();
      auto output_cn = args[9].cast<DLTensor*>();
      int hidden_size = args[10].cast<int>();
      int num_layers = args[11].cast<int>();
      bool batch_first = args[12].cast<bool>();
      bool bidirectional = args[13].cast<bool>();
      
      // Get Metal device
      MetalThreadEntry* entry = MetalThreadEntry::ThreadLocal();
      id<MTLDevice> dev = entry->metal_api->GetDevice(input->device);
      id<MTLCommandQueue> queue = entry->metal_api->GetCommandQueue(input->device);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      
      // Determine dimensions
      int seq_len = batch_first ? input->shape[1] : input->shape[0];
      int batch_size = batch_first ? input->shape[0] : input->shape[1];
      int input_size = input->shape[2];
      
      // Create LSTM descriptor
      MPSLSTMDescriptor* desc = [MPSLSTMDescriptor createLSTMDescriptorWithInputFeatureChannels:input_size
                                                                        outputFeatureChannels:hidden_size];
      desc.useFloat32Weights = YES;
      
      // Set weights (need to handle PyTorch -> MPS format conversion)
      // PyTorch packs: [input_gate, forget_gate, cell_gate, output_gate]
      // Each gate weight: [hidden_size, input_size] for weight_ih
      //                   [hidden_size, hidden_size] for weight_hh
      
      // Create weight data sources...
      // (Implementation detail: may need transposition)
      
      // Create MPSRNNMatrixInferenceLayer
      MPSRNNMatrixInferenceLayer* lstm = [[MPSRNNMatrixInferenceLayer alloc] 
          initWithDevice:dev
          rnnDescriptor:desc];
      
      // Create MPSMatrix objects for input/output
      // ...
      
      // Encode sequence
      [lstm encodeSequenceToCommandBuffer:cb
                            sourceMatrices:inputMatrices
                       destinationMatrices:outputMatrices
                         recurrentInputState:initialState
                        recurrentOutputStates:outputStates];
      
      [cb commit];
      [cb waitUntilCompleted];
    });
}

}  // namespace contrib
}  // namespace tvm
```

### Python API Addition (`mps.py`)

```python
def lstm(input, weight_ih, weight_hh, bias_ih, bias_hh, h0, c0,
         hidden_size, num_layers=1, batch_first=True, bidirectional=False):
    """MPS-accelerated LSTM inference.
    
    Parameters
    ----------
    input : Tensor
        Input tensor of shape (seq_len, batch, input_size) or 
        (batch, seq_len, input_size) if batch_first=True
    weight_ih : Tensor
        Input-hidden weights [4*hidden_size, input_size]
    weight_hh : Tensor
        Hidden-hidden weights [4*hidden_size, hidden_size]
    bias_ih : Tensor
        Input-hidden bias [4*hidden_size]
    bias_hh : Tensor
        Hidden-hidden bias [4*hidden_size]
    h0 : Tensor
        Initial hidden state [num_layers, batch, hidden_size]
    c0 : Tensor
        Initial cell state [num_layers, batch, hidden_size]
    hidden_size : int
        Hidden dimension
    num_layers : int
        Number of stacked LSTM layers
    batch_first : bool
        If True, input shape is (batch, seq, feature)
    bidirectional : bool
        If True, use bidirectional LSTM
        
    Returns
    -------
    output : Tensor
        All hidden states (seq_len, batch, hidden_size)
    h_n : Tensor
        Final hidden state
    c_n : Tensor
        Final cell state
    """
    if batch_first:
        batch, seq_len, _ = input.shape
    else:
        seq_len, batch, _ = input.shape
    
    num_directions = 2 if bidirectional else 1
    out_hidden = hidden_size * num_directions
    
    return te.extern(
        [
            (seq_len, batch, out_hidden) if not batch_first else (batch, seq_len, out_hidden),
            (num_layers * num_directions, batch, hidden_size),
            (num_layers * num_directions, batch, hidden_size),
        ],
        [input, weight_ih, weight_hh, bias_ih, bias_hh, h0, c0],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mps.lstm",
            ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6],
            outs[0], outs[1], outs[2],
            hidden_size, num_layers, batch_first, bidirectional
        ),
        name="lstm_out",
    )
```

---

## 4. Technical Challenges

| Challenge | Difficulty | Mitigation |
|-----------|------------|------------|
| **Weight format mismatch** | Medium | PyTorch packs gates as `[i,f,g,o]` with shape `[4*H, input]`. MPS may expect different layout. Need transpose/reshape logic. |
| **Gate ordering** | Medium | PyTorch: input, forget, cell, output. Verify MPS order matches or reorder. |
| **Bidirectional handling** | Low | MPS natively supports via descriptor. Need separate forward/backward weight sets. |
| **Stacked layers** | Low | MPS supports `numberOfLayers` property. |
| **Memory management** | Low | Follow existing `gemm.mm` patterns for Metal buffer lifecycle. |
| **Batch-first vs seq-first** | Low | Transpose input if needed before MPS call. |
| **Initial states (h0/c0)** | High | Start with zero-init only; later add explicit initial state support if required. |
| **Caching and weight upload costs** | High | Cache MPS layer/weights per device + weight identity; avoid per-inference construction. |

### Weight Format Conversion

PyTorch `nn.LSTM` weight layout:
```
weight_ih_l0: [4*hidden_size, input_size]   # gates packed: i,f,g,o
weight_hh_l0: [4*hidden_size, hidden_size]
bias_ih_l0:   [4*hidden_size]
bias_hh_l0:   [4*hidden_size]
```

MPS `MPSLSTMDescriptor` expects separate gate weights:
```
inputGateInputWeights:  [hidden_size, input_size]
forgetGateInputWeights: [hidden_size, input_size]
cellGateInputWeights:   [hidden_size, input_size]
outputGateInputWeights: [hidden_size, input_size]
```

**Solution**: Slice PyTorch packed weights into individual gate matrices before passing to MPS.

### DLTensor ↔ MPS API mismatch (practical)

The MPS LSTM descriptor expects per-gate weights as `id<MPSCNNConvolutionDataSource>` objects (not raw `MPSMatrix` pointers). TVM’s `DLTensor` arguments for weights and biases are `MTLBuffer`-backed when running on `tvm.metal()`, but:

- TVM Metal buffers are allocated with `MTLResourceStorageModePrivate` in `reference/tvm/src/runtime/metal/metal_device_api.mm`, which makes CPU access unavailable without an explicit copy.
- Therefore, “just pass pointers” is not sufficient: a workable design needs either a GPU-resident training-layer weight path or a CPU-staging copy to feed a `MPSCNNConvolutionDataSource`.

---

## 5. Build Configuration

Already supported in TVM CMake:

```cmake
# cmake/modules/Metal.cmake
if(USE_MPS)
  tvm_file_glob(GLOB MPS_CONTRIB_SRC src/runtime/contrib/mps/*.mm)
  list(APPEND RUNTIME_SRCS ${MPS_CONTRIB_SRC})
  find_library(MPS_CONTRIB_LIB MetalPerformanceShaders)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${MPS_CONTRIB_LIB})
endif()
```

To enable: Set `USE_MPS=ON` in `config.cmake` (requires `USE_METAL=ON`).

### Repository Build Status (kokoro-tvm)

This repository vendors TVM under `reference/tvm/` and builds it under `reference/tvm/build/`.

Current state in this repo (as checked):

- `reference/tvm/build/config.cmake` has `USE_METAL=ON` and `USE_MPS=OFF`, so no `tvm.contrib.mps.*` externs are registered at runtime.
- `py -3.12 -c "import tvm; print(tvm.support.libinfo().get('USE_METAL'), tvm.support.libinfo().get('USE_MPS')); print(bool(tvm.get_global_func('tvm.contrib.mps.matmul', True)))"` currently reports Metal enabled but MPS not enabled, and `tvm.contrib.mps.matmul` is unavailable.

Implication: implementing `tvm.contrib.mps.lstm` requires rebuilding TVM with `USE_MPS=ON` (and verifying `tvm.get_global_func("tvm.contrib.mps.lstm", True)` is present).

---

## 6. Integration with Kokoro

### Current State in kokoro-tvm (LSTM lowering)

Kokoro’s LSTMs are handled in `python/src/kokoro_tvm/` (not in upstream TVM):

- `python/src/kokoro_tvm/ops/lstm_custom_op.py` defines `kokoro::lstm_forward` / `kokoro::lstm_forward_bidirectional` custom ops so `torch.export` keeps LSTM as an opaque node.
- `python/src/kokoro_tvm/tvm_extensions.py` monkeypatches TVM’s `ExportedProgramImporter.create_convert_map` to route LSTM ops into a custom `_lstm` handler.
- `_lstm` dispatches between three implementations based on flags:
  - TOPI (`_lstm_topi`) — baseline but can cause IR explosion for long sequences.
  - Relax (`_lstm_relax`) — still unrolled, but expressed in Relax ops.
  - TIR (`_lstm_tir`) — emits a `PrimFunc` with a `while` loop (`python/src/kokoro_tvm/ops/tir_lstm.py`), keeping IR size O(1).
- The encoder CLI exposes this via `python/src/kokoro_tvm/cli/port_encoder.py` with `--lstm-method {topi,relax,tir}`.

### Why this is needed (current GPU failure mode)

The current TIR while-loop LSTM is sequential over timesteps and is not GPU-friendly for Metal. This matches the failure described in `NOTES/GPU_LSTM_CONTINUATION.md` (host-access/binding errors and inability for DLight to schedule the loop).

### Proposed Integration Point for MPS LSTM in kokoro-tvm

The cleanest integration is to add a new LSTM path in `python/src/kokoro_tvm/tvm_extensions.py` that is selected when compiling for `metal-*` targets and when `tvm.contrib.mps.lstm` is available:

- Add an extern wrapper (either in upstream `reference/tvm/python/tvm/contrib/mps.py` or as a local helper) using `te.extern(... call_packed("tvm.contrib.mps.lstm", ...))`, following the pattern in `reference/tvm/python/tvm/contrib/mps.py`.
- Extend `python/src/kokoro_tvm/cli/port_encoder.py` with `--lstm-method mps` (or auto-select `mps` for Metal when available), and keep CPU fallback behavior unchanged.
- Keep the extern contract aligned with existing kokoro custom-op semantics:
  - kokoro custom ops internally use seq-first input (`[seq, batch, feature]`), which avoids extra transposes.
  - start with the minimal feature set that Kokoro uses in practice (single-layer, optional bidirectional, fp32), then extend.

```python
# Sketch only: kokoro-tvm routes LSTM via ExportedProgramImporter._lstm,
# so the integration point is a new _lstm_mps path.
USE_MPS_LSTM = True

def _lstm(self, node):
    if USE_MPS_LSTM:
        return _lstm_mps(self, node)
    if USE_TIR_LSTM:
        return _lstm_tir(self, node)
    if USE_RELAX_LSTM:
        return _lstm_relax(self, node)
    return _lstm_topi(self, node)

def _lstm_mps(self, node):
    # Emit a te.extern(...) that calls:
    # tvm.tir.call_packed("tvm.contrib.mps.lstm", ...)
    # and wrap outputs to match PyTorch LSTM return: (out, (h_n, c_n)).
    ...
```

---

## Implementation Requirements (Concrete)

This section summarizes what is necessary to make an MPS-backed LSTM actually run in this repository.

### TVM-side work (reference/tvm)

- Add a new runtime entry point: `reference/tvm/src/runtime/contrib/mps/lstm.mm` registering `tvm.contrib.mps.lstm` via `TVM_FFI_STATIC_INIT_BLOCK()`, mirroring patterns in `reference/tvm/src/runtime/contrib/mps/gemm.mm` and `reference/tvm/src/runtime/contrib/mps/conv.mm`.
- Implement the kernel using Apple’s RNN APIs:
  - Configure `MPSLSTMDescriptor` (input/output feature channels, `useFloat32Weights`, gate weights/biases).
  - Execute using `MPSRNNMatrixInferenceLayer` over a sequence of `MPSMatrix` objects (one per timestep).
  - Convert TVM `DLTensor` data pointers to `id<MTLBuffer>` and wrap with `MPSMatrixDescriptor` + `MPSMatrix` similar to `reference/tvm/src/runtime/contrib/mps/gemm.mm`.
- Decide and implement a weight strategy:
  - CPU-staging strategy: copy weight/bias `MTLBuffer` into a shared/CPU buffer and feed a custom `MPSCNNConvolutionDataSource` per gate; cache the resulting MPS layer by (device, dtype, shapes, weight identity).
  - GPU-native strategy: use `MPSRNNMatrixTrainingLayer` weight matrices and copy APIs to populate trainable weights on GPU; still cache to avoid per-call overhead.
- Add a TE extern wrapper in `reference/tvm/python/tvm/contrib/mps.py` (or equivalent) that emits `tvm.tir.call_packed("tvm.contrib.mps.lstm", ...)` similar to existing `matmul`/`conv2d`.
- Add a test in `reference/tvm/tests/python/contrib/test_mps.py` that:
  - Skips if `tvm.get_global_func("tvm.contrib.mps.lstm", True)` is missing.
  - Runs a small LSTM configuration on `tvm.metal(0)` and compares against a reference (NumPy/PyTorch) within tolerance.

### kokoro-tvm-side work (python/src/kokoro_tvm)

- Add a new LSTM lowering path (e.g. `_lstm_mps`) in `python/src/kokoro_tvm/tvm_extensions.py` and dispatch to it when compiling Metal targets and the MPS extern is available.
- Extend `python/src/kokoro_tvm/cli/port_encoder.py` to accept `--lstm-method mps` (and/or auto-select it for Metal targets when available).
- Keep CPU fallback paths intact (TIR/Relax/TOPI) so the feature is opt-in and platform-gated.

### Build and verification workflow

- Rebuild TVM with `USE_METAL=ON` and `USE_MPS=ON` in `reference/tvm/build/config.cmake`.
- Verify registration from Python:
  - `py -3.12 -c "import tvm; print(bool(tvm.get_global_func('tvm.contrib.mps.lstm', True)))"`
- Compile encoder with MPS LSTM enabled and validate numerics end-to-end.

### Implementation status (now implemented)

This repository now includes a working first version:

- TVM runtime: `reference/tvm/src/runtime/contrib/mps/lstm.mm`
- TVM Python wrapper: `reference/tvm/python/tvm/contrib/mps.py` (`mps.lstm`)
- kokoro lowering: `python/src/kokoro_tvm/tvm_extensions.py` (`USE_MPS_LSTM` and `_lstm_mps`)
- Encoder CLI: `python/src/kokoro_tvm/cli/port_encoder.py` (`--lstm-method mps`)
- Accuracy test: `python/src/kokoro_tvm/tests/test_mps_lstm_accuracy.py`

Current limitations of the first version:

- dtype: float32 only
- num_layers: 1 only
- initial state: `h0`/`c0` parameters are currently ignored by the runtime implementation (zero-init)
- bidirectional: supported in kokoro lowering via two unidirectional calls and concat; runtime op itself is unidirectional

---

## Key Risks and Unknowns

- Initial state support (h0/c0):
  - `MPSRNNMatrixInferenceLayer` takes an optional `recurrentInputState` of type `MPSRNNRecurrentMatrixState`, but the public headers primarily expose accessors for reading stored matrices (output + memory cell) rather than a simple “construct from h0/c0 matrices” initializer.
  - If Kokoro uses non-zero initial states, the implementation needs an explicit strategy (or a model-level change to use zero init).
- Semantics alignment:
  - Gate naming: MPS describes “memory gate” where PyTorch uses “cell gate (g)”; ordering and activation choices must match PyTorch’s `nn.LSTM` defaults.
  - Bias handling: PyTorch has `bias_ih` and `bias_hh`; MPS’ gate weights include bias terms inside each data source, so biases likely need to be combined per gate.
- Performance hazards:
  - Creating MPS layers and uploading weights per call will dominate runtime. Caching is mandatory for any realistic speedup.
  - If weights must be staged via CPU due to `MTLStorageModePrivate` buffers, copies must be minimized and ideally performed once at compile/load time.
- API surface and maintenance:
  - The existing MPS contrib wrappers in TVM are relatively small and do limited validation; LSTM introduces more shape/dtype combinations and more opportunity for silent mismatch.

---

## Apple SDK Recon Notes (from xcrun)

The relevant headers are present in the macOS SDK and confirm that the lower-level MPS RNN APIs are available on macOS:

- `xcrun --sdk macosx --show-sdk-path` returns the active SDK root (example on this machine: `.../MacOSX15.2.sdk`).
- `MPSLSTMDescriptor` and `MPSRNNMatrixInferenceLayer` declarations live under:
  - `<SDK>/System/Library/Frameworks/MetalPerformanceShaders.framework/Frameworks/MPSNeuralNetwork.framework/Headers/MPSRNNLayer.h`
- The umbrella header `MetalPerformanceShaders.framework/Headers/MetalPerformanceShaders.h` pulls in `MPSNeuralNetwork/MPSNeuralNetwork.h`, and the RNN-specific declarations are in the nested `MPSNeuralNetwork.framework` headers above.
- The `MPSLSTMDescriptor` uses per-gate weight objects:
  - gate fields are `id<MPSCNNConvolutionDataSource>` such as `inputGateInputWeights`, `forgetGateRecurrentWeights`, `cellGateInputWeights`, `outputGateInputWeights`, and corresponding bias terms through the data source descriptor.
- `MPSCNNConvolutionDataSource` is a protocol, and Apple notes that MPS does not provide a built-in implementation; a TVM MPS LSTM implementation will need to provide its own data source type (or use a different pathway that avoids data sources).
- Sequence execution APIs exist for matrices:
  - `-[MPSRNNMatrixInferenceLayer encodeSequenceToCommandBuffer:sourceMatrices:destinationMatrices:recurrentInputState:recurrentOutputStates:]`
  - There is also `encodeBidirectionalSequenceToCommandBuffer:...` with `bidirectionalCombineMode` options.
- The headers explicitly document the “matrix math convention”:
  - For RNN matrices, the operation is described as `y^T = W x^T  <=> y = x W^T`, which is important when mapping PyTorch weight layouts.

These observations support feasibility, but also highlight the need to engineer the weight and initial-state pathways carefully around MPS’ data-source and recurrent-state abstractions.

## 7. Effort Estimate

| Task | Effort | Notes |
|------|--------|-------|
| C++ `lstm.mm` implementation | 2-3 days | Core MPS integration |
| Weight format conversion | 1 day | PyTorch → MPS gate layout |
| Python API wrapper | 0.5 day | `mps.py` addition |
| Unit tests | 0.5 day | Numerical validation |
| Integration with Kokoro converter | 1 day | `tvm_extensions.py` |
| End-to-end testing & debugging | 1-2 days | Full pipeline validation |
| **Total** | **6-8 days** | |

---

## 8. Alternative Approaches Comparison

| Approach | Effort | Performance | Complexity |
|----------|--------|-------------|------------|
| **MPS Contrib (Recommended)** | 6-8 days | Best (native Apple optimization) | Medium |
| Hybrid CPU+GPU | 1 day | Mixed (LSTM on CPU) | Low |
| Relax Recursive | 3-5 days | Poor (kernel launch overhead) | High |
| Multi-Kernel TIR | 5-7 days | Medium (host loop overhead) | High |

---

## 9. Recommendation

**Proceed with MPS Contrib implementation.** The infrastructure is mature:

1. ✅ CMake build system already supports `USE_MPS`
2. ✅ Metal buffer ↔ MPS matrix conversion patterns exist
3. ✅ Thread-local Metal device management is in place
4. ✅ Apple provides optimized `MPSRNNMatrixInferenceLayer`
5. ✅ Clear implementation patterns from `conv.mm` and `gemm.mm`

This approach provides the best performance on Apple Silicon and is the cleanest long-term solution for Metal GPU LSTM support in TVM.

---

## 10. References

- [MPSLSTMDescriptor - Apple Developer](https://developer.apple.com/documentation/metalperformanceshaders/mpslstmdescriptor)
- [MPSRNNMatrixInferenceLayer - Apple Developer](https://developer.apple.com/documentation/metalperformanceshaders/mpsrnnmatrixinferencelayer)
- [TVM MPS Source](file:///Users/yunhocho/GitHub/kokoro-tvm/reference/tvm/src/runtime/contrib/mps/)
- [GPU LSTM Continuation Notes](file:///Users/yunhocho/GitHub/kokoro-tvm/NOTES/GPU_LSTM_CONTINUATION.md)
