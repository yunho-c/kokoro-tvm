### Expected behavior

Building TVM with Metal Performance Shaders enabled (`USE_MPS=ON`) should compile and link successfully on macOS/arm64.

### Actual behavior

Previously, the build failed during compilation of the MPS runtime sources:

- `reference/tvm/src/runtime/contrib/mps/conv.mm`
- `reference/tvm/src/runtime/contrib/mps/gemm.mm`
- `reference/tvm/src/runtime/contrib/mps/mps_utils.mm`

Representative errors:

```
error: 'CopyDataFromTo' is a protected member of 'tvm::runtime::metal::MetalWorkspace'
error: no member named 'GetCommandQueue' in 'tvm::runtime::metal::MetalWorkspace'
error: no member named 'IsContiguous' in namespace 'tvm::ffi'
error: no viable conversion from 'Any' to 'void *'
```

With `USE_MPS=OFF`, the build succeeded because these MPS sources were not compiled.

### Current status (resolved in this repository)

This repository now builds TVM with `USE_MPS=ON` and the MPS contrib runtime compiles and runs:

- `tvm.get_global_func("tvm.contrib.mps.matmul", True)` is present and a small numerical sanity check matches NumPy.
- The repo also adds `tvm.contrib.mps.lstm` (see `reference/tvm/src/runtime/contrib/mps/lstm.mm`) and a corresponding wrapper `reference/tvm/python/tvm/contrib/mps.py:mps.lstm`.

### Environment

- OS: macOS 15.6 (24G84)
- Arch: arm64 (`arm64-apple-darwin24.6.0`)
- Compiler: Apple clang 16.0.0
- CMake: 4.2.0
- TVM commit: `6248b5db43505fbcfb13cc289d11877d5d2649e8`
- `tvm-ffi` submodule: `ae346ec92a3c386f1376064ae086aae72947c329`

### Steps to reproduce

- Enable MPS in `reference/tvm/cmake/config.cmake`:
  - Set `set(USE_MPS ON)`
- Configure and build:

```bash
cd reference/tvm/build
cmake ..
make -j"$(sysctl -n hw.ncpu)"
```

If you add new `.mm` files under `src/runtime/contrib/mps/`, CMake may report a “GLOB mismatch”. Rerun `cmake ..` in `reference/tvm/build` to regenerate build files, then rebuild.

### Triage

* needs-triage
* type: bug
* device: metal
* area: runtime

### Root cause analysis

The MPS contrib runtime (`reference/tvm/src/runtime/contrib/mps/*.mm`) was written against older TVM runtime and tvm-ffi APIs. Recent refactors changed the accessible/public API surface in a few key ways:

- `device_api.metal` is registered as a `tvm::ffi` global function and returns an `ffi::Any`. The MPS code previously assumed the call expression was implicitly convertible to `void*`, which no longer holds. The correct usage is `get_metal_api().cast<void*>()`.
- `tvm::runtime::metal::MetalWorkspace::CopyDataFromTo(...)` is a `protected` override of the `DeviceAPI` implementation detail. Calling it directly from contrib code is no longer permitted, which is why `conv.mm` fails with “protected member”.
- The Metal runtime no longer exposes a public `GetCommandQueue(...)` API on `MetalWorkspace`. Instead, it routes execution through streams (`Stream`) and `GetCurrentStream(...)`/`CastStreamOrGetDefault(...)`, and command buffers are created from the stream (`Stream::GetCommandBuffer(...)`).
- `tvm::ffi::IsContiguous` is defined in `tvm/ffi/container/tensor.h`. The MPS sources referenced `ffi::IsContiguous` but did not include the header that declares it, so the name lookup failed.

Net effect: enabling `USE_MPS` compiles code that’s out of sync with the current Metal runtime + tvm-ffi interfaces.

### Proposed solutions

Update the MPS contrib implementation to use only public, stable-ish runtime interfaces:

- Obtain the Metal workspace pointer via `ffi::Any` cast:
  - `void* ret = get_metal_api().cast<void*>();`
- Replace any use of the removed `GetCommandQueue(...)` with stream-based command buffer creation:
  - `Stream* s = metal_ws->CastStreamOrGetDefault(metal_ws->GetCurrentStream(dev), dev.device_id);`
  - `id<MTLCommandBuffer> cb = s->GetCommandBuffer("label");`
- Avoid calling protected `MetalWorkspace::CopyDataFromTo(...)` from contrib code:
  - For MTLBuffer-to-MTLBuffer copies, encode an explicit `MTLBlitCommandEncoder` copy on a command buffer.
  - For CPU-visible staging, use `runtime::metal::MetalThreadEntry::GetTempBuffer(...)` (shared storage) and then `memcpy`/`readBytes`/`writeBytes` as appropriate.
- Include the correct header for contiguity checks:
  - Add `#include <tvm/ffi/container/tensor.h>`

Alternative approaches (less ideal):

- Reintroduce compatibility shims in the Metal runtime (e.g., a public wrapper for copy and command-queue access). This increases API surface and encourages new callers to depend on internal mechanisms.
- Pin to a TVM commit where the existing MPS sources still compile, but this blocks upgrades and diverges from current runtime semantics.
- Keep `USE_MPS=OFF` and rely on Metal only, if MPS is not required.

### Implemented fix in this repository

This repository updates the MPS sources to align with current TVM + tvm-ffi:

- `reference/tvm/src/runtime/contrib/mps/mps_utils.mm` uses `get_metal_api().cast<void*>()`.
- `reference/tvm/src/runtime/contrib/mps/gemm.mm` uses Metal streams to create command buffers.
- `reference/tvm/src/runtime/contrib/mps/conv.mm` replaces direct calls to `MetalWorkspace::CopyDataFromTo` with explicit `MTLBlitCommandEncoder` copies on stream-backed command buffers.
- `reference/tvm/src/runtime/contrib/mps/mps_utils.h` includes `tvm/ffi/container/tensor.h` so `ffi::IsContiguous` resolves.

Validation:

- Building in `reference/tvm/build` with `USE_MPS=ON` completes successfully (`make -j$(sysctl -n hw.ncpu)`).
- `py -3.12 -c "import tvm; print(bool(tvm.get_global_func('tvm.contrib.mps.matmul', True)))"` prints `True`.
