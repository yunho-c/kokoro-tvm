# Rust TVM FFI: Findings and Limitations

Status: **Work in Progress** | Date: 2024-12-18

## Overview

This document captures learnings from attempting to build a native Rust inference
pipeline for Kokoro TTS using the `tvm-ffi` crate and TVM-compiled Relax modules.

## TVM Architecture (Post-Unity)

TVM Unity introduces a new architecture with three key libraries:

| Library | Size | Purpose |
|---------|------|---------|
| `libtvm_ffi.dylib` | ~650KB | Lightweight FFI layer - minimal, no Relax |
| `libtvm_runtime.dylib` | ~4.5MB | Runtime for executing compiled models |
| `libtvm.dylib` | ~65MB | Full compiler + runtime |

The `tvm-ffi` Rust crate (0.1.0-alpha.0) links against `libtvm_ffi.dylib`.

## Key Findings

### 1. Relax VM Requires Full Runtime

Kokoro modules are compiled using **TVM Relax IR**, resulting in `.so` files with
format `{relax.VMExecutable}`. Loading these requires:

```python
# Python approach (works)
lib = tvm.runtime.load_module("bert_compiled.so")
vm = tvm.relax.VirtualMachine(lib, tvm.cpu())
func = vm["bert_forward"]
```

The `relax.VirtualMachine` constructor is registered in `libtvm.dylib` or
`libtvm_runtime.dylib`, but **NOT** in `libtvm_ffi.dylib`.

### 2. tvm-ffi Has Separate Global Registry

The `tvm-ffi` crate's global function registry (`Function::get_global()`) is
**separate** from the registry used by `libtvm.dylib`:

```rust
// This fails even after loading libtvm.dylib
let vm_fn = tvm_ffi::Function::get_global("relax.VirtualMachine")?;
// Error: Function relax.VirtualMachine not found
```

Loading `libtvm.dylib` via `libloading` doesn't merge registrations because:
- Each library has its own static registry
- Symbol loading doesn't trigger cross-library registration
- TVM Unity's design separates FFI layer from runtime

### 3. Module Loading Works, VM Wrapping Doesn't

```rust
// This works - the Relax deserializer IS registered
let lib = tvm_ffi::Module::load_from_file("bert_compiled.so")?;

// This fails - relax.VirtualMachine not in tvm-ffi registry
let vm_fn = tvm_ffi::Function::get_global("relax.VirtualMachine")?;
let vm = vm_fn.call_tuple((&lib, device_type, device_id))?;

// This also fails - direct function access returns None
let func = lib.get_function("bert_forward")?;
// Error: Cannot convert from type `None` to `ffi.Function`
```

### 4. tvm-ffi API Notes

#### Working APIs
```rust
// Load a module (triggers Relax deserialization via libtvm_runtime)
let module = tvm_ffi::Module::load_from_file("module.so")?;

// Create CPU tensors from slices
let tensor = tvm_ffi::Tensor::from_slice(&data, &[1, 512])?;

// Call functions with tuples
func.call_tuple((&input, &output))?;
```

#### Device tensors and data movement (CPU <-> GPU)

The `tvm-ffi` Rust crate exposes CPU-only helpers like `Tensor::from_slice` and
`Tensor::data_as_slice`, which assume CPU-contiguous memory. To work with GPU
devices (Metal/CUDA), you need to allocate device tensors explicitly and copy
data in/out using runtime functions registered by `libtvm_runtime`:

```rust
// Allocate a device tensor (works for Metal/CUDA)
let alloc = tvm_ffi::Function::get_global("runtime.TVMTensorAllocWithScope")?;
let shape = tvm_ffi::Shape::from(&[1, 512][..]);
let mem_scope: Option<tvm_ffi::String> = None;
let tensor_any = alloc.call_tuple((shape, dtype, device, mem_scope))?;
let tensor: tvm_ffi::Tensor = tensor_any.try_into()?;

// Copy host data into device tensor
let copy_from = tvm_ffi::Function::get_global("runtime.TVMTensorCopyFromBytes")?;
copy_from.call_tuple((&tensor, host_ptr, nbytes))?;

// Copy device data back to host
let copy_to = tvm_ffi::Function::get_global("runtime.TVMTensorCopyToBytes")?;
copy_to.call_tuple((&tensor, host_ptr, nbytes))?;
```

Notes:
- `Tensor::data_as_slice()` is CPU-only; it fails on non-CPU devices.
- These runtime copy helpers require `libtvm_runtime` (or `libtvm`) to be loaded.
- The copy APIs expect contiguous buffers and exact byte sizes; mismatches error out.
- `call_tuple` does not accept `DLDataType`, `DLDevice`, `Option<String>`, or raw pointers;
  use `Function::call_packed` with `AnyView` for these arguments.

Example `call_packed` usage:

```rust
let alloc = tvm_ffi::Function::get_global("runtime.TVMTensorAllocWithScope")?;
let shape = tvm_ffi::Shape::from(&[1, 512][..]);
let mem_scope: Option<tvm_ffi::String> = None;
let alloc_args = [
    tvm_ffi::AnyView::from(&shape),
    tvm_ffi::AnyView::from(&dtype),
    tvm_ffi::AnyView::from(&device),
    tvm_ffi::AnyView::from(&mem_scope),
];
let tensor_any = alloc.call_packed(&alloc_args)?;

let copy_from = tvm_ffi::Function::get_global("runtime.TVMTensorCopyFromBytes")?;
let copy_args = [
    tvm_ffi::AnyView::from(&tensor),
    tvm_ffi::AnyView::from(&host_ptr),
    tvm_ffi::AnyView::from(&nbytes),
];
copy_from.call_packed(&copy_args)?;
```

#### Non-Working APIs for Relax
```rust
// Global function lookup doesn't find Relax-registered functions
tvm_ffi::Function::get_global("relax.VirtualMachine") // Error

// Direct function access on Relax modules returns None
module.get_function("bert_forward") // None
```

### 5. Error Handling Quirk

`tvm_ffi::Error` doesn't implement `Send + Sync`, so it can't be used with
`anyhow::Context`. Workaround:

```rust
macro_rules! tvm_err {
    ($expr:expr, $msg:expr) => {
        $expr.map_err(|e| anyhow::anyhow!("{}: {:?}", $msg, e))
    };
}
```

## Build Requirements

### Compile Time
```bash
export LIBRARY_PATH="/path/to/tvm-ffi/build/lib:$LIBRARY_PATH"
cargo build --release
```

### Run Time
```bash
export DYLD_LIBRARY_PATH="/path/to/tvm/build:/path/to/tvm-ffi/build/lib:$DYLD_LIBRARY_PATH"
./target/release/kokoro-tvm ...
```

### Submodule Setup
If `libtvm_ffi.dylib` needs rebuilding:
```bash
cd reference/tvm/3rdparty/tvm-ffi
git submodule update --init --recursive  # For libbacktrace
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++
make -j8
```

## Potential Solutions

### Option 1: Build Unified TVM + tvm-ffi
Build TVM with `libtvm_ffi` that shares the global registry with `libtvm_runtime`.
Requires modifying TVM's CMake configuration.

### Option 2: Use TVM's C API Directly
Bypass `tvm-ffi` crate and call TVM's C API via `bindgen`:
```rust
extern "C" {
    fn TVMFuncGetGlobal(name: *const c_char, out: *mut TVMFunctionHandle) -> c_int;
}
```

### Option 3: PyO3 Python Bindings
Call Python's TVM from Rust:
```rust
use pyo3::prelude::*;

Python::with_gil(|py| {
    let tvm = py.import("tvm")?;
    let relax = tvm.getattr("relax")?;
    // ...
});
```

### Option 4: Wait for tvm-ffi Maturity
The crate is `0.1.0-alpha.0`. Future versions may add Relax VM support.

### Option 5: Non-Relax Compilation
Recompile Kokoro modules without Relax (use `tvm.build()` instead of
`relax.build()`). However, this loses dynamic shape support.

## Current State

The Rust project compiles and:
- ✅ Loads vocabulary JSON
- ✅ Loads voice pack NPY files (handles 3D→2D squeeze)
- ✅ Tokenizes IPA phonemes
- ✅ Loads `libtvm.dylib` / `libtvm_runtime.dylib` dynamically
- ✅ Loads compiled `.so` modules (Relax deserializer works)
- ❌ Cannot wrap modules with `relax.VirtualMachine`
- ❌ Cannot access Relax entry point functions

## Files Modified

- `rust/Cargo.toml` - Added `libloading` dependency
- `rust/src/pipeline.rs` - VirtualMachine wrapper approach (blocked)
- `rust/src/voice.rs` - Handle 3D→2D array conversion

## References

- [tvm-ffi Rust Guide](https://tvm.apache.org/ffi/guides/rust_guide.html)
- [TVM Relax VM](https://tvm.apache.org/docs/reference/api/python/relax/vm.html)
- [tvm-ffi crate](https://crates.io/crates/tvm-ffi)
