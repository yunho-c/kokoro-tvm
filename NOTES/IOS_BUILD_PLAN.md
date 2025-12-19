# iOS Build Plan (TVM + Rust Pipeline)

This note sketches the steps to run the Rust pipeline on iOS using Metal.
It focuses on static linking and app-bundle deployment to avoid `dlopen`
restrictions on iOS.

## Overview

- Build TVM + tvm-ffi for iOS (arm64) with Metal enabled.
- Compile Kokoro Relax modules for Metal iOS targets.
- Link TVM/tvm-ffi statically into the Rust crate.
- Load model artifacts from the app bundle.

## 1) Build TVM for iOS + Metal

### Prereqs

- Xcode + iOS SDK
- CMake
- Rust toolchain with iOS target (`aarch64-apple-ios`)

### CMake config (example)

```bash
mkdir -p reference/tvm/build-ios
cd reference/tvm/build-ios

cmake .. \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_METAL=ON \
  -DUSE_MPS=ON \
  -DUSE_LLVM=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_OPENCL=OFF \
  -DUSE_VULKAN=OFF \
  -DUSE_RPC=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DTVM_FFI_USE_LIBBACKTRACE=OFF \
  -DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF
# NOTE: backtrace related options may not work

cmake --build . --config Release -j8
```

Outputs: static `libtvm_runtime.a` (and possibly `libtvm.a` if enabled).

## 2) Build tvm-ffi for iOS

If you depend on the tvm-ffi C layer, build it for iOS as well.
In this repo it lives under `reference/tvm/3rdparty/tvm-ffi`.

```bash
mkdir -p reference/tvm/3rdparty/tvm-ffi/build-ios
cd reference/tvm/3rdparty/tvm-ffi/build-ios

cmake .. \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DTVM_FFI_USE_LIBBACKTRACE=OFF \
  -DTVM_FFI_BACKTRACE_ON_SEGFAULT=OFF
# NOTE: backtrace related options may not work

cmake --build . --config Release -j8

```

Outputs: static `libtvm_ffi.a`.

## 3) Compile Relax modules for iOS Metal

The compiled modules must match the target triple and Metal backend.
Use a target similar to `metal -mtriple=arm64-apple-ios`.

Example (Python CLI in this repo):

```bash
py -3.12 python/src/kokoro_tvm/cli/compile_kokoro.py \
  --target "metal -mtriple=arm64-apple-ios"
```

Place the resulting `.dylib` or `.so` in your iOS bundle resources
(or convert to a format your loader expects).

## 4) Rust toolchain + linking

### Add iOS target

```bash
rustup target add aarch64-apple-ios
```

### Static linking via build.rs

Create `rust/build.rs` that points to the iOS static libs:

```rust
fn main() {
    println!("cargo:rustc-link-search=native=../reference/tvm/build-ios");
    println!("cargo:rustc-link-search=native=../reference/tvm/3rdparty/tvm-ffi/build-ios");
    println!("cargo:rustc-link-lib=static=tvm_runtime");
    println!("cargo:rustc-link-lib=static=tvm_ffi");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
```

Environment overrides:

- `TVM_BUILD_DIR`: path to the TVM build output directory (default: `reference/tvm/build-ios`)
- `TVM_FFI_BUILD_DIR`: path to the tvm-ffi build output directory (default: `reference/tvm/3rdparty/tvm-ffi/build-ios`)
- `KOKORO_TVM_LINK_TVM=1`: enable linking TVM on non-iOS targets (disabled by default)

### Disable `libloading` on iOS

`libloading`/`dlopen` is restricted on iOS. Use conditional compilation:

```rust
#[cfg(not(target_os = "ios"))]
fn init_relax_runtime() -> Result<()> { ... }

#[cfg(target_os = "ios")]
fn init_relax_runtime() -> Result<()> { Ok(()) }
```

## 5) Bundle model artifacts

Place compiled artifacts in the app bundle (e.g., `Resources/models`).
The Rust pipeline should accept a bundle path (or a list of paths) rather
than relying on the working directory.

## 6) Runtime data movement

Metal requires device tensors and host/device copies. The Rust pipeline
already uses `runtime.TVMTensorAllocWithScope`, `runtime.TVMTensorCopyFromBytes`,
and `runtime.TVMTensorCopyToBytes` via `call_packed`.

Make sure all:
- inputs are created on the Metal device
- outputs are copied back to CPU before reading

## 7) Build the Rust lib for iOS

```bash
cd rust
cargo build --release --target aarch64-apple-ios
```

You can package the resulting static library into an Xcode project or
an xcframework.

## Notes

- Ensure `STATIC_TEXT_LEN`, `STATIC_AUDIO_LEN`, and `STYLE_DIM` match
  the compiled model shapes.
- On iOS you should not use `libloading` or other runtime dynamic loading.
- Audio output should write to an app-writable directory, not the repo root.
