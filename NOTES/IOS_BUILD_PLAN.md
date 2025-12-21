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
  -DBUILD_STATIC_RUNTIME=ON \
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
# py -3.12 python/src/kokoro_tvm/cli/compile_kokoro.py \
#   --target "metal -mtriple=arm64-apple-ios"

py -3.12 python/src/kokoro_tvm/cli/port_encoder.py --component "all" --target "metal-ios" --lstm-method mps --output-dir ktvm_ios --
seq-len 512 --aligned-len 512

# py -3.12 python/src/kokoro_tvm/cli/port_decoder.py --target "metal-ios" --output-dir ktvm_ios --seq-len 512
py -3.12 python/src/kokoro_tvm/cli/port_decoder.py --target "metal-ios" --output-dir ktvm_ios --seq-len "128,256,512,1024"
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
    println!("cargo:rustc-link-search=native=../reference/tvm/build-ios/lib");
    println!("cargo:rustc-link-search=native=../reference/tvm/3rdparty/tvm-ffi/build-ios/lib");
    println!("cargo:rustc-link-lib=static=tvm_runtime");
    println!("cargo:rustc-link-lib=static=tvm_ffi_static");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
```

#### Force-load for iOS

Even when you link static tvm-ffi, iOS dead-stripping can remove the
registration code that defines `ffi.ModuleLoadFromFile`. If you see:

```
RuntimeError: Function ffi.ModuleLoadFromFile not found
```

you likely need `-Wl,-force_load` for the static archives:

```rust
println!("cargo:rustc-link-arg=-Wl,-force_load,/abs/path/libtvm_runtime.a");
println!("cargo:rustc-link-arg=-Wl,-force_load,/abs/path/libtvm_ffi_static.a");
```

### Dynamic linking (optional, more complex)

If you choose `.dylib` linking instead of static archives, you must:
- link against `libtvm_ffi.dylib`/`libtvm_runtime.dylib` at build time
- embed an `rpath` so dyld can find those dylibs at runtime
- bundle the dylibs in `Frameworks/` and ensure code signing

The current `rust/build.rs` only configures link search paths and
`-l` flags by default; enable dynamic linking on iOS by setting:

```
set -x KOKORO_TVM_IOS_DYNAMIC 1
```

When this flag is set, `rust/build.rs` emits `-rpath` entries for
`@executable_path/Frameworks` and `@loader_path/Frameworks`, but you
still need to bundle the dylibs in `Frameworks/` and ensure they are
signed. Dynamic linking requires extra Xcode or build steps to copy
the libraries into the app bundle.

Environment overrides:

- `TVM_BUILD_DIR`: path to the TVM build output directory (default: `reference/tvm/build-ios/lib`)
- `TVM_FFI_BUILD_DIR`: path to the tvm-ffi build output directory (default: `reference/tvm/3rdparty/tvm-ffi/build-ios/lib`)
- `KOKORO_TVM_LINK_TVM=1`: enable linking TVM on non-iOS targets (disabled by default)

### Disable `libloading` on iOS

`libloading`/`dlopen` is restricted on iOS. Use conditional compilation:

```rust
#[cfg(not(target_os = "ios"))]
fn init_relax_runtime() -> Result<()> { ... }

#[cfg(target_os = "ios")]
fn init_relax_runtime() -> Result<()> { Ok(()) }
```

This means iOS expects the TVM runtime to be *already loaded* at process
startup (via static linking or a linked dylib). If the runtime isn't
present, `ffi.ModuleLoadFromFile` will be missing.

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
set -x LIBRARY_PATH "/Users/yunhocho/GitHub/kokoro-tvm/reference/tvm/3rdparty/tvm-ffi/build/lib" $LIBRARY_PATH
cargo build --release --target aarch64-apple-ios --features frb
```

You can package the resulting static library into an Xcode project or
an xcframework.

## Flutter + rust-flutter-bridge checklist

Use the same TVM/tvm-ffi build steps, then integrate into the Flutter iOS app.

- Build Rust as a static library or XCFramework (arm64 + simulator slices).
- Ensure the iOS Runner target links:
  - `libtvm_runtime.a`
  - `libtvm_ffi.a`
  - Metal / MetalPerformanceShaders / Foundation frameworks
- Keep the `#[cfg(target_os = "ios")]` guard for `libloading`.
- Bundle model artifacts in the iOS app resources and pass the bundle path
  from Swift/Dart into Rust (via rust-flutter-bridge).
- If you use a Podspec, add the TVM static libs to `vendored_libraries`
  and the Metal frameworks to `s.frameworks`.

## Notes

- Ensure `STATIC_TEXT_LEN`, `STATIC_AUDIO_LEN`, and `STYLE_DIM` match
  the compiled model shapes.
- On iOS you should not use `libloading` or other runtime dynamic loading.
- Audio output should write to an app-writable directory, not the repo root.
