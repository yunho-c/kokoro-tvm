# Kokoro TVM

This project aims to compile and run the Kokoro TTS model using Apache TVM.

## Prerequisites

- **CMake** (`brew install cmake`)
- **LLVM** (`brew install llvm`) (optional)
- **Python**

## Build Instructions

### 1. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 2. Build TVM (Core)

```bash
mkdir -p reference/tvm/build
cd reference/tvm/build
cp ../cmake/config.cmake .

cmake ..
make -j$(sysctl -n hw.ncpu)
cd ../../..
```

### 3. Build TVM FFI (Manual)

Due to some issues on macOS with Homebrew LLVM, you might need to use the system compiler for `tvm-ffi`.

```bash
mkdir -p reference/tvm/3rdparty/tvm-ffi/build
cd reference/tvm/3rdparty/tvm-ffi/build

# Use system clang/clang++ to avoid linker errors with Homebrew LLVM
cmake .. \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DTVM_FFI_BUILD_PYTHON_MODULE=ON

make -j$(sysctl -n hw.ncpu)
cd ../../../../..
```

### 4. Install Python Packages

Install the packages in editable mode:

```bash
# Install tvm-ffi
# Note: We specify compilers again to ensure pip uses the same environment if it triggers a rebuild
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e reference/tvm/3rdparty/tvm-ffi

# Install tvm
pip install -e reference/tvm/python
```

## Usage

### VS Code Tasks

This project is configured with VS Code tasks to simplify the workflow:

1.  **Initialize TVM Submodules**: Runs the git submodule update.
2.  **Build TVM**: Runs the core TVM build.
3.  **Compile ONNX with TVM**: Runs the compilation script.

### Manual Compilation

To compile the ONNX model manually:

```bash
python scripts/compile_onnx.py
```

This will generate `build/kokoro.so` and `build/kokoro_params.npz`.
