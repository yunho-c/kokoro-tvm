# Kokoro TVM

This project aims to compile and run the Kokoro TTS model using Apache TVM.

## Prerequisites

- **CMake**
- **LLVM** (Required for CPU code generation in TVM)
- **Python**

### Installing LLVM

#### Linux/Ubuntu

Install LLVM from the default Ubuntu repository:

```bash
sudo apt-get update
sudo apt-get install -y llvm clang libclang-dev
```

This will install the default LLVM version for your Ubuntu release (LLVM 15 on Ubuntu 22.04, LLVM 18 on Ubuntu 24.04). TVM works well with LLVM versions 11-18.

#### macOS

```bash
brew install cmake llvm
```

## Build Instructions

### 1. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 2. Build TVM (Core)

**IMPORTANT:** You must enable LLVM support in the TVM configuration before building. LLVM is required for CPU code generation.

```bash
mkdir -p reference/tvm/build
cd reference/tvm/build
cp ../cmake/config.cmake .

# Edit config.cmake to enable LLVM
# Change: set(USE_LLVM OFF)
# To:     set(USE_LLVM ON)
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' config.cmake

# For Linux: if you have a specific llvm-config, you can specify it:
# set(USE_LLVM /usr/bin/llvm-config-17)

cmake ..
make -j$(nproc)  # Linux
# make -j$(sysctl -n hw.ncpu)  # macOS
cd ../../..
```

### 3. Build TVM FFI

**Prerequisites:**
Ensure `cython` is installed:
```bash
pip install cython
```

#### Linux

```bash
mkdir -p reference/tvm/3rdparty/tvm-ffi/build
cd reference/tvm/3rdparty/tvm-ffi/build

# Use GCC to avoid libstdc++ linking issues
cmake .. \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DTVM_FFI_BUILD_PYTHON_MODULE=ON

make -j$(nproc)
cd ../../../../..
```

#### macOS

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

#### Linux

```bash
# Install tvm-ffi
pip install -e reference/tvm/3rdparty/tvm-ffi

# Install tvm
pip install -e reference/tvm/python
```

#### macOS

```bash
# Install tvm-ffi
# Note: We specify compilers to ensure pip uses the same environment if it triggers a rebuild
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
