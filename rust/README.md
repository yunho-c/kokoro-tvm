# Kokoro TVM Rust Inference

Rust-based inference pipeline for Kokoro TTS using TVM-compiled modules.

## Features

- **Pure Rust**: No Python runtime dependency for inference
- **TVM Backend**: Uses `tvm-ffi` crate for native TVM module loading
- **IPA Input**: Direct phoneme input (no G2P - bring your own)
- **Multi-device**: Supports LLVM (CPU), Metal (macOS GPU), CUDA

## Prerequisites

1. **Compiled TVM modules**: Run the Python compilation pipeline first
2. **Converted assets**: Voice packs (.npy) and vocabulary (vocab.json)
3. **Rust toolchain**: Install via [rustup](https://rustup.rs/)

## Quick Start

### 1. Convert Assets (Python)

```bash
cd python
uv run python scripts/convert_assets.py --voice af_bella --output ../assets/
```

This creates:
- `assets/vocab.json` - Character to token ID mapping
- `assets/af_bella.npy` - Voice pack in NumPy format

### 2. Build

```bash
cd rust
cargo build --release
```

### 3. Run Inference

```bash
cargo run --release -- \
  --phonemes "həlˈoʊ wˈɜːld" \
  --voice ../assets/af_bella.npy \
  --vocab ../assets/vocab.json \
  --lib-dir ../tvm_output_llvm \
  --output hello.wav \
  --speed 1.0
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --phonemes` | IPA phoneme string | (required) |
| `-v, --voice` | Path to voice pack .npy | (required) |
| `--vocab` | Path to vocab.json | (required) |
| `-l, --lib-dir` | TVM module directory | `tvm_output` |
| `-o, --output` | Output WAV path | `output.wav` |
| `-s, --speed` | Speech speed (1.0 = normal) | `1.0` |
| `--device` | Target: llvm, metal, cuda | `llvm` |

## Project Structure

```
rust/
├── Cargo.toml
├── src/
│   ├── main.rs           # CLI entry point
│   ├── lib.rs            # Public API
│   ├── pipeline.rs       # TVM inference orchestration
│   ├── vocab.rs          # Vocabulary loading
│   ├── voice.rs          # Voice pack loading
│   ├── preprocessing.rs  # Padding, masking, alignment
│   └── audio.rs          # WAV output
└── README.md
```

## Implementation Status

| Component | Status |
|-----------|--------|
| Vocab loading | ✅ Complete |
| Voice pack loading | ✅ Complete |
| Preprocessing | ✅ Complete |
| Alignment matrix | ✅ Complete |
| WAV output | ✅ Complete |
| TVM module loading | ⚠️ Needs testing |
| Tensor conversion | ⚠️ API TBD |

### Known Issues

The `tensor_to_*` helper functions in `pipeline.rs` are marked with `todo!()` 
because the exact tvm-ffi Rust API for reading tensor data is not fully 
documented. Once the crate is compiled, check the actual API and implement:

1. `Tensor::as_slice()` or similar for data access
2. `Tensor::shape()` for dimension info
3. Or use DLPack interop if direct access isn't available

See [NOTES/RUST_INFERENCE_PORT.md](../NOTES/RUST_INFERENCE_PORT.md) for details.

## License

Apache-2.0
