# Rust TVM Inference Port

This document details the architecture and implementation notes for porting the Kokoro TVM inference pipeline from Python to Rust.

## Overview

The Rust port aims to provide a native, high-performance TTS inference implementation that:
- Loads pre-compiled TVM modules (.so/.dylib files)
- Takes IPA phoneme input directly (no G2P)
- Uses pre-converted voice packs (.npy format)
- Outputs WAV audio files

## Architecture Comparison

### Python Pipeline (`pipeline.py`)
```
IPA Text → text_to_ids() → input_ids [1, seq_len]
                              ↓
            +------------------------------------------+
            |            KokoroPipeline                |
            |------------------------------------------|
            | BERT: input_ids, mask → d_en             |
            | Duration: d_en, style → duration, d      |
            | Alignment: Python/NumPy logic            |
            | F0N: en, style → F0, N                   |
            | TextEncoder: input_ids → t_en            |
            | Decoder: asr, F0, N, style → audio       |
            +------------------------------------------+
                              ↓
                        WAV output
```

### Rust Pipeline (target)
```
IPA Text → phonemes_to_ids() → input_ids [1, seq_len]
                              ↓
            +------------------------------------------+
            |            KokoroPipeline (Rust)         |
            |------------------------------------------|
            | Same TVM modules via tvm-ffi             |
            | Alignment logic in pure Rust (ndarray)   |
            +------------------------------------------+
                              ↓
                        WAV output (hound)
```

## Key Data Structures

### Vocabulary
- JSON map: `{ "a": 1, "b": 2, ... }`
- Special tokens: `0` = start/end token
- Source: `hexgrad/Kokoro-82M` config.json → `vocab` field

### Voice Pack
- Original format: PyTorch `.pt` file, shape `[N, 256]` where N varies by pack (~500+ entries)
- Converted format: NumPy `.npy` file, shape `[N, 256]`, dtype float32
- Selection logic: `ref_s = pack[min(phoneme_len - 1, pack.shape[0] - 1)]`
- Split: `style_128 = ref_s[:, :128]`, `style_s = ref_s[:, 128:]`

### Static Constants (must match compilation!)
```rust
const STATIC_TEXT_LEN: usize = 512;
const STATIC_AUDIO_LEN: usize = 5120;
const STYLE_DIM: usize = 128;
const SAMPLES_PER_FRAME: usize = 600;  // For audio trimming
```

## TVM Module Signatures

| Module | Function | Inputs | Outputs |
|--------|----------|--------|---------|
| bert_compiled | `bert_forward` | `(input_ids[1,512] int64, attention_mask[1,512] int64)` | `Array(d_en[1,512,512] float32)` |
| duration_compiled | `duration_forward` | `(d_en, style[1,128] float32, lengths[1] int64, mask[1,512] **bool**)` | `(duration[1,512,bins], d[1,512,640])` |
| f0n_compiled | `f0n_forward` | `(en[1,640,audio_len], style[1,128])` | `(F0[1,audio_len*2], N[1,audio_len*2])` |
| text_encoder_compiled | `text_encoder_forward` | `(input_ids[1,512], lengths[1], mask[1,512])` | `t_en[1,512,512]` |
| decoder_compiled | `decoder_forward` | `(asr[1,512,audio_len], F0[1,audio_len*2], N[1,audio_len*2], style[1,128])` | `audio[1,audio_len*600]` |

> **Important**: The `mask` parameter for duration_forward expects `dtype=bool`, not `int8`. This is a common source of errors.

## Alignment Algorithm (Pure Rust Implementation)

```rust
// 1. Compute predicted durations from logits
let duration_probs = sigmoid(duration_logits);  // [1, seq_len, bins]
let duration_sum = duration_probs.sum_axis(Axis(2)) / speed;  // [1, seq_len]
let pred_dur: Vec<i64> = duration_sum
    .iter()
    .map(|&x| x.round().max(1.0) as i64)
    .take(cur_len)
    .collect();

// 2. Build alignment matrix
let indices: Vec<usize> = pred_dur.iter()
    .enumerate()
    .flat_map(|(i, &dur)| std::iter::repeat(i).take(dur as usize))
    .collect();

let actual_audio_len = indices.len().min(STATIC_AUDIO_LEN);

let mut pred_aln_trg = Array2::<f32>::zeros((cur_len, STATIC_AUDIO_LEN));
for (frame_idx, &text_idx) in indices.iter().take(actual_audio_len).enumerate() {
    pred_aln_trg[[text_idx, frame_idx]] = 1.0;
}

// 3. Pad to full text length and compute en
let mut full_aln = Array3::<f32>::zeros((1, STATIC_TEXT_LEN, STATIC_AUDIO_LEN));
full_aln.slice_mut(s![0, ..cur_len, ..]).assign(&pred_aln_trg);

// en = d.transpose @ full_aln
let d_transposed = d.permuted_axes([0, 2, 1]);  // [1, 640, 512]
let en = d_transposed.dot(&full_aln);  // [1, 640, audio_len]
```

## Dependencies

```toml
[dependencies]
tvm-ffi = { path = "../../external/tvm-ffi/rust/tvm-ffi" }
ndarray = "0.16"
ndarray-npy = "0.9"           # For loading .npy files
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"            # For vocab loading
clap = { version = "4.0", features = ["derive"] }
hound = "3.5"                 # For WAV output
anyhow = "1.0"                # Error handling
```

## File Structure

```
rust/
├── Cargo.toml
├── src/
│   ├── main.rs           # CLI entry point
│   ├── lib.rs            # Public API
│   ├── pipeline.rs       # KokoroPipeline implementation
│   ├── vocab.rs          # Vocabulary loading and tokenization
│   ├── voice.rs          # Voice pack loading (.npy)
│   ├── preprocessing.rs  # Padding, masking, alignment
│   └── audio.rs          # WAV output
└── README.md
```

## Voice Pack Conversion

A Python utility script is provided to convert `.pt` voice packs to `.npy`:

```python
# python/scripts/convert_voice_pack.py
import torch
import numpy as np
from pathlib import Path

def convert_voice_pack(pt_path: str, npy_path: str):
    pack = torch.load(pt_path, weights_only=True)
    np.save(npy_path, pack.numpy())
```

## Known Limitations

1. **No G2P**: Input must be IPA phonemes, not text
2. **Offline voice packs**: Must pre-convert from `.pt` to `.npy`
3. **Offline vocab**: Must extract from HuggingFace config.json
4. **Single voice only**: No voice mixing (can be added later)

## Performance Considerations

- No Python GIL overhead
- Native SIMD via ndarray
- Direct TVM FFI calls without Python wrapper
- Potential for async/parallel module loading

## Relax VM Integration (Rust)

### Initialization Sequence

The `tvm-ffi` crate does **not** expose a high-level `VirtualMachine` wrapper. You must manually invoke the VM functions:

```rust
// 1. Load the compiled library
let lib = Module::load_from_file("bert_compiled.so")?;

// 2. Get and call vm_load_executable (returns Executable module)
let vm_load_exec = lib.get_function("vm_load_executable")?;
let exec_module: Module = vm_load_exec.call_tuple(())?.try_into()?;

// 3. Initialize the VM with devices and allocator
let vm_init = exec_module.get_function("vm_initialization")?;
vm_init.call_tuple((device_type, device_id, POOLED_ALLOCATOR))?;

// 4. Get entry point function
let func = exec_module.get_function("bert_forward")?;
```

**Allocator Types:**
- `NAIVE_ALLOCATOR = 1` - Simple malloc/free (slow)
- `POOLED_ALLOCATOR = 2` - Memory pool (recommended)

### Handling Array Outputs

Relax functions often return `ffi.Array` (tuples) instead of single tensors. Use `ffi.ArrayGetItem` to extract elements:

```rust
fn extract_tensor_from_output(output: tvm_ffi::Any, index: usize) -> Result<Tensor> {
    if output.type_index() == TypeIndex::kTVMFFITensor as i32 {
        // Direct tensor
        return output.try_into();
    }
    // Array - extract element
    let array_get_item = Function::get_global("ffi.ArrayGetItem")?;
    let output_view = AnyView::from(&output);
    let args = [output_view, AnyView::from(&(index as i64))];
    let element = array_get_item.call_packed(&args)?;
    element.try_into()
}
```

### Runtime Library Requirements

```bash
export DYLD_LIBRARY_PATH="/path/to/tvm-ffi/build/lib:/path/to/tvm/build:$DYLD_LIBRARY_PATH"
```

Both `libtvm_ffi.dylib` (FFI layer) and `libtvm.dylib` (runtime with Relax VM) must be loadable.

## Known Issues

### 1. `ffi.Array` vs `ffi.Tensor`

**Error**: `TypeError: Cannot convert from type 'ffi.Array' to 'ffi.Tensor'`

**Cause**: Relax functions often return tuples/arrays of outputs.

**Solution**: Use `ffi.ArrayGetItem(array, index)` to extract individual tensors.

### 2. dtype Mismatch: `bool` vs `int8`

**Error**: `expect Tensor with dtype bool but get int8`

**Cause**: Rust `i8` is not the same as TVM's `bool` dtype.

**Solution**: Create tensors with explicit bool dtype:
```rust
// NOT: tvm_ffi::Tensor::from_slice(&[0i8, 1i8, ...], shape)
// Instead, create with bool dtype (requires custom allocation)
```

### 3. `relax.VirtualMachine` Not Found

**Error**: `Failed to get relax.VirtualMachine from registry`

**Cause**: `tvm-ffi` has a separate global function registry from `libtvm_runtime`.

**Solution**: Use module-level `vm_load_executable` instead of global functions.
