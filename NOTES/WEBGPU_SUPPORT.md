# WebGPU Support for Kokoro-TVM

## Status: Integrated into Existing Scripts

### Summary

WebGPU compilation is now integrated into the existing `port_decoder.py` CLI via `config.py` target configuration.

### Usage

```bash
# Compile decoder for WebGPU (requires emscripten for WASM export)
py -3.12 python/src/kokoro_tvm/cli/port_decoder.py --target webgpu --seq-len 256 --output-dir web_output

# Other existing targets still work
py -3.12 python/src/kokoro_tvm/cli/port_decoder.py --target llvm --seq-len 256
py -3.12 python/src/kokoro_tvm/cli/port_decoder.py --target metal-macos --seq-len 256
```

### Configuration

WebGPU target added to `config.py`:
```python
"webgpu": {
    "target_host": "llvm -mtriple=wasm32-unknown-unknown-wasm",
    "target": "webgpu",
    "extension": ".wasm",
    "description": "WebGPU (Browser via WASM)",
    "export_func": "tvmjs",
}
```

### Known Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| Emscripten | Not installed | Needed for WASM export |
| LSTM on WebGPU | Planned | Will use CPU WASM for LSTMs |
| Full decoder export | Pending | Needs emscripten + testing |

### Next Steps

1. Install emscripten: `git clone https://github.com/emscripten-core/emsdk.git && cd emsdk && ./emsdk install latest && ./emsdk activate latest`
2. Run full decoder export: `py -3.12 python/src/kokoro_tvm/cli/port_webgpu.py --seq-len 256`
3. Create web demo application
4. Add LSTM handling for encoder (hybrid CPU+WebGPU)

### TVM WebGPU Architecture

```
Compilation Pipeline:
  PyTorch Model → torch.export → Relax IR → relax.build(target="webgpu")
                                                    ↓
                                      ┌─────────────┴─────────────┐
                                      │                           │
                                decoder.wasm              decoder.wgsl
                                (WASM runtime)          (WebGPU shaders)

Browser Runtime:
  tvmjs (JS) → WASM runtime → WebGPU Context → GPU execution
```

### References

- TVM WebGPU codegen: `reference/tvm/src/target/source/codegen_webgpu.cc`
- TVM web runtime: `reference/tvm/web/src/webgpu.ts`
- WASM export: `tvm.contrib.tvmjs.create_tvmjs_wasm`
