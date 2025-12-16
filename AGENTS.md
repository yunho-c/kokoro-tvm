# Agent Instructions for kokoro-tvm

## Python Environment

**Always use `py -3.12`** for running scripts in this project:

```bash
py -3.12 <script>
```

This provides access to TVM and all required dependencies.

### Example Usage

```bash
# Compile the decoder
py -3.12 python/src/kokoro_tvm/cli/port_decoder.py --target metal-macos --seq-len 5120 --output tvm_output/decoder_compiled.dylib

# Run other scripts
py -3.12 python/src/kokoro_tvm/cli/compile_kokoro.py
```

## Project Structure

- `python/src/kokoro_tvm/cli/` - Compilation, porting, and debug CLIs
- `python/src/kokoro_tvm/` - Pipeline and model wrappers
- `external/kokoro/` - Kokoro TTS model (submodule)
- `reference/tvm/` - TVM source (submodule)
- `tvm_output/` - Default output dir for compiled libraries (large, usually untracked)
- `NOTES/` - Debugging writeups and investigation logs

## Code Style

- **No Numbered Headings in Comments**: Do not use numbers (e.g., "1. Step One", "2. Step Two") in comment headings. Use descriptive text only.

## Static Shape Conventions

The pipeline expects statically-compiled shapes that match `python/src/kokoro_tvm/pipeline.py`:

- `STATIC_TEXT_LEN=512` (token sequence length for encoder modules)
- `STATIC_AUDIO_LEN=5120` (frame length for decoder inputs; `f0/n` use `2 * STATIC_AUDIO_LEN`)
- `STYLE_DIM=128` (style vector passed into decoder is the first 128 dims of the 256-dim ref style)

If you compile the decoder with a different `--seq-len` (e.g. 150) and try to run the full pipeline, TVM will fail with a `match_cast` shape mismatch (e.g. `5120 vs. 150`).

## Decoder NaN Debugging

Decoder NaN investigation notes live in `NOTES/DECODER_NAN_DEBUG_PLAN.md`.

Useful CLIs:

```bash
py -3.12 python/src/kokoro_tvm/cli/validate_steps.py --text "Hello world" --device metal --lib-dir tvm_output
py -3.12 python/src/kokoro_tvm/cli/probe_decoder.py --target llvm --seq-len 150
py -3.12 python/src/kokoro_tvm/cli/probe_decoder_full.py --target llvm --seq-len 150
```

There is a stability workaround for TVM lowering of `nn.InstanceNorm1d` inside decoder AdaIN:

- `python/src/kokoro_tvm/patches/adain.py` patches `kokoro.istftnet.AdaIN1d.forward` during export to avoid non-finite outputs.
- If you change this patch, recompile the decoder to observe its effect.

## Submodules and Artifacts

- Submodules (`external/kokoro/`, `reference/tvm/`) can show as “dirty” if local changes exist inside them; avoid committing those changes unless intentional.
- Large artifacts like `tvm_output/`, `decoder_*_opt.py`, and generated `.wav` traces should generally stay untracked.
