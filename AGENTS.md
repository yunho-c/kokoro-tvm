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
py -3.12 scripts/port_decoder.py --seq-len 150

# Run other scripts
py -3.12 scripts/compile_kokoro.py
```

## Project Structure

- `scripts/` - Compilation and porting scripts
- `python/` - Python package with pixi environment
- `external/kokoro/` - Kokoro TTS model (submodule)
- `reference/tvm/` - TVM source (submodule)

## Code Style

- **No Numbered Headings in Comments**: Do not use numbers (e.g., "1. Step One", "2. Step Two") in comment headings. Use descriptive text only.
