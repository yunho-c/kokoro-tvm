# Agent Instructions for kokoro-tvm

## Python Environment

**Always use the pixi Python environment** for running scripts in this project:

```bash
./python/.pixi/envs/default/bin/python <script>
```

This environment has TVM and all required dependencies properly configured.

### Example Usage

```bash
# Compile the decoder
./python/.pixi/envs/default/bin/python scripts/port_decoder.py --seq-len 150

# Run other scripts
./python/.pixi/envs/default/bin/python scripts/compile_kokoro.py
```

## Project Structure

- `scripts/` - Compilation and porting scripts
- `python/` - Python package with pixi environment
- `external/kokoro/` - Kokoro TTS model (submodule)
- `reference/tvm/` - TVM source (submodule)

## Code Style

- **No Numbered Headings in Comments**: Do not use numbers (e.g., "1. Step One", "2. Step Two") in comment headings. Use descriptive text only.
