- Move `scripts` to inside `src/kokoro_tvm`, to make things invokeable from CLI without repo clone. (Let's do this when the git tree is clean, so we can have a clean, isolated commit.)

## Kokoro Component Porting (Unit Tests)

Prioritized from low-level building blocks to high-level modules.

- [x] **AdainResBlk1d** (`kokoro.istftnet`)
    - Role: Residual block with Adaptive Instance Normalization.
    - Status: Verified (`experiments/adain_import.py`)
- [x] **DurationEncoder** (`kokoro.modules`)
    - Role: LSTM-based duration encoding.
    - Status: Verified (`experiments/duration_encoder_import.py`). Requires `tvm_extensions` and packed sequence monkeypatching.
- [x] **TextEncoder** (`kokoro.modules`)
    - Role: Main text encoder with CNNs and LSTMs.
    - Status: Verified (`experiments/text_encoder_import.py`)
- [x] **ProsodyPredictor** (`kokoro.modules`)
    - Role: Predicts duration, F0, N. Contains LSTMs and AdainResBlk1d.
    - Status: Verified (`experiments/prosody_import.py`). Required tuple unpacking fix.
- [x] **CustomAlbert** (`kokoro.modules`)
    - Role: ALBERT transformer (High Complexity).
    - Status: Verified (`experiments/albert_import.py`). Required upstream TVM fix for boolean max/min.

## Package Integration

- [x] **Encoder Integration**
    - Created `models/encoder.py`, `patches/lstm.py`, and `cli/port_encoder.py`.
    - Refactored target config to `config.py`.
    - Verified compilation of all encoder components.