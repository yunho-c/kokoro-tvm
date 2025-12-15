LSTM semantics mismatch (why MPS LSTM “passes” but Kokoro still sounds wrong)

Summary
The MPS-backed LSTM kernel (`tvm.contrib.mps.lstm`) can be numerically accurate in isolation, yet the end-to-end Kokoro encoder/decoder pipeline can still diverge badly. The current evidence points to a semantic mismatch around variable-length handling (packing/masking) in bidirectional LSTMs, not an arithmetic bug in the LSTM kernel itself.

Observed symptoms
- End-to-end TVM audio has very low correlation with reference Kokoro audio even when using the same G2P phonemes and voice style.
- Step validation (`kokoro_tvm/cli/validate_steps.py`) shows:
  - BERT output (`d_en`) matches closely.
  - Large divergence begins in LSTM-heavy components: duration logits/`d`, text encoder output, F0/N, and finally the decoder waveform.

Why isolated LSTM tests can pass
The standalone tests for MPS LSTM accuracy (e.g., `test_mps_lstm_accuracy.py`) compare:
- PyTorch `nn.LSTM` run over a full sequence tensor
vs
- `tvm.contrib.mps.lstm` run over the same full tensor, same weights

Those tests do not exercise variable-length behavior. They assume every timestep in the tensor is “real” input and should influence the recurrence.

What Kokoro relies on: length-aware LSTMs
Reference Kokoro uses `pack_padded_sequence` / `pad_packed_sequence` to ensure LSTMs only process valid timesteps per sample and do not incorporate padded timesteps into recurrent state updates.

Two places where this matters a lot:
- `TextEncoder.lstm` (bidirectional)
- `DurationEncoder` inside `ProsodyPredictor` (multiple bidirectional LSTMs)

In bidirectional LSTMs, the reverse direction is particularly sensitive: if the padded tail is treated as real timesteps, the reverse recurrence “sees” padding first and that state bleeds into earlier timesteps, even if you later zero out masked positions.

What kokoro-tvm currently does for export
To enable static export for TVM, `apply_lstm_patch()` monkeypatches:
- `nn.utils.rnn.pack_padded_sequence` -> identity
- `nn.utils.rnn.pad_packed_sequence` -> identity

This removes the PackedSequence machinery but also removes its semantics. As a result:
- The bidirectional LSTMs process the full static length (e.g., 512) instead of only `cur_len`.
- The padded region (often zeros) still affects recurrent states due to bias terms and recurrent dynamics.
- Even if the code later masks outputs, the internal hidden state evolution has already diverged.

Why masking after the fact is insufficient
Kokoro often performs `masked_fill_` on activations. That can zero the outputs at padded positions, but it cannot undo the effect that the padded steps had on hidden/cell states (especially in the backward direction of a bidirectional LSTM). Packing avoids that by not running those timesteps at all.

How this shows up downstream
Once text encoder and duration predictor diverge, everything after them compounds:
- Duration prediction changes alignment length (`frames`), shifting the decoding timebase.
- Alignment differences change `en`, `F0`, `N`, and `asr`.
- The decoder is extremely sensitive to these conditioning signals, so the waveform can become noise-like even if the decoder itself is “correct.”

Next debugging steps (recommended)
- Add targeted validation to compare these three regimes for the same LSTM:
  - PyTorch with packing (true reference)
  - PyTorch with packing disabled but masked outputs (current export approximation)
  - TVM/MPS implementation
- Focus first on one bidirectional LSTM in `DurationEncoder` and one in `TextEncoder`, since step validation already indicates large divergence there.

Potential fixes (high level)
- Implement a length-aware LSTM op for TVM export (either:
  - extend the custom op to accept sequence lengths and enforce early termination, or
  - provide a TVMScript/TIR LSTM that respects lengths, or
  - restructure the model to avoid PackedSequence while preserving semantics).
- Alternatively, move certain LSTMs back to CPU PyTorch as a hybrid fallback while validating correctness.

