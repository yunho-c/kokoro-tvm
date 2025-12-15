# LSTM Packed Semantics Issue (Kokoro on TVM/MPS)

## Summary

Kokoro’s reference PyTorch implementation relies on length-aware LSTMs via `pack_padded_sequence` / `pad_packed_sequence`. If we export/compile a version where packing is replaced with padded full-length execution, end-to-end audio can become noise-like even if the underlying LSTM kernel is numerically accurate for full sequences.

This document consolidates:
- Root cause analysis (why the mismatch happens)
- Validation evidence (how we measured it)
- Implementation progress (how we started fixing it on MPS)

## Observed symptoms

- End-to-end TVM audio can have very low similarity to reference Kokoro audio even when using the same phonemes and voice style.
- Step validation (`python/src/kokoro_tvm/cli/validate_steps.py`) showed:
  - BERT output (`d_en`) matches closely.
  - Large divergence begins in LSTM-heavy components: duration logits/`d`, text encoder output, F0/N, and then the decoder waveform.

## Root cause

### What reference Kokoro relies on

Reference Kokoro uses `pack_padded_sequence` / `pad_packed_sequence` to ensure LSTMs only process valid timesteps per sample and do not incorporate padded timesteps into recurrent state updates.

Two places where this matters a lot:
- `TextEncoder.lstm` (bidirectional)
- `DurationEncoder` inside `ProsodyPredictor` (multiple bidirectional LSTMs)

### Why bidirectional LSTMs are especially sensitive

In bidirectional LSTMs, the reverse direction is particularly sensitive: if the padded tail is treated as real timesteps, the reverse recurrence “sees” padding first and that state bleeds into earlier timesteps, even if you later zero out masked positions.

### What kokoro-tvm did for export (historically)

To enable static export for TVM, `apply_lstm_patch()` monkeypatched:
- `nn.utils.rnn.pack_padded_sequence` → identity
- `nn.utils.rnn.pad_packed_sequence` → identity

This removes the PackedSequence machinery and its semantics. As a result:
- Bidirectional LSTMs process the full static length (e.g. 512) instead of only `cur_len`.
- The padded region (often zeros) still affects recurrent states due to bias terms and recurrent dynamics.

### Why masking after the fact is insufficient

Kokoro frequently uses `masked_fill_` on activations. That can zero the outputs at padded positions, but it cannot undo the effect that padded steps had on hidden/cell states (especially in the backward direction of a bidirectional LSTM). Packing avoids those padded timesteps being processed at all.

## How the error propagates to audio

Once text encoder and duration prediction diverge, everything after them compounds:
- Duration prediction changes alignment length (`frames`) and alignment structure.
- Alignment differences change conditioning signals (`en`, `asr`, `F0`, `N`).
- The decoder is extremely sensitive to these conditioning signals, so the waveform can become noise-like even if the decoder itself is correct.

## Dynamic vs static audio-length behavior

Even after fixing text-axis packing, you can still observe “dynamic sounds good, static sounds bad” if any **aligned-length** bidirectional LSTM is executed over a padded fixed length.

One concrete example is `ProsodyPredictor.F0Ntrain`, which begins with a bidirectional LSTM (`predictor.shared`). If `en` is padded to `STATIC_AUDIO_LEN` and then passed through a bidirectional LSTM, the backward direction “sees” the padded tail first and can contaminate states for earlier (real) frames. Trimming the waveform afterward does not fix this, because the conditioning signals (`F0`, `N`) for the early frames were computed from already-contaminated recurrent state.

The correctness-first fix is to make the aligned-length recurrence length-aware too: pass `frames` (or per-sample aligned lengths) into the compiled `f0n` path and run the bidirectional LSTM with packed semantics for only the valid prefix, then pad back to `STATIC_AUDIO_LEN` for downstream static-shape ops.

## Why isolated LSTM tests can still pass

Standalone tests for MPS LSTM accuracy (e.g. `python/src/kokoro_tvm/tests/test_mps_lstm_accuracy.py`) compare:
- PyTorch `nn.LSTM` run over a full sequence tensor
- `tvm.contrib.mps.lstm` run over the same full tensor, same weights

Those tests do not exercise variable-length behavior. They assume every timestep in the tensor is “real” input and should influence the recurrence.

## Validation workflow (what to compare)

The step validator (`python/src/kokoro_tvm/cli/validate_steps.py`) compares four regimes:
- Dynamic reference Kokoro (true packed semantics, no static padding)
- Static PyTorch reference (static shapes, but still packed semantics)
- Static PyTorch no-pack (packing disabled; padded full-length semantics)
- TVM pipeline trace

Two comparisons are the key to avoid getting fooled:
- Packed vs no-pack (semantic delta)
- TVM vs packed (fidelity to reference)

## Export strategy that preserves packing semantics

### Why `torch.export` needs a different representation

`torch.export` cannot reliably carry a PyTorch `PackedSequence` object through tracing because it encodes shape/data-dependent control flow.

### What we do instead

We introduce export-safe custom ops that carry `lengths` explicitly as tensor input:
- `kokoro::lstm_forward_packed`
- `kokoro::lstm_forward_packed_bidirectional`

At export time, we patch Kokoro modules to call these ops instead of constructing a `PackedSequence`. The exported graph remains static and the LSTM stays a single opaque node, but length information is still present for lowering.

## MPS runtime progress (bridge code)

### Packed stepping-stone op

We added a packed runtime entrypoint:
- `tvm.contrib.mps.lstm_packed` in `reference/tvm/src/runtime/contrib/mps/lstm.mm`

Current behavior:
- Implemented for `batch=1` as a first milestone: run only the first `lengths[0]` timesteps and zero-fill the padded region.
- Added a `reverse` flag so the reverse direction of bidirectional runs can traverse only valid timesteps without flipping padded sequences.

### Practical lessons learned

- `te.extern` cannot infer output dtype if inputs include mixed dtypes (e.g. float tensors plus `int64 lengths`); the wrapper must provide `dtype=[...]`.
- The compiled Metal function expects the `lengths` tensor on `kDLMetal` as well. Passing a CPU `lengths` tensor into a Metal-compiled extern triggers a runtime device constraint failure.
- Implementing bidirectional packed semantics as “flip input then run packed forward” is wrong; it puts padding at the start. Reverse traversal must skip padding by construction.

## Next debugging steps (recommended)

- Add targeted validation to compare these three regimes for the same LSTM:
  - PyTorch with packing (true reference)
  - PyTorch with packing disabled but masked outputs (export approximation)
  - TVM/MPS implementation
- Focus first on one bidirectional LSTM in `DurationEncoder` and one in `TextEncoder`, since step validation amplifies issues there.

## Potential fixes (high level)

- Implement a length-aware LSTM op for TVM export by:
  - extending the MPS extern path to accept more complete packed information (e.g. `batch_sizes`), or
  - restructuring the exported graph to avoid PackedSequence objects but still carry lengths and preserve semantics, or
  - using a correctness-first non-MPS backend for LSTMs while iterating.

## Current limitations

- `tvm.contrib.mps.lstm_packed` is `batch=1` only today.
  - This matches the current CLI inference use case (single utterance).
  - Full `batch>1` packed semantics require a PackedSequence-like representation (`batch_sizes`) or a per-sample length-driven construction of ragged per-timestep matrices.

## Next steps for full packed semantics on GPU

- Extend the packed runtime to support `batch>1` using MPSRNN’s native ragged-batch representation (decreasing rows per timestep).
- Add bidirectional support at the runtime level (either via MPS bidirectional encode APIs or two packed unidirectional passes with correct reverse traversal and output concatenation).
- Apply packed/length-aware semantics on the aligned axis for `ProsodyPredictor.F0Ntrain` (`predictor.shared`) by plumbing `frames`/aligned lengths into the compiled `f0n` module.

## How to reproduce and measure

- Compile encoder with packed semantics:
  - `py -3.12 python/src/kokoro_tvm/cli/port_encoder.py --component all --target metal-macos --lstm-method mps --lstm-semantics packed --output-dir tvm_output_packed --validate`
- Validate stage-by-stage:
  - `py -3.12 python/src/kokoro_tvm/cli/validate_steps.py --device metal --lib-dir tvm_output_packed --text "Hello world"`
- Run the MPS LSTM tests (includes a packed-length case):
  - `py -3.12 python/src/kokoro_tvm/tests/test_mps_lstm_accuracy.py`

## Related notes

- MPS feasibility background: `NOTES/MPS_LSTM_FEASIBILITY.md`
- Relax/TIR alternatives and IR size considerations: `NOTES/RELAX_LSTM_FIX_PROPOSAL.md`
