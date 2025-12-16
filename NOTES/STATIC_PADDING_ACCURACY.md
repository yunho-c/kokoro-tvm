# Making `pt.static` / TVM match `pt.dynamic` end-to-end

## What we are seeing

From `validate_steps.py` runs (e.g. “Hello world”, `frames≈63` vs `STATIC_AUDIO_LEN=5120`):

- Text-side components are effectively identical across PyTorch and TVM:
  - `bert`, `duration`, `text_encoder`, and reconstructed `pred_dur` / `asr` match at ~1e-6 scale.
- The mismatch appears on the *audio/aligned-length* side:
  - `pt.dynamic vs tvm (aligned-length F0/N)` is excellent when comparing the valid prefix.
  - `pt.dynamic vs pt.static` shows a huge deviation for `f0/n` even on the prefix.
  - `pt.static vs tvm` also shows a huge deviation for `f0/n` on the prefix.
- Crossed decoder tests show the decoder is “internally consistent” with whichever semantics you feed it:
  - `tvm(dec<-pt.static)` correlates well with `pt.static`.
  - `tvm(dec<-pt.dynamic)` does not correlate well with `pt.dynamic`.

This strongly suggests that the dominant error is not “TVM is wrong”, but rather:

- `pt.static` is not a faithful proxy for `pt.dynamic` once we introduce massive padding on the aligned axis, and
- the TVM pipeline is also operating under those static/padded semantics.

## Working hypothesis: padding sensitivity

The static path pads aligned-length tensors to `STATIC_AUDIO_LEN=5120` frames even when the true length is ~63 frames. That means ~99% of the aligned axis is padding (zeros).

Many audio-side operations are not padding-invariant, especially anything that:

- computes statistics over time (InstanceNorm/LayerNorm/AdaIN variants), or
- has receptive fields and accumulates effects from padded regions (conv stacks without careful masking).

Even if the padding is “just zeros”, the mean/variance over time changes dramatically, and that can change the *prefix outputs* in a way that cannot be fixed by trimming after the fact.

## What “accurate” should mean

There are two reasonable targets:

- **Match `pt.dynamic`**: preserve the original model semantics (variable-length, no giant padding influencing statistics).
- **Match `pt.static`**: ensure TVM faithfully reproduces the static/padded execution (useful for debugging compilation, but may diverge from `pt.dynamic`).

If our goal is “sounds right like reference Kokoro”, we generally want to match `pt.dynamic`.

## Options to make TVM accurate vs `pt.dynamic`

### Make audio-side modules length-aware (best fidelity)

Core idea: keep static shapes (for TVM compilation) but ensure computations *only use the valid prefix* for any time-dependent statistics.

Concrete steps:

- **Fix the PyTorch “static” reference first** (fast sanity check):
  - Compute `en`, then run `F0N` only on the valid prefix (`frames`), then pad `f0/n` outputs to static length.
  - If this makes `pt.static` align with `pt.dynamic`, we have strong confirmation that padding is the cause.
- **Extend this idea to TVM**:
  - `F0N` already has a `frame_lengths`-aware signature in the pipeline; verify the exported module actually uses it internally (not just accepts it).
  - For the **decoder**, add an explicit `frame_lengths` (or mask) input and implement masked normalization:
    - for each timewise normalization op, compute mean/variance using only `t < frames`, and ignore/preserve padded positions.
  - If the decoder has AdaIN/InstanceNorm-like paths, the masking must happen inside the norm math (trimming output afterwards is too late).

This is the most likely way to reach near-`pt.dynamic` fidelity while retaining static compilation.

### Bucketed compilation (good engineering tradeoff)

Compile multiple decoder variants at smaller aligned lengths (e.g. 256/512/1024/2048/5120 frames), and pick the smallest bucket that fits `frames`.

Benefits:

- Reduces the padding fraction substantially for typical utterances.
- Often “good enough” without rewriting model internals.

Costs:

- More artifacts to compile/store, and some routing logic at runtime.
- Not a perfect semantic match (still padded, just less).

### Treat static semantics as the target (already achievable)

If the goal is “TVM matches `pt.static`”, we already see strong evidence this is true (e.g. `tvm(dec<-pt.static)` correlates well).

This is still useful for compiler debugging, but it does not guarantee good fidelity to `pt.dynamic`.

## Recommended next experiments

- **Experiment A: prefix-only `F0N` in the PyTorch static reference**
  - Modify the `pt.static` trace path so `F0N` runs on `en[..., :frames]` and then pads outputs.
  - Re-run `validate_steps.py` to see if `pt.dynamic vs pt.static` (F0/N prefix) collapses to ~1e-2 scale.
- **Experiment B: isolate decoder padding sensitivity**
  - Feed the same decoder inputs but vary only the padded tail (e.g. pad with zeros vs pad with repeated last value) and see how much the prefix audio changes.
  - If prefix audio changes significantly, the decoder is computing timewise stats that are padding-sensitive.
- **Experiment C: bucketed decoder feasibility**
  - Compile a small decoder bucket (e.g. aligned length 256 or 512) and compare against `pt.dynamic` for short texts.
  - This gives a quick read on whether bucketing alone is sufficient for perceptual fidelity.

## Notes about saving WAVs for comparison

If debug traces are written as `PCM_16` WAV, clipping can make different raw waveforms look/sound artificially similar. Prefer saving as float WAV (`subtype="FLOAT"`) or normalize with headroom before writing for meaningful listening tests.

