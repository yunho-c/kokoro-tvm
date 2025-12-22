# Progress Offsets Feasibility (kokoro-tvm)

This note evaluates whether we can provide progress events with text offsets
using the current Kokoro TVM pipeline.

## What exists today

- `build_alignment_with_pred` in `rust/src/preprocessing.rs` produces:
  - `pred_dur`: per‑token predicted durations (frames)
  - `actual_audio_len`: audio frame length after clamping
- `KokoroPipeline::forward_trace` in `rust/src/pipeline.rs` already computes
  and returns `pred_dur` and `frames`.
- `SAMPLES_PER_FRAME` and `SAMPLE_RATE` are defined in `rust/src/lib.rs`,
  so frame counts can be converted to samples/time.

## What’s feasible now

### Phoneme‑level offsets
Feasible with moderate effort. We can compute a cumulative sum over `pred_dur`
to assign start/end frames per phoneme token, then convert to samples.

### Offsets for phoneme string input
Feasible. We can extend `Vocab::encode` (or add a helper) to return the
original phoneme character indices that survived vocab filtering. This gives
`token_index -> char_offset` for a phoneme string.

## What’s hard

### Offsets for raw text input
Not reliably available with the current G2P. `voirs-g2p` does not emit original
text spans for each phoneme. To get precise text offsets, we would need either:

- G2P changes to return `(phoneme, start_char, end_char)` spans, or
- A heuristic mapping (word‑level, punctuation‑level), which is lossy
  after normalization and pronunciation rules.

### Long utterances
Alignment is capped by `STATIC_AUDIO_LEN` (currently 512 in `rust/src/lib.rs`).
For longer sequences, offsets would only cover the truncated audio unless
static shapes are restored to 5120 or bucketed.

## Streaming considerations

Chunk‑level streaming provides per‑chunk offsets only. We can compute offsets
per chunk and add a running sample offset, but cancellation remains
chunk‑boundary only, and prosody artifacts can occur at chunk splits.

## Suggested minimal API (phoneme‑level first)

Expose a new API returning phoneme offsets instead of text offsets, e.g.
`synthesize_with_alignment`:

- audio, sample_rate
- token_offsets: `[{token_index, start_sample, end_sample, is_word_boundary}]`

If raw text offsets are required, add a G2P path that returns span metadata
and translate those spans into the phoneme‑level alignment.

## Conclusion

Phoneme‑level progress offsets are feasible with modest changes using existing
alignment data. Accurate raw‑text offsets are higher effort and depend on G2P
instrumentation or a lossy heuristic mapping. The static audio length cap is
another practical blocker for long‑form progress reporting.
