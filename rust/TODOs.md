# kokoro-tvm Rust TODOs

## Streaming + progress

- Implement true streaming (incremental inference) instead of chunked phoneme segmentation.
- Improve cancellation to interrupt inference mid‑forward (not just chunk boundaries).
- Add a progress offsets API (phoneme‑level first, text offsets once G2P spans exist).

## Voice metadata

- Add `synthesize_with_voice_id` helper that maps manifest IDs to voice indices.
- Ensure `voices.json` is bundled with runtime assets and matches the `voices.npy` index order.

## Error model + docs

- Decide on a public error taxonomy (keep `TtsError` stable as Dart API).
- Document error variants and any fallback behavior in the Rust crate README.

## FRB + Dart surfacing

- Regenerate FRB bindings after any API change.
- Update Dart wrapper to expose `frb_synthesize_text`, `frb_get_voices`, `frb_get_languages`,
  streaming APIs, and `TtsError`.
