# kokoro-tvm Rust TODOs

## Streaming + progress

- Implement true streaming (incremental inference) instead of chunked phoneme segmentation.
- Improve cancellation to interrupt inference mid‑forward (not just chunk boundaries).
- Add a progress offsets API (phoneme‑level first, text offsets once G2P spans exist).

## Voice metadata

- Ensure `voices.json` is bundled with runtime assets and matches the `voices.npy` index order.

## FRB + Dart surfacing

- Update Dart wrapper for request‑based APIs: `SynthesisRequest`/`SynthesisInput`/`VoiceSelection`,
  plus streaming APIs and `TtsError`.
