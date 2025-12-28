# kokoro-tvm Rust TODOs

## Streaming + progress

- Non-goal: true streaming (token-level incremental inference) for Kokoro. See
  `NOTES/TTS_API_STREAMING_API_SKETCH.md` for rationale (non-causal context and
  convolutions, modest latency benefit vs. chunking).
- Improve cancellation to interrupt inference mid-forward (not just chunk boundaries).
- Add a progress offsets API (phoneme-level first, text offsets once G2P spans exist).

## Voice metadata

- Ensure `voices.json` is bundled with runtime assets and matches the `voices.npy` index order.
