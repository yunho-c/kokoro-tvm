# voices.json schema (kokoro-tvm)

This repo supports an optional voice manifest to expose metadata for voice selection.
If present, it should live next to the voice pack (`voices.npy`) or share its stem
and be named one of:

- `voices.json`
- `voice_manifest.json`
- `<voice_pack_stem>.json` (for example `voices.json` alongside `voices.npy`)

The manifest is parsed by `rust/src/voice.rs` and used by:

- `frb_get_voices`
- `frb_get_languages`

## Accepted shapes

Either of the following JSON shapes are accepted:

1) **Array of objects** (recommended)
```json
[
  {
    "id": "af_bella",
    "name": "Bella",
    "language": "en-US",
    "gender": "female",
    "quality": "A-",
    "tags": ["warm", "friendly"],
    "index": 0
  }
]
```

2) **Object map** (keys are voice IDs)
```json
{
  "af_bella": {
    "name": "Bella",
    "language": "en-US",
    "gender": "female",
    "quality": "A-",
    "tags": ["warm", "friendly"],
    "index": 0
  }
}
```

## Field definitions

All fields are optional except `id` (required for list entries). Unknown fields are ignored.

- `id` (string, required): Stable voice ID (e.g., `af_bella`, `am_michael`).
- `name` (string, optional): Display name for UI.
- `language` (string, optional): BCP-47 style language tag (e.g., `en-US`, `ja`).
- `gender` (string, optional): `female` or `male`.
- `quality` (string, optional): Free-form grade (e.g., `A-`, `C+`).
- `tags` (array of strings, optional): Traits or descriptors (e.g., `["warm", "broadcast"]`).
- `index` (integer, optional): Explicit index into the voice pack. If missing, indices are assigned sequentially.

## Notes

- If `language`, `gender`, or `name` are omitted, `kokoro-tvm` attempts to infer them from the ID
  schema `[language][gender]_[name]` (for example `af_bella` -> `en-US`, `female`, `bella`).
- If the manifest is missing entirely, `frb_get_voices` returns a fallback list of `Voice {index}`.
