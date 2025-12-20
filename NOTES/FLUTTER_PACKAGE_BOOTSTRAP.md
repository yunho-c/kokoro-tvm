# Flutter Package Bootstrap (for Agent Handoff)

## Scope
This note covers:
1) Scaffolding a minimal Flutter plugin repo that depends on the Rust crate in this repo.
2) Defining the asset packaging contract (paths and layout) for iOS/Android.

## Prereqs
- Rust crate lives in `rust/` in this repo.
- FRB exports are in `rust/src/frb_api.rs`, gated by `--features frb`.
- Use the runtime API (init/warmup/status/synthesize) to avoid reloading artifacts.

## Task #2: Flutter Plugin Scaffold

### Goal
Create a separate Flutter plugin repo that:
- Depends on this Rust crate by git tag.
- Generates FRB bindings from `rust/src/frb_api.rs`.
- Ships platform glue (iOS/Android) and Dart wrapper.

### Recommended Repo Layout
- `kokoro_tvm_flutter/`
  - `pubspec.yaml`
  - `lib/ffi/kokoro_tvm.dart` (generated)
  - `lib/kokoro_tvm.dart` (manual wrapper)
  - `ios/Classes/frb_generated.h` (generated)
  - `android/src/main/kotlin/...` (minimal plugin)
  - `rust/` (optional subdir only if you embed Rust build here)

### Dependency on Rust Core
Prefer git tag dependency in `Cargo.toml` (inside plugin repo if it builds Rust):
- `kokoro-tvm = { git = "git@.../kokoro-tvm.git", tag = "v0.1.0", features = ["frb"] }`

### FRB Codegen (from this repo)
From `kokoro-tvm` root:
- `cargo build --features frb`
- `flutter_rust_bridge_codegen --rust-input rust/src/frb_api.rs --dart-output <flutter_repo>/lib/ffi/kokoro_tvm.dart --c-output <flutter_repo>/ios/Classes/frb_generated.h`

### Dart Wrapper
Create `lib/kokoro_tvm.dart`:
- Wrap `KokoroTvmImpl` (generated) with a small API:
  - `init(artifactsDir, device, vocabPath, voicePath)`
  - `warmup()`, `status()`, `synthesize(phonemes, speed, voiceIndex)`
- Keep async/await consistent; return audio as `Float32List`.

## Task #3: Asset Packaging Contract

### Artifact Layout (Bundle)
The Rust loader expects compiled artifacts in a directory. The helper
`KokoroPipeline::load_from_artifacts_dir()` tries these candidates:
- `<base>/`
- `<base>/tvm_output/`
- `<base>/Resources/`
- `<base>/Resources/tvm_output/`
- `<base>/Contents/Resources/`
- `<base>/Contents/Resources/tvm_output/`
- plus parent directory variants

### Required Files
In the chosen directory:
- `bert_compiled.{so|dylib}`
- `duration_compiled.{so|dylib}`
- `f0n_compiled.{so|dylib}`
- `text_encoder_compiled.{so|dylib}`
- decoder files (bucketed or `decoder_compiled.{so|dylib}`)

### iOS
- Bundle artifacts under `Runner.app/Frameworks` or `Runner.app/Resources`.
- Expose the base resource path to Dart (e.g., via a small platform channel).
- Pass that path into `frbInit(artifactsDir, device, vocabPath, voicePath)`.

### Android
- Store artifacts in `android/app/src/main/assets/`.
- On first run, copy from APK assets to app internal storage and use that path.
- Pass the resolved directory into `frbInit(...)`.

### Vocab + Voice Pack
Keep `vocab.json` and `voice.npy` alongside artifacts, or store them in
a separate asset directory and pass their absolute paths into `frbInit`.

### API Contract (Dart -> Rust)
- `artifactsDir`: absolute path
- `device`: "cpu" or "metal" (future: "cuda")
- `vocabPath`: absolute path to `vocab.json`
- `voicePath`: absolute path to `voice.npy`
- `voiceIndex`: optional index into voice pack

## Suggested Milestones
1) Build plugin skeleton + FRB codegen.
2) Basic init/warmup/status call works in Dart.
3) Artifacts and vocab/voice loaded from assets on iOS/Android.
4) Audio output returned to Dart and played.
