# Decoder NaNs: investigation log and fix direction

This note documents the full investigation into decoder NaNs, what we learned, and what to do next.

## Executive summary

- The NaNs are not coming from F0/N padding (TVM pads with zeros, as expected).
- The TVM decoder was producing all-NaN audio even when given finite inputs.
- The first non-finite values originate inside the decoder generator’s AdaIN blocks, specifically `AdaIN1d.norm = nn.InstanceNorm1d`.
- In TVM, `InstanceNorm1d` lowering can produce non-finite outputs for a subset of channels; the first downstream convolution then propagates NaNs everywhere.
- A practical workaround is to avoid `InstanceNorm1d` in the exported graph by rewriting `AdaIN1d.forward` to compute instance norm explicitly with a stable variance formula. This removes NaNs in probes and makes the decoder output finite on Metal, but output fidelity is still poor (very large waveform magnitude).

## Repro (current baseline)

The original NaN repro (before applying the AdaIN workaround) is:

- `py -3.12 python/src/kokoro_tvm/cli/validate_steps.py --text "Hello world" --device metal --lib-dir tvm_output`
  - `tvm audio_trimmed finite_frac=0.0000` (all NaNs)

The run is saved in `logs/decoder_nan_repro_validate_steps.txt`.

## F0/N padding inspection

`validate_steps.py` prints a “padded tail” report that confirms TVM padding is not injecting weird values:

- `tvm.f0` and `tvm.n` tail beyond valid frames are exactly zeros (`nonzero_frac=0.000`)
- Therefore, “TVM tensor creation / padding API” is not the immediate NaN culprit.

See `logs/decoder_nan_repro_validate_steps.txt` for the exact output.

## Decoder inputs are finite, yet decoder output was all NaN

`validate_steps.py` also prints finiteness/range stats for the decoder inputs coming from the TVM pipeline:

- `asr`, `f0`, `n`, `s` are all finite (`finite_frac=1.0`)
- Yet `tvm audio_trimmed` was entirely non-finite (`finite_frac=0.0`) with the original decoder build

This points at a decoder-internal numerical/runtime issue rather than bad encoder outputs.

## Localization: where does the first non-finite value appear?

### Lightweight slice probe (initial, slightly misleading)

We added a “slice-only” probe that returns head/tail slices of intermediate tensors:

- `python/src/kokoro_tvm/models/decoder_probe.py`
- `py -3.12 python/src/kokoro_tvm/cli/probe_decoder.py --target llvm --seq-len 150`

This suggested the first NaNs appeared at `Generator.noise_res[0].convs1[0]` output (`nr0_conv1`), with upstream slices looking finite.

However, this was a sampling artifact: the slices didn’t include the specific channels that were already non-finite earlier.

### Full-tensor probe (definitive)

To remove sampling ambiguity, we compiled a probe that returns full tensors for the first noise-res block:

- `python/src/kokoro_tvm/cli/probe_decoder_full.py`
- It returns `(x_source0_conv, nr0_n1, nr0_snake1, nr0_conv1_w, nr0_conv1_b, nr0_conv1)`

Running it on LLVM with random decoder weights:

- `py -3.12 python/src/kokoro_tvm/cli/probe_decoder_full.py --target llvm --seq-len 150 --no-weights --no-fuse-ops --no-fuse-tir`

showed:

- `x_source0_conv` is fully finite (`finite_frac=1.0`)
- `nr0_n1` and `nr0_snake1` are partially non-finite (`finite_frac=0.960938`)
  - `0.960938 == 246/256`, strongly suggesting exactly 10 out of 256 channels are entirely non-finite across time (a per-channel scalar becoming NaN/Inf in InstanceNorm)
- `nr0_conv1` becomes completely non-finite (`finite_frac=0.0`) because convolution sums across all input channels and any NaN input contaminates the reduction

See `logs/decoder_probe_full_llvm_no_weights_no_fuse.txt` for the exact stats.

## Root cause hypothesis (best fit to evidence)

The first non-finite values appear at `AdaIN1d.norm(x)` where `AdaIN1d.norm` is `nn.InstanceNorm1d` in `external/kokoro/kokoro/istftnet.py`.

The pattern “exactly N channels fully non-finite” is consistent with a per-channel normalization factor becoming NaN/Inf, e.g.:

- variance computed negative (and epsilon not applied correctly) → `sqrt(var + eps)` becomes NaN
- unstable variance formulation (`E[x^2] - E[x]^2`) producing negative values for some channels
- a TVM lowering bug for InstanceNorm reductions on this target

## Workaround implemented: patch AdaIN to avoid InstanceNorm lowering

### Patch details

We added `python/src/kokoro_tvm/patches/adain.py`:

- Monkey-patches `kokoro.istftnet.AdaIN1d.forward`
- Replaces `self.norm(x)` with an explicit instance norm:
  - `mean = x.mean(dim=2)`
  - `var = ((x - mean)**2).mean(dim=2)`
  - `x_hat = (x - mean) / sqrt(var + eps)`
  - apply affine `weight/bias` if present

This avoids going through `nn.InstanceNorm1d` / `aten.instance_norm` lowering in TVM.

The patch is applied during export in:

- `python/src/kokoro_tvm/models/decoder.py`
- `python/src/kokoro_tvm/models/decoder_probe.py`
- `python/src/kokoro_tvm/cli/port_decoder.py`
- `python/src/kokoro_tvm/cli/compile_kokoro.py`

### Evidence the workaround removes NaNs

Full-tensor probe after the patch:

- `py -3.12 python/src/kokoro_tvm/cli/probe_decoder_full.py --target llvm --seq-len 150 --no-weights --no-fuse-ops --no-fuse-tir`
- `nr0_n1`, `nr0_snake1`, and `nr0_conv1` become fully finite (`finite_frac=1.0`)

See `logs/decoder_probe_full_llvm_no_weights_no_fuse_adain_patch.txt`.

Slice probe after the patch:

- `py -3.12 python/src/kokoro_tvm/cli/probe_decoder.py --target llvm --seq-len 150 --no-weights --no-fuse-ops --no-fuse-tir`
- all stages (including `audio_out`) are finite

See `logs/decoder_nan_probe_llvm_no_weights_no_fuse_adain_patch.txt`.

## End-to-end status on Metal after recompiling decoder

To validate against the pipeline’s static shapes, decoder must be compiled with `--seq-len 5120` (not 150).

We created `tvm_output_adain_fix/` by copying the existing compiled encoder libs and recompiling only the decoder:

- `py -3.12 python/src/kokoro_tvm/cli/port_decoder.py --target metal-macos --seq-len 5120 --output tvm_output_adain_fix/decoder_compiled.dylib`
- `py -3.12 python/src/kokoro_tvm/cli/validate_steps.py --text "Hello world" --device metal --lib-dir tvm_output_adain_fix`

Result:

- `tvm audio_trimmed finite_frac=1.0000` (NaNs are gone)
- Output fidelity is still poor:
  - waveform magnitude is extremely large (on the order of `1e9`)
  - correlation vs PyTorch is near zero

See `logs/validate_steps_metal_adain_fix_seq5120.txt`.

## What’s next (to get correct audio, not just “finite audio”)

Now that the decoder runs without NaNs, the next debugging loop is about correctness and scale.

Most likely suspects:

- `exp` magnitude path inside generator (`spec = exp(x_post[..., :])`): a small mismatch earlier can explode after `exp`.
- precision/dtype issues on Metal (implicit FP16 in fused kernels, or reductions done in reduced precision).
- other ops with unstable reductions similar to the InstanceNorm issue.

Concrete next steps:

- Add a “decoder internal fidelity” compare for a small fixed input:
  - run PyTorch decoder and TVM decoder on identical `(asr, f0, n, s)` inputs (use CPU first), and compare internal tensors around:
    - `x_post` (pre-exp)
    - `spec` (post-exp)
    - ISTFT inverse output
  - the first large divergence will identify the next op family to focus on.
- Extend `decoder_probe.py` to return (or slice more robustly) `x_post` and `spec` under real weights and real pipeline inputs, then compute summary stats (min/max/percentiles).
- If divergence is mainly “too large before exp”:
  - inspect normalizations and residual scaling
  - verify TVM uses float32 for reductions and `exp` on Metal
- If divergence is “reasonable before exp but exp overflows”:
  - confirm dtype for `exp` on Metal and consider forcing float32 evaluation at that site for diagnosis.

## Files and tools added during this investigation

- `python/src/kokoro_tvm/pipeline.py` and `python/src/kokoro_tvm/cli/validate_steps.py`: extra finiteness/tail stats printing.
- `python/src/kokoro_tvm/models/decoder_probe.py`, `python/src/kokoro_tvm/cli/probe_decoder.py`: slice-based probe.
- `python/src/kokoro_tvm/cli/probe_decoder_full.py`: full-tensor probe for the first noise-res block.
- `python/src/kokoro_tvm/patches/adain.py`: AdaIN workaround to avoid InstanceNorm lowering.
- `python/src/kokoro_tvm/cli/probe_weightnorm_conv1d.py`, `python/src/kokoro_tvm/models/weightnorm_conv1d_probe.py`: minimal conv(+weight_norm) probe (showed conv alone is fine).

