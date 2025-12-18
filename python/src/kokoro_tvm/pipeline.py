# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Kokoro Pipeline orchestration using TVM compiled modules.

This module provides a pure NumPy inference pipeline that runs compiled TVM
modules for the Kokoro text-to-speech model. No PyTorch dependency is required
for inference.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tvm
import tvm.runtime

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Static compilation constants - must match what was used during compilation!
STATIC_TEXT_LEN = 512
STATIC_AUDIO_LEN = 5120
STYLE_DIM = 128  # Kokoro-82M uses 128

# Duration/alignment is in decoder frames, but decoder returns waveform samples.
# For Kokoro-82M, each decoder frame corresponds to 600 samples at 24kHz:
# 2 (F0/N are 2x frames) * prod(upsample_rates=10*6) * gen_istft_hop_size=5.
SAMPLES_PER_FRAME = 600

# Debug flag - set to True to enable timing prints
DEBUG = True


def _sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class KokoroPipeline:
    """TVM-based inference pipeline for Kokoro TTS.

    This pipeline orchestrates the following TVM-compiled modules:
    - BERT encoder: input_ids, attention_mask -> d_en
    - Duration predictor: d_en, style, lengths, mask -> (duration_logits, d)
    - F0N predictor: en, style, frame_lengths -> (F0, N)
    - Text encoder: input_ids, lengths, mask -> t_en
    - Decoder: asr, F0, N, style -> audio
    """

    def __init__(self, lib_dir: str, device_target: str = "llvm", hybrid: bool = False):
        """Initialize pipeline with compiled TVM modules.

        Args:
            lib_dir: Directory containing .so/.dylib files
            device_target: 'llvm', 'metal', 'cuda' (for decoder in hybrid mode)
            hybrid: If True, encoder runs on CPU, decoder runs on device_target
        """
        self.lib_dir = lib_dir
        self.hybrid = hybrid
        self.device_type = device_target.split()[0] if " " in device_target else device_target
        self._decoder_bucket_lens: list[int] = []

        # Set up devices
        self.cpu_dev = tvm.cpu(0)

        if self.device_type == "llvm":
            self.gpu_dev = tvm.cpu(0)  # No GPU, use CPU
        elif self.device_type == "metal":
            self.gpu_dev = tvm.metal(0)
        elif self.device_type == "cuda":
            self.gpu_dev = tvm.cuda(0)
        else:
            msg = f"Unsupported device: {device_target}"
            raise ValueError(msg)

        if hybrid:
            print(f"[HYBRID MODE] Encoder on CPU, Decoder on {self.gpu_dev}")
            self._load_hybrid()
        else:
            print(f"Loading modules on {self.gpu_dev}...")
            self._load_single_device()

    def _load_single_device(self) -> None:
        """Load all modules on the same device."""
        prefer_dylib = self.device_type == "metal"
        self.bert = self._load_mod("bert_compiled", self.gpu_dev, prefer_dylib=prefer_dylib)
        self.duration = self._load_mod("duration_compiled", self.gpu_dev, prefer_dylib=prefer_dylib)
        self.f0n = self._load_mod("f0n_compiled", self.gpu_dev, prefer_dylib=prefer_dylib)
        self.text_encoder = self._load_mod("text_encoder_compiled", self.gpu_dev, prefer_dylib=prefer_dylib)

        self.f_bert = self.bert["bert_forward"]
        self.f_duration = self.duration["duration_forward"]
        self.f_f0n = self.f0n["f0n_forward"]
        self.f_text_enc = self.text_encoder["text_encoder_forward"]
        self._load_decoders(self.gpu_dev, prefer_dylib=prefer_dylib)

    def _load_hybrid(self) -> None:
        """Load encoder on CPU, decoder on GPU."""
        # Encoder modules on CPU (uses .so files)
        print("  Loading encoder modules on CPU...")
        self.bert = self._load_mod("bert_compiled", self.cpu_dev, prefer_dylib=False)
        self.duration = self._load_mod("duration_compiled", self.cpu_dev, prefer_dylib=False)
        self.f0n = self._load_mod("f0n_compiled", self.cpu_dev, prefer_dylib=False)
        self.text_encoder = self._load_mod("text_encoder_compiled", self.cpu_dev, prefer_dylib=False)

        # Decoder on GPU (uses .dylib for Metal)
        print(f"  Loading decoder on {self.gpu_dev}...")

        self.f_bert = self.bert["bert_forward"]
        self.f_duration = self.duration["duration_forward"]
        self.f_f0n = self.f0n["f0n_forward"]
        self.f_text_enc = self.text_encoder["text_encoder_forward"]
        self._load_decoders(self.gpu_dev, prefer_dylib=(self.device_type == "metal"))

    def _load_mod(self, name: str, dev: tvm.runtime.Device, prefer_dylib: bool = False) -> tvm.runtime.Module:
        path_so = os.path.join(self.lib_dir, f"{name}.so")
        path_dylib = os.path.join(self.lib_dir, f"{name}.dylib")

        if prefer_dylib:
            path = path_dylib if os.path.exists(path_dylib) else path_so
        else:
            path = path_so if os.path.exists(path_so) else path_dylib

        if not os.path.exists(path):
            msg = f"Module {name} not found at {path_so} or {path_dylib}"
            raise FileNotFoundError(msg)
        if DEBUG:
            print(f"    Loading {path}...")
        lib = tvm.runtime.load_module(path)
        return tvm.relax.VirtualMachine(lib, dev)

    def _find_decoder_bucket_paths(self, *, prefer_dylib: bool) -> dict[int, str]:
        lib_dir = Path(self.lib_dir)
        if not lib_dir.exists():
            return {}

        exts = [".dylib", ".so"] if prefer_dylib else [".so", ".dylib"]
        out: dict[int, str] = {}
        for ext in exts:
            for p in lib_dir.glob(f"decoder_compiled_seq*{ext}"):
                m = re.match(rf"decoder_compiled_seq(\d+){re.escape(ext)}$", p.name)
                if not m:
                    continue
                seq_len = int(m.group(1))
                if seq_len in out:
                    continue
                out[seq_len] = str(p)
        return out

    def _find_decoder_default_path(self, *, prefer_dylib: bool) -> str | None:
        lib_dir = Path(self.lib_dir)
        exts = [".dylib", ".so"] if prefer_dylib else [".so", ".dylib"]
        for ext in exts:
            p = lib_dir / f"decoder_compiled{ext}"
            if p.exists():
                return str(p)
        return None

    def _load_decoders(self, dev: tvm.runtime.Device, *, prefer_dylib: bool) -> None:
        """Eagerly load all available decoder buckets.

        Bucketed decoders are named: decoder_compiled_seq{N}.{so,dylib}.
        If no bucketed decoder exists, fall back to decoder_compiled.{so,dylib}.
        """
        self._decoder_vms: dict[int, tvm.relax.VirtualMachine] = {}
        self._decoder_fns: dict[int, object] = {}

        bucket_paths = self._find_decoder_bucket_paths(prefer_dylib=prefer_dylib)
        if bucket_paths:
            for seq_len, path in sorted(bucket_paths.items()):
                if DEBUG:
                    print(f"    Loading {path}...")
                lib = tvm.runtime.load_module(path)
                vm = tvm.relax.VirtualMachine(lib, dev)
                self._decoder_vms[seq_len] = vm
                self._decoder_fns[seq_len] = vm["decoder_forward"]

            default_path = self._find_decoder_default_path(prefer_dylib=prefer_dylib)
            if default_path is not None and STATIC_AUDIO_LEN not in self._decoder_fns:
                if DEBUG:
                    print(f"    Loading {default_path}...")
                lib = tvm.runtime.load_module(default_path)
                vm = tvm.relax.VirtualMachine(lib, dev)
                self._decoder_vms[STATIC_AUDIO_LEN] = vm
                self._decoder_fns[STATIC_AUDIO_LEN] = vm["decoder_forward"]

            self._decoder_bucket_lens = sorted(self._decoder_fns.keys())
            # Backward-compatible single-decoder aliases.
            max_bucket = self._decoder_bucket_lens[-1]
            self.decoder = self._decoder_vms[max_bucket]
            self.f_decoder = self._decoder_fns[max_bucket]
            print(f"Loaded {len(self._decoder_bucket_lens)} decoder bucket(s): {self._decoder_bucket_lens}")
            return

        default_path = self._find_decoder_default_path(prefer_dylib=prefer_dylib)
        if default_path is None:
            msg = f"Decoder module not found in {self.lib_dir} (decoder_compiled.* or decoder_compiled_seq*.*)"
            raise FileNotFoundError(msg)
        if DEBUG:
            print(f"    Loading {default_path}...")
        lib = tvm.runtime.load_module(default_path)
        vm = tvm.relax.VirtualMachine(lib, dev)
        self._decoder_vms[STATIC_AUDIO_LEN] = vm
        self._decoder_fns[STATIC_AUDIO_LEN] = vm["decoder_forward"]
        self._decoder_bucket_lens = [STATIC_AUDIO_LEN]
        self.decoder = vm
        self.f_decoder = vm["decoder_forward"]

    def _select_decoder_bucket(self, frames: int) -> int:
        """Select the smallest decoder bucket that fits the given frame count."""
        frames = int(frames)
        if not self._decoder_bucket_lens:
            return STATIC_AUDIO_LEN
        if frames <= 0:
            return self._decoder_bucket_lens[0]
        for b in self._decoder_bucket_lens:
            if b >= frames:
                return b
        msg = f"Decoder bucket too small for frames={frames}; max_bucket={self._decoder_bucket_lens[-1]}."
        raise ValueError(msg)

    def _debug(self, msg: str, start_time: float | None = None) -> None:
        """Print debug message with optional elapsed time."""
        if DEBUG:
            if start_time is not None:
                elapsed = time.time() - start_time
                print(f"[DEBUG] {msg} ({elapsed:.3f}s)")
            else:
                print(f"[DEBUG] {msg}")

    def _run_inference(
        self,
        input_ids: NDArray[np.integer],
        ref_s: NDArray[np.floating],
        speed: float = 1.0,
        trace: bool = False,
    ) -> tuple[NDArray[np.floating], dict[str, object] | None]:
        """Core inference logic shared by forward() and trace().

        Args:
            input_ids: [1, seq_len] int64 array
            ref_s: [1, 256] float32 array (style embedding)
            speed: Speech speed multiplier
            trace: If True, collect intermediate tensors for debugging

        Returns:
            Tuple of (audio, trace_dict) where trace_dict is None if trace=False.
        """
        t0 = time.time()
        self._debug("Starting inference...")

        enc_dev = self.cpu_dev if self.hybrid else self.gpu_dev
        dec_dev = self.gpu_dev

        out: dict[str, object] = {}

        # --- Preprocess Inputs ---
        cur_len = int(input_ids.shape[1])
        if cur_len > STATIC_TEXT_LEN:
            input_ids = input_ids[:, :STATIC_TEXT_LEN]
            cur_len = STATIC_TEXT_LEN
        elif cur_len < STATIC_TEXT_LEN:
            pad = np.zeros((1, STATIC_TEXT_LEN - cur_len), dtype=np.int64)
            input_ids = np.concatenate([input_ids, pad], axis=1)

        # Text mask: True where position > length
        text_mask = np.zeros((1, STATIC_TEXT_LEN), dtype=np.bool_)
        text_mask[:, cur_len:] = True

        attention_mask = (~text_mask).astype(np.int64)  # BERT expects int64
        input_lengths = np.array([cur_len], dtype=np.int64)
        self._debug("Preprocessing done", t0)

        if trace:
            out["cur_len"] = cur_len
            out["input_ids"] = input_ids
            out["attention_mask"] = attention_mask
            out["text_mask"] = text_mask
            out["input_lengths"] = input_lengths

        # --- BERT: input_ids, attention_mask -> d_en [B, 512, seq_len] ---
        t1 = time.time()
        self._debug("Running BERT (CPU)..." if self.hybrid else "Running BERT...")
        bert_inputs = [
            tvm.runtime.tensor(input_ids, device=enc_dev),
            tvm.runtime.tensor(attention_mask, device=enc_dev),
        ]
        d_en_tvm = self.f_bert(*bert_inputs)[0]  # Single output
        self._debug("BERT done", t1)

        if trace:
            out["d_en"] = d_en_tvm.numpy()

        # --- Style embedding: ref_s[:, 128:] -> s [B, 128] ---
        s = ref_s[:, 128:]

        # --- Duration module: d_en, s, lengths, mask -> (duration_logits, d) ---
        t2 = time.time()
        self._debug("Running Duration module (CPU)..." if self.hybrid else "Running Duration module...")
        duration_inputs = [
            d_en_tvm,
            tvm.runtime.tensor(s, device=enc_dev),
            tvm.runtime.tensor(input_lengths, device=enc_dev),
            tvm.runtime.tensor(text_mask, device=enc_dev),
        ]
        duration_out = self.f_duration(*duration_inputs)
        duration_logits = duration_out[0].numpy()  # [B, seq_len, bins]
        d_np = duration_out[1].numpy()  # [B, seq_len, d_hid + style_dim]
        self._debug("Duration done", t2)

        if trace:
            out["duration_logits"] = duration_logits
            out["d"] = d_np

        # --- Alignment logic (pure NumPy) ---
        t3 = time.time()
        self._debug("Computing alignment...")

        # Compute predicted durations from logits
        duration_probs = _sigmoid(duration_logits.astype(np.float32))
        duration_sum = duration_probs.sum(axis=-1) / speed  # [B, seq_len]
        pred_dur = np.round(duration_sum).clip(min=1).astype(np.int64).squeeze()  # [seq_len]
        if pred_dur.ndim == 0:
            pred_dur = pred_dur.reshape(1)
        pred_dur = pred_dur[:cur_len]

        # Build alignment matrix
        indices = np.repeat(np.arange(cur_len), pred_dur)
        actual_audio_len = len(indices)
        if actual_audio_len > STATIC_AUDIO_LEN:
            indices = indices[:STATIC_AUDIO_LEN]
            actual_audio_len = STATIC_AUDIO_LEN

        pred_aln_trg = np.zeros((cur_len, STATIC_AUDIO_LEN), dtype=np.float32)
        pred_aln_trg[indices, np.arange(len(indices))] = 1

        # Pad to static text length
        full_aln = np.zeros((1, STATIC_TEXT_LEN, STATIC_AUDIO_LEN), dtype=np.float32)
        full_aln[0, :cur_len, :] = pred_aln_trg

        # Compute en = d.T @ alignment
        # d is [B, T, d_hid + style_dim] = [B, T, 640] from DurationEncoder
        d = d_np if d_np is not None else np.zeros((1, STATIC_TEXT_LEN, 640), dtype=np.float32)
        en = np.transpose(d, (0, 2, 1)) @ full_aln  # [B, 640, audio_len]
        self._debug("Alignment done", t3)

        if trace:
            out["pred_dur"] = pred_dur
            out["frames"] = actual_audio_len

        # --- F0N module: en, s, frame_lengths -> (F0, N) ---
        t4 = time.time()
        self._debug("Running F0N module (CPU)..." if self.hybrid else "Running F0N module...")
        frame_lengths = np.array([actual_audio_len], dtype=np.int64)
        f0n_out = self.f_f0n(
            tvm.runtime.tensor(en, device=enc_dev),
            tvm.runtime.tensor(s, device=enc_dev),
            tvm.runtime.tensor(frame_lengths, device=enc_dev),
        )
        f0_np = f0n_out[0].numpy()  # [B, audio_len * 2]
        n_np = f0n_out[1].numpy()  # [B, audio_len * 2]
        self._debug("F0N done", t4)

        if trace:
            out["f0"] = f0_np
            out["n"] = n_np
            out["f0n_frame_lengths"] = frame_lengths

        # --- Text Encoder: input_ids, lengths, mask -> t_en ---
        t5 = time.time()
        self._debug("Running Text Encoder (CPU)..." if self.hybrid else "Running Text Encoder...")
        text_enc_inputs = [
            tvm.runtime.tensor(input_ids, device=enc_dev),
            tvm.runtime.tensor(input_lengths, device=enc_dev),
            tvm.runtime.tensor(text_mask, device=enc_dev),
        ]
        t_en_np = self.f_text_enc(*text_enc_inputs)[0].numpy()  # Single output
        self._debug("Text Encoder done", t5)

        if trace:
            out["t_en"] = t_en_np

        # --- ASR = t_en @ alignment ---
        asr = t_en_np @ full_aln

        # --- Decoder: asr, F0, N, style[:128] -> audio ---
        t6 = time.time()
        self._debug("Running Decoder (GPU)..." if self.hybrid else "Running Decoder...")
        bucket_len = self._select_decoder_bucket(actual_audio_len)
        if bucket_len != STATIC_AUDIO_LEN:
            self._debug(f"Decoder bucket selected: {bucket_len} (frames={actual_audio_len})")

        asr_b = asr[:, :, :bucket_len]
        f0_b = f0_np[:, : bucket_len * 2]
        n_b = n_np[:, : bucket_len * 2]
        s128 = ref_s[:, :128]

        if trace:
            out["decoder_bucket_len"] = int(bucket_len)

        f_decoder = self._decoder_fns[bucket_len]
        decoder_inputs = [
            tvm.runtime.tensor(asr_b, device=dec_dev),
            tvm.runtime.tensor(f0_b, device=dec_dev),
            tvm.runtime.tensor(n_b, device=dec_dev),
            tvm.runtime.tensor(s128, device=dec_dev),
        ]

        audio_np = f_decoder(*decoder_inputs)[0].numpy()  # Single output
        self._debug("Decoder done", t6)

        self._debug("Total inference time", t0)

        # Trim to actual waveform length (convert frames -> samples)
        audio_1d = audio_np.squeeze()
        target_samples = min(int(audio_1d.size), int(actual_audio_len) * SAMPLES_PER_FRAME)
        audio_trimmed = audio_1d[:target_samples]

        if trace:
            out["audio_trimmed"] = audio_trimmed

        return audio_trimmed, out if trace else None

    def forward(
        self,
        input_ids: NDArray[np.integer],
        ref_s: NDArray[np.floating],
        speed: float = 1.0,
    ) -> NDArray[np.floating]:
        """Run full inference.

        Args:
            input_ids: [1, seq_len] integer array or torch.LongTensor
            ref_s: [1, 256] float array or torch.FloatTensor (style embedding)
            speed: Speech speed multiplier

        Returns:
            audio: [audio_len] float32 array
        """
        # Convert torch tensors to numpy if needed
        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
        if hasattr(ref_s, "numpy"):
            ref_s = ref_s.numpy()

        audio, _ = self._run_inference(input_ids, ref_s, speed, trace=False)
        return audio

    def trace(
        self,
        input_ids: NDArray[np.integer],
        ref_s: NDArray[np.floating],
        speed: float = 1.0,
        *,
        include_full_audio: bool = False,
    ) -> dict[str, object]:
        """Run inference and return intermediate tensors for debugging/validation.

        Values are returned as CPU numpy arrays or Python scalars.
        """
        # Lazy import debug utilities
        from kokoro_tvm.debug_utils import stats_summary, tail_summary

        # Convert torch tensors to numpy if needed
        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
        if hasattr(ref_s, "numpy"):
            ref_s = ref_s.numpy()

        audio, out = self._run_inference(input_ids, ref_s, speed, trace=True)
        assert out is not None

        # Add debug summaries
        actual_audio_len = out["frames"]
        d_np = out.get("d")
        if d_np is not None:
            # Summarize alignment region beyond valid frames
            out["en_pad_summary"] = tail_summary(
                np.zeros((1, 640, STATIC_AUDIO_LEN - actual_audio_len)),
                0,
                preview=8,
                atol=0.0,
            )

        # F0/N are typically 2x aligned frames; summarize the padded tail
        valid_f0n = int(actual_audio_len) * 2
        f0_np = out.get("f0")
        n_np = out.get("n")
        if f0_np is not None:
            out["f0_pad_summary"] = tail_summary(f0_np, valid_f0n, preview=8, atol=1e-6)
        if n_np is not None:
            out["n_pad_summary"] = tail_summary(n_np, valid_f0n, preview=8, atol=1e-6)

        # Decoder input stats
        out["decoder_input_stats"] = {
            "f0": stats_summary(out.get("f0", np.array([]))),
            "n": stats_summary(out.get("n", np.array([]))),
            "style_128": stats_summary(ref_s[:, :128]),
        }

        if include_full_audio:
            # Re-run to get full untrimmmed audio (optional)
            pass  # audio_trimmed is already included

        return out
