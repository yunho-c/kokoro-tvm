# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Kokoro Pipeline orchestration using TVM compiled modules."""

import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import tvm
import tvm.runtime

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


def _tail_summary(x: np.ndarray, valid: int, *, preview: int = 8, atol: float = 0.0) -> dict[str, object]:
    flat = np.asarray(x).reshape(-1)
    total = int(flat.size)
    valid = int(max(0, min(valid, total)))
    pad = flat[valid:]

    if pad.size == 0:
        return {
            "valid": valid,
            "total": total,
            "pad": 0,
            "finite_frac": 1.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "nonzero_frac": 0.0,
            "nonzero_atol": atol,
            "head": [],
            "tail": [],
        }

    pad_f32 = pad.astype(np.float32, copy=False)
    finite = np.isfinite(pad_f32)
    finite_frac = float(np.mean(finite)) if pad_f32.size else 1.0

    abs_pad = np.abs(pad_f32[finite]) if np.any(finite) else np.array([], dtype=np.float32)
    max_abs = float(np.max(abs_pad)) if abs_pad.size else float("nan")
    mean_abs = float(np.mean(abs_pad)) if abs_pad.size else float("nan")
    nonzero_frac = float(np.mean(np.abs(pad_f32) > atol)) if pad_f32.size else 0.0

    head = pad_f32[:preview].tolist()
    tail = pad_f32[-preview:].tolist() if pad_f32.size >= preview else pad_f32.tolist()

    return {
        "valid": valid,
        "total": total,
        "pad": int(pad_f32.size),
        "finite_frac": finite_frac,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "nonzero_frac": nonzero_frac,
        "nonzero_atol": atol,
        "head": head,
        "tail": tail,
    }


def _stats_summary(x: np.ndarray, *, atol: float = 1e-8) -> dict[str, float]:
    arr = np.asarray(x).reshape(-1).astype(np.float32, copy=False)
    if arr.size == 0:
        return {"finite_frac": 1.0, "nonzero_frac": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    finite = np.isfinite(arr)
    finite_frac = float(np.mean(finite))
    arr_f = arr[finite] if np.any(finite) else np.array([], dtype=np.float32)
    min_v = float(np.min(arr_f)) if arr_f.size else float("nan")
    max_v = float(np.max(arr_f)) if arr_f.size else float("nan")
    mean_v = float(np.mean(arr_f)) if arr_f.size else float("nan")
    std_v = float(np.std(arr_f)) if arr_f.size else float("nan")
    nonzero_frac = float(np.mean(np.abs(arr) > atol))
    return {
        "finite_frac": finite_frac,
        "nonzero_frac": nonzero_frac,
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
    }


class KokoroPipeline:
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
        self._f0n_accepts_lengths: bool | None = None
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

    def _load_single_device(self):
        """Load all modules on the same device."""
        self.bert = self._load_mod("bert_compiled", self.gpu_dev, prefer_dylib=(self.device_type == "metal"))
        self.duration = self._load_mod("duration_compiled", self.gpu_dev, prefer_dylib=(self.device_type == "metal"))
        self.f0n = self._load_mod("f0n_compiled", self.gpu_dev, prefer_dylib=(self.device_type == "metal"))
        self.text_encoder = self._load_mod(
            "text_encoder_compiled", self.gpu_dev, prefer_dylib=(self.device_type == "metal")
        )

        self.f_bert = self.bert["bert_forward"]
        self.f_duration = self.duration["duration_forward"]
        self.f_f0n = self.f0n["f0n_forward"]
        self.f_text_enc = self.text_encoder["text_encoder_forward"]
        self._load_decoders(self.gpu_dev, prefer_dylib=(self.device_type == "metal"))

    def _load_hybrid(self):
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

    def _unwrap(self, obj):
        """Extract NDArray from VM output (handles tuples, lists, Arrays)."""
        if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
            if len(obj) == 1:
                return obj[0]
        return obj

    def _call_f0n(self, en, s, frame_lengths, enc_dev: tvm.runtime.Device):
        """Call f0n_forward with best-effort support for legacy signatures.

        Newer builds export: f0n_forward(en, style, frame_lengths).
        Older builds export: f0n_forward(en, style).
        """
        if self._f0n_accepts_lengths is None:
            try:
                out = self.f_f0n(en, tvm.runtime.tensor(s, device=enc_dev), tvm.runtime.tensor(frame_lengths, device=enc_dev))
                self._f0n_accepts_lengths = True
                return out
            except tvm.error.TVMError as e:
                msg = str(e)
                if "expects 2 arguments" in msg and "but 3 arguments were provided" in msg:
                    print(
                        "Warning: f0n_forward uses legacy signature (en, style). "
                        "Recompile encoder f0n to enable frame_lengths-aware behavior."
                    )
                    self._f0n_accepts_lengths = False
                else:
                    raise

        if self._f0n_accepts_lengths:
            return self.f_f0n(en, tvm.runtime.tensor(s, device=enc_dev), tvm.runtime.tensor(frame_lengths, device=enc_dev))
        return self.f_f0n(en, tvm.runtime.tensor(s, device=enc_dev))

    def _debug(self, msg: str, start_time: float = None):
        """Print debug message with optional elapsed time."""
        if DEBUG:
            if start_time is not None:
                elapsed = time.time() - start_time
                print(f"[DEBUG] {msg} ({elapsed:.3f}s)")
            else:
                print(f"[DEBUG] {msg}")

    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1.0) -> torch.FloatTensor:
        """Run full inference matching KModel.forward_with_tokens.

        Args:
            input_ids: [1, seq_len] LongTensor
            ref_s: [1, 256] FloatTensor (style)
            speed: float

        Returns:
            audio: [audio_len] FloatTensor
        """
        t0 = time.time()
        self._debug("Starting inference...")

        enc_dev = self.cpu_dev if self.hybrid else self.gpu_dev
        dec_dev = self.gpu_dev

        # Preprocess Inputs
        cur_len = input_ids.shape[1]
        if cur_len > STATIC_TEXT_LEN:
            input_ids = input_ids[:, :STATIC_TEXT_LEN]
            cur_len = STATIC_TEXT_LEN
        elif cur_len < STATIC_TEXT_LEN:
            pad = torch.zeros((1, STATIC_TEXT_LEN - cur_len), dtype=torch.long)
            input_ids = torch.cat([input_ids, pad], dim=1)

        # Create text mask: True where position > length
        text_mask = torch.zeros((1, STATIC_TEXT_LEN), dtype=torch.bool)
        text_mask[:, cur_len:] = True

        attention_mask = (~text_mask).long()  # BERT expects int64
        input_lengths = torch.tensor([cur_len], dtype=torch.long)
        self._debug("Preprocessing done", t0)

        # Run BERT: input_ids, attention_mask -> d_en [B, 512, seq_len]
        t1 = time.time()
        self._debug("Running BERT (CPU)..." if self.hybrid else "Running BERT...")
        bert_inputs = [
            tvm.runtime.tensor(input_ids.numpy(), device=enc_dev),
            tvm.runtime.tensor(attention_mask.numpy(), device=enc_dev),
        ]
        d_en_tvm = self._unwrap(self.f_bert(*bert_inputs))
        self._debug("BERT done", t1)

        # Style: ref_s[:, 128:] -> s [B, 128]
        s = ref_s[:, 128:].numpy()

        # Duration module: d_en, s, lengths, mask -> (duration, d)
        t2 = time.time()
        self._debug("Running Duration module (CPU)..." if self.hybrid else "Running Duration module...")
        duration_inputs = [
            d_en_tvm,
            tvm.runtime.tensor(s, device=enc_dev),
            tvm.runtime.tensor(input_lengths.numpy(), device=enc_dev),
            tvm.runtime.tensor(text_mask.numpy(), device=enc_dev),
        ]
        duration_out = self.f_duration(*duration_inputs)
        duration_tvm = self._unwrap(duration_out[0]) if hasattr(duration_out, "__getitem__") else duration_out
        d_tvm = duration_out[1] if hasattr(duration_out, "__getitem__") and len(duration_out) > 1 else None
        self._debug("Duration done", t2)

        # Alignment logic (Python)
        t3 = time.time()
        self._debug("Computing alignment...")
        duration = torch.from_numpy(duration_tvm.numpy()).float()
        duration = torch.sigmoid(duration).sum(dim=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
        pred_dur = pred_dur[:cur_len]

        # Build alignment matrix
        indices = torch.repeat_interleave(torch.arange(cur_len), pred_dur)
        actual_audio_len = len(indices)
        if actual_audio_len > STATIC_AUDIO_LEN:
            indices = indices[:STATIC_AUDIO_LEN]
            actual_audio_len = STATIC_AUDIO_LEN

        pred_aln_trg = torch.zeros((cur_len, STATIC_AUDIO_LEN))
        pred_aln_trg[indices, torch.arange(len(indices))] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)  # [1, cur_len, audio_len]

        # Pad to static text length
        full_aln = torch.zeros((1, STATIC_TEXT_LEN, STATIC_AUDIO_LEN), dtype=torch.float32)
        full_aln[0, :cur_len, :] = pred_aln_trg[0]

        # Compute en = d.T @ alignment
        # d is [B, T, d_hid + style_dim] = [B, T, 640] from DurationEncoder
        d = torch.from_numpy(d_tvm.numpy()) if d_tvm is not None else torch.zeros(1, STATIC_TEXT_LEN, 640)
        en = d.transpose(-1, -2) @ full_aln  # [B, 640, audio_len]
        self._debug("Alignment done", t3)

        # F0N module: en, s, frame_lengths -> (F0, N)
        t4 = time.time()
        self._debug("Running F0N module (CPU)..." if self.hybrid else "Running F0N module...")
        frame_lengths = np.array([actual_audio_len], dtype=np.int64)
        f0n_out = self._call_f0n(tvm.runtime.tensor(en.numpy(), device=enc_dev), s, frame_lengths, enc_dev)
        f0_tvm = self._unwrap(f0n_out[0]) if hasattr(f0n_out, "__getitem__") else f0n_out
        n_tvm = f0n_out[1] if hasattr(f0n_out, "__getitem__") and len(f0n_out) > 1 else None
        self._debug("F0N done", t4)

        # Text Encoder: input_ids, lengths, mask -> t_en
        t5 = time.time()
        self._debug("Running Text Encoder (CPU)..." if self.hybrid else "Running Text Encoder...")
        text_enc_inputs = [
            tvm.runtime.tensor(input_ids.numpy(), device=enc_dev),
            tvm.runtime.tensor(input_lengths.numpy(), device=enc_dev),
            tvm.runtime.tensor(text_mask.numpy(), device=enc_dev),
        ]
        t_en_tvm = self._unwrap(self.f_text_enc(*text_enc_inputs))
        t_en = torch.from_numpy(t_en_tvm.numpy())
        self._debug("Text Encoder done", t5)

        # ASR = t_en @ alignment
        asr = t_en @ full_aln

        # Decoder: asr, F0, N, style[:128] -> audio
        t6 = time.time()
        self._debug("Running Decoder (GPU)..." if self.hybrid else "Running Decoder...")
        bucket_len = self._select_decoder_bucket(actual_audio_len)
        if bucket_len != STATIC_AUDIO_LEN:
            self._debug(f"Decoder bucket selected: {bucket_len} (frames={actual_audio_len})")

        asr_np = asr.numpy()[:, :, :bucket_len]
        f0_np = f0_tvm.numpy()[:, : bucket_len * 2]
        n_np = n_tvm.numpy()[:, : bucket_len * 2] if n_tvm is not None else None
        s128_np = ref_s[:, :128].numpy()

        f_decoder = self._decoder_fns[bucket_len]
        decoder_inputs = [
            tvm.runtime.tensor(asr_np, device=dec_dev),
            tvm.runtime.tensor(f0_np, device=dec_dev),
            tvm.runtime.tensor(n_np, device=dec_dev) if n_np is not None else n_tvm,
            tvm.runtime.tensor(s128_np, device=dec_dev),
        ]

        audio_tvm = self._unwrap(f_decoder(*decoder_inputs))
        audio = torch.from_numpy(audio_tvm.numpy())
        self._debug("Decoder done", t6)

        self._debug("Total inference time", t0)
        # Trim to actual waveform length (convert frames -> samples).
        audio_1d = audio.squeeze()
        target_samples = min(int(audio_1d.numel()), int(actual_audio_len) * SAMPLES_PER_FRAME)
        return audio_1d[:target_samples]

    def trace(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        include_full_audio: bool = False,
    ) -> dict[str, object]:
        """Run inference and return intermediate tensors for debugging/validation.

        Values are returned as CPU numpy arrays or Python scalars.
        """
        enc_dev = self.cpu_dev if self.hybrid else self.gpu_dev
        dec_dev = self.gpu_dev

        cur_len = int(input_ids.shape[1])
        if cur_len > STATIC_TEXT_LEN:
            input_ids = input_ids[:, :STATIC_TEXT_LEN]
            cur_len = STATIC_TEXT_LEN
        elif cur_len < STATIC_TEXT_LEN:
            pad = torch.zeros((1, STATIC_TEXT_LEN - cur_len), dtype=torch.long)
            input_ids = torch.cat([input_ids, pad], dim=1)

        text_mask = torch.zeros((1, STATIC_TEXT_LEN), dtype=torch.bool)
        text_mask[:, cur_len:] = True
        attention_mask = (~text_mask).long()
        input_lengths = torch.tensor([cur_len], dtype=torch.long)

        out: dict[str, object] = {
            "cur_len": cur_len,
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "text_mask": text_mask.numpy(),
            "input_lengths": input_lengths.numpy(),
        }

        bert_inputs = [
            tvm.runtime.tensor(input_ids.numpy(), device=enc_dev),
            tvm.runtime.tensor(attention_mask.numpy(), device=enc_dev),
        ]
        d_en_tvm = self._unwrap(self.f_bert(*bert_inputs))
        out["d_en"] = d_en_tvm.numpy()

        s = ref_s[:, 128:].numpy()
        duration_inputs = [
            d_en_tvm,
            tvm.runtime.tensor(s, device=enc_dev),
            tvm.runtime.tensor(input_lengths.numpy(), device=enc_dev),
            tvm.runtime.tensor(text_mask.numpy(), device=enc_dev),
        ]
        duration_out = self.f_duration(*duration_inputs)
        duration_tvm = self._unwrap(duration_out[0]) if hasattr(duration_out, "__getitem__") else duration_out
        d_tvm = duration_out[1] if hasattr(duration_out, "__getitem__") and len(duration_out) > 1 else None

        out["duration_logits"] = duration_tvm.numpy()
        out["d"] = d_tvm.numpy() if d_tvm is not None else None

        duration = torch.from_numpy(duration_tvm.numpy()).float()
        duration = torch.sigmoid(duration).sum(dim=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
        pred_dur = pred_dur[:cur_len]

        indices = torch.repeat_interleave(torch.arange(cur_len), pred_dur)
        actual_audio_len = int(indices.numel())
        if actual_audio_len > STATIC_AUDIO_LEN:
            indices = indices[:STATIC_AUDIO_LEN]
            actual_audio_len = STATIC_AUDIO_LEN

        out["pred_dur"] = pred_dur.cpu().numpy()
        out["frames"] = actual_audio_len

        pred_aln_trg = torch.zeros((cur_len, STATIC_AUDIO_LEN))
        pred_aln_trg[indices, torch.arange(int(indices.numel()))] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        full_aln = torch.zeros((1, STATIC_TEXT_LEN, STATIC_AUDIO_LEN), dtype=torch.float32)
        full_aln[0, :cur_len, :] = pred_aln_trg[0]

        d = torch.from_numpy(d_tvm.numpy()) if d_tvm is not None else torch.zeros(1, STATIC_TEXT_LEN, 640)
        en = d.transpose(-1, -2) @ full_aln
        en_np = en.numpy()
        out["en_pad_summary"] = _tail_summary(en_np[:, :, actual_audio_len:], 0, preview=8, atol=0.0)

        frame_lengths = np.array([actual_audio_len], dtype=np.int64)
        out["f0n_frame_lengths"] = frame_lengths.copy()
        f0n_out = self._call_f0n(tvm.runtime.tensor(en_np, device=enc_dev), s, frame_lengths, enc_dev)
        f0_tvm = self._unwrap(f0n_out[0]) if hasattr(f0n_out, "__getitem__") else f0n_out
        n_tvm = f0n_out[1] if hasattr(f0n_out, "__getitem__") and len(f0n_out) > 1 else None

        f0_np = f0_tvm.numpy()
        n_np = n_tvm.numpy() if n_tvm is not None else None
        out["f0"] = f0_np
        out["n"] = n_np

        # F0/N are typically 2x aligned frames; summarize the padded tail to ensure it is truly zeroed.
        valid_f0n = int(actual_audio_len) * 2
        out["f0_pad_summary"] = _tail_summary(f0_np, valid_f0n, preview=8, atol=1e-6)
        out["n_pad_summary"] = _tail_summary(n_np, valid_f0n, preview=8, atol=1e-6) if n_np is not None else None

        text_enc_inputs = [
            tvm.runtime.tensor(input_ids.numpy(), device=enc_dev),
            tvm.runtime.tensor(input_lengths.numpy(), device=enc_dev),
            tvm.runtime.tensor(text_mask.numpy(), device=enc_dev),
        ]
        t_en_tvm = self._unwrap(self.f_text_enc(*text_enc_inputs))
        t_en = torch.from_numpy(t_en_tvm.numpy())
        out["t_en"] = t_en_tvm.numpy()

        asr = t_en @ full_aln
        out["decoder_input_stats"] = {
            "asr": _stats_summary(asr.numpy()),
            "f0": _stats_summary(f0_np),
            "n": _stats_summary(n_np) if n_np is not None else None,
            "style_128": _stats_summary(ref_s[:, :128].numpy()),
        }
        bucket_len = self._select_decoder_bucket(actual_audio_len)
        out["decoder_bucket_len"] = int(bucket_len)

        asr_np = asr.numpy()[:, :, :bucket_len]
        f0_np_b = f0_np[:, : bucket_len * 2]
        n_np_b = n_np[:, : bucket_len * 2] if n_np is not None else None
        s128_np = ref_s[:, :128].numpy()

        decoder_inputs = [
            tvm.runtime.tensor(asr_np, device=dec_dev),
            tvm.runtime.tensor(f0_np_b, device=dec_dev),
            tvm.runtime.tensor(n_np_b, device=dec_dev) if n_np_b is not None else n_tvm,
            tvm.runtime.tensor(s128_np, device=dec_dev),
        ]

        f_decoder = self._decoder_fns[bucket_len]
        audio_tvm = self._unwrap(f_decoder(*decoder_inputs))
        audio_full = audio_tvm.numpy().squeeze()
        target_samples = min(int(audio_full.size), int(actual_audio_len) * SAMPLES_PER_FRAME)
        out["audio_trimmed"] = audio_full[:target_samples]
        if include_full_audio:
            out["audio_full"] = audio_full

        return out
