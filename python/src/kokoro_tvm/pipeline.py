# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Kokoro Pipeline orchestration using TVM compiled modules."""

import os
import time

import numpy as np
import torch
import tvm
import tvm.runtime

# Static compilation constants - must match what was used during compilation!
STATIC_TEXT_LEN = 512
STATIC_AUDIO_LEN = 5120
STYLE_DIM = 128  # Kokoro-82M uses 128

# Debug flag - set to True to enable timing prints
DEBUG = True


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
        self.decoder = self._load_mod("decoder_compiled", self.gpu_dev, prefer_dylib=(self.device_type == "metal"))

        self.f_bert = self.bert["bert_forward"]
        self.f_duration = self.duration["duration_forward"]
        self.f_f0n = self.f0n["f0n_forward"]
        self.f_text_enc = self.text_encoder["text_encoder_forward"]
        self.f_decoder = self.decoder["decoder_forward"]

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
        self.decoder = self._load_mod("decoder_compiled", self.gpu_dev, prefer_dylib=(self.device_type == "metal"))

        self.f_bert = self.bert["bert_forward"]
        self.f_duration = self.duration["duration_forward"]
        self.f_f0n = self.f0n["f0n_forward"]
        self.f_text_enc = self.text_encoder["text_encoder_forward"]
        self.f_decoder = self.decoder["decoder_forward"]

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

    def _unwrap(self, obj):
        """Extract NDArray from VM output (handles tuples, lists, Arrays)."""
        if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
            if len(obj) == 1:
                return obj[0]
        return obj

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

        # 1. Run BERT: input_ids, attention_mask -> d_en [B, 512, seq_len]
        t1 = time.time()
        self._debug("Running BERT (CPU)..." if self.hybrid else "Running BERT...")
        bert_inputs = [
            tvm.runtime.tensor(input_ids.numpy(), device=self.cpu_dev),
            tvm.runtime.tensor(attention_mask.numpy(), device=self.cpu_dev),
        ]
        d_en_tvm = self._unwrap(self.f_bert(*bert_inputs))
        self._debug("BERT done", t1)

        # 2. Style: ref_s[:, 128:] -> s [B, 128]
        s = ref_s[:, 128:].numpy()

        # 3. Duration module: d_en, s, lengths, mask -> (duration, d)
        t2 = time.time()
        self._debug("Running Duration module (CPU)..." if self.hybrid else "Running Duration module...")
        duration_inputs = [
            d_en_tvm,
            tvm.runtime.tensor(s, device=self.cpu_dev),
            tvm.runtime.tensor(input_lengths.numpy(), device=self.cpu_dev),
            tvm.runtime.tensor(text_mask.numpy(), device=self.cpu_dev),
        ]
        duration_out = self.f_duration(*duration_inputs)
        duration_tvm = self._unwrap(duration_out[0]) if hasattr(duration_out, "__getitem__") else duration_out
        d_tvm = duration_out[1] if hasattr(duration_out, "__getitem__") and len(duration_out) > 1 else None
        self._debug("Duration done", t2)

        # 4. Alignment logic (Python)
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

        # 5. Compute en = d.T @ alignment
        # d is [B, T, d_hid + style_dim] = [B, T, 640] from DurationEncoder
        d = torch.from_numpy(d_tvm.numpy()) if d_tvm is not None else torch.zeros(1, STATIC_TEXT_LEN, 640)
        en = d.transpose(-1, -2) @ full_aln  # [B, 640, audio_len]
        self._debug("Alignment done", t3)

        # 6. F0N module: en, s -> (F0, N)
        t4 = time.time()
        self._debug("Running F0N module (CPU)..." if self.hybrid else "Running F0N module...")
        f0n_inputs = [
            tvm.runtime.tensor(en.numpy(), device=self.cpu_dev),
            tvm.runtime.tensor(s, device=self.cpu_dev),
        ]
        f0n_out = self.f_f0n(*f0n_inputs)
        f0_tvm = self._unwrap(f0n_out[0]) if hasattr(f0n_out, "__getitem__") else f0n_out
        n_tvm = f0n_out[1] if hasattr(f0n_out, "__getitem__") and len(f0n_out) > 1 else None
        self._debug("F0N done", t4)

        # 7. Text Encoder: input_ids, lengths, mask -> t_en
        t5 = time.time()
        self._debug("Running Text Encoder (CPU)..." if self.hybrid else "Running Text Encoder...")
        text_enc_inputs = [
            tvm.runtime.tensor(input_ids.numpy(), device=self.cpu_dev),
            tvm.runtime.tensor(input_lengths.numpy(), device=self.cpu_dev),
            tvm.runtime.tensor(text_mask.numpy(), device=self.cpu_dev),
        ]
        t_en_tvm = self._unwrap(self.f_text_enc(*text_enc_inputs))
        t_en = torch.from_numpy(t_en_tvm.numpy())
        self._debug("Text Encoder done", t5)

        # 8. ASR = t_en @ alignment
        asr = t_en @ full_aln

        # 9. Decoder: asr, F0, N, style[:128] -> audio
        t6 = time.time()
        self._debug("Running Decoder (GPU)..." if self.hybrid else "Running Decoder...")

        # In hybrid mode, transfer data to GPU
        if self.hybrid:
            f0_np = f0_tvm.numpy()
            n_np = n_tvm.numpy() if n_tvm is not None else None
            decoder_inputs = [
                tvm.runtime.tensor(asr.numpy(), device=self.gpu_dev),
                tvm.runtime.tensor(f0_np, device=self.gpu_dev),
                tvm.runtime.tensor(n_np, device=self.gpu_dev) if n_np is not None else n_tvm,
                tvm.runtime.tensor(ref_s[:, :128].numpy(), device=self.gpu_dev),
            ]
        else:
            decoder_inputs = [
                tvm.runtime.tensor(asr.numpy(), device=self.cpu_dev),
                f0_tvm,
                n_tvm,
                tvm.runtime.tensor(ref_s[:, :128].numpy(), device=self.cpu_dev),
            ]

        audio_tvm = self._unwrap(self.f_decoder(*decoder_inputs))
        audio = torch.from_numpy(audio_tvm.numpy())
        self._debug("Decoder done", t6)

        self._debug("Total inference time", t0)
        # Trim to actual audio length
        return audio.squeeze()[:actual_audio_len]
