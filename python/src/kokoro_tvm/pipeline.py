# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Kokoro Pipeline orchestration using TVM compiled modules."""

import os
import torch
import numpy as np
import tvm
import tvm.runtime
from typing import Tuple, Optional

# Static compilation constants
# Must match what was used during compilation!
STATIC_TEXT_LEN = 512
STATIC_AUDIO_LEN = 5120 # Matches seq_len*10 in encoder.py

class KokoroPipeline:
    def __init__(self, lib_dir: str, device_target: str = "llvm"):
        """Initialize pipeline with compiled TVM modules.
        
        Args:
            lib_dir: Directory containing .so/.dylib files
            device_target: 'llvm', 'metal', 'cuda'
        """
        self.lib_dir = lib_dir
        self.device_type = device_target.split()[0] if " " in device_target else device_target
        
        # Load Device
        if self.device_type == "llvm":
            self.dev = tvm.cpu(0)
        elif self.device_type == "metal":
            self.dev = tvm.metal(0)
        elif self.device_type == "cuda":
            self.dev = tvm.cuda(0)
        else:
            raise ValueError(f"Unsupported device: {device_target}")
            
        print(f"Loading modules on {self.dev}...")
        self.bert = self._load_mod("bert_compiled")
        self.prosody = self._load_mod("prosody_compiled")
        self.text_encoder = self._load_mod("text_encoder_compiled")
        self.decoder = self._load_mod("decoder_compiled")
        
        self.f_bert = self.bert["bert_forward"]
        self.f_prosody = self.prosody["prosody_forward"]
        self.f_text_enc = self.text_encoder["text_encoder_forward"]
        self.f_decoder = self.decoder["decoder_forward"]

    def _load_mod(self, name: str) -> tvm.runtime.Module:
        path_so = os.path.join(self.lib_dir, f"{name}.so")
        path_dylib = os.path.join(self.lib_dir, f"{name}.dylib")
        path = path_so if os.path.exists(path_so) else path_dylib
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Module {name} not found at {path_so} or {path_dylib}")
        lib = tvm.runtime.load_module(path)
        return tvm.relax.VirtualMachine(lib, self.dev)

    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1.0) -> torch.FloatTensor:
        """Run full inference.
        
        Args:
            input_ids: [1, seq_len] LongTensor
            ref_s: [1, 256] FloatTensor (style)
            speed: float
            
        Returns:
            audio: [audio_len] FloatTensor
        """
        # Preprocess Inputs
        cur_len = input_ids.shape[1]
        if cur_len > STATIC_TEXT_LEN:
             input_ids = input_ids[:, :STATIC_TEXT_LEN]
             cur_len = STATIC_TEXT_LEN
        elif cur_len < STATIC_TEXT_LEN:
             pad = torch.zeros((1, STATIC_TEXT_LEN - cur_len), dtype=torch.long)
             input_ids = torch.cat([input_ids, pad], dim=1)
             
        attention_mask = torch.zeros((1, STATIC_TEXT_LEN), dtype=torch.long)
        attention_mask[:, :cur_len] = 1
        
        input_lengths = torch.tensor([cur_len], dtype=torch.long)
        
        # Run BERT
        bert_inputs = [
            tvm.runtime.tensor(input_ids.numpy(), device=self.dev),
            tvm.runtime.tensor(attention_mask.numpy(), device=self.dev)
        ]
        d_en_tvm = self.f_bert(*bert_inputs)
        
        # Predict Duration (Pass 1)
        s = ref_s[:, 128:].numpy()
        
        dummy_aln = np.zeros((1, STATIC_TEXT_LEN, STATIC_AUDIO_LEN), dtype=np.float32)
        dummy_m = np.zeros((1, STATIC_TEXT_LEN), dtype=bool)
        
        prosody_inputs = [
            d_en_tvm, # d_en
            tvm.runtime.tensor(s, device=self.dev), # input_style
            tvm.runtime.tensor(input_lengths.numpy(), device=self.dev), # input_lengths
            tvm.runtime.tensor(dummy_aln, device=self.dev), # input_alignment
            tvm.runtime.tensor(dummy_m, device=self.dev) # input_m
        ]
        
        duration_tvm, _, _ = self.f_prosody(*prosody_inputs)
        duration = torch.from_numpy(duration_tvm.numpy()).float()
        
        # Alignment Logic
        duration = torch.sigmoid(duration).sum(dim=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        
        if pred_dur.dim() == 0: pred_dur = pred_dur.unsqueeze(0)
        
        pred_dur = pred_dur[:cur_len]
        
        # Build Alignment
        indices = torch.repeat_interleave(torch.arange(cur_len), pred_dur)
        if len(indices) > STATIC_AUDIO_LEN:
            indices = indices[:STATIC_AUDIO_LEN]
        
        pred_aln_trg = torch.zeros((cur_len, STATIC_AUDIO_LEN))
        pred_aln_trg[indices, torch.arange(len(indices))] = 1
        
        # Pad back to [STATIC_TEXT, STATIC_AUDIO]
        full_aln = torch.zeros((1, STATIC_TEXT_LEN, STATIC_AUDIO_LEN), dtype=torch.float32)
        full_aln[0, :cur_len, :] = pred_aln_trg
        
        # Prosody (Pass 2)
        prosody_inputs[3] = tvm.runtime.tensor(full_aln.numpy(), device=self.dev)
        _, f0_tvm, n_tvm = self.f_prosody(*prosody_inputs)
        
        # Text Encoder
        text_enc_inputs = [
           tvm.runtime.tensor(input_ids.numpy(), device=self.dev),
           tvm.runtime.tensor(input_lengths.numpy(), device=self.dev),
           tvm.runtime.tensor(dummy_m, device=self.dev)
        ]
        t_en_tvm = self.f_text_enc(*text_enc_inputs)
        t_en = torch.from_numpy(t_en_tvm.numpy())
        
        asr = torch.bmm(t_en, torch.from_numpy(full_aln.numpy()))
        
        # Decoder
        decoder_inputs = [
            tvm.runtime.tensor(asr.numpy(), device=self.dev),
            f0_tvm,
            n_tvm,
            tvm.runtime.tensor(ref_s[:, :128].numpy(), device=self.dev)
        ]
        
        audio_tvm = self.f_decoder(*decoder_inputs)
        return torch.from_numpy(audio_tvm.numpy())
