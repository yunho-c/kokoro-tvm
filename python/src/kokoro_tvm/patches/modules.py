# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Kokoro module patches for TVM export compatibility.

This module patches kokoro modules to remove pack_padded_sequence/pad_packed_sequence
which are not compatible with TVM export.
"""

import torch
import torch.nn.functional as F
from torch import nn

_text_encoder_patched = False
_prosody_predictor_patched = False
_duration_encoder_patched = False
_adain_patched = False


def apply_text_encoder_patch():
    """Patch TextEncoder to remove pack/unpack operations."""
    global _text_encoder_patched
    if _text_encoder_patched:
        return

    from kokoro.modules import TextEncoder

    def text_encoder_forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)

        # Skip packing
        x, _ = self.lstm(x)
        # Skip unpacking

        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x

    TextEncoder.forward = text_encoder_forward
    _text_encoder_patched = True


def apply_prosody_predictor_patch():
    """Patch ProsodyPredictor to remove pack/unpack operations."""
    global _prosody_predictor_patched
    if _prosody_predictor_patched:
        return

    from kokoro.modules import ProsodyPredictor

    def prosody_predictor_forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        m = m.unsqueeze(1)

        # Skip packing
        x, _ = self.lstm(d)
        # Skip unpacking

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, :x.shape[1], :] = x
        x = x_pad
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en

    ProsodyPredictor.forward = prosody_predictor_forward
    _prosody_predictor_patched = True


def apply_duration_encoder_patch():
    """Patch DurationEncoder to remove pack/unpack operations."""
    global _duration_encoder_patched
    if _duration_encoder_patched:
        return

    from kokoro.modules import AdaLayerNorm, DurationEncoder

    def duration_encoder_forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                # block is LSTM
                x = x.transpose(-1, -2)
                # Skip packing
                x, _ = block(x)
                # Skip unpacking

                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)

    DurationEncoder.forward = duration_encoder_forward
    _duration_encoder_patched = True


def apply_adain_patch():
    """Patch AdaIN1d.forward to handle dynamic shape check."""
    global _adain_patched
    if _adain_patched:
        return

    from kokoro.istftnet import AdaIN1d

    original_adain_forward = AdaIN1d.forward

    def new_adain_forward(self, x, s):
        # Hint that sequence length is > 1 (we set min=2)
        if x.dim() == 3:
            torch._check(x.size(2) > 1)
        return original_adain_forward(self, x, s)

    AdaIN1d.forward = new_adain_forward
    _adain_patched = True


def apply_all_module_patches():
    """Apply all module patches needed for TVM export.
    
    This is a convenience function to apply all patches at once.
    """
    apply_text_encoder_patch()
    apply_prosody_predictor_patch()
    apply_duration_encoder_patch()
    apply_adain_patch()
