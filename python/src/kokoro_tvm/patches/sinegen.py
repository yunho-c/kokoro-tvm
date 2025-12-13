# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""SineGen patches for dynamic shape compatibility with TVM export."""

import torch
import torch.nn.functional as F

# Lazy import to avoid circular imports at module load time
_patched = False


def _f02sine_friendly(self, f0_values):
    """Dynamic-shape-friendly replacement for SineGen._f02sine.
    
    This patch avoids scale_factor interpolation which confuses torch.export,
    instead using explicit size-based interpolation.
    """
    rad_values = (f0_values / self.sampling_rate) % 1
    rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    
    if not self.flag_for_pulse:
        tgt_len = f0_values.shape[1] // self.upsample_scale
        rad_values = F.interpolate(
            rad_values.transpose(1, 2), size=tgt_len, mode="linear"
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale, 
            size=f0_values.shape[1], 
            mode="linear"
        ).transpose(1, 2)
        sines = torch.sin(phase)
    else:
        # Fallback to original for pulse mode
        from kokoro.istftnet import SineGen
        return SineGen._old_f02sine(self, f0_values)
    return sines


def apply_sinegen_patch():
    """Apply the SineGen._f02sine patch if not already applied.
    
    This must be called before torch.export to ensure dynamic shapes work.
    """
    global _patched
    if _patched:
        return
    
    from kokoro.istftnet import SineGen
    
    if not hasattr(SineGen, '_old_f02sine'):
        SineGen._old_f02sine = SineGen._f02sine
        SineGen._f02sine = _f02sine_friendly
        _patched = True
