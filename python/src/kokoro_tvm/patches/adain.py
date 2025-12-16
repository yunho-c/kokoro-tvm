# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""AdaIN/InstanceNorm patches to improve numerical stability under TVM export.

TVM's lowering of `nn.InstanceNorm1d` can produce non-finite values for some
channels (likely via an unstable variance formulation). Kokoro's decoder uses
InstanceNorm1d inside AdaIN blocks, and a single NaN in an input channel
propagates through subsequent convolutions.

This patch replaces `AdaIN1d.forward` to compute InstanceNorm1d explicitly via
a stable variance definition: `mean((x - mean(x))^2)`.
"""

from __future__ import annotations

import torch

_patched = False


def _instance_norm1d_stable(x: torch.Tensor, norm: torch.nn.InstanceNorm1d) -> torch.Tensor:
    mean = x.mean(dim=2, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=2, keepdim=True)
    y = (x - mean) / torch.sqrt(var + norm.eps)
    if getattr(norm, "affine", False):
        y = y * norm.weight.view(1, -1, 1) + norm.bias.view(1, -1, 1)
    return y


def _adain1d_forward_stable(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    h = self.fc(s)
    h = h.view(h.size(0), h.size(1), 1)
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    x_norm = _instance_norm1d_stable(x, self.norm)
    return (1 + gamma) * x_norm + beta


def apply_adain_patch() -> None:
    global _patched
    if _patched:
        return

    from kokoro.istftnet import AdaIN1d

    if not hasattr(AdaIN1d, "_old_forward"):
        AdaIN1d._old_forward = AdaIN1d.forward
        AdaIN1d.forward = _adain1d_forward_stable

    _patched = True

