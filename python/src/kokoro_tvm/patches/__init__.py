# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Patches for TVM export compatibility."""

from kokoro_tvm.patches.sinegen import apply_sinegen_patch
from kokoro_tvm.patches.lstm import apply_lstm_patch
from kokoro_tvm.patches.modules import (
    apply_text_encoder_patch,
    apply_prosody_predictor_patch,
    apply_duration_encoder_patch,
    apply_adain_patch,
    apply_all_module_patches,
)
from kokoro_tvm.ops.lstm_custom_op import patch_lstm_modules as apply_lstm_custom_op_patch

__all__ = [
    "apply_sinegen_patch",
    "apply_lstm_patch",
    "apply_lstm_custom_op_patch",
    "apply_text_encoder_patch",
    "apply_prosody_predictor_patch",
    "apply_duration_encoder_patch",
    "apply_adain_patch",
    "apply_all_module_patches",
]
