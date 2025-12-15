# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Patches for torch.nn.LSTM to support TVM export.

This module provides a monkeypatch for pack_padded_sequence and pad_packed_sequence
to avoid creating PackedSequence objects, which are not supported by torch.export due to
dynamic data-dependent shapes.
"""

from torch import nn


def apply_lstm_patch():
    """Apply mokeypatch to torch.nn.utils.rnn for PackedSequence handling.
    
    This replaces pack_padded_sequence and pad_packed_sequence with pass-through mocks,
    forcing the model to use padded tensors directly. This is necessary for static graph
    export to TVM.
    """
    print("Applying LSTM PackedSequence monkeypatch for TVM export...")

    def mock_pack(input, lengths, batch_first=False, enforce_sorted=True):
        # Simply return the input tensor, ignoring packing
        return input

    def mock_pad(sequence, batch_first=False, padding_value=0.0, total_length=None):
        # Return the sequence (which is the input tensor from mock_pack)
        # and None for lengths.
        return sequence, None

    nn.utils.rnn.pack_padded_sequence = mock_pack
    nn.utils.rnn.pad_packed_sequence = mock_pad
    print("LSTM patch applied.")
