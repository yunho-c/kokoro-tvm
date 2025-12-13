# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""LSTM patches for TVM export compatibility.

This module provides:
1. A custom kokoro::lstm op to prevent LSTM decomposition during export
2. A patched nn.LSTM.forward that uses this custom op
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

_lstm_op_registered = False
_lstm_forward_patched = False


def register_kokoro_lstm_op():
    """Register the kokoro::lstm custom op to prevent decomposition.
    
    This custom op wraps torch.ops.aten.lstm.input and prevents it from
    being decomposed into primitive operations during torch.export.
    """
    global _lstm_op_registered
    if _lstm_op_registered:
        return
    
    try:
        @torch.library.custom_op("kokoro::lstm", mutates_args=())
        def kokoro_lstm(
            input: Tensor, 
            hx: List[Tensor], 
            params: List[Tensor], 
            has_biases: bool, 
            num_layers: int, 
            dropout: float, 
            train: bool, 
            bidirectional: bool, 
            batch_first: bool
        ) -> Tuple[Tensor, Tensor, Tensor]:
            return torch.ops.aten.lstm.input(
                input, hx, params, has_biases, num_layers, 
                dropout, train, bidirectional, batch_first
            )

        @kokoro_lstm.register_fake
        def kokoro_lstm_fake(
            input, hx, params, has_biases, num_layers, 
            dropout, train, bidirectional, batch_first
        ):
            if batch_first:
                batch_size = input.size(0)
                seq_len = input.size(1)
            else:
                seq_len = input.size(0)
                batch_size = input.size(1)
                
            num_directions = 2 if bidirectional else 1
            hidden_size = hx[0].size(2)
            
            if batch_first:
                output_shape = (batch_size, seq_len, num_directions * hidden_size)
            else:
                output_shape = (seq_len, batch_size, num_directions * hidden_size)
                
            h_n_shape = (num_layers * num_directions, batch_size, hidden_size)
            c_n_shape = (num_layers * num_directions, batch_size, hidden_size)
            
            output = input.new_empty(output_shape)
            h_n = input.new_empty(h_n_shape)
            c_n = input.new_empty(c_n_shape)
            
            return output, h_n, c_n
        
        _lstm_op_registered = True
            
    except AttributeError:
        # Fallback for older PyTorch if custom_op is missing
        print("Warning: torch.library.custom_op not found. Using direct call (might fail).")


def apply_lstm_forward_patch():
    """Patch nn.LSTM.forward to handle hidden state init and use kokoro::lstm.
    
    This patch:
    1. Initializes hidden state if None (required for export)
    2. Uses the custom kokoro::lstm op to prevent decomposition
    """
    global _lstm_forward_patched
    if _lstm_forward_patched:
        return
    
    original_lstm_forward = nn.LSTM.forward
    
    def new_lstm_forward(self, input, hx=None):
        # Handle hidden state init
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            if self.batch_first:
                batch_size = input.size(0)
            else:
                batch_size = input.size(1)
                
            h_0 = torch.zeros(
                self.num_layers * num_directions, batch_size, self.hidden_size, 
                device=input.device, dtype=input.dtype
            )
            c_0 = torch.zeros(
                self.num_layers * num_directions, batch_size, self.hidden_size, 
                device=input.device, dtype=input.dtype
            )
            hx = (h_0, c_0)

        # Use custom op to prevent decomposition
        params = self._flat_weights
        has_biases = self.bias
        num_layers = self.num_layers
        dropout = self.dropout
        train = self.training
        bidirectional = self.bidirectional
        batch_first = self.batch_first
        
        # hx must be a list for custom op if defined as List[Tensor]
        hx_list = list(hx)
        
        output, h_n, c_n = torch.ops.kokoro.lstm(
            input, 
            hx_list, 
            params, 
            has_biases, 
            num_layers, 
            dropout, 
            train, 
            bidirectional, 
            batch_first
        )
        
        return output, (h_n, c_n)
    
    nn.LSTM.forward = new_lstm_forward
    _lstm_forward_patched = True
