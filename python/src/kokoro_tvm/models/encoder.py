# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Encoder model creation for TVM compilation.

This module provides functions to create Relax IRModules for the Kokoro encoder
components (BERT, ProsodyPredictor, TextEncoder).
"""

import json
import os
from typing import Optional, Tuple, Dict, Any

import torch
import tvm
from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig

from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
from kokoro_tvm import tvm_extensions  # noqa: F401
from kokoro_tvm.patches.lstm import apply_lstm_patch


def get_kokoro_config() -> Dict[str, Any]:
    """Get the full Kokoro model configuration."""
    repo_id = 'hexgrad/Kokoro-82M'
    config_path = hf_hub_download(repo_id=repo_id, filename='config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _export_and_import(
    model: torch.nn.Module,
    args: Tuple[Any, ...],
    func_name: str,
    dump_ir: Optional[str] = None
) -> tvm.IRModule:
    """Helper to export PyTorch model and import to TVM Relax."""
    
    print(f"Exporting {func_name} with torch.export...")
    exported_program = torch.export.export(model, args)
    
    print("Importing into TVM Relax...")
    # Use the top-level function which is more stable across versions/patches
    from tvm.relax.frontend.torch import from_exported_program
    mod = from_exported_program(
        exported_program,
        keep_params_as_input=False,
        unwrap_unit_return_tuple=False, 
        no_bind_return_tuple=False
    )
    
    # Rename main function
    new_funcs = {}
    for gv, func in mod.functions.items():
        if gv.name_hint == "main":
            new_gv = tvm.ir.GlobalVar(func_name)
            if hasattr(func, "attrs") and func.attrs is not None and "global_symbol" in func.attrs:
                new_attrs = dict(func.attrs)
                new_attrs["global_symbol"] = func_name
                func = func.with_attrs(new_attrs)
            new_funcs[new_gv] = func
        else:
            new_funcs[gv] = func
            
    mod = tvm.IRModule(new_funcs, attrs=mod.attrs)
    
    if dump_ir:
        with open(dump_ir, "w") as f:
            f.write(mod.script())
        print(f"Dumped IR to {dump_ir}")
        
    return mod


def create_bert_module(
    vocab_size: int = 100, # Mock/Testing defaults, should be from config
    dump_ir: Optional[str] = None
) -> tvm.IRModule:
    """Create Relax IRModule for CustomAlbert (Text processing)."""
    
    # Apply patches (though Albert usually doesn't need LSTM patch, harmless)
    
    print("Initializing CustomAlbert...")
    # NOTE: In real usage we should load config from HF. 
    # For now, we reuse the config structure or defaults if feasible.
    # But Kokoro uses a specific AlbertConfig.
    # We will fetch it from 'hexgrad/Kokoro-82M' config.json if possible, 
    # but the config.json there is for the custom assembly, not raw AlbertConfig.
    # Actually, CustomAlbert in kokoro/modules.py seems to take a standard AlbertConfig?
    # No, it inherits from AlbertModel.
    # We should stick to what `kokoro.model.KModel` does.
    # KModel: self.bert = CustomAlbert(base_model.config) where base_model is loaded via transformers?
    # "bert = load_model(bert_path)" ?
    # Let's use a standard config for portability or use values from model config if available.
    # For now, let's use the parameters compatible with Kokoro-82M.
    
    # Assuming standard albert-base-v2 or similar?
    # Kokoro-82M uses a specific style.
    # Let's load the model config parameters if we can.
    
    # For simplicity in this porting step, we use explicit params matching known Kokoro sizes if known,
    # or rely on load_weights logic.
    # But to create the *structure* for compilation, we need the config.
    
    # We'll use a standard config for now suitable for 82M logic or mock it.
    # Let's use inputs compatible with experiments/albert_import.py
    
    config = AlbertConfig(
        vocab_size=30000, # Approx for english/multi
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    
    model = CustomAlbert(config)
    model.eval()
    
    # Static Inputs
    BATCH_SIZE = 1
    SEQ_LEN = 50 # Example static length
    input_ids = torch.randint(0, 30000, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    
    return _export_and_import(model, (input_ids, attention_mask), "bert_forward", dump_ir)


def create_predictor_module(
    style_dim: int = 64,
    d_hid: int = 512,
    nlayers: int = 8,
    max_dur: int = 50,
    dropout: float = 0.1,
    seq_len: int = 50,
    dump_ir: Optional[str] = None
) -> tvm.IRModule:
    """Create Relax IRModule for ProsodyPredictor."""
    
    apply_lstm_patch()
    
    print("Initializing ProsodyPredictor...")
    model = ProsodyPredictor(
        style_dim=style_dim, 
        d_hid=d_hid, 
        nlayers=nlayers, 
        max_dur=max_dur, 
        dropout=0.0 # Deterministic
    )
    model.eval()
    
    # Static Inputs
    BATCH_SIZE = 1
    # Note: Prosody uses TextEncoder output which has dim d_hid (if using same as DurationEncoder?)
    # or d_model?
    # In Kokoro-82M config: `predictor_params` usually has d_hid.
    
    input_texts = torch.randn(BATCH_SIZE, d_hid, seq_len)
    input_style = torch.randn(BATCH_SIZE, style_dim)
    input_lengths = torch.tensor([seq_len] * BATCH_SIZE, dtype=torch.long)
    
    # Alignment: mapping text seq_len to audio frames.
    # Let's assume audio frames = seq_len * 2 for dummy.
    aligned_len = seq_len * 2
    input_alignment = torch.randn(BATCH_SIZE, seq_len, aligned_len)
    
    input_m = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.bool)
    
    return _export_and_import(
        model, 
        (input_texts, input_style, input_lengths, input_alignment, input_m), 
        "prosody_forward", 
        dump_ir
    )


def create_text_encoder_module(
    channels: int = 512,
    kernel_size: int = 5,
    depth: int = 3,
    n_symbols: int = 178, # Kokoro default
    seq_len: int = 50,
    dump_ir: Optional[str] = None
) -> tvm.IRModule:
    """Create Relax IRModule for TextEncoder."""
    
    apply_lstm_patch()
    
    print("Initializing TextEncoder...")
    model = TextEncoder(
        channels=channels, 
        kernel_size=kernel_size, 
        depth=depth, 
        n_symbols=n_symbols
    )
    model.eval()
    
    # Static Inputs
    BATCH_SIZE = 1
    input_x = torch.randint(0, n_symbols, (BATCH_SIZE, seq_len), dtype=torch.long)
    input_lengths = torch.tensor([seq_len] * BATCH_SIZE, dtype=torch.long)
    input_m = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.bool)
    
    return _export_and_import(
        model, 
        (input_x, input_lengths, input_m), 
        "text_encoder_forward", 
        dump_ir
    )
