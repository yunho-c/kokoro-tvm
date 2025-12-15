# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Encoder model creation for TVM compilation.

This module provides functions to create Relax IRModules for the Kokoro encoder
components (BERT, ProsodyPredictor, TextEncoder).
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import tvm
from huggingface_hub import hf_hub_download
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
from transformers import AlbertConfig
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter

from kokoro_tvm import tvm_extensions  # noqa: F401
from kokoro_tvm.patches.lstm import apply_lstm_patch
from kokoro_tvm.ops.lstm_custom_op import patch_lstm_modules as apply_lstm_custom_op_patch


_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_CHECKPOINT_CACHE: Dict[str, Dict[str, Any]] = {}


def get_kokoro_config() -> Dict[str, Any]:
    """Get the full Kokoro model configuration."""
    repo_id = "hexgrad/Kokoro-82M"
    if repo_id in _CONFIG_CACHE:
        return _CONFIG_CACHE[repo_id]
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _CONFIG_CACHE[repo_id] = cfg
    return cfg


def _load_kokoro_checkpoint(repo_id: str = "hexgrad/Kokoro-82M") -> Dict[str, Any]:
    if repo_id in _CHECKPOINT_CACHE:
        return _CHECKPOINT_CACHE[repo_id]
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename="kokoro-v1_0.pth")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _CHECKPOINT_CACHE[repo_id] = state
    return state


def _load_state_dict(module: torch.nn.Module, state_dict: Dict[str, Any], name: str) -> None:
    try:
        module.load_state_dict(state_dict)
        return
    except Exception:
        first_key = next(iter(state_dict.keys()), "")
        if first_key.startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            module.load_state_dict(state_dict, strict=False)
            return
        module.load_state_dict(state_dict, strict=False)
        return


def _export_and_import(
    model: torch.nn.Module, args: Tuple[Any, ...], func_name: str, dump_ir: Optional[str] = None
) -> tvm.IRModule:
    """Helper to export PyTorch model and import to TVM Relax."""

    print(f"Exporting {func_name} with torch.export...")
    exported_program = torch.export.export(model, args)

    print("Importing into TVM Relax...")
    mod = from_exported_program(
        exported_program, keep_params_as_input=False, unwrap_unit_return_tuple=False, no_bind_return_tuple=False
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

    # Apply DCE to reduce IR size
    print("Applying DeadCodeElimination...")
    mod = relax.transform.DeadCodeElimination()(mod)

    if dump_ir:
        with open(dump_ir, "w") as f:
            f.write(mod.script())
        print(f"Dumped IR to {dump_ir}")

    return mod


def create_bert_module(load_weights: bool = True, dump_ir: Optional[str] = None) -> tvm.IRModule:
    """Create Relax IRModule for CustomAlbert with Linear projection."""

    print("Initializing CustomAlbert...")

    cfg = get_kokoro_config()
    config = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])

    class BertWrapper(torch.nn.Module):
        """Wraps CustomAlbert + Linear projection to match KModel structure."""

        def __init__(self, config):
            super().__init__()
            self.bert = CustomAlbert(config)
            self.bert_encoder = torch.nn.Linear(config.hidden_size, cfg["hidden_dim"])

        def forward(self, input_ids, attention_mask):
            bert_dur = self.bert(input_ids, attention_mask=attention_mask)
            return self.bert_encoder(bert_dur).transpose(-1, -2)

    model = BertWrapper(config)
    model.eval()

    if load_weights:
        state = _load_kokoro_checkpoint()
        _load_state_dict(model.bert, state["bert"], "bert")
        _load_state_dict(model.bert_encoder, state["bert_encoder"], "bert_encoder")

    BATCH_SIZE = 1
    SEQ_LEN = model.bert.config.max_position_embeddings
    input_ids = torch.randint(0, cfg["n_token"], (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)

    return _export_and_import(model, (input_ids, attention_mask), "bert_forward", dump_ir)


def create_duration_module(
    style_dim: int = 128,
    d_hid: int = 512,
    nlayers: int = 3,
    max_dur: int = 50,
    dropout: float = 0.0,
    seq_len: int = 512,
    load_weights: bool = True,
    dump_ir: Optional[str] = None,
) -> tvm.IRModule:
    """Create Relax IRModule for ProsodyPredictor duration computation.

    This exports the path: text_encoder -> lstm -> duration_proj
    Returns: (duration, d) where d is the intermediate tensor for computing en.
    """
    cfg = get_kokoro_config()
    if load_weights:
        style_dim = cfg["style_dim"]
        d_hid = cfg["hidden_dim"]
        nlayers = cfg["n_layer"]
        max_dur = cfg["max_dur"]
        dropout = cfg["dropout"]

    print("Initializing ProsodyPredictor for duration...")
    predictor = ProsodyPredictor(style_dim=style_dim, d_hid=d_hid, nlayers=nlayers, max_dur=max_dur, dropout=dropout)
    predictor.eval()

    if load_weights:
        state = _load_kokoro_checkpoint()
        _load_state_dict(predictor, state["predictor"], "predictor")

    apply_lstm_patch()

    # Replace nn.LSTM with our custom op wrapper to preserve as single node during export
    apply_lstm_custom_op_patch(predictor)

    class DurationWrapper(torch.nn.Module):
        """Wraps the duration path of ProsodyPredictor (text_encoder -> lstm -> duration_proj)."""

        def __init__(self, predictor, seq_len):
            super().__init__()
            self.text_encoder = predictor.text_encoder
            self.lstm = predictor.lstm
            self.duration_proj = predictor.duration_proj
            self.seq_len = seq_len

        def forward(self, d_en, style, text_lengths, m):
            # text_encoder returns d: [B, T, d_hid + style_dim] = [B, T, 640]
            d = self.text_encoder(d_en, style, text_lengths, m)

            # LSTM already expects [B, T, input_size] which d is
            self.lstm.flatten_parameters()
            x, _ = self.lstm(d)

            # Pad to static length
            x_pad = torch.zeros([x.shape[0], self.seq_len, x.shape[-1]], device=x.device)
            x_pad[:, : x.shape[1], :] = x
            x = x_pad

            # Project to duration
            duration = self.duration_proj(x)
            return duration.squeeze(-1), d

    model = DurationWrapper(predictor, seq_len)
    model.eval()

    batch_size = 1
    input_d_en = torch.randn(batch_size, d_hid, seq_len)
    input_style = torch.randn(batch_size, style_dim)
    input_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.long)
    input_m = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    return _export_and_import(model, (input_d_en, input_style, input_lengths, input_m), "duration_forward", dump_ir)


def create_f0n_module(
    style_dim: int = 128,
    d_hid: int = 512,
    nlayers: int = 3,
    max_dur: int = 50,
    dropout: float = 0.0,
    aligned_len: int = 5120,
    load_weights: bool = True,
    dump_ir: Optional[str] = None,
) -> tvm.IRModule:
    """Create Relax IRModule for ProsodyPredictor F0N computation.

    This exports F0Ntrain(en, s) -> (F0, N)
    Note: en has shape [B, d_hid + style_dim, aligned_len] = [B, 640, aligned_len]
    because DurationEncoder output includes concatenated style.
    """
    cfg = get_kokoro_config()
    if load_weights:
        style_dim = cfg["style_dim"]
        d_hid = cfg["hidden_dim"]
        nlayers = cfg["n_layer"]
        max_dur = cfg["max_dur"]
        dropout = cfg["dropout"]

    print("Initializing ProsodyPredictor for F0N...")
    predictor = ProsodyPredictor(style_dim=style_dim, d_hid=d_hid, nlayers=nlayers, max_dur=max_dur, dropout=dropout)
    predictor.eval()

    if load_weights:
        state = _load_kokoro_checkpoint()
        _load_state_dict(predictor, state["predictor"], "predictor")

    apply_lstm_patch()

    # Replace nn.LSTM with our custom op wrapper to preserve as single node during export
    apply_lstm_custom_op_patch(predictor)

    class F0NWrapper(torch.nn.Module):
        """Wraps the F0Ntrain path of ProsodyPredictor."""

        def __init__(self, predictor):
            super().__init__()
            self.shared = predictor.shared
            self.F0 = predictor.F0
            self.N = predictor.N
            self.F0_proj = predictor.F0_proj
            self.N_proj = predictor.N_proj

        def forward(self, en, style):
            # en shape: [B, d_hid + style_dim, aligned_len] = [B, 640, aligned_len]
            # transpose to [B, aligned_len, 640] for LSTM
            self.shared.flatten_parameters()
            x, _ = self.shared(en.transpose(-1, -2))  # [B, aligned, d_hid]
            F0 = x.transpose(-1, -2)
            for block in self.F0:
                F0 = block(F0, style)
            F0 = self.F0_proj(F0)
            N = x.transpose(-1, -2)
            for block in self.N:
                N = block(N, style)
            N = self.N_proj(N)
            return F0.squeeze(1), N.squeeze(1)

    model = F0NWrapper(predictor)
    model.eval()

    batch_size = 1
    # en is d @ alignment: [B, d_hid + style_dim, aligned_len]
    # Because DurationEncoder output includes style concatenation
    en_dim = d_hid + style_dim
    input_en = torch.randn(batch_size, en_dim, aligned_len)
    input_style = torch.randn(batch_size, style_dim)

    return _export_and_import(model, (input_en, input_style), "f0n_forward", dump_ir)


def create_text_encoder_module(
    channels: int = 512,
    kernel_size: int = 5,
    depth: int = 3,
    n_symbols: int = 178,
    seq_len: int = 50,
    load_weights: bool = True,
    dump_ir: Optional[str] = None,
) -> tvm.IRModule:
    """Create Relax IRModule for TextEncoder."""

    cfg = get_kokoro_config()
    if load_weights:
        channels = cfg["hidden_dim"]
        kernel_size = cfg["text_encoder_kernel_size"]
        depth = cfg["n_layer"]
        n_symbols = cfg["n_token"]

    print("Initializing TextEncoder...")
    model = TextEncoder(channels=channels, kernel_size=kernel_size, depth=depth, n_symbols=n_symbols)
    model.eval()

    if load_weights:
        state = _load_kokoro_checkpoint()
        _load_state_dict(model, state["text_encoder"], "text_encoder")

    apply_lstm_patch()

    # Replace nn.LSTM with our custom op wrapper to preserve as single node during export
    apply_lstm_custom_op_patch(model)

    # Static Inputs
    BATCH_SIZE = 1
    input_x = torch.randint(0, n_symbols, (BATCH_SIZE, seq_len), dtype=torch.long)
    input_lengths = torch.tensor([seq_len] * BATCH_SIZE, dtype=torch.long)
    input_m = torch.zeros(BATCH_SIZE, seq_len, dtype=torch.bool)

    return _export_and_import(model, (input_x, input_lengths, input_m), "text_encoder_forward", dump_ir)
