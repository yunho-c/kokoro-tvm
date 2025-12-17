# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Decoder model creation for TVM compilation.

This module provides functions to create Relax IRModules from the Kokoro
decoder, ready for target-specific compilation and optimization.
"""

import json
from typing import Optional

import torch
import tvm
from huggingface_hub import hf_hub_download
from kokoro.istftnet import Decoder
from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter

from kokoro_tvm import tvm_extensions  # noqa: F401 - apply TVM patches
from kokoro_tvm.patches.adain import apply_adain_patch
from kokoro_tvm.patches.sinegen import apply_sinegen_patch


def create_decoder_module(
    seq_len: int = 150,
    load_weights: bool = True,
    func_name: str = "decoder_forward",
    dump_ir: Optional[str] = None,
) -> tvm.IRModule:
    """Create a Relax IRModule for the Kokoro decoder.

    This function handles the full pipeline:
    1. Load model config from HuggingFace
    2. Initialize decoder with optional pretrained weights
    3. Export with torch.export
    4. Import to TVM Relax

    The resulting module is target-agnostic and can be compiled
    for any supported target (CPU, Metal, CUDA, etc.).

    Args:
        seq_len: Static sequence length for compilation
        load_weights: Whether to load pretrained weights (default: True)
        func_name: Name for the main function (default: "decoder_forward")
        dump_ir: Optional path to dump the IR before optimization

    Returns:
        tvm.IRModule ready for compilation/tuning

    Example:
        >>> mod = create_decoder_module(seq_len=150)
        >>> # Now compile for a specific target
        >>> target = tvm.target.Target("metal", host="llvm -mtriple=arm64-apple-macos")
        >>> ex = relax.build(mod, target)
    """
    # Apply patches before export
    apply_sinegen_patch()
    apply_adain_patch()

    # Load config from HuggingFace
    repo_id = "hexgrad/Kokoro-82M"
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    hidden_dim = config["hidden_dim"]
    style_dim = config["style_dim"]
    n_mels = config["n_mels"]
    istftnet_params = config["istftnet"]

    # Initialize decoder
    decoder = Decoder(dim_in=hidden_dim, style_dim=style_dim, dim_out=n_mels, disable_complex=True, **istftnet_params)

    # Load pretrained weights if requested
    if load_weights:
        model_filename = "kokoro-v1_0.pth"
        print(f"Downloading pretrained weights: {model_filename}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

        print(f"Loading decoder weights from {model_path}...")
        state_dicts = torch.load(model_path, map_location="cpu", weights_only=True)

        if "decoder" in state_dicts:
            decoder_state_dict = state_dicts["decoder"]
            first_key = next(iter(decoder_state_dict.keys()), "")
            if first_key.startswith("module."):
                print("Stripping 'module.' prefix from weight keys...")
                decoder_state_dict = {k[7:]: v for k, v in decoder_state_dict.items()}
            decoder.load_state_dict(decoder_state_dict, strict=False)
            print("Successfully loaded pretrained decoder weights!")
        else:
            print("Warning: 'decoder' key not found in checkpoint. Using random weights.")
    else:
        print("Skipping weight loading. Using random weights.")

    decoder.eval()

    # Create static inputs
    batch_size = 1
    asr = torch.randn(batch_size, hidden_dim, seq_len)
    f0 = torch.randn(batch_size, seq_len * 2)
    n = torch.randn(batch_size, seq_len * 2)
    s = torch.randn(batch_size, style_dim)

    print(f"Static inputs: asr={asr.shape}, f0={f0.shape}, n={n.shape}, s={s.shape}")

    # Export with torch.export
    print(f"Exporting model with static seq_len={seq_len}...")
    exported_program = torch.export.export(
        decoder,
        (asr, f0, n, s),
    )

    # Run decompositions
    print("Running decompositions...")
    try:
        exported_program = exported_program.run_decompositions()
        print("Decompositions complete.")
    except Exception as e:
        print(f"Warning: run_decompositions failed: {e}. Continuing anyway.")

    # Import to TVM Relax
    print("Importing into TVM Relax...")
    print("Importing into TVM Relax...")
    from tvm.relax.frontend.torch import from_exported_program

    mod = from_exported_program(
        exported_program, keep_params_as_input=False, unwrap_unit_return_tuple=False, no_bind_return_tuple=False
    )

    print("Relax Module created.")

    # Dump IR for debugging if requested
    if dump_ir:
        with open(dump_ir, "w") as f:
            f.write(mod.script())
        print(f"Dumped IR to {dump_ir}")

    # Rename 'main' to specified function name
    new_funcs = {}
    for gv, func in mod.functions.items():
        if gv.name_hint == "main":
            new_gv = tvm.ir.GlobalVar(func_name)
            if hasattr(func, "attrs") and func.attrs is not None and "global_symbol" in func.attrs:
                new_attrs = dict(func.attrs)
                new_attrs["global_symbol"] = func_name
                func = func.with_attrs(new_attrs)
                print(f"Updated function global_symbol to '{func_name}'")
            new_funcs[new_gv] = func
            print(f"Renamed function 'main' to '{func_name}'")
        else:
            new_funcs[gv] = func

    mod = tvm.IRModule(new_funcs, attrs=mod.attrs)
    print(f"Module has {len(mod.functions)} functions")

    return mod


def get_decoder_config() -> dict:
    """Get the Kokoro decoder configuration.

    Returns:
        Dictionary with hidden_dim, style_dim, n_mels, and istftnet params
    """
    repo_id = "hexgrad/Kokoro-82M"
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)
