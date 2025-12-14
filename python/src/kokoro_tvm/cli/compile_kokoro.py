# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for compiling full Kokoro model to TVM Relax."""

import argparse
import os

import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from torch.nn.attention import sdpa_kernel, SDPBackend

from kokoro import KModel
from kokoro.model import KModelForONNX

# Import extensions (applies TVM patches on import)
from kokoro_tvm import tvm_extensions  # noqa: F401
from kokoro_tvm.patches import (
    apply_all_module_patches,
    apply_sinegen_patch,
)
from kokoro_tvm.patches.lstm import apply_lstm_patch


def compile_kokoro(model, output_dir: str):
    """Compile Kokoro model to TVM Relax IR.
    
    Args:
        model: KModelForONNX instance
        output_dir: Directory to save compiled artifacts
    """
    print("Tracing model with torch.export...")
    
    # Define symbolic dimensions
    seq_len = torch.export.Dim("seq_len", min=2, max=512)
    
    # Create dummy inputs
    dummy_input_ids = torch.randint(0, 100, (1, 50), dtype=torch.long)
    dummy_style = torch.randn(1, 256, dtype=torch.float32)
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)

    # Specify dynamic shapes
    dynamic_shapes = {
        "input_ids": {1: seq_len},
        "ref_s": None,
        "speed": None,
    }

    # Disable mkldnn to avoid potential dynamic shape issues
    torch.backends.mkldnn.enabled = False

    # Apply all patches
    apply_sinegen_patch()
    apply_lstm_patch()
    apply_all_module_patches()

    # Export the program
    with sdpa_kernel(SDPBackend.MATH):
        exported_program = torch.export.export(
            model,
            (dummy_input_ids, dummy_style, dummy_speed),
            dynamic_shapes=dynamic_shapes
        )
    
    print("Importing to TVM Relax...")
    mod = from_exported_program(exported_program)
    
    # Basic optimization pipeline
    print("Applying optimizations...")
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        relax.transform.DeadCodeElimination(),
        relax.transform.CanonicalizeBindings(),
    ])
    
    mod = seq(mod)
    
    # Print the module to verify
    print("Compilation successful!")
    print(mod.script(show_meta=False)[:1000] + "\n...")
    
    # Save the module
    output_path = os.path.join(output_dir, "kokoro_relax.json")
    with open(output_path, "w") as f:
        f.write(tvm.ir.save_json(mod))
    print(f"Saved Relax module to {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser("Compile Kokoro Model to TVM Relax", add_help=True)
    parser.add_argument(
        "--config_file", "-c", type=str, required=False, 
        help="path to config file"
    )
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=False, 
        help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="tvm_output", 
        help="output directory"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model...")
    kmodel = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True)
    model = KModelForONNX(kmodel).eval()

    compile_kokoro(model, args.output_dir)


if __name__ == "__main__":
    main()
