import argparse
import os
import sys
import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# # Add external/kokoro to path to import kokoro package
# current_dir = os.path.dirname(os.path.abspath(__file__))
# kokoro_path = os.path.join(current_dir, "..", "external", "kokoro")
# sys.path.append(kokoro_path)

from kokoro import KModel
from kokoro.model import KModelForONNX

def compile_kokoro(model, output_dir):
    print("Tracing model with torch.export...")
    
    # Define symbolic dimensions
    batch = torch.export.Dim("batch", min=1)
    seq_len = torch.export.Dim("seq_len", min=1)
    
    # Create dummy inputs
    # input_ids: (batch, seq_len)
    dummy_input_ids = torch.randint(0, 100, (1, 50), dtype=torch.long)
    # ref_s: (batch, 256) - style vector
    dummy_style = torch.randn(1, 256, dtype=torch.float32)
    # speed: (batch,) - speed factor
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)

    # Specify dynamic shapes
    dynamic_shapes = {
        "input_ids": {0: batch, 1: seq_len},
        "ref_s": {0: batch},
        "speed": {0: batch},
    }

    # Export the program
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compile Kokoro Model to TVM Relax", add_help=True)
    parser.add_argument(
        # "--config_file", "-c", type=str, default="checkpoints/config.json", help="path to config file"
        "--config_file", "-c", type=str, required=False, help="path to config file"
    )
    parser.add_argument(
        # "--checkpoint_path", "-p", type=str, default="checkpoints/kokoro-v1_0.pth", help="path to checkpoint file"
        "--checkpoint_path", "-p", type=str, required=False, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="tvm_output", help="output directory"
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {checkpoint_path}...")
    # Initialize KModel
    # Note: KModel expects config to be a dict or path.
    # We assume the user provides valid paths.
    # if not os.path.exists(config_file):
    #     print(f"Warning: Config file {config_file} not found. KModel might try to download it.")
    
    # if not os.path.exists(checkpoint_path):
    #     print(f"Warning: Checkpoint file {checkpoint_path} not found. KModel might try to download it.")

    kmodel = KModel(config=config_file, model=checkpoint_path, disable_complex=True)
    model = KModelForONNX(kmodel).eval()

    compile_kokoro(model, output_dir)
