import argparse
import os
from pathlib import Path

import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

def compile_model(onnx_path: str, output_dir: str, target_str: str = "llvm"):
    """
    Compiles an ONNX model to a TVM Relax executable.
    
    Args:
        onnx_path: Path to the input ONNX model.
        output_dir: Directory to save the compiled artifacts.
        target_str: TVM target string (e.g., "llvm", "cuda").
    """
    print(f"Loading ONNX model from: {onnx_path}")
    model = onnx.load(onnx_path)
    
    print("Converting to TVM Relax IRModule...")
    # keep_params_in_input=True separates weights from the graph, 
    # allowing for cleaner deployment and potentially smaller .so files
    mod = from_onnx(model, keep_params_in_input=True)
    
    # Separate parameters from the module
    mod, params = relax.frontend.detach_params(mod)
    
    target = tvm.target.Target(target_str)
    print(f"Compiling for target: {target}")
    
    # Apply default optimizations
    pipeline = relax.get_pipeline()
    with target:
        mod = pipeline(mod)
        
    # Compile to executable
    ex = tvm.compile(mod, target=target)
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the executable (shared library)
    lib_name = "kokoro.so"
    lib_path = output_path / lib_name
    ex.export_library(str(lib_path))
    print(f"Exported library to: {lib_path}")
    
    # Save the parameters
    # We save them as a dictionary of numpy arrays for easy loading
    import numpy as np
    params_name = "kokoro_params.npz"
    params_path = output_path / params_name
    
    # Convert TVM arrays to numpy for saving
    param_dict = {}
    for func_name, func_params in params.items():
        # func_params is a list of TVM NDArrays
        for i, param in enumerate(func_params):
            param_dict[f"{func_name}_p{i}"] = param.numpy()
            
    np.savez(str(params_path), **param_dict)
    print(f"Exported parameters to: {params_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Kokoro ONNX model with TVM")
    parser.add_argument("--onnx-path", type=str, default="external/kokoro-onnx-export/kokoro.onnx", help="Path to input ONNX file")
    parser.add_argument("--output-dir", type=str, default="build", help="Directory for output artifacts")
    parser.add_argument("--target", type=str, default="llvm", help="TVM target (e.g., llvm, cuda)")
    
    args = parser.parse_args()
    
    # Resolve paths relative to workspace root if needed, but here we assume CWD is root
    compile_model(args.onnx_path, args.output_dir, args.target)
