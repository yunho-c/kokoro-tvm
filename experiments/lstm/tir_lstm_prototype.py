#!/usr/bin/env python3
"""Prototype: Demonstrating loop-based TIR vs unrolled emit_te for LSTM.

This shows the IR size difference between:
1. Using emit_te(topi.nn.lstm) - which unrolls the scan
2. Using call_tir to a TIR PrimFunc with explicit loops

Run with: py -3.12 experiments/tir_lstm_prototype.py
"""

import numpy as np
import tvm
from tvm import te, tir
from tvm.script import tir as T, relax as R, ir as I


def create_looped_matmul_module():
    """Create a simple module showing loop structure is preserved.
    
    This demonstrates the key concept: if we define computation as loops
    in TIR, they stay as loops and don't get unrolled like te.scan does.
    """
    
    @I.ir_module
    class LoopedModule:
        @T.prim_func
        def sequential_matmul(
            # Input sequence: [seq_len, batch, dim]
            x: T.Buffer((64, 1, 256), "float32"),
            # Weight: [dim, dim]  
            W: T.Buffer((256, 256), "float32"),
            # Output: [seq_len, batch, dim]
            y: T.Buffer((64, 1, 256), "float32"),
        ):
            """Apply W to each timestep sequentially - demonstrates loop preservation."""
            # This loop will NOT be unrolled - it stays as a loop in the IR
            for t in range(64):
                for b in range(1):
                    for i in range(256):
                        y[t, b, i] = T.float32(0)
                        for k in range(256):
                            y[t, b, i] = y[t, b, i] + x[t, b, k] * W[k, i]
        
        @R.function
        def forward(
            x: R.Tensor((64, 1, 256), dtype="float32"),
            W: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((64, 1, 256), "float32"):
            cls = LoopedModule
            y = R.call_tir(
                cls.sequential_matmul,
                (x, W),
                out_sinfo=R.Tensor((64, 1, 256), "float32"),
            )
            return y
    
    return LoopedModule


def create_emit_te_module():
    """Create equivalent using emit_te - this should produce larger IR."""
    from tvm import relax
    
    # Use block builder
    bb = relax.BlockBuilder()
    
    # Define inputs
    x = relax.Var("x", relax.TensorStructInfo((64, 1, 256), "float32"))
    W = relax.Var("W", relax.TensorStructInfo((256, 256), "float32"))
    
    with bb.function("forward", [x, W]):
        with bb.dataflow():
            # emit_te for each timestep separately - simulates what happens with te.scan
            results = []
            for t in range(64):  # This loop IS unrolled during construction
                def slice_and_matmul(x_data, w_data, timestep=t):
                    # Slice out timestep t
                    x_t = te.compute(
                        (1, 256),
                        lambda b, k: x_data[timestep, b, k],
                        name=f"slice_{timestep}"
                    )
                    # Matmul
                    y_t = te.compute(
                        (1, 256),
                        lambda b, i: te.sum(x_t[b, te.reduce_axis((0, 256), "k")] * 
                                           w_data[te.reduce_axis((0, 256), "k"), i], axis=[]),
                        name=f"matmul_{timestep}"
                    )
                    return y_t
                
                y_t = bb.emit_te(slice_and_matmul, x, W)
                results.append(y_t)
            
            # Stack results
            y = bb.emit(relax.op.concat(
                [bb.emit(relax.op.expand_dims(r, 0)) for r in results],
                axis=0
            ))
            bb.emit_output(y)
        bb.emit_func_output(y)
    
    return bb.finalize()


def main():
    print("="*60)
    print("Loop Preservation Test")
    print("="*60)
    
    # Approach 1: TIR with explicit loops
    print("\n📦 Approach 1: TIR PrimFunc with loops")
    mod1 = create_looped_matmul_module()
    script1 = mod1.script()
    lines1 = len(script1.split('\n'))
    bytes1 = len(script1.encode('utf-8'))
    print(f"   IR: {lines1} lines, {bytes1/1024:.1f} KB")
    
    with open("/tmp/looped_module.py", "w") as f:
        f.write(script1)
    print("   Saved to /tmp/looped_module.py")
    
    # Approach 2: emit_te (unrolled)
    print("\n📦 Approach 2: emit_te for each timestep (unrolled)")
    mod2 = create_emit_te_module()
    script2 = mod2.script()
    lines2 = len(script2.split('\n'))
    bytes2 = len(script2.encode('utf-8'))
    print(f"   IR: {lines2} lines, {bytes2/1024:.1f} KB")
    
    with open("/tmp/emit_te_module.py", "w") as f:
        f.write(script2)
    print("   Saved to /tmp/emit_te_module.py")
    
    # Compile both
    print("\n� Compiling both modules...")
    target = tvm.target.Target("llvm")
    
    with tvm.transform.PassContext(opt_level=3):
        ex1 = tvm.relax.build(mod1, target)
    ex1.export_library("/tmp/looped_compiled.so")
    import os
    size1 = os.path.getsize("/tmp/looped_compiled.so")
    
    # mod2 might be too big or need transforms, try anyway
    try:
        mod2_transformed = mod2
        mod2_transformed = tvm.relax.transform.LegalizeOps()(mod2_transformed)
        with tvm.transform.PassContext(opt_level=3):
            ex2 = tvm.relax.build(mod2_transformed, target)
        ex2.export_library("/tmp/emit_te_compiled.so")
        size2 = os.path.getsize("/tmp/emit_te_compiled.so")
    except Exception as e:
        print(f"   emit_te compilation failed: {e}")
        size2 = None
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"TIR with loops:      {lines1:,} lines, {bytes1/1024:.1f} KB IR, {size1/1024:.1f} KB compiled")
    if size2:
        print(f"emit_te (unrolled):  {lines2:,} lines, {bytes2/1024:.1f} KB IR, {size2/1024:.1f} KB compiled")
        print(f"\nRatio: {lines2/lines1:.1f}x more IR, {bytes2/bytes1:.1f}x larger")
    else:
        print(f"emit_te (unrolled):  {lines2:,} lines, {bytes2/1024:.1f} KB IR (compilation failed)")
    
    print("\n💡 Key Insight:")
    print("   When using TIR PrimFunc, loops stay as loops.")
    print("   When using emit_te with te.scan or per-timestep compute,")
    print("   the loop gets unrolled into explicit operations.")
    print("\n   For LSTM: Use TIR PrimFunc or call_packed instead of emit_te(topi.nn.lstm)")


if __name__ == "__main__":
    main()
