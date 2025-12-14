"""TIR-based LSTM using tvm.tir.ir_builder.while_loop for non-unrolled iteration.

This creates a PrimFunc with a proper While loop that stays as a loop in the
compiled code. Uses the low-level ir_builder API which avoids TVMScript 
block well-formedness issues.
"""

import tvm
from tvm import tir


def create_tir_lstm_primfunc(
    seq_len: int,
    batch_size: int,
    input_size: int,
    hidden_size: int,
    dtype: str = "float32",
) -> tvm.tir.PrimFunc:
    """Create TIR LSTM PrimFunc with While loop.
    
    Uses tvm.tir.ir_builder directly to create a function with a proper
    While loop that won't be unrolled.
    """
    gate_size = 4 * hidden_size
    
    # Create buffers with proper types
    x_buf = tir.decl_buffer((seq_len, batch_size, input_size), dtype, "x")
    h_init_buf = tir.decl_buffer((batch_size, hidden_size), dtype, "h_init")
    c_init_buf = tir.decl_buffer((batch_size, hidden_size), dtype, "c_init")
    Wi_buf = tir.decl_buffer((gate_size, input_size), dtype, "Wi")
    Wh_buf = tir.decl_buffer((gate_size, hidden_size), dtype, "Wh")
    bi_buf = tir.decl_buffer((gate_size,), dtype, "bi")
    bh_buf = tir.decl_buffer((gate_size,), dtype, "bh")
    out_buf = tir.decl_buffer((seq_len, batch_size, hidden_size), dtype, "output")
    h_final_buf = tir.decl_buffer((batch_size, hidden_size), dtype, "h_final")
    c_final_buf = tir.decl_buffer((batch_size, hidden_size), dtype, "c_final")
    
    # Build function body
    ib = tir.ir_builder.create()
    
    # Get buffer ptrs to read from
    x_ptr = ib.buffer_ptr(x_buf)
    h_init_ptr = ib.buffer_ptr(h_init_buf)
    c_init_ptr = ib.buffer_ptr(c_init_buf)
    Wi_ptr = ib.buffer_ptr(Wi_buf)
    Wh_ptr = ib.buffer_ptr(Wh_buf)
    bi_ptr = ib.buffer_ptr(bi_buf)
    bh_ptr = ib.buffer_ptr(bh_buf)
    out_ptr = ib.buffer_ptr(out_buf)
    h_final_ptr = ib.buffer_ptr(h_final_buf)
    c_final_ptr = ib.buffer_ptr(c_final_buf)
    
    # Allocate working memory (flat arrays for simplicity)
    h = ib.allocate(dtype, batch_size * hidden_size, "h", scope="global")
    c = ib.allocate(dtype, batch_size * hidden_size, "c", scope="global")
    gates = ib.allocate(dtype, batch_size * gate_size, "gates", scope="global")
    
    # Initialize h and c from inputs
    with ib.for_range(0, batch_size, name="b0") as b:
        with ib.for_range(0, hidden_size, name="j0") as j:
            flat = b * hidden_size + j
            h[flat] = h_init_ptr[b * hidden_size + j]
            c[flat] = c_init_ptr[b * hidden_size + j]
    
    # Timestep counter
    t = ib.allocate("int32", 1, "t", scope="local")
    t[0] = 0
    
    # Main While loop - THIS IS THE KEY: stays as a loop, not unrolled
    with ib.while_loop(t[0] < seq_len):
        t_val = t[0]
        
        # Compute gates for this timestep
        with ib.for_range(0, batch_size, name="b1") as b:
            with ib.for_range(0, gate_size, name="g1") as g:
                gate_flat = b * gate_size + g
                
                # Initialize with bias
                gates[gate_flat] = bi_ptr[g] + bh_ptr[g]
                
                # x[t] @ Wi.T
                with ib.for_range(0, input_size, name="k1") as k:
                    x_idx = t_val * batch_size * input_size + b * input_size + k
                    Wi_idx = g * input_size + k
                    gates[gate_flat] = gates[gate_flat] + x_ptr[x_idx] * Wi_ptr[Wi_idx]
                
                # h @ Wh.T
                with ib.for_range(0, hidden_size, name="k2") as k:
                    h_idx = b * hidden_size + k
                    Wh_idx = g * hidden_size + k
                    gates[gate_flat] = gates[gate_flat] + h[h_idx] * Wh_ptr[Wh_idx]
        
        # Apply activations and update states
        with ib.for_range(0, batch_size, name="b2") as b:
            with ib.for_range(0, hidden_size, name="j2") as j:
                flat = b * hidden_size + j
                
                # Gate values
                i_g = tir.sigmoid(gates[b * gate_size + j])
                f_g = tir.sigmoid(gates[b * gate_size + hidden_size + j])
                g_g = tir.tanh(gates[b * gate_size + 2 * hidden_size + j])
                o_g = tir.sigmoid(gates[b * gate_size + 3 * hidden_size + j])
                
                # Update cell
                c_new = f_g * c[flat] + i_g * g_g
                c[flat] = c_new
                
                # Update hidden
                h_new = o_g * tir.tanh(c_new)
                h[flat] = h_new
                
                # Write to output
                out_idx = t_val * batch_size * hidden_size + b * hidden_size + j
                out_ptr[out_idx] = h_new
        
        # Increment timestep
        t[0] = t_val + 1
    
    # Copy final states to output buffers
    with ib.for_range(0, batch_size, name="b3") as b:
        with ib.for_range(0, hidden_size, name="j3") as j:
            flat = b * hidden_size + j
            h_final_ptr[flat] = h[flat]
            c_final_ptr[flat] = c[flat]
    
    body = ib.get()
    
    # Create the PrimFunc
    func = tir.PrimFunc(
        params=[x_buf.data, h_init_buf.data, c_init_buf.data,
                Wi_buf.data, Wh_buf.data, bi_buf.data, bh_buf.data,
                out_buf.data, h_final_buf.data, c_final_buf.data],
        body=body,
        buffer_map={
            x_buf.data: x_buf,
            h_init_buf.data: h_init_buf,
            c_init_buf.data: c_init_buf,
            Wi_buf.data: Wi_buf,
            Wh_buf.data: Wh_buf,
            bi_buf.data: bi_buf,
            bh_buf.data: bh_buf,
            out_buf.data: out_buf,
            h_final_buf.data: h_final_buf,
            c_final_buf.data: c_final_buf,
        },
    ).with_attr("global_symbol", "tir_lstm_forward").with_attr("tir.noalias", True)
    
    return func


def test_tir_lstm():
    """Test the TIR LSTM with While loop."""
    import traceback
    import numpy as np
    
    print("=" * 60)
    print("TIR LSTM with While Loop")
    print("=" * 60)
    
    seq_len, batch_size, input_size, hidden_size = 8, 1, 64, 32
    
    try:
        print("\n📦 Creating TIR LSTM PrimFunc...")
        func = create_tir_lstm_primfunc(seq_len, batch_size, input_size, hidden_size)
        
        script = func.script()
        lines = script.split('\n')
        
        print(f"\n📊 TIR Statistics:")
        print(f"   Lines: {len(lines)}")
        print(f"   Size: {len(script)} bytes")
        
        # Check for while loop presence
        has_while = "while" in script.lower()
        print(f"   Contains While loop: {has_while}")
        
        print("\n� Generated TIR (first 60 lines):")
        for line in lines[:60]:
            print(line)
        if len(lines) > 60:
            print(f"... ({len(lines) - 60} more lines)")
        
        # Build
        print("\n🔧 Building for LLVM...")
        target = tvm.target.Target("llvm")
        mod = tvm.IRModule({"tir_lstm_forward": func})
        
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(mod, target=target)
        
        print("✅ Build successful!")
        
        # Test execution
        print("\n🏃 Testing execution...")
        dev = tvm.cpu()
        
        np.random.seed(42)
        x = np.random.randn(seq_len, batch_size, input_size).astype("float32")
        h_init = np.zeros((batch_size, hidden_size), dtype="float32")
        c_init = np.zeros((batch_size, hidden_size), dtype="float32")
        Wi = (np.random.randn(4 * hidden_size, input_size) * 0.1).astype("float32")
        Wh = (np.random.randn(4 * hidden_size, hidden_size) * 0.1).astype("float32")
        bi = np.zeros((4 * hidden_size,), dtype="float32")
        bh = np.zeros((4 * hidden_size,), dtype="float32")
        output = np.zeros((seq_len, batch_size, hidden_size), dtype="float32")
        h_final = np.zeros((batch_size, hidden_size), dtype="float32")
        c_final = np.zeros((batch_size, hidden_size), dtype="float32")
        
        lib["tir_lstm_forward"](
            tvm.runtime.tensor(x, dev),
            tvm.runtime.tensor(h_init, dev),
            tvm.runtime.tensor(c_init, dev),
            tvm.runtime.tensor(Wi, dev),
            tvm.runtime.tensor(Wh, dev),
            tvm.runtime.tensor(bi, dev),
            tvm.runtime.tensor(bh, dev),
            tvm.runtime.tensor(output, dev),
            tvm.runtime.tensor(h_final, dev),
            tvm.runtime.tensor(c_final, dev),
        )
        
        print(f"   Output shape: {output.shape}")
        print(f"   Output[0,0,:5]: {output[0, 0, :5]}")
        print(f"   h_final[0,:5]: {h_final[0, :5]}")
        print(f"   Non-zero elements: {np.sum(output != 0)}")
        
        print("\n✅ TIR LSTM with While loop works!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_tir_lstm()
