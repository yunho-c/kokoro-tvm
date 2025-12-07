import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter
import operator
from typing import Callable

# Save original method
original_create_convert_map = ExportedProgramImporter.create_convert_map

def new_create_convert_map(self):
    # Get original map
    convert_map = original_create_convert_map(self)
    
    # Add custom converters
    custom_map = {
        "kokoro.lstm.default": self._lstm, # Reuse existing LSTM implementation
        "lstm.default": self._lstm, # Alias for custom op if name matches
        
        # Missing ops
        "atan2.default": lambda node: self.block_builder.emit_te(tvm.tir.atan2, self.env[node.args[0]], self.env[node.args[1]]),
        "_assert_scalar.default": lambda node: None, # Ignore assertions
        "le.Scalar": self._binary_op(relax.op.less_equal, operator.le),
        "le.Tensor": self._binary_op(relax.op.less_equal, operator.le),
        "truediv.Scalar": self._binary_op(relax.op.divide, operator.truediv),
        "truediv.Tensor": self._binary_op(relax.op.divide, operator.truediv),
        "rand.default": self._rand,
        "sym_float.default": lambda node: self.block_builder.emit(relax.op.astype(self.env[node.args[0]], "float32")),
        "sym_float": lambda node: self.block_builder.emit(relax.op.astype(self.env[node.args[0]], "float32")),
        "eq.Scalar": self._binary_op(relax.op.equal, operator.eq),
        "eq.Tensor": self._binary_op(relax.op.equal, operator.eq),
        "repeat_interleave.Tensor": self._repeat_interleave,
        "randn_like.default": self._randn_like,
        "add.default": self._binary_op(relax.op.add, operator.add), # Fallback
        "sym_constrain_range_for_size.default": lambda node: None,
        "gt.Scalar": self._binary_op(relax.op.greater, operator.gt),
        "gt.Tensor": self._binary_op(relax.op.greater, operator.gt),
        "ge.Scalar": self._binary_op(relax.op.greater_equal, operator.ge),
        "ge.Tensor": self._binary_op(relax.op.greater_equal, operator.ge),
        "sub.default": self._binary_op(relax.op.subtract, operator.sub),
        "full.default": self._full, # Monkeypatch full to handle dynamic value
        "item.default": self._item, # Monkeypatch item to handle scalar tensors
        "arange.default": self._arange, # Monkeypatch arange to handle PrimStructInfo
        "arange.start": self._arange,
        "arange.start_step": self._arange,
        "slice.Tensor": self._slice, # Monkeypatch slice to handle dynamic indices
        "reshape.default": self._reshape, # Monkeypatch reshape to handle None shape
        "expand.default": self._expand, # Monkeypatch expand to use broadcast_to
        
        # Add generic fallbacks for ops that might be missing specific overloads
        "le": self._binary_op(relax.op.less_equal, operator.le),
        "eq": self._binary_op(relax.op.equal, operator.eq),
        "gt": self._binary_op(relax.op.greater, operator.gt),
        "ge": self._binary_op(relax.op.greater_equal, operator.ge),
        "add": self._binary_op(relax.op.add, operator.add),
        "sub": self._binary_op(relax.op.subtract, operator.sub),
        "truediv": self._binary_op(relax.op.divide, operator.truediv),
    }
    
    convert_map.update(custom_map)
    return convert_map

# Define custom converter methods to be added to the class
def _item(self, node):
    x = self.env[node.args[0]]
    if len(self.shape_of(x)) == 0:
        return x
    return self.block_builder.emit(relax.op.take(x, relax.const(0, "int64"), axis=0))

def _full(self, node):
    import torch
    args = self.retrieve_args(node)
    size = relax.ShapeExpr(args[0] if isinstance(args[0], (list, tuple)) else (args[0],))
    dtype = self._convert_data_type(
        node.kwargs.get("dtype", torch.get_default_dtype()), self.env
    )
    value = args[1]
    if isinstance(value, tvm.tir.PrimExpr):
        # Use TE to create filled tensor for dynamic value
        return self.block_builder.emit_te(
            lambda shape, val: tvm.te.compute(shape, lambda *i: val, name="full"),
            size, value
        )
    
    if not isinstance(value, relax.Expr):
        try:
            value = relax.const(value, dtype)
        except ValueError:
            print(f"DEBUG: _full value type: {type(value)}, value: {value}")
            raise
    
    return self.block_builder.emit(
        relax.op.full(
            size,
            value,
            dtype,
        )
    )

def _rand(self, node):
    # args: size, dtype, layout, device, pin_memory
    size = node.args[0]
    dtype = self._convert_data_type(node.args[1] if len(node.args) > 1 else "float32")
    # Relax random uniform expects shape
    # We might need to handle symbolic shapes
    return self.block_builder.emit(relax.op.random.uniform(relax.ShapeExpr(size), dtype=dtype))

def _randn_like(self, node):
    input_tensor = self.env[node.args[0]]
    dtype = input_tensor.struct_info.dtype
    return self.block_builder.emit(relax.op.random.normal(relax.op.shape_of(input_tensor), dtype=dtype))

def _repeat_interleave(self, node):
    # args: input, repeats, dim, output_size
    x = self.env[node.args[0]]
    repeats = node.args[1]
    dim = node.args[2] if len(node.args) > 2 else None
    
    # Relax doesn't have direct repeat_interleave.
    # If repeats is scalar, we can use repeat.
    if isinstance(repeats, int):
        return self.block_builder.emit(relax.op.repeat(x, repeats, axis=dim))
    
    # If repeats is a tensor, it's harder.
    # For now, assume scalar or try to handle it.
    # If it's a tensor, we might need dynamic support or a loop.
    # But for Kokoro, it might be scalar.
    
    # If repeats is a Node, try to get constant value
    if hasattr(repeats, "meta"): # It's a node
         # We can't easily get value if it's dynamic.
         pass
         
    # Fallback to repeat if possible
    return self.block_builder.emit(relax.op.repeat(x, repeats, axis=dim))

# Attach methods to class
ExportedProgramImporter._rand = _rand
ExportedProgramImporter._randn_like = _randn_like
ExportedProgramImporter._repeat_interleave = _repeat_interleave
ExportedProgramImporter._full = _full
ExportedProgramImporter._item = _item

def _create_scalar_tensor(self, value, dtype):
    return self.block_builder.emit_te(
        lambda val: tvm.te.compute((), lambda *i: tvm.tir.Cast(dtype, val), name="scalar"),
        value
    )

ExportedProgramImporter._create_scalar_tensor = _create_scalar_tensor

def _binary_op(self, relax_op: Callable, intrinsic_op: Callable) -> Callable:
    from torch import fx

    def convert(node: fx.Node) -> relax.Var:
        def promote_binary_op_args(lhs, rhs):
            if isinstance(lhs, tvm.tir.PrimExpr):
                lhs = self._create_scalar_tensor(lhs, lhs.dtype)
            if isinstance(rhs, tvm.tir.PrimExpr):
                rhs = self._create_scalar_tensor(rhs, rhs.dtype)

            if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
                lhs_si = getattr(lhs, "struct_info", None)
                rhs_si = getattr(rhs, "struct_info", None)
                if isinstance(lhs_si, relax.TensorStructInfo) and isinstance(
                    rhs_si, relax.TensorStructInfo
                ):
                    target_dtype = self._promote_common_dtype(lhs_si.dtype, rhs_si.dtype)
                    if target_dtype is not None:
                        if lhs_si.dtype != target_dtype:
                            lhs = self.block_builder.emit(relax.op.astype(lhs, target_dtype))
                        if rhs_si.dtype != target_dtype:
                            rhs = self.block_builder.emit(relax.op.astype(rhs, target_dtype))
                return lhs, rhs
            elif isinstance(lhs, relax.Expr):
                assert isinstance(lhs.struct_info, relax.TensorStructInfo)
                return lhs, relax.const(rhs, lhs.struct_info.dtype)
            elif isinstance(rhs, relax.Expr):
                assert isinstance(rhs.struct_info, relax.TensorStructInfo)
                return relax.const(lhs, rhs.struct_info.dtype), rhs
            else:
                assert False

        def call_binary_op(op, lhs, rhs):
            lhs, rhs = promote_binary_op_args(lhs, rhs)
            return self.block_builder.emit(op(lhs, rhs))

        lhs, rhs = self.retrieve_args(node)
        if isinstance(lhs, (relax.Var, relax.expr.Call, tvm.tir.PrimExpr)) or isinstance(rhs, (relax.Var, relax.expr.Call, tvm.tir.PrimExpr)):
            return call_binary_op(relax_op, lhs, rhs)
        elif isinstance(lhs, relax.expr.Constant):
            return call_binary_op(relax_op, lhs, relax.const(rhs, dtype=lhs.struct_info.dtype))
        elif isinstance(rhs, relax.expr.Constant):
            return call_binary_op(relax_op, relax.const(lhs, dtype=rhs.struct_info.dtype), rhs)
        return intrinsic_op(lhs, rhs)

    return convert

ExportedProgramImporter._binary_op = _binary_op

def _arange(self, node):
    print(f"DEBUG: _arange node.args: {node.args}")
    start_end_step = []
    for x in node.args:
        if isinstance(x, torch.fx.Node):
            val = self.env[x]
            print(f"DEBUG: _arange env[x]: {val}, type: {type(val)}")
            start_end_step.append(val)
        else:
            print(f"DEBUG: _arange const arg: {x}, type: {type(x)}")
            start_end_step.append(x)
            
    dtype = self._convert_data_type(
        node.kwargs.get("dtype", torch.get_default_dtype()), self.env
    )
    
    new_args = []
    for arg in start_end_step:
        if isinstance(arg, relax.Var) and isinstance(arg.struct_info, relax.PrimStructInfo):
            # Extract PrimExpr using match_cast
            m = tvm.tir.Var("m", "int64")
            self.block_builder.match_cast(arg, relax.PrimStructInfo(value=m))
            new_args.append(m)
        elif isinstance(arg, relax.Var) and isinstance(arg.struct_info, relax.TensorStructInfo):
             # Convert scalar tensor to PrimExpr
             # Reshape to 1D
             arg_1d = self.block_builder.emit(relax.op.reshape(arg, [1]))
             # Convert to shape
             s = self.block_builder.emit(relax.op.tensor_to_shape(arg_1d))
             # Extract PrimExpr using match_cast
             m = tvm.tir.Var("m", "int64")
             self.block_builder.match_cast(s, relax.ShapeStructInfo([m]))
             new_args.append(m)
        else:
            new_args.append(arg)
            
    return self.block_builder.emit(relax.op.arange(*new_args, dtype=dtype))

ExportedProgramImporter._arange = _arange

def _slice(self, node):
    import sys
    x = self.env[node.args[0]]
    dim = node.args[1] if len(node.args) > 1 else 0
    start = node.args[2] if len(node.args) > 2 else None
    end_val = node.args[3] if len(node.args) > 3 else None
    step = node.args[4] if len(node.args) > 4 else 1

    # Helper to resolve arg to int or PrimExpr
    def resolve(arg, default):
        if arg is None:
            return default
        if isinstance(arg, torch.fx.Node):
            val = self.env[arg]
            # Convert val to PrimExpr
            if isinstance(val, tvm.tir.PrimExpr):
                return val
            if isinstance(val, relax.Var) and isinstance(val.struct_info, relax.PrimStructInfo):
                m = tvm.tir.Var("m", "int64")
                self.block_builder.match_cast(val, relax.PrimStructInfo(value=m))
                return m
            if isinstance(val, relax.Var) and isinstance(val.struct_info, relax.TensorStructInfo):
                 # Scalar tensor -> PrimExpr
                 arg_1d = self.block_builder.emit(relax.op.reshape(val, [1]))
                 s = self.block_builder.emit(relax.op.tensor_to_shape(arg_1d))
                 m = tvm.tir.Var("m", "int64")
                 self.block_builder.match_cast(s, relax.ShapeStructInfo([m]))
                 return m
            # Fallback
            return val
        return arg

    start = resolve(start, 0)
    
    if end_val is None:
        # Get shape of x
        shape = self.shape_of(x)
        end_val = shape[dim]
    else:
        end_val = resolve(end_val, sys.maxsize)
        
    step = resolve(step, 1)

    axes = [dim]
    begin = [start]
    end = [end_val]
    stride = [step]
    
    return self.block_builder.emit(relax.op.strided_slice(x, axes, begin, end, stride))

ExportedProgramImporter._slice = _slice

def _reshape(self, node):
    x = self.env[node.args[0]]
    dims = node.args[1]
    
    # Resolve dims if they are Nodes
    new_dims = []
    for d in dims:
        if isinstance(d, torch.fx.Node):
            val = self.env[d]
            if isinstance(val, tvm.tir.PrimExpr):
                new_dims.append(val)
            elif isinstance(val, relax.Var) and isinstance(val.struct_info, relax.PrimStructInfo):
                m = tvm.tir.Var("m", "int64")
                self.block_builder.match_cast(val, relax.PrimStructInfo(value=m))
                new_dims.append(m)
            elif isinstance(val, relax.Var) and isinstance(val.struct_info, relax.TensorStructInfo):
                 arg_1d = self.block_builder.emit(relax.op.reshape(val, [1]))
                 s = self.block_builder.emit(relax.op.tensor_to_shape(arg_1d))
                 m = tvm.tir.Var("m", "int64")
                 self.block_builder.match_cast(s, relax.ShapeStructInfo([m]))
                 new_dims.append(m)
            else:
                new_dims.append(val) # Hope it's compatible
        else:
            new_dims.append(d)
            
    current_shape = self.shape_of(x)
    if current_shape is not None and list(current_shape) == list(new_dims):
        return x
    return self.block_builder.emit(relax.op.reshape(x, new_dims))

ExportedProgramImporter._reshape = _reshape

def _expand(self, node):
    x = self.env[node.args[0]]
    sizes = node.args[1]
    
    # Resolve sizes if they are Nodes
    new_sizes = []
    for s in sizes:
        if isinstance(s, torch.fx.Node):
            val = self.env[s]
            if isinstance(val, tvm.tir.PrimExpr):
                new_sizes.append(val)
            elif isinstance(val, relax.Var) and isinstance(val.struct_info, relax.PrimStructInfo):
                m = tvm.tir.Var("m", "int64")
                self.block_builder.match_cast(val, relax.PrimStructInfo(value=m))
                new_sizes.append(m)
            elif isinstance(val, relax.Var) and isinstance(val.struct_info, relax.TensorStructInfo):
                 arg_1d = self.block_builder.emit(relax.op.reshape(val, [1]))
                 s_shape = self.block_builder.emit(relax.op.tensor_to_shape(arg_1d))
                 m = tvm.tir.Var("m", "int64")
                 self.block_builder.match_cast(s_shape, relax.ShapeStructInfo([m]))
                 new_sizes.append(m)
            else:
                new_sizes.append(val)
        else:
            new_sizes.append(s)
            
    # Handle -1 in sizes (keep dimension)
    x_shape = self.shape_of(x)
    if x_shape is not None:
        resolved_sizes = []
        diff = len(new_sizes) - len(x_shape)
        for i, s in enumerate(new_sizes):
            if isinstance(s, int) and s == -1:
                if i - diff >= 0:
                    resolved_sizes.append(x_shape[i - diff])
                else:
                    # Should not happen for valid expand
                    resolved_sizes.append(s)
            else:
                resolved_sizes.append(s)
        new_sizes = resolved_sizes
            
    return self.block_builder.emit(relax.op.broadcast_to(x, new_sizes))

ExportedProgramImporter._expand = _expand

# Wrap _lstm to handle dynamic shapes using TOPI
def _lstm(self, node):
    import tvm.topi.nn
    args = self.retrieve_args(node)
    input_tensor = args[0]
    hx = args[1] if len(args) > 1 else None
    params = args[2] if len(args) > 2 else None
    has_biases = args[3] if len(args) > 3 else True
    num_layers = args[4] if len(args) > 4 else 1
    bidirectional = args[7] if len(args) > 7 else False
    batch_first = args[8] if len(args) > 8 else False
    
    # Transpose input if batch_first
    if batch_first:
        input_tensor = self.block_builder.emit(relax.op.permute_dims(input_tensor, [1, 0, 2]))
        
    current_input = input_tensor
    all_layer_h = []
    all_layer_c = []
    
    param_idx = 0
    num_directions = 2 if bidirectional else 1
    
    for layer in range(num_layers):
        layer_outputs = []
        layer_h_n = []
        layer_c_n = []
        
        for direction in range(num_directions):
            reverse = (direction == 1)
            
            Wi = params[param_idx]
            Wh = params[param_idx+1]
            Bi = params[param_idx+2] if has_biases else None
            Bh = params[param_idx+3] if has_biases else None
            param_idx += 4
            
            # Initial states for this layer/direction
            idx = layer * num_directions + direction
            
            h_init = None
            c_init = None
            if hx is not None:
                print(f"DEBUG: _lstm hx type: {type(hx)}")
                if isinstance(hx, (list, tuple)):
                    h0 = hx[0]
                    c0 = hx[1]
                else:
                    h0 = self.block_builder.emit(relax.TupleGetItem(hx, 0))
                    c0 = self.block_builder.emit(relax.TupleGetItem(hx, 1))
                    
                h_init = self.block_builder.emit(relax.op.take(h0, relax.const(idx, "int64"), axis=0))
                c_init = self.block_builder.emit(relax.op.take(c0, relax.const(idx, "int64"), axis=0))
            
            # Call TOPI LSTM
            lstm_out = self.block_builder.emit_te(
                tvm.topi.nn.lstm,
                current_input,
                Wi, Wh, Bi, Bh,
                h_init, c_init,
                None, None, None, None, # proj, p_i, p_f, p_o
                tvm.tir.sigmoid, tvm.tir.tanh, tvm.tir.tanh,
                reverse # reverse argument
            )
            print(f"DEBUG: _lstm lstm_out type: {type(lstm_out)}")
            
            if isinstance(lstm_out, (list, tuple)) or "Array" in str(type(lstm_out)):
                 out_seq = lstm_out[0]
                 out_cell = lstm_out[1]
            else:
                 out_seq = self.block_builder.emit(relax.TupleGetItem(lstm_out, 0))
                 out_cell = self.block_builder.emit(relax.TupleGetItem(lstm_out, 1))
            
            layer_outputs.append(out_seq)
            
            # Extract last state
            h_n_squeezed = self.block_builder.emit(relax.op.take(out_seq, relax.const(-1, "int64"), axis=0))
            c_n_squeezed = self.block_builder.emit(relax.op.take(out_cell, relax.const(-1, "int64"), axis=0))
            
            layer_h_n.append(h_n_squeezed)
            layer_c_n.append(c_n_squeezed)

        # Combine directions
        if bidirectional:
            current_input = self.block_builder.emit(relax.op.concat(layer_outputs, axis=2))
        else:
            current_input = layer_outputs[0]
            
        all_layer_h.extend(layer_h_n)
        all_layer_c.extend(layer_c_n)
        
    # Stack final states
    final_h = self.block_builder.emit(relax.op.concat([
        self.block_builder.emit(relax.op.expand_dims(h, 0)) for h in all_layer_h
    ], axis=0))
    
    final_c = self.block_builder.emit(relax.op.concat([
        self.block_builder.emit(relax.op.expand_dims(c, 0)) for c in all_layer_c
    ], axis=0))
    
    # Transpose output if batch_first
    if batch_first:
        current_input = self.block_builder.emit(relax.op.permute_dims(current_input, [1, 0, 2]))
        
    states = self.block_builder.emit(relax.op.tuple([final_h, final_c]))
    return self.block_builder.emit(relax.op.tuple([current_input, states]))

ExportedProgramImporter._lstm = _lstm

# Apply monkeypatch
ExportedProgramImporter.create_convert_map = new_create_convert_map
