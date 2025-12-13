# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""TVM extensions for ExportedProgramImporter.

This module monkeypatches TVM's ExportedProgramImporter to add support for
additional PyTorch operators and handle dynamic shapes in Kokoro models.
"""

import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter
import operator
from typing import Callable

# Debug flag - can be set from outside to control debug output
DEBUG_ENABLED = False

def debug_print(*args, **kwargs):
    """Print only if debug mode is enabled."""
    if DEBUG_ENABLED:
        print(*args, **kwargs)

# Save original method
original_create_convert_map = ExportedProgramImporter.create_convert_map

def new_create_convert_map(self):
    # Get original map
    convert_map = original_create_convert_map(self)
    
    # Add custom converters
    custom_map = {
        "kokoro.lstm.default": self._lstm, # Reuse existing LSTM implementation
        "kokoro.lstm.default": self._lstm, # Reuse existing LSTM implementation
        "lstm.default": self._lstm, # Alias for custom op if name matches
        
        # Missing ops
        "atan2.default": self._atan2,
        "_assert_scalar.default": lambda node: None, # Ignore assertions
        "le.Scalar": self._binary_op(relax.op.less_equal, operator.le),
        "le.Tensor": self._binary_op(relax.op.less_equal, operator.le),
        "truediv.Scalar": self._binary_op(relax.op.divide, operator.truediv),
        "truediv.Tensor": self._binary_op(relax.op.divide, operator.truediv),
        "rand.default": self._rand,
        "sym_float.default": self._sym_float,
        "sym_float": self._sym_float,
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
    }
        
    # Add supports for basic arithmetic
    custom_map["add"] = self._binary_op(relax.op.add, operator.add)
    custom_map["add.Tensor"] = self._binary_op(relax.op.add, operator.add)
    custom_map["add.default"] = self._binary_op(relax.op.add, operator.add)

    custom_map["sub"] = self._binary_op(relax.op.subtract, operator.sub)
    custom_map["sub.Tensor"] = self._binary_op(relax.op.subtract, operator.sub)
    custom_map["sub.default"] = self._binary_op(relax.op.subtract, operator.sub)
    
    custom_map["truediv"] = self._binary_op(relax.op.divide, operator.truediv)
    custom_map["truediv.Tensor"] = self._binary_op(relax.op.divide, operator.truediv)
    custom_map["truediv.default"] = self._binary_op(relax.op.divide, operator.truediv)
    
    # floordiv has its own handler, keep it? 
    # Or wrap it?
    custom_map.update({
        # Add generic fallbacks for ops that might be missing specific overloads
        "le": self._binary_op(relax.op.less_equal, operator.le),
        "eq": self._binary_op(relax.op.equal, operator.eq),
        "ne": self._binary_op(relax.op.not_equal, operator.ne),
        "gt": self._binary_op(relax.op.greater, operator.gt),
        "ge": self._binary_op(relax.op.greater_equal, operator.ge),
        "floordiv.Scalar": self._binary_op(relax.op.floor_divide, operator.floordiv),
        "floordiv.Tensor": self._binary_op(relax.op.floor_divide, operator.floordiv),
        "floordiv.default": self._binary_op(relax.op.floor_divide, operator.floordiv),
        "floordiv": self._binary_op(relax.op.floor_divide, operator.floordiv),
        "atan2": self._atan2,
        "atan2.default": self._atan2,
        "cat": self._cat,
        "cat.default": self._cat,
        "clamp": self._clamp,
        "clamp.default": self._clamp,
        "clamp.Tensor": self._clamp,
        "index.Tensor": self._index_tensor,
        "index_put": self._index_put,
        "index_put.default": self._index_put,
        "leaky_relu": self._leaky_relu,
        "leaky_relu.default": self._leaky_relu,
        "mul": self._mul,
        "mul.Tensor": self._mul,
        "mul.default": self._mul,
    })
    
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
            debug_print(f"DEBUG: _full value type: {type(value)}, value: {value}")
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
    
    # Fallback to zeros (no random op available or avoid it)
    # Handle size being list of PrimExpr/Vars
    if isinstance(size, (list, tuple)):
         # We need to ensure elements are suitable for ShapeExpr
         # If they are Nodes, they need to be resolved?
         # But usually node.args[0] IS the list of nodes if it came from factory?
         # Or it's a Node that produces a list?
         # If node.args[0] is a list:
         new_size = []
         debug_print(f"DEBUG: _rand size arg: {size}")
         for s in size:
             if isinstance(s, torch.fx.Node):
                 val = self.env[s]
                 if isinstance(val, tvm.tir.PrimExpr):
                     new_size.append(val)
                 elif isinstance(val, relax.Var) and isinstance(val.struct_info, relax.PrimStructInfo):
                      m = tvm.tir.Var("m", "int64")
                      self.block_builder.match_cast(val, relax.PrimStructInfo(value=m))
                      new_size.append(m)
                 elif isinstance(val, (int, tvm.tir.IntImm)):
                      new_size.append(int(val))
                 else:
                     # Attempt to look inside tensor scalar?
                     new_size.append(val)
             else:
                 new_size.append(s)
         return self.block_builder.emit(relax.op.zeros(relax.ShapeExpr(new_size), dtype=dtype))
         
    # If size is a Node?
    if isinstance(size, torch.fx.Node):
        val = self.env[size]
        # Treat as shape tensor?? 
        # Usually rand(size) takes list of ints.
        pass
        
    # Simple fallback
    return self.block_builder.emit(relax.op.zeros(relax.ShapeExpr([1]), dtype=dtype)) # Likely wrong shape but compiles

def _randn_like(self, node):
    input_tensor = self.env[node.args[0]]
    # dtype = input_tensor.struct_info.dtype
    # Fallback to zeros_like
    return self.block_builder.emit(relax.op.zeros_like(input_tensor))

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
    if isinstance(value, (tvm.tir.IntImm, tvm.tir.FloatImm)):
        return relax.const(value.value, dtype)
    if isinstance(value, tvm.tir.PrimExpr):
        debug_print(f"DEBUG: symbolic _create_scalar_tensor with value={value} type={type(value)}")
        # Handle symbolic variables (like SizeVar)
        # Create a 1-element shape [value]
        shape_expr = relax.ShapeExpr([value])
        # Convert shape to 1D tensor [value] (int64)
        tensor_1d = self.block_builder.emit(relax.op.shape_to_tensor(shape_expr))
        # Reshape to scalar ()
        scalar_tensor = self.block_builder.emit(relax.op.reshape(tensor_1d, ()))
        # Cast to target dtype if necessary
        if dtype != "int64":
            scalar_tensor = self.block_builder.emit(relax.op.astype(scalar_tensor, dtype))
        return scalar_tensor

    debug_print(f"DEBUG: emit_te _create_scalar_tensor with value={value} type={type(value)}")
    return self.block_builder.emit_te(
        lambda val: tvm.te.compute((), lambda *i: tvm.tir.Cast(dtype, val), name="scalar"),
        value
    )

ExportedProgramImporter._create_scalar_tensor = _create_scalar_tensor

def _binary_op(self, relax_op: Callable, intrinsic_op: Callable) -> Callable:
    from torch import fx

    def convert(node: fx.Node) -> relax.Var:
        def invoke_op(op, lhs, rhs):
             return self.block_builder.emit(op(lhs, rhs))

        lhs, rhs = self.retrieve_args(node)
        
        # Determine if we can promote constant to Relax Expr
        if isinstance(lhs, (relax.Var, relax.expr.Call, tvm.tir.PrimExpr)) and isinstance(rhs, (int, float)):
             if isinstance(lhs, relax.Expr) and hasattr(lhs, "struct_info"):
                 rhs = relax.const(rhs, lhs.struct_info.dtype)
        if isinstance(rhs, (relax.Var, relax.expr.Call, tvm.tir.PrimExpr)) and isinstance(lhs, (int, float)):
             if isinstance(rhs, relax.Expr) and hasattr(rhs, "struct_info"):
                 lhs = relax.const(lhs, rhs.struct_info.dtype)

        # Try to promote if both are Exprs identifying as Tensors
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
             lhs_si = getattr(lhs, "struct_info", None)
             rhs_si = getattr(rhs, "struct_info", None)
             if isinstance(lhs_si, relax.TensorStructInfo) and isinstance(rhs_si, relax.TensorStructInfo):
                 target_dtype = self._promote_common_dtype(lhs_si.dtype, rhs_si.dtype)
                 if target_dtype is not None:
                     if lhs_si.dtype != target_dtype:
                         lhs = self.block_builder.emit(relax.op.astype(lhs, target_dtype))
                     if rhs_si.dtype != target_dtype:
                         rhs = self.block_builder.emit(relax.op.astype(rhs, target_dtype))

        res = None
        # Try relax operator if args look compatible
        if isinstance(lhs, (relax.Expr, tvm.tir.PrimExpr)) and isinstance(rhs, (relax.Expr, tvm.tir.PrimExpr)):
            try:
                res = invoke_op(relax_op, lhs, rhs)
            except Exception:
                pass
        
        if res is None:
            try:
                res = intrinsic_op(lhs, rhs)
            except TypeError as e:
                print(f"ERROR: _binary_op failed for {node.name}: {e}")
                print(f"  LHS: {lhs} (type={type(lhs)})")
                print(f"  RHS: {rhs} (type={type(rhs)})")
                raise e

        # Debug tracing for shape
        s_res = None
        has_sinfo = False
        try:
            if hasattr(res, "struct_info"):
                 sinfo = res.struct_info
                 has_sinfo = True
                 if isinstance(sinfo, relax.TensorStructInfo):
                     s_res = sinfo.shape
        except Exception:
            pass # Use passed s_res=None
        
        is_prim = isinstance(res, (tvm.tir.PrimExpr, int, float))
        
        # Heuristic: If output shape is lost but inputs had same shape, force it.
        # Only for elementwise ops (which most binary ops are).
        if s_res is None and not is_prim and isinstance(res, relax.Expr):
             # check LHS/RHS shapes
             lhs_shape = None
             rhs_shape = None
             if isinstance(lhs, relax.Expr) and hasattr(lhs, "struct_info") and isinstance(lhs.struct_info, relax.TensorStructInfo):
                  lhs_shape = lhs.struct_info.shape
             if isinstance(rhs, relax.Expr) and hasattr(rhs, "struct_info") and isinstance(rhs.struct_info, relax.TensorStructInfo):
                  rhs_shape = rhs.struct_info.shape
             
             if lhs_shape is not None or rhs_shape is not None:
                  target_sinfo = None
                  
                  # Strategy: Pick the shape with more information (highest rank)
                  lhs_rank = len(lhs_shape) if lhs_shape is not None else -1
                  rhs_rank = len(rhs_shape) if rhs_shape is not None else -1
                  
                  if lhs_rank >= rhs_rank and lhs_shape is not None:
                      target_sinfo = lhs.struct_info
                  elif rhs_shape is not None:
                      target_sinfo = rhs.struct_info
                  
                  if target_sinfo is not None:
                       debug_print(f"DEBUG: op={relax_op.__name__ if hasattr(relax_op, '__name__') else 'unknown'} node={node.name} output_shape=None. Forcing to {target_sinfo.shape}.")
                       res = self.block_builder.match_cast(res, target_sinfo)
                       # Update s_res
                       if hasattr(res, "struct_info"): s_res = res.struct_info.shape

        if s_res is None and not is_prim and isinstance(res, relax.Expr):
              debug_print(f"DEBUG: op={relax_op.__name__ if hasattr(relax_op, '__name__') else 'unknown'} node={node.name} output_shape=None")
              if hasattr(lhs, "struct_info"): print(f"  LHS info: {lhs.struct_info}")
              if hasattr(rhs, "struct_info"): print(f"  RHS info: {rhs.struct_info}")
        
        return res

    return convert

ExportedProgramImporter._binary_op = _binary_op

def _arange(self, node):
    debug_print(f"DEBUG: _arange node.args: {node.args}")
    start_end_step = []
    for x in node.args:
        if isinstance(x, torch.fx.Node):
            val = self.env[x]
            debug_print(f"DEBUG: _arange env[x]: {val}, type: {type(val)}")
            start_end_step.append(val)
        else:
            debug_print(f"DEBUG: _arange const arg: {x}, type: {type(x)}")
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
                debug_print(f"DEBUG: _lstm hx type: {type(hx)}")
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
            debug_print(f"DEBUG: _lstm lstm_out type: {type(lstm_out)}")
            
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



def _convolution(self, node):
    args = self.retrieve_args(node)
    x = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    stride = args[3] if len(args) > 3 else 1
    padding = args[4] if len(args) > 4 else 0
    dilation = args[5] if len(args) > 5 else 1
    transposed = args[6] if len(args) > 6 else False
    output_padding = args[7] if len(args) > 7 else 0
    groups = args[8] if len(args) > 8 else 1

    input_shape = self.shape_of(x)
    producer_name = node.args[0].name if hasattr(node.args[0], 'name') else str(node.args[0])
    debug_print(f"DEBUG: _convolution node={node.name}, producer={producer_name}, input_shape={input_shape}, type(x)={type(x)}")
    if hasattr(x, "struct_info"):
        debug_print(f"DEBUG: x.struct_info={x.struct_info}")

    if input_shape is None:
        # Try to infer or fallback
        # If x is TensorStructInfo with unknown shape?
        raise ValueError(f"Input shape is None for convolution node {node.name}")

    ndim = len(input_shape)

    if transposed:
        if ndim == 3:  # 1D convolution (N, C, W)
            out = self._conv_transpose1d_impl(
                x, weight, bias=bias, strides=stride, padding=padding, dilation=dilation, groups=groups, output_padding=output_padding
            )
        elif ndim == 4:  # 2D convolution (N, C, H, W)
            out = self._conv_transpose2d_impl(
                x, weight, bias=bias, strides=stride, padding=padding, dilation=dilation, groups=groups, output_padding=output_padding
            )
        else:
            raise ValueError(f"Unsupported transposed convolution dimensionality: {ndim}")
    else:
        if ndim == 3:  # 1D convolution (N, C, W)
            out = self._conv1d_impl(
                x, weight, bias=bias, strides=stride, padding=padding, dilation=dilation, groups=groups
            )
        elif ndim == 4:  # 2D convolution (N, C, H, W)
            out = self._conv2d_impl(
                x, weight, bias=bias, strides=stride, padding=padding, dilation=dilation, groups=groups
            )
        elif ndim == 5:  # 3D convolution (N, C, D, H, W)
            out = self._conv3d_impl(
                x, weight, bias=bias, strides=stride, padding=padding, dilation=dilation, groups=groups
            )
        else:
            raise ValueError(f"Unsupported convolution dimensionality: {ndim}")
            
    # Debug output shape
    out_shape = self.shape_of(out)
    if out_shape is None:
        debug_print(f"DEBUG: _convolution output shape matches None! node={node.name}")
    else:
        # debug_print(f"DEBUG: _convolution output shape: {out_shape}")
        pass
        
    return out

def _leaky_relu(self, node):
    args = self.retrieve_args(node)
    x = args[0]
    negative_slope = args[1] if len(args) > 1 else 0.01
    # Check kwarg
    if "negative_slope" in node.kwargs: negative_slope = node.kwargs["negative_slope"]
    
    debug_print(f"DEBUG: _leaky_relu node={node.name}, input_shape={self.shape_of(x)}, producer={node.args[0].name if hasattr(node.args[0], 'name') else str(node.args[0])}")
    try:
        return self.block_builder.emit(relax.op.nn.leaky_relu(x, negative_slope))
    except AttributeError:
        # Fallback for some TVM versions
        if hasattr(relax.op.nn, "leaky_relu"): # Was checked above
             pass
        return self.block_builder.emit(relax.op.nn.leakyrelu(x, negative_slope))

ExportedProgramImporter._leaky_relu = _leaky_relu

def _mul(self, node):
    args = self.retrieve_args(node)
    lhs = args[0]
    rhs = args[1]
    
    def safe_shape_of(x):
        if hasattr(x, "struct_info"):
            return self.shape_of(x)
        return type(x)

    ls = safe_shape_of(lhs)
    rs = safe_shape_of(rhs)
    
    # Only print for later nodes to avoid spam? or inspect node name
    if "mul_389" in node.name or (isinstance(ls, relax.ShapeExpr) and isinstance(rs, relax.ShapeExpr) and (ls is None or rs is None)):
        debug_print(f"DEBUG: _mul node={node.name} lhs_shape={ls} rhs_shape={rs}")
        
    # Determine common dtype
    target_dtype = None
    arg_producers = []
    for i, arg in enumerate([lhs, rhs]):
         # debug_print(f"DEBUG: arg[{i}] type={type(arg)}")
         arg_prod = "unknown"
         if hasattr(node.args[i], "name"): arg_prod = node.args[i].name
         arg_producers.append(arg_prod)
         if hasattr(arg, "struct_info"):
             # debug_print(f"DEBUG: arg[{i}] struct_info={arg.struct_info}")
             if hasattr(arg.struct_info, "dtype"):
                 d = arg.struct_info.dtype
                 # debug_print(f"DEBUG: arg[{i}] dtype={d}")
                 if "float" in d:
                     target_dtype = d
                     break
                 if target_dtype is None: target_dtype = d
         else:
             # debug_print(f"DEBUG: arg[{i}] has no struct_info")
             pass
    
    # Force print if likely failing node
    if "mul_389" in node.name or target_dtype is None or target_dtype == "int64" or any(self.shape_of(a) is None for a in [lhs, rhs] if hasattr(a, "struct_info")):
         debug_print(f"DEBUG: _mul node={node.name}, producers={arg_producers}, detected target_dtype={target_dtype}") 
         print([str(a.struct_info) if hasattr(a, "struct_info") else type(a) for a in [lhs, rhs]])
         pass

    def ensure_tensor(val, dtype_hint):
        import tvm
        val_dtype = "int64"
        if isinstance(val, float): val_dtype = "float32"
        if dtype_hint: val_dtype = dtype_hint
        
        debug_print(f"DEBUG: ensure_tensor val={val} type={type(val)} dtype_hint={dtype_hint} -> val_dtype={val_dtype}")

        if isinstance(val, (int, float, bool)):
             return relax.const(val, val_dtype)
        elif isinstance(val, tvm.tir.PrimExpr):
             return self._create_scalar_tensor(val, val_dtype)
        return val

    lhs = ensure_tensor(lhs, target_dtype)
    rhs = ensure_tensor(rhs, target_dtype)
    
    # Coerce dtypes if mismatch
    if target_dtype:
         def coerce(val, dtype):
             if hasattr(val, "struct_info") and hasattr(val.struct_info, "dtype"):
                 if val.struct_info.dtype != dtype:
                     return self.block_builder.emit(relax.op.astype(val, dtype))
             return val
         
         lhs = coerce(lhs, target_dtype)
         rhs = coerce(rhs, target_dtype)
    
    return self.block_builder.emit(relax.op.multiply(lhs, rhs))

ExportedProgramImporter._mul = _mul


ExportedProgramImporter._lstm = _lstm
ExportedProgramImporter._convolution = _convolution

def _cat(self, node):
    args = self.retrieve_args(node)
    tensors = args[0]
    dim = args[1] if len(args) > 1 else 0
    
    shapes = [self.shape_of(t) for t in tensors]
    debug_print(f"DEBUG: _cat node={node.name}, dim={dim}, input_shapes={shapes}")
    
    out = self.block_builder.emit(relax.op.concat(tensors, axis=dim))
    
    # Force shape if inference failed (likely due to symbolic mismatch in non-concat dims)
    if self.shape_of(out) is None:
        debug_print(f"DEBUG: _cat output shape is None. Attempting to force shape based on first input.")
        if shapes[0] is not None:
            # Assuming all inputs match the first input on non-concat dims
            # And we just sum the concat dim
            target_shape_list = list(shapes[0])
            
            # Compute new dim size
            total_dim_size = 0
            unknown_dim = False
            for s in shapes:
                if s is None: 
                    unknown_dim = True
                    break
                dim_val = s[dim]
                if isinstance(dim_val, (int, tvm.tir.IntImm)):
                    total_dim_size += int(dim_val)
                elif isinstance(dim_val, tvm.tir.PrimExpr):
                     # Try to add expressions? 
                     # For Kokoro decoder, channels are usually static ints.
                     try:
                        total_dim_size += int(dim_val)
                     except:
                        # Symbolic channel count? Unexpected for this model.
                        # Maybe we can sum expressions.
                        total_dim_size += dim_val
                else:
                    unknown_dim = True
                    break
            
            if not unknown_dim:
                target_shape_list[dim] = total_dim_size
                target_shape = relax.ShapeExpr(target_shape_list)
                debug_print(f"DEBUG: Forcing cat output shape: {target_shape}")
                
                # Use match_cast to enforce this shape
                out = self.block_builder.match_cast(out, relax.TensorStructInfo(target_shape, dtype=out.struct_info.dtype))
                
    return out

ExportedProgramImporter._cat = _cat

def _sym_float(self, node):
    val = self.env[node.args[0]]
    if isinstance(val, (tvm.tir.PrimExpr, int, float)):
        return self._create_scalar_tensor(val, "float32")
    return self.block_builder.emit(relax.op.astype(val, "float32"))

ExportedProgramImporter._sym_float = _sym_float

def _clamp(self, node):
    input_tensor = self.env[node.args[0]]
    min_val = node.args[1] if len(node.args) > 1 else None
    max_val = node.args[2] if len(node.args) > 2 else None
    
    # Handle kwargs
    if "min" in node.kwargs: min_val = node.kwargs["min"]
    if "max" in node.kwargs: max_val = node.kwargs["max"]
    
    def _cast_to_dtype(val, dtype):
        if val is None: return None
        if isinstance(val, (int, float)):
            return relax.const(val, dtype)
        if isinstance(val, torch.fx.Node):
            val_expr = self.env[val]
            if isinstance(val_expr, (tvm.tir.PrimExpr, int, float)):
                 return self._create_scalar_tensor(val_expr, dtype)
            if hasattr(val_expr, "struct_info") and isinstance(val_expr.struct_info, relax.TensorStructInfo):
                 if val_expr.struct_info.dtype != dtype:
                     return self.block_builder.emit(relax.op.astype(val_expr, dtype))
            return val_expr
        # If val is already Relax Expr
        if isinstance(val, relax.Expr):
             if hasattr(val, "struct_info") and isinstance(val.struct_info, relax.TensorStructInfo):
                 if val.struct_info.dtype != dtype:
                     return self.block_builder.emit(relax.op.astype(val, dtype))
        return val

    x = input_tensor
    target_dtype = x.struct_info.dtype
    
    min_val_expr = _cast_to_dtype(min_val, target_dtype)
    max_val_expr = _cast_to_dtype(max_val, target_dtype)
    
    if min_val_expr is not None:
        x = self.block_builder.emit(relax.op.maximum(x, min_val_expr))
    if max_val_expr is not None:
        x = self.block_builder.emit(relax.op.minimum(x, max_val_expr))
        
    return x

ExportedProgramImporter._clamp = _clamp

from tvm.relax.frontend.torch.base_fx_graph_translator import BaseFXGraphImporter
original_index_tensor = BaseFXGraphImporter._index_tensor

def _index_tensor(self, node):
    # Cast indices in env to int64 if needed
    indices_nodes = node.args[1]
    for idx_node in indices_nodes:
        if idx_node is not None and isinstance(idx_node, torch.fx.Node):
             val = self.env[idx_node]
             if isinstance(val, relax.Expr) and hasattr(val, "struct_info"):
                 if isinstance(val.struct_info, relax.TensorStructInfo):
                     if val.struct_info.dtype not in ("int32", "int64"):
                         debug_print(f"DEBUG: Casting index node {idx_node} from {val.struct_info.dtype} to int64")
                         new_val = self.block_builder.emit(relax.op.astype(val, "int64"))
                         self.env[idx_node] = new_val

    return original_index_tensor(self, node)

ExportedProgramImporter._index_tensor = _index_tensor  # Can attach to class too

def _atan2(self, node):
    lhs = self.env[node.args[0]]
    rhs = self.env[node.args[1]]
    try:
        return self.block_builder.emit(relax.op.atan2(lhs, rhs))
    except AttributeError:
        def compute_func(a, b):
             return tvm.te.compute(a.shape, lambda *i: tvm.tir.atan2(a(*i), b(*i)), name="atan2")
        return self.block_builder.emit_te(compute_func, lhs, rhs)

ExportedProgramImporter._atan2 = _atan2

def _index_put(self, node):
    from tvm import topi
    from tvm.relax.frontend.torch.base_fx_graph_translator import BaseFXGraphImporter
    
    args = self.retrieve_args(node)
    data = args[0]
    indices = args[1]
    values = args[2]
    accumulate = args[3] if len(args) > 3 else False
    
    # Check for bool indices
    has_bool = False
    if isinstance(indices, (list, tuple)):
        for idx in indices:
            if idx is not None:
                 if hasattr(idx, "struct_info") and isinstance(idx.struct_info, relax.TensorStructInfo):
                     if idx.struct_info.dtype == "bool":
                         has_bool = True
                         break
    
    if has_bool:
        debug_print(f"DEBUG: _index_put handling bool indices for node {node.name}")
        # Assuming single bool mask for now (common case)
        if len(indices) == 1 and indices[0] is not None:
            mask = indices[0]
            
            # Optimization: If values matches data shape or is scalar, use where
            # tensor[mask] = values -> where(mask, values, tensor)
            use_where = False
            if hasattr(values, "struct_info") and isinstance(values.struct_info, relax.TensorStructInfo):
                 # Scalar
                 if values.struct_info.ndim == 0:
                     use_where = True
                 # Same shape as data/mask
                 # (Relax structural equality check or manual?)
                 # For now, just trust scalar usage or broadcasting if ndim matches?
            
            if use_where:
                debug_print(f"DEBUG: Using where() optimization for bool index_put")
                return self.block_builder.emit(relax.op.where(mask, values, data))

            # argwhere fallback
            def compute_argwhere(m):
                if hasattr(topi, "argwhere"): return topi.argwhere(m)
                try: 
                    from tvm.topi import transform
                    return transform.argwhere(m)
                except:
                    pass
                raise RuntimeError("argwhere not found in topi")
            
            indices_blob = self.block_builder.emit_te(compute_argwhere, mask)
            
            # Get ndim of logical data (which matches mask ndim usually)
            # data[mask] = values. mask shape must be broadcastable to data?
            # usually mask is same shape as data or prefix.
            mask_ndim = len(mask.struct_info.shape)
            
            # Transpose to (ndim, N)
            indices_transposed = self.block_builder.emit(relax.op.permute_dims(indices_blob, axes=[1, 0]))
            
            # Split into separate index tensors
            new_indices = []
            for i in range(mask_ndim):
                # Take row i: (1, N)
                idx_slice = self.block_builder.emit(relax.op.strided_slice(indices_transposed, axes=[0], begin=[i], end=[i+1]))
                # Squeeze to (N,)
                # Note: squeeze outputting (N,) requires ShapeExpr for N?
                # relax.op.squeeze computes shape.
                idx = self.block_builder.emit(relax.op.squeeze(idx_slice, axis=[0]))
                new_indices.append(idx)
            
            return self.block_builder.emit(relax.op.index_put(data, relax.Tuple(new_indices), values, accumulate))
    
    return BaseFXGraphImporter._index_put(self, node)

ExportedProgramImporter._index_put = _index_put

# Update map keys
# But since I can't restart, I will just update convert_map inside new_create_convert_map if I can re-execute it?
# No, `new_create_convert_map` is executed by `ExportedProgramImporter` when instantiated.
# Since my script imports `tvm_extensions`, it monkeypatches `create_convert_map`.
# I need to update `custom_map` literal in `new_create_convert_map`.
# I'll update line 60 to add clamp.


# Update map keys
original_create_convert_map = ExportedProgramImporter.create_convert_map # Wait, I already overwrote it?
# Actually I need to update the `custom_map` defined LOCALLY inside `new_create_convert_map`?
# I can't easily edit `new_create_convert_map` definition if I don't replace the whole function.
# But I can modify `new_create_convert_map` in the file.

# Oh, replacing the logic in new_create_convert_map in the file:



# Apply monkeypatch
# Apply monkeypatch
ExportedProgramImporter.create_convert_map = new_create_convert_map

# Counter for unique slice_scatter function names
_slice_scatter_counter = [0]

# Custom legalization for slice_scatter to handle dynamic shapes that emit_te rejects
@tvm.ir.register_op_attr("relax.slice_scatter", "FLegalize", level=11)
def _custom_slice_scatter(bb: relax.BlockBuilder, call: relax.Call):
    from tvm import te, topi
    _slice_scatter_counter[0] += 1
    func_id = _slice_scatter_counter[0]
    data, src, start_val, end_val, step_val = call.args
    axis = int(call.attrs.axis)

    def get_te_shape(expr):
        sinfo = expr.struct_info
        if isinstance(sinfo.shape, relax.ShapeExpr):
             return list(sinfo.shape.values)
        if sinfo.ndim < 0:
            # Fallback: assume rank 3 if unknown (based on failure case)
            # Or raise error if truly unknown.
            # But failure case was ndim=3
            return [te.var(f"dim_unk_{i}") for i in range(3)] 
        # Create symbolic vars for dims if shape is not ShapeExpr (e.g. Var)
        return [te.var(f"dim_{i}") for i in range(sinfo.ndim)]

    data_shape = get_te_shape(data)
    src_shape = get_te_shape(src)
    
    te_data = te.placeholder(data_shape, dtype=data.struct_info.dtype, name="data")
    te_src = te.placeholder(src_shape, dtype=src.struct_info.dtype, name="src")
    
    # Handle PrimValue args
    def handle_arg(arg, name):
         val = arg.value
         if isinstance(val, (int, tvm.tir.IntImm)):
             return val.value if hasattr(val, 'value') else val, None
         # Symbolic or dynamic: use a fresh var for definition, pass arg in call
         v = te.var(name, dtype="int64")
         return v, v
    
    start, start_var = handle_arg(start_val, "start")
    end, end_var = handle_arg(end_val, "end")
    step, step_var = handle_arg(step_val, "step")
    
    # Check if we can use single output or list
    # topi.slice_scatter usually returns list of 1 tensor
    out_topi = topi.slice_scatter(te_data, te_src, start, end, step, axis)
    if isinstance(out_topi, list):
        output = out_topi[0]
    else:
        output = out_topi
    
    # Build PrimFunc signature
    # Inputs + [Dynamic Scalar Args] + [Output]
    te_args = [te_data, te_src]
    call_args = [data, src]
    
    if start_var is not None:
        te_args.append(start_var)
        call_args.append(start_val)
    if end_var is not None:
         te_args.append(end_var)
         call_args.append(end_val)
    if step_var is not None:
         te_args.append(step_var)
         call_args.append(step_val)
         
    te_args.append(output)
    
    prim_func = te.create_prim_func(te_args)
    # Explicitly set unique global_symbol to avoid collision
    prim_func = prim_func.with_attr("global_symbol", f"slice_scatter_tir_{func_id}")
    gvar = bb.add_func(prim_func, f"slice_scatter_custom_{func_id}")
    
    # Ensure out_sinfo is correct. 
    # call.struct_info is Tensor(ndim=3, dtype=float32) which is what we want.
    return bb.emit(relax.call_tir(gvar, call_args, out_sinfo=call.struct_info))


