# [Bug] LLVM module verification fails with `#dbg_declare must be a pointer or int` on LLVM 15+

## Expected behavior

TVM should successfully compile and build Relax models targeting LLVM backends when using LLVM 15+ (including LLVM 21).

## Actual behavior

When building a model with fused kernels containing operations like `instance_norm`, LLVM module verification fails with:

```
LLVM module verification failed: location of #dbg_declare must be a pointer or int
```

This occurs specifically with fused functions like:
- `fused_add*_instance_norm*_multiply*_leaky_relu*_compute_`

## Environment

- **TVM Version**: main branch (commit hash: `fc2bdfe`)
- **LLVM Version**: 21.1.6 (also likely affects LLVM 15-20)
- **Operating System**: macOS 15 (also likely affects Linux)
- **Target**: `llvm -mtriple=arm64-apple-macosx`
- **Python**: 3.12

### TVM Configuration

```python
import tvm
print(tvm.support.libinfo()["LLVM_VERSION"])  # Output: 21.1.6
```

## Root Cause Analysis

The issue is in `src/target/llvm/codegen_llvm.cc`, function `AddDebugInformation` (lines 2311-2342).

### Problem

`AddDebugInformation` calls `insertDeclare` on **any** `llvm::Value*`, including floating-point values:

```cpp
void CodeGenLLVM::AddDebugInformation(llvm::Value* llvm_value, const Var& tir_var, ...) {
  // ...
  // No check if llvm_value is a pointer or integer!
  dbg_info_->di_builder_->insertDeclare(llvm_value, local_var, ...);  // ← fails on floats
}
```

LLVM 15+ added stricter verification for `#dbg_declare` intrinsics, requiring that the first operand be a pointer or integer type. When TVM emits debug info for floating-point reduction temporaries (e.g., in `instance_norm` fused kernels), LLVM's verifier rejects the module.

### Call Sites

The function is called from:
1. `VisitExpr_(const LetNode* op)` - line 1722
2. `VisitStmt_(const LetStmtNode* op)` - lines 2203, 2210  
3. `VisitStmt_(const AllocateNode* op)` - line 2137

When processing `LetNode` for float temporaries in reduction operations, `AddDebugInformation` receives a float `llvm::Value*` and attempts to attach debug info, which fails verification.

## Steps to reproduce

```python
import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Any model with InstanceNorm will trigger this
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(256)
    
    def forward(self, x):
        return self.norm(x)

model = Model().eval()
example_input = torch.randn(1, 256, 100)

# Export and import to TVM
with torch.no_grad():
    exported = torch.export.export(model, (example_input,))

mod = from_exported_program(exported)

# Apply standard optimizations
mod = relax.transform.DecomposeOpsForInference()(mod)
mod = relax.transform.LegalizeOps()(mod)
mod = relax.transform.FuseOps()(mod)
mod = relax.transform.FuseTIR()(mod)

# Build - this crashes with LLVM 15+
target = tvm.target.Target("llvm")
ex = relax.build(mod, target)  # LLVM verification error here
```

## Triage

* llvm
* type: bug
