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