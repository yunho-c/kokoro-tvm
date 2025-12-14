"""Diagnostic script to identify where IR explosion occurs in duration/f0 compilation.

This script measures IR size and time at each stage of the compilation pipeline:
1. torch.export output (ExportedProgram graph size)
2. from_exported_program output (initial Relax IR)
3. After each transform pass

Run with: python experiments/ir_explosion_diagnosis.py
"""

import sys
import os
import time

# Add paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external", "kokoro"))

import torch
import tvm
from tvm import relax

# Import extensions first
import kokoro_tvm.tvm_extensions  # noqa: F401
from kokoro_tvm.patches.lstm import apply_lstm_patch

# Apply LSTM patch
apply_lstm_patch()

from kokoro.modules import ProsodyPredictor


def count_ir_nodes(mod):
    """Count approximate number of IR nodes in a module."""
    script = mod.script()
    # Rough heuristic: count lines and significant tokens
    lines = len(script.split("\n"))
    bytes_size = len(script.encode("utf-8"))
    return lines, bytes_size


def count_exported_program_nodes(ep):
    """Count nodes in ExportedProgram graph."""
    return len(list(ep.graph.nodes))


def measure_stage(name, func):
    """Measure time and return result."""
    print(f"\n{'=' * 60}")
    print(f"STAGE: {name}")
    print("=" * 60)

    start = time.time()
    result = func()
    elapsed = time.time() - start

    print(f"⏱️  Time: {elapsed:.2f}s")
    return result, elapsed


def main():
    SEQ_LEN = 64  # Smaller for faster diagnosis
    STYLE_DIM = 128
    D_HID = 512
    NLAYERS = 3

    print("=" * 60)
    print("IR EXPLOSION DIAGNOSIS")
    print(f"SEQ_LEN={SEQ_LEN}, NLAYERS={NLAYERS}, D_HID={D_HID}")
    print("=" * 60)

    # Stage 0: Create model
    print("\n📦 Creating ProsodyPredictor model...")
    predictor = ProsodyPredictor(
        style_dim=STYLE_DIM, d_hid=D_HID, nlayers=NLAYERS, max_dur=50, dropout=0.0
    )
    predictor.eval()

    class DurationWrapper(torch.nn.Module):
        def __init__(self, predictor, seq_len):
            super().__init__()
            self.text_encoder = predictor.text_encoder
            self.lstm = predictor.lstm
            self.duration_proj = predictor.duration_proj
            self.seq_len = seq_len

        def forward(self, d_en, style, text_lengths, m):
            d = self.text_encoder(d_en, style, text_lengths, m)
            self.lstm.flatten_parameters()
            x, _ = self.lstm(d)
            x_pad = torch.zeros(
                [x.shape[0], self.seq_len, x.shape[-1]], device=x.device
            )
            x_pad[:, : x.shape[1], :] = x
            x = x_pad
            duration = self.duration_proj(x)
            return duration.squeeze(-1), d

    model = DurationWrapper(predictor, SEQ_LEN)
    model.eval()

    # Create inputs
    batch_size = 1
    input_d_en = torch.randn(batch_size, D_HID, SEQ_LEN)
    input_style = torch.randn(batch_size, STYLE_DIM)
    input_lengths = torch.tensor([SEQ_LEN] * batch_size, dtype=torch.long)
    input_m = torch.zeros(batch_size, SEQ_LEN, dtype=torch.bool)
    args = (input_d_en, input_style, input_lengths, input_m)

    # Stage 1: torch.export
    def do_export():
        return torch.export.export(model, args)

    ep, t1 = measure_stage("torch.export", do_export)
    n_nodes = count_exported_program_nodes(ep)
    print(f"📊 ExportedProgram nodes: {n_nodes}")

    # Save graph for inspection
    with open("/tmp/exported_graph.txt", "w") as f:
        f.write(str(ep.graph))
    print("   Saved graph to /tmp/exported_graph.txt")

    # Stage 2: from_exported_program
    from tvm.relax.frontend.torch import from_exported_program

    def do_import():
        return from_exported_program(ep, keep_params_as_input=False)

    mod, t2 = measure_stage("from_exported_program", do_import)
    lines, bytes_ = count_ir_nodes(mod)
    print(f"📊 Initial Relax IR: {lines} lines, {bytes_ // 1024}KB")

    # Save initial IR
    with open("/tmp/ir_1_initial.py", "w") as f:
        f.write(mod.script())
    print("   Saved to /tmp/ir_1_initial.py")

    # Stage 3: DecomposeOpsForInference
    target = tvm.target.Target("llvm")

    def do_decompose():
        with target:
            return relax.transform.DecomposeOpsForInference()(mod)

    mod2, t3 = measure_stage("DecomposeOpsForInference", do_decompose)
    lines, bytes_ = count_ir_nodes(mod2)
    print(f"📊 After DecomposeOps: {lines} lines, {bytes_ // 1024}KB")

    with open("/tmp/ir_2_decompose.py", "w") as f:
        f.write(mod2.script())
    print("   Saved to /tmp/ir_2_decompose.py")

    # Stage 4: LegalizeOps
    def do_legalize():
        with target:
            return relax.transform.LegalizeOps()(mod2)

    mod3, t4 = measure_stage("LegalizeOps", do_legalize)
    lines, bytes_ = count_ir_nodes(mod3)
    print(f"📊 After LegalizeOps: {lines} lines, {bytes_ // 1024}KB")

    with open("/tmp/ir_3_legalize.py", "w") as f:
        f.write(mod3.script())
    print("   Saved to /tmp/ir_3_legalize.py")

    # Stage 5: AnnotateTIROpPattern
    def do_annotate():
        with target:
            return relax.transform.AnnotateTIROpPattern()(mod3)

    mod4, t5 = measure_stage("AnnotateTIROpPattern", do_annotate)
    lines, bytes_ = count_ir_nodes(mod4)
    print(f"📊 After AnnotateTIR: {lines} lines, {bytes_ // 1024}KB")

    # Stage 6: FuseOps
    def do_fuse():
        with target:
            m = relax.transform.DeadCodeElimination()(mod4)
            return relax.transform.FuseOps()(m)

    mod5, t6 = measure_stage("DCE + FuseOps", do_fuse)
    lines, bytes_ = count_ir_nodes(mod5)
    print(f"📊 After FuseOps: {lines} lines, {bytes_ // 1024}KB")

    # Stage 7: FuseTIR
    def do_fuse_tir():
        with target:
            return relax.transform.FuseTIR()(mod5)

    mod6, t7 = measure_stage("FuseTIR", do_fuse_tir)
    lines, bytes_ = count_ir_nodes(mod6)
    print(f"📊 After FuseTIR: {lines} lines, {bytes_ // 1024}KB")

    with open("/tmp/ir_6_final.py", "w") as f:
        f.write(mod6.script())
    print("   Saved to /tmp/ir_6_final.py")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Stage':<30} {'Time':>10}")
    print("-" * 42)
    print(f"{'torch.export':<30} {t1:>10.2f}s")
    print(f"{'from_exported_program':<30} {t2:>10.2f}s")
    print(f"{'DecomposeOpsForInference':<30} {t3:>10.2f}s")
    print(f"{'LegalizeOps':<30} {t4:>10.2f}s")
    print(f"{'AnnotateTIROpPattern':<30} {t5:>10.2f}s")
    print(f"{'DCE + FuseOps':<30} {t6:>10.2f}s")
    print(f"{'FuseTIR':<30} {t7:>10.2f}s")
    print("-" * 42)
    print(f"{'TOTAL':<30} {t1 + t2 + t3 + t4 + t5 + t6 + t7:>10.2f}s")

    print("\n💡 Check /tmp/ir_*.py files to see IR at each stage")
    print("   The largest jump in size indicates where explosion happens")


if __name__ == "__main__":
    main()
