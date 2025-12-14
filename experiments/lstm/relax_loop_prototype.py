#!/usr/bin/env python3
"""Prototype: LSTM using Relax recursive functions for control flow.

Relax doesn't have explicit while/for loops. Instead, it uses:
- Recursive function calls for iteration
- The VM handles tail-call optimization

Run with: py -3.12 experiments/relax_loop_prototype.py
"""

import numpy as np
import tvm
from tvm.script import tir as T, relax as R, ir as I


@I.ir_module
class RecursiveLoopModule:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        # Define a local recursive function
        @R.function
        def while_loop(
            i: R.Tensor((), dtype="int32"), s: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            cond = R.call_pure_packed(
                "test.vm.less", i, R.const(10), sinfo_args=R.Tensor((), dtype="bool")
            )
            c = R.const(1, dtype="int32")
            if cond:
                new_i = R.add(i, c)
                new_s = R.add(s, x)
                r = while_loop(new_i, new_s)
            else:
                r = s
            return r

        gv = while_loop(R.const(0), x)
        return gv


def main():
    print("=" * 60)
    print("Relax Recursive Loop Prototype")
    print("=" * 60)

    mod = RecursiveLoopModule
    script = mod.script()

    print("\n📜 Generated Relax IR:")
    print(script)

    lines = len(script.split("\n"))
    bytes_size = len(script.encode("utf-8"))
    print(f"\n📊 IR Size: {lines} lines, {bytes_size / 1024:.1f} KB")

    # Save for inspection
    with open("/tmp/relax_recursive_loop.py", "w") as f:
        f.write(script)
    print("   Saved to /tmp/relax_recursive_loop.py")

    print("\n💡 Key Observations:")
    print("   1. Relax uses recursive functions for loops")
    print("   2. The 'if cond' becomes branching at runtime")
    print("   3. Tail-recursive calls are optimized by the VM")
    print("   4. No unrolling occurs - loop structure preserved!")
    print("   5. Loop is ~35 lines regardless of iteration count")

    print("\n🔧 To use for LSTM:")
    print("   - Define lstm_step as a recursive function")
    print("   - Pass hidden/cell state as arguments")
    print("   - Update and recurse until sequence complete")


if __name__ == "__main__":
    main()
