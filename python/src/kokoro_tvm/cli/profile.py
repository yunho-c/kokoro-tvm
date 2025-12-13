# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for profiling compiled modules."""

import argparse
from pathlib import Path

from kokoro_tvm.profiling import benchmark_module, create_benchmark_report
from kokoro_tvm.cli.port_decoder import TARGET_CONFIGS


def main():
    """Main CLI entry point for profiling."""
    parser = argparse.ArgumentParser(description="Profile compiled Kokoro Decoder")
    parser.add_argument("module", type=str,
                        help="Path to compiled module (.so or .dylib)")
    parser.add_argument("--target", type=str, default="llvm",
                        choices=list(TARGET_CONFIGS.keys()),
                        help=f"Target: {', '.join(TARGET_CONFIGS.keys())} (default: llvm)")
    parser.add_argument("--seq-len", type=int, default=150,
                        help="Sequence length for benchmark inputs (default: 150)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup runs (default: 5)")
    parser.add_argument("--repeat", type=int, default=20,
                        help="Number of timed runs (default: 20)")
    parser.add_argument("--func-name", type=str, default="decoder_forward",
                        help="Function name to benchmark (default: decoder_forward)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for JSON report")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to another module to compare against")
    args = parser.parse_args()
    
    module_path = Path(args.module)
    if not module_path.exists():
        print(f"Error: Module not found: {module_path}")
        return 1
    
    print(f"Profiling: {module_path}")
    print(f"Target: {args.target}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print()
    
    # Run benchmark
    result = benchmark_module(
        module_path,
        target=args.target,
        func_name=args.func_name,
        seq_len=args.seq_len,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    
    print("Results:")
    print(result)
    print()
    
    # Create report
    report = create_benchmark_report(
        module_path,
        result,
        args.target,
        args.seq_len,
        output_path=args.output,
    )
    
    # Compare if requested
    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f"Warning: Comparison module not found: {compare_path}")
        else:
            print(f"\nComparing with: {compare_path}")
            compare_result = benchmark_module(
                compare_path,
                target=args.target,
                func_name=args.func_name,
                seq_len=args.seq_len,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            
            print("Comparison results:")
            print(compare_result)
            print()
            
            from kokoro_tvm.profiling.benchmark import compare_results
            comparison = compare_results(result, compare_result)
            
            print("Comparison:")
            print(f"  Speedup: {comparison['speedup']:.2f}x")
            print(f"  Latency reduction: {comparison['latency_reduction_pct']:.1f}%")
            print(f"  Throughput improvement: {comparison['throughput_improvement_pct']:.1f}%")
    
    return 0


if __name__ == "__main__":
    exit(main())
