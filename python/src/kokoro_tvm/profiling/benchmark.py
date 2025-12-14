# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Benchmarking utilities for TVM compiled modules.

This module provides functions for measuring inference latency, throughput,
and generating performance reports.
"""

import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import tvm
from tvm import relax
from tvm.runtime import Tensor


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    throughput_per_sec: float
    num_runs: int
    warmup_runs: int
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"Latency: {self.mean_ms:.3f} ± {self.std_ms:.3f} ms "
            f"(min={self.min_ms:.3f}, max={self.max_ms:.3f}, median={self.median_ms:.3f})\n"
            f"Throughput: {self.throughput_per_sec:.1f} inferences/sec"
        )


def get_device(target_str: str) -> tvm.runtime.Device:
    """Get TVM device from target string."""
    if "metal" in target_str.lower():
        return tvm.metal()
    elif "cuda" in target_str.lower():
        return tvm.cuda()
    else:
        return tvm.cpu()


def benchmark_inference(
    vm: relax.VirtualMachine,
    func_name: str,
    inputs: List[Tensor],
    warmup: int = 5,
    repeat: int = 20,
) -> BenchmarkResult:
    """Benchmark a single function in the VM.
    
    Args:
        vm: Relax VirtualMachine instance
        func_name: Name of the function to benchmark
        inputs: List of input NDArrays
        warmup: Number of warmup runs
        repeat: Number of timed runs
        
    Returns:
        BenchmarkResult with timing statistics
    """
    func = vm[func_name]
    
    # Warmup
    for _ in range(warmup):
        _ = func(*inputs)
    
    # Sync before timing
    tvm.runtime.device_sync(inputs[0].device)
    
    # Timed runs
    times_ms = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = func(*inputs)
        tvm.runtime.device_sync(inputs[0].device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    
    times_ms = np.array(times_ms)
    
    return BenchmarkResult(
        mean_ms=float(np.mean(times_ms)),
        std_ms=float(np.std(times_ms)),
        min_ms=float(np.min(times_ms)),
        max_ms=float(np.max(times_ms)),
        median_ms=float(np.median(times_ms)),
        throughput_per_sec=1000.0 / float(np.mean(times_ms)),
        num_runs=repeat,
        warmup_runs=warmup,
    )


def benchmark_module(
    lib_path: Union[str, Path],
    target: str = "llvm",
    func_name: str = "decoder_forward",
    seq_len: int = 150,
    hidden_dim: int = 512,
    style_dim: int = 128,
    warmup: int = 5,
    repeat: int = 20,
) -> BenchmarkResult:
    """Benchmark a compiled TVM module.
    
    Loads the module and runs inference benchmarks with synthetic inputs.
    
    Args:
        lib_path: Path to compiled .so/.dylib file
        target: Target string for device selection
        func_name: Name of the function to benchmark
        seq_len: Sequence length for synthetic inputs
        hidden_dim: Hidden dimension for ASR input
        style_dim: Style dimension
        warmup: Number of warmup runs
        repeat: Number of timed runs
        
    Returns:
        BenchmarkResult with timing statistics
    """
    lib = tvm.runtime.load_module(str(lib_path))
    dev = get_device(target)
    vm = relax.VirtualMachine(lib, dev)
    
    # Create synthetic inputs matching decoder signature
    inputs = [
        tvm.runtime.tensor(np.random.randn(1, hidden_dim, seq_len).astype("float32"), device=dev),
        tvm.runtime.tensor(np.random.randn(1, seq_len * 2).astype("float32"), device=dev),
        tvm.runtime.tensor(np.random.randn(1, seq_len * 2).astype("float32"), device=dev),
        tvm.runtime.tensor(np.random.randn(1, style_dim).astype("float32"), device=dev),
    ]
    
    return benchmark_inference(vm, func_name, inputs, warmup, repeat)


def create_benchmark_report(
    lib_path: Union[str, Path],
    result: BenchmarkResult,
    target: str,
    seq_len: int,
    output_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Create a comprehensive benchmark report.
    
    Args:
        lib_path: Path to the benchmarked module
        result: Benchmark results
        target: Target used
        seq_len: Sequence length used
        output_path: Optional path to save JSON report
        
    Returns:
        Report dictionary
    """
    report = {
        "module": str(lib_path),
        "target": target,
        "seq_len": seq_len,
        "system": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "results": result.to_dict(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_path}")
    
    return report


def compare_results(
    baseline: BenchmarkResult,
    optimized: BenchmarkResult,
) -> dict:
    """Compare two benchmark results.
    
    Args:
        baseline: Result before optimization
        optimized: Result after optimization
        
    Returns:
        Comparison dictionary with speedup metrics
    """
    speedup = baseline.mean_ms / optimized.mean_ms
    latency_reduction_pct = (1 - optimized.mean_ms / baseline.mean_ms) * 100
    throughput_improvement_pct = (optimized.throughput_per_sec / baseline.throughput_per_sec - 1) * 100
    
    return {
        "speedup": speedup,
        "latency_reduction_pct": latency_reduction_pct,
        "throughput_improvement_pct": throughput_improvement_pct,
        "baseline_mean_ms": baseline.mean_ms,
        "optimized_mean_ms": optimized.mean_ms,
    }
