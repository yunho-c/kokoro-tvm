# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""MetaSchedule auto-tuning for TVM Relax modules.

This module provides utilities for auto-tuning TIR primitives using TVM's
MetaSchedule framework to optimize performance for specific hardware targets.
"""

import os
from pathlib import Path
from typing import Optional, Union

import tvm
from tvm import relax
from tvm.ir import IRModule


def get_tuning_pipeline(
    work_dir: Union[str, Path],
    max_trials: int = 2000,
) -> tvm.transform.Sequential:
    """Get the tuning transformation pipeline.
    
    This creates a pipeline that:
    1. Legalizes Relax ops to TIR
    2. Tunes TIR primitives with MetaSchedule
    3. Applies the best schedules from the database
    
    Args:
        work_dir: Directory for tuning artifacts (logs, database)
        max_trials: Maximum number of tuning trials
        
    Returns:
        Sequential transform pipeline
    """
    work_dir = str(work_dir)
    
    return tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.MetaScheduleTuneTIR(
            work_dir=work_dir,
            max_trials_global=max_trials,
        ),
        relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir),
    ])


def tune_module(
    mod: IRModule,
    target: tvm.target.Target,
    work_dir: Union[str, Path] = "tuning_logs",
    max_trials: int = 2000,
    num_trials_per_iter: int = 64,
) -> IRModule:
    """Auto-tune a Relax module with MetaSchedule.
    
    This function tunes the TIR primitives in the module using MetaSchedule's
    search algorithms to find optimal schedules for the target hardware.
    
    Args:
        mod: The Relax IRModule to tune
        target: TVM target (e.g., "metal", "llvm")
        work_dir: Directory for tuning artifacts
        max_trials: Maximum number of tuning trials globally
        num_trials_per_iter: Number of trials per iteration
        
    Returns:
        Tuned IRModule with optimized schedules applied
        
    Example:
        >>> target = tvm.target.Target("metal", host="llvm -mtriple=arm64-apple-macos")
        >>> tuned_mod = tune_module(mod, target, work_dir="tuning_logs", max_trials=2000)
    """
    from tvm import meta_schedule as ms
    
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting MetaSchedule tuning...")
    print(f"  Target: {target}")
    print(f"  Work dir: {work_dir}")
    print(f"  Max trials: {max_trials}")
    
    # Run with profiling to track tuning time
    with ms.Profiler() as profiler:
        with target:
            pipeline = get_tuning_pipeline(work_dir, max_trials)
            tuned_mod = pipeline(mod)
    
    # Print profiling results
    print("\nTuning completed!")
    print(profiler.table())
    
    return tuned_mod


def apply_tuned_database(
    mod: IRModule,
    target: tvm.target.Target,
    work_dir: Union[str, Path],
) -> IRModule:
    """Apply pre-tuned schedules from a database.
    
    Use this to apply previously tuned schedules without re-running tuning.
    Useful for deployment or when transferring tuned schedules.
    
    Args:
        mod: The Relax IRModule
        target: TVM target (must match the target used for tuning)
        work_dir: Directory containing tuning database
        
    Returns:
        Module with tuned schedules applied
        
    Raises:
        FileNotFoundError: If tuning database doesn't exist
    """
    work_dir = Path(work_dir)
    
    if not work_dir.exists():
        raise FileNotFoundError(f"Tuning database not found: {work_dir}")
    
    print(f"Applying tuned schedules from: {work_dir}")
    
    with target:
        pipeline = tvm.transform.Sequential([
            relax.transform.LegalizeOps(),
            relax.transform.MetaScheduleApplyDatabase(work_dir=str(work_dir)),
        ])
        tuned_mod = pipeline(mod)
    
    return tuned_mod


def estimate_tuning_time(
    mod: IRModule,
    target: tvm.target.Target,
    max_trials: int = 2000,
    seconds_per_trial: float = 0.5,
) -> dict:
    """Estimate tuning time based on module complexity.
    
    Args:
        mod: The module to tune
        target: Target hardware
        max_trials: Maximum trials
        seconds_per_trial: Estimated time per trial
        
    Returns:
        Dictionary with estimation details
    """
    # Count tunable TIR functions
    num_tir_funcs = sum(
        1 for gv, func in mod.functions.items()
        if isinstance(func, tvm.tir.PrimFunc)
    )
    
    estimated_seconds = num_tir_funcs * max_trials * seconds_per_trial
    
    return {
        "num_tir_functions": num_tir_funcs,
        "max_trials": max_trials,
        "seconds_per_trial": seconds_per_trial,
        "estimated_seconds": estimated_seconds,
        "estimated_minutes": estimated_seconds / 60,
        "estimated_hours": estimated_seconds / 3600,
    }
