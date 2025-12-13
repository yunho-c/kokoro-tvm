# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Profiling utilities for Kokoro TVM."""

from kokoro_tvm.profiling.benchmark import (
    benchmark_module,
    benchmark_inference,
    create_benchmark_report,
)

__all__ = [
    "benchmark_module",
    "benchmark_inference",
    "create_benchmark_report",
]
