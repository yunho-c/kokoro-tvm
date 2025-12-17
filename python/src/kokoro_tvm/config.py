# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration and utilities for Kokoro TVM compilation."""

import tvm

# Target configurations for different platforms
TARGET_CONFIGS = {
    "llvm": {
        "target": "llvm -opt-level=3",
        "extension": ".so",
        "description": "CPU (LLVM)",
    },
    "metal-macos": {
        "target_host": "llvm -mtriple=arm64-apple-macos",
        "target": "metal",
        "extension": ".dylib",
        "description": "macOS (Metal GPU + ARM64 CPU)",
    },
    "metal-ios": {
        "target_host": "llvm -mtriple=arm64-apple-ios",
        "target": "metal",
        "extension": ".dylib",
        "description": "iOS (Metal GPU + ARM64 CPU)",
    },
    "webgpu": {
        "target_host": "llvm -mtriple=wasm32-unknown-unknown-wasm",
        "target": "webgpu",
        "extension": ".wasm",
        "description": "WebGPU (Browser via WASM)",
        "export_func": "tvmjs",  # Special flag for WASM export
    },
}


def resolve_target(target_name: str) -> tuple:
    """Resolve target name to TVM target objects.

    Returns:
        Tuple of (target, target_host, extension, description, export_func)
        export_func is None for native targets, "tvmjs" for WASM targets.
    """
    if target_name not in TARGET_CONFIGS:
        available = list(TARGET_CONFIGS.keys())
        msg = f"Unknown target: {target_name}. Available: {available}"
        raise ValueError(msg)

    config = TARGET_CONFIGS[target_name]

    if "target_host" in config:
        # Metal/WebGPU targets have separate host target
        target_host = tvm.target.Target(config["target_host"])
        target = tvm.target.Target(config["target"], host=target_host)
    else:
        # CPU-only targets
        target = tvm.target.Target(config["target"])
        target_host = None

    export_func = config.get("export_func", None)

    return target, target_host, config["extension"], config["description"], export_func
