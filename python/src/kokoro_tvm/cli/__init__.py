# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry points for kokoro-tvm."""

from kokoro_tvm.cli.compile_kokoro import main as compile_kokoro_main
from kokoro_tvm.cli.port_decoder import main as port_decoder_main
from kokoro_tvm.cli.tune import main as tune_main
from kokoro_tvm.cli.profile import main as profile_main

__all__ = [
    "compile_kokoro_main", 
    "port_decoder_main",
    "tune_main",
    "profile_main",
]
