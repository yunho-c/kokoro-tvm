# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""kokoro-tvm: TVM compilation for the Kokoro text-to-speech model."""

__version__ = "0.0.1"

# Re-export commonly used items
from kokoro_tvm import tvm_extensions  # noqa: F401 - ensures patches are applied on import
