# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal Conv1d(+weight_norm) probe for NaN debugging.

This isolates a single Conv1d layer from the decoder generator to determine
whether NaNs originate from conv lowering itself or from upstream tensor values.
"""

from __future__ import annotations

import json
from typing import Optional

import torch
import tvm
from huggingface_hub import hf_hub_download
from torch.nn.utils.parametrizations import weight_norm

from kokoro.istftnet import Decoder

from kokoro_tvm import tvm_extensions  # noqa: F401 - apply TVM patches
from kokoro_tvm.patches.adain import apply_adain_patch
from kokoro_tvm.patches.sinegen import apply_sinegen_patch


def _probe_slice(x: torch.Tensor, *, k: int = 16) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    flat = x.reshape(-1)
    k = min(int(k), int(flat.numel()))
    if k == 0:
        return flat
    head = flat[:k]
    tail = flat[-k:] if flat.numel() >= k else flat
    return torch.cat([head, tail], dim=0)


class WeightNormConv1dProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        kernel_size: int,
        dilation: int,
        padding: int,
        use_weight_norm: bool,
        load_decoder_weights: bool,
    ):
        super().__init__()
        base = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding,
            bias=True,
        )
        self.conv = weight_norm(base) if use_weight_norm else base

        if load_decoder_weights:
            repo_id = "hexgrad/Kokoro-82M"
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            istftnet_params = config["istftnet"]

            decoder = Decoder(
                dim_in=config["hidden_dim"],
                style_dim=config["style_dim"],
                dim_out=config["n_mels"],
                disable_complex=True,
                **istftnet_params,
            ).eval()

            model_path = hf_hub_download(repo_id=repo_id, filename="kokoro-v1_0.pth")
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            dec_state = state.get("decoder", {})
            first_key = next(iter(dec_state.keys()), "")
            if first_key.startswith("module."):
                dec_state = {k[7:]: v for k, v in dec_state.items()}
            decoder.load_state_dict(dec_state, strict=False)

            src = decoder.generator.noise_res[0].convs1[0]
            self.conv.load_state_dict(src.state_dict(), strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if hasattr(self.conv, "parametrizations"):
            g = self.conv.parametrizations.weight.original0
            v = self.conv.parametrizations.weight.original1
            w = self.conv.weight
        else:
            g = torch.zeros(1, dtype=torch.float32)
            v = torch.zeros(1, dtype=torch.float32)
            w = self.conv.weight
        return torch.stack([_probe_slice(x), _probe_slice(g), _probe_slice(v), _probe_slice(w), _probe_slice(y)], dim=0)


def create_weightnorm_conv1d_probe_module(
    *,
    length: int,
    channels: int = 256,
    kernel_size: int = 7,
    dilation: int = 1,
    padding: int = 3,
    use_weight_norm: bool = True,
    load_decoder_weights: bool = True,
    func_name: str = "weightnorm_conv1d_probe_forward",
    dump_ir: Optional[str] = None,
) -> tvm.IRModule:
    apply_sinegen_patch()
    apply_adain_patch()

    model = WeightNormConv1dProbe(
        channels=channels,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        use_weight_norm=use_weight_norm,
        load_decoder_weights=load_decoder_weights,
    ).eval()

    x = torch.zeros((1, channels, length), dtype=torch.float32)
    exported_program = torch.export.export(model, (x,))
    try:
        exported_program = exported_program.run_decompositions()
    except Exception:
        pass

    from tvm.relax.frontend.torch import from_exported_program

    mod = from_exported_program(
        exported_program,
        keep_params_as_input=False,
        unwrap_unit_return_tuple=False,
        no_bind_return_tuple=False,
    )

    new_funcs = {}
    for gv, func in mod.functions.items():
        if gv.name_hint == "main":
            new_gv = tvm.ir.GlobalVar(func_name)
            if hasattr(func, "attrs") and func.attrs is not None and "global_symbol" in func.attrs:
                new_attrs = dict(func.attrs)
                new_attrs["global_symbol"] = func_name
                func = func.with_attrs(new_attrs)
            new_funcs[new_gv] = func
        else:
            new_funcs[gv] = func
    mod = tvm.IRModule(new_funcs, attrs=mod.attrs)

    if dump_ir:
        with open(dump_ir, "w", encoding="utf-8") as f:
            f.write(mod.script())

    return mod
