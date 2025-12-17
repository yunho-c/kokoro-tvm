# SPDX-FileCopyrightText: 2025-present Yunho Cho <opensource@yunhocho.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Decoder probe module for locating NaN/INF sources.

This exports a decoder variant that returns lightweight per-stage statistics
instead of the full waveform only. The intent is to identify the earliest stage
that becomes non-finite in TVM.
"""

from __future__ import annotations

import json
from typing import Optional

import torch
import torch.nn.functional as F
import tvm
from huggingface_hub import hf_hub_download
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


class DecoderProbe(torch.nn.Module):
    def __init__(self, decoder: Decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self, asr: torch.Tensor, f0_curve: torch.Tensor, n_curve: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        dec = self.decoder
        f0 = dec.F0_conv(f0_curve.unsqueeze(1))
        n = dec.N_conv(n_curve.unsqueeze(1))
        x_in = torch.cat([asr, f0, n], axis=1)

        x = dec.encode(x_in, s)
        asr_res = dec.asr_res(asr)
        res = True
        for block in dec.decode:
            if res:
                x = torch.cat([x, asr_res, f0, n], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False

        gen = dec.generator
        f0_up = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

        xg = x
        x_source0_conv = None
        x_source0 = None
        nr0_n1 = None
        nr0_snake1 = None
        nr0_conv1_w = None
        nr0_conv1_b = None
        nr0_conv1 = None
        x_up0 = None
        xg0_pre = None
        xg0_post = None
        xg1_pre = None
        xg1_post = None
        for i in range(gen.num_upsamples):
            xg = F.leaky_relu(xg, negative_slope=0.1)
            x_source = gen.noise_convs[i](har)
            if i == 0:
                x_source0_conv = x_source
                block0 = gen.noise_res[0]
                n1 = block0.adain1[0](x_source, s)
                a1 = block0.alpha1[0]
                snake1 = n1 + (1.0 / a1) * (torch.sin(a1 * n1) ** 2)
                conv1_w = block0.convs1[0].weight
                conv1_b = block0.convs1[0].bias
                conv1 = block0.convs1[0](snake1)
                nr0_n1 = n1
                nr0_snake1 = snake1
                nr0_conv1_w = conv1_w
                nr0_conv1_b = conv1_b
                nr0_conv1 = conv1

            x_source = gen.noise_res[i](x_source, s)
            xg = gen.ups[i](xg)
            if i == 0:
                x_source0 = x_source
                x_up0 = xg
            if i == gen.num_upsamples - 1:
                xg = gen.reflection_pad(xg)
            xg = xg + x_source
            if i == 0:
                xg0_pre = xg
            elif i == 1:
                xg1_pre = xg

            xs = None
            for j in range(gen.num_kernels):
                idx = i * gen.num_kernels + j
                xs = gen.resblocks[idx](xg, s) if xs is None else (xs + gen.resblocks[idx](xg, s))
            xg = xs / gen.num_kernels
            if i == 0:
                xg0_post = xg
            elif i == 1:
                xg1_post = xg

        xg = F.leaky_relu(xg, negative_slope=0.1)
        x_pre = xg
        x_post = gen.conv_post(x_pre)
        spec = torch.exp(x_post[:, : gen.post_n_fft // 2 + 1, :])
        phase = torch.sin(x_post[:, gen.post_n_fft // 2 + 1 :, :])
        audio = gen.stft.inverse(spec, phase)

        return torch.stack(
            [
                _probe_slice(f0_curve),
                _probe_slice(n_curve),
                _probe_slice(x_in),
                _probe_slice(x),
                _probe_slice(har),
                _probe_slice(x_source0_conv if x_source0_conv is not None else x),
                _probe_slice(nr0_n1 if nr0_n1 is not None else x),
                _probe_slice(nr0_snake1 if nr0_snake1 is not None else x),
                _probe_slice(nr0_conv1_w if nr0_conv1_w is not None else x),
                _probe_slice(nr0_conv1_b if nr0_conv1_b is not None else x),
                _probe_slice(nr0_conv1 if nr0_conv1 is not None else x),
                _probe_slice(x_source0 if x_source0 is not None else x),
                _probe_slice(x_up0 if x_up0 is not None else x),
                _probe_slice(xg0_pre if xg0_pre is not None else x_pre),
                _probe_slice(xg0_post if xg0_post is not None else x_pre),
                _probe_slice(xg1_pre if xg1_pre is not None else x_pre),
                _probe_slice(xg1_post if xg1_post is not None else x_pre),
                _probe_slice(x_post),
                _probe_slice(spec),
                _probe_slice(audio),
            ],
            dim=0,
        )


class DecoderProbeFull(torch.nn.Module):
    def __init__(self, decoder: Decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, asr: torch.Tensor, f0_curve: torch.Tensor, n_curve: torch.Tensor, s: torch.Tensor):
        dec = self.decoder
        f0 = dec.F0_conv(f0_curve.unsqueeze(1))
        n = dec.N_conv(n_curve.unsqueeze(1))
        x_in = torch.cat([asr, f0, n], axis=1)

        x = dec.encode(x_in, s)
        asr_res = dec.asr_res(asr)
        res = True
        for block in dec.decode:
            if res:
                x = torch.cat([x, asr_res, f0, n], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False

        gen = dec.generator
        f0_up = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

        x_source0_conv = gen.noise_convs[0](har)
        block0 = gen.noise_res[0]
        n1 = block0.adain1[0](x_source0_conv, s)
        a1 = block0.alpha1[0]
        snake1 = n1 + (1.0 / a1) * (torch.sin(a1 * n1) ** 2)
        conv1_w = block0.convs1[0].weight
        conv1_b = block0.convs1[0].bias
        conv1 = block0.convs1[0](snake1)

        return (x_source0_conv, n1, snake1, conv1_w, conv1_b, conv1)


def create_decoder_probe_module(
    seq_len: int = 150,
    load_weights: bool = True,
    func_name: str = "decoder_probe_forward",
    dump_ir: Optional[str] = None,
) -> tvm.IRModule:
    apply_sinegen_patch()
    apply_adain_patch()

    repo_id = "hexgrad/Kokoro-82M"
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    hidden_dim = config["hidden_dim"]
    style_dim = config["style_dim"]
    n_mels = config["n_mels"]
    istftnet_params = config["istftnet"]

    decoder = Decoder(dim_in=hidden_dim, style_dim=style_dim, dim_out=n_mels, disable_complex=True, **istftnet_params)
    if load_weights:
        model_path = hf_hub_download(repo_id=repo_id, filename="kokoro-v1_0.pth")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        dec_state = state.get("decoder", {})
        first_key = next(iter(dec_state.keys()), "")
        if first_key.startswith("module."):
            dec_state = {k[7:]: v for k, v in dec_state.items()}
        decoder.load_state_dict(dec_state, strict=False)

    decoder.eval()
    model = DecoderProbe(decoder).eval()

    batch_size = 1
    asr = torch.zeros(batch_size, hidden_dim, seq_len)
    f0 = torch.zeros(batch_size, seq_len * 2)
    n = torch.zeros(batch_size, seq_len * 2)
    s = torch.zeros(batch_size, style_dim)

    exported_program = torch.export.export(model, (asr, f0, n, s))
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


def create_decoder_probe_full_module(
    seq_len: int = 150,
    load_weights: bool = True,
    func_name: str = "decoder_probe_full_forward",
    dump_ir: Optional[str] = None,
) -> tvm.IRModule:
    apply_sinegen_patch()
    apply_adain_patch()

    repo_id = "hexgrad/Kokoro-82M"
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    hidden_dim = config["hidden_dim"]
    style_dim = config["style_dim"]
    n_mels = config["n_mels"]
    istftnet_params = config["istftnet"]

    decoder = Decoder(dim_in=hidden_dim, style_dim=style_dim, dim_out=n_mels, disable_complex=True, **istftnet_params)
    if load_weights:
        model_path = hf_hub_download(repo_id=repo_id, filename="kokoro-v1_0.pth")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        dec_state = state.get("decoder", {})
        first_key = next(iter(dec_state.keys()), "")
        if first_key.startswith("module."):
            dec_state = {k[7:]: v for k, v in dec_state.items()}
        decoder.load_state_dict(dec_state, strict=False)

    decoder.eval()
    model = DecoderProbeFull(decoder).eval()

    batch_size = 1
    asr = torch.zeros(batch_size, hidden_dim, seq_len)
    f0 = torch.zeros(batch_size, seq_len * 2)
    n = torch.zeros(batch_size, seq_len * 2)
    s = torch.zeros(batch_size, style_dim)

    exported_program = torch.export.export(model, (asr, f0, n, s))
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
