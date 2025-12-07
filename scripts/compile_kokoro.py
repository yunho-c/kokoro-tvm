import argparse
import os
import sys
import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from kokoro import KModel
from kokoro.model import KModelForONNX
import tvm_extensions


def compile_kokoro(model, output_dir):
    print("Tracing model with torch.export...")
    
    # Define symbolic dimensions
    # batch = torch.export.Dim("batch", min=1) # Treat batch as static 1 for now
    seq_len = torch.export.Dim("seq_len", min=2, max=512)
    
    # Create dummy inputs
    # input_ids: (batch, seq_len)
    dummy_input_ids = torch.randint(0, 100, (1, 50), dtype=torch.long)
    # ref_s: (batch, 256) - style vector
    dummy_style = torch.randn(1, 256, dtype=torch.float32)
    # speed: (batch,) - speed factor
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)

    # Specify dynamic shapes
    dynamic_shapes = {
        "input_ids": {1: seq_len},
        "ref_s": None,
        "speed": None,
    }

    # Monkey patch to avoid packing and handle LSTM hidden state
    from kokoro.modules import TextEncoder, ProsodyPredictor, DurationEncoder, AdaLayerNorm
    import torch.nn as nn
    import torch.nn.functional as F

    # Disable mkldnn to avoid potential dynamic shape issues
    torch.backends.mkldnn.enabled = False

    # Define custom op to prevent decomposition
    from torch import Tensor
    from typing import List, Tuple
    
    try:
        @torch.library.custom_op("kokoro::lstm", mutates_args=())
        def kokoro_lstm(input: Tensor, hx: List[Tensor], params: List[Tensor], has_biases: bool, num_layers: int, dropout: float, train: bool, bidirectional: bool, batch_first: bool) -> Tuple[Tensor, Tensor, Tensor]:
            return torch.ops.aten.lstm.input(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)

        @kokoro_lstm.register_fake
        def kokoro_lstm_fake(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
            if batch_first:
                batch_size = input.size(0)
                seq_len = input.size(1)
                input_size = input.size(2)
            else:
                seq_len = input.size(0)
                batch_size = input.size(1)
                input_size = input.size(2)
                
            num_directions = 2 if bidirectional else 1
            hidden_size = hx[0].size(2)
            
            if batch_first:
                output_shape = (batch_size, seq_len, num_directions * hidden_size)
            else:
                output_shape = (seq_len, batch_size, num_directions * hidden_size)
                
            h_n_shape = (num_layers * num_directions, batch_size, hidden_size)
            c_n_shape = (num_layers * num_directions, batch_size, hidden_size)
            
            output = input.new_empty(output_shape)
            h_n = input.new_empty(h_n_shape)
            c_n = input.new_empty(c_n_shape)
            
            return output, h_n, c_n
            
    except AttributeError:
        # Fallback for older PyTorch if custom_op is missing
        print("Warning: torch.library.custom_op not found. Using direct call (might fail).")
        def kokoro_lstm(*args):
            return torch.ops.aten.lstm.input(*args)

    # 1. Patch nn.LSTM.forward to handle hidden state init and use kokoro::lstm
    original_lstm_forward = nn.LSTM.forward
    
    def new_lstm_forward(self, input, hx=None):
        # Handle hidden state init
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            if self.batch_first:
                batch_size = input.size(0)
            else:
                batch_size = input.size(1)
                
            h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=input.device, dtype=input.dtype)
            c_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=input.device, dtype=input.dtype)
            hx = (h_0, c_0)

        # Use custom op to prevent decomposition
        params = self._flat_weights
        has_biases = self.bias
        num_layers = self.num_layers
        dropout = self.dropout
        train = self.training
        bidirectional = self.bidirectional
        batch_first = self.batch_first
        
        # hx must be a list for custom op if defined as List[Tensor]
        hx_list = list(hx)
        
        output, h_n, c_n = torch.ops.kokoro.lstm(
            input, 
            hx_list, 
            params, 
            has_biases, 
            num_layers, 
            dropout, 
            train, 
            bidirectional, 
            batch_first
        )
        
        return output, (h_n, c_n)
    
    nn.LSTM.forward = new_lstm_forward

    # 2. Patch AdaIN1d.forward to handle dynamic shape check
    from kokoro.istftnet import AdaIN1d
    original_adain_forward = AdaIN1d.forward
    
    def new_adain_forward(self, x, s):
        # Hint that sequence length is > 1 (we set min=2)
        if x.dim() == 3:
            torch._check(x.size(2) > 1)
        return original_adain_forward(self, x, s)
        
    AdaIN1d.forward = new_adain_forward

    # 3. Patch SineGen._f02sine to handle interpolation with dynamic shapes
    from kokoro.istftnet import SineGen
    
    original_f02sine = SineGen._f02sine
    
    def new_f02sine(self, f0_values):
        # Re-implement _f02sine to avoid scale_factor interpolation which confuses export
        rad_values = (f0_values / self.sampling_rate) % 1
        
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        
        if not self.flag_for_pulse:
            in_len = rad_values.shape[1]
            target_len = in_len // int(self.upsample_scale)
            torch._check(target_len > 0)
            
            rad_values = rad_values.transpose(1, 2)
            rad_values = F.interpolate(rad_values, size=target_len, mode="linear")
            rad_values = rad_values.transpose(1, 2)
            
            phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            
            phase = phase.transpose(1, 2) * self.upsample_scale
            phase = F.interpolate(phase, size=in_len, mode="linear")
            phase = phase.transpose(1, 2)
            
            sines = torch.sin(phase)
        else:
            return original_f02sine(self, f0_values)
            
        return sines
        
    SineGen._f02sine = new_f02sine

    # 4. Patch TextEncoder to remove packing
    def text_encoder_forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)
        
        # Skip packing
        x, _ = self.lstm(x)
        # Skip unpacking
        
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x

    TextEncoder.forward = text_encoder_forward

    # 5. Patch ProsodyPredictor to remove packing
    def prosody_predictor_forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        m = m.unsqueeze(1)
        
        # Skip packing
        x, _ = self.lstm(d) 
        # Skip unpacking
        
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, :x.shape[1], :] = x
        x = x_pad
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en

    ProsodyPredictor.forward = prosody_predictor_forward

    # 6. Patch DurationEncoder to remove packing
    def duration_encoder_forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                # block is LSTM
                x = x.transpose(-1, -2)
                # Skip packing
                x, _ = block(x)
                # Skip unpacking
                
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)

    DurationEncoder.forward = duration_encoder_forward

    # Export the program
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        exported_program = torch.export.export(
            model,
            (dummy_input_ids, dummy_style, dummy_speed),
            dynamic_shapes=dynamic_shapes
        )
    
    print("Importing to TVM Relax...")
    mod = from_exported_program(exported_program)
    
    # Basic optimization pipeline
    print("Applying optimizations...")
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        relax.transform.DeadCodeElimination(),
        relax.transform.CanonicalizeBindings(),
    ])
    
    mod = seq(mod)
    
    # Print the module to verify
    print("Compilation successful!")
    print(mod.script(show_meta=False)[:1000] + "\n...")
    
    # Save the module
    output_path = os.path.join(output_dir, "kokoro_relax.json")
    with open(output_path, "w") as f:
        f.write(tvm.ir.save_json(mod))
    print(f"Saved Relax module to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compile Kokoro Model to TVM Relax", add_help=True)
    parser.add_argument(
        # "--config_file", "-c", type=str, default="checkpoints/config.json", help="path to config file"
        "--config_file", "-c", type=str, required=False, help="path to config file"
    )
    parser.add_argument(
        # "--checkpoint_path", "-p", type=str, default="checkpoints/kokoro-v1_0.pth", help="path to checkpoint file"
        "--checkpoint_path", "-p", type=str, required=False, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="tvm_output", help="output directory"
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {checkpoint_path}...")
    # Initialize KModel
    # Note: KModel expects config to be a dict or path.
    # We assume the user provides valid paths.
    # if not os.path.exists(config_file):
    #     print(f"Warning: Config file {config_file} not found. KModel might try to download it.")
    
    # if not os.path.exists(checkpoint_path):
    #     print(f"Warning: Checkpoint file {checkpoint_path} not found. KModel might try to download it.")

    kmodel = KModel(config=config_file, model=checkpoint_path, disable_complex=True)
    model = KModelForONNX(kmodel).eval()

    compile_kokoro(model, output_dir)
