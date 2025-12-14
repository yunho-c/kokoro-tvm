# LSTM to Transformer Distillation for Kokoro TVM

## Problem Statement

The ProsodyPredictor module in Kokoro uses a 3-layer bidirectional LSTM (`d_hid=512`) for duration and F0/N prediction. When compiling to TVM via `torch.export`, this creates severe issues:

1. **Compile-time explosion**: 10+ minutes for `seq_len=128`, scaling poorly with sequence length
2. **Massive IR size**: The LSTM recurrence is statically unrolled, creating millions of IR nodes
3. **Sequential `te.scan` execution**: TVM's TOPI LSTM uses `te.scan()` which is inherently serial

## Proposed Solution: Knowledge Distillation to Transformer

Replace the LSTM with a Transformer encoder trained via knowledge distillation. The student Transformer learns to replicate the LSTM's output distribution.

### Why This Works for Kokoro

| Property | LSTM (Current) | Transformer (Proposed) |
|----------|----------------|------------------------|
| Context | Bidirectional (sees full sequence) | Bidirectional (sees full sequence) |
| Parallelism | Sequential across time | Fully parallel |
| TVM Compilation | `te.scan` → massive unrolled IR | MatMul + Softmax → clean IR |
| Causality | Not required (inference only) | Not required (no masking needed) |

The bidirectional LSTM already processes the full context before outputting prosody features—this is functionally equivalent to a bidirectional Transformer encoder.

## Architecture Design

### Current LSTM Path
```
d_en [B, 512, T] → TextEncoder → d [B, T, 640]
                                    ↓
                        LSTM (3-layer bidir, hidden=512)
                                    ↓
                              x [B, T, 1024]
                                    ↓
                            duration_proj → [B, T, 1]
```

### Proposed Transformer Path
```
d_en [B, 512, T] → TextEncoder → d [B, T, 640]
                                    ↓
                     TransformerEncoder (3-layer, d=640, heads=8)
                                    ↓
                              x [B, T, 640]
                                    ↓
                        duration_proj (adjusted) → [B, T, 1]
```

### Student Model Definition

```python
import torch
import torch.nn as nn

class ProsodyTransformer(nn.Module):
    """Transformer replacement for ProsodyPredictor LSTM."""
    
    def __init__(
        self, 
        d_model: int = 640,      # Match TextEncoder output (d_hid + style_dim)
        nhead: int = 8,
        num_layers: int = 3,      # Match LSTM depth
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Match LSTM output dimension: bidir LSTM outputs 2*hidden
        # We project to match if needed
        self.output_proj = nn.Linear(d_model, 1024)  # Optional: match LSTM dim
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] - TextEncoder output (d tensor)
            mask: [B, T] - Padding mask (True = padded)
        Returns:
            out: [B, T, 1024] - Prosody features matching LSTM output shape
        """
        # Convert bool mask to attention mask (0 = attend, -inf = ignore)
        if mask is not None:
            attn_mask = mask.float().masked_fill(mask, float('-inf'))
        else:
            attn_mask = None
            
        out = self.transformer(x, src_key_padding_mask=attn_mask)
        return self.output_proj(out)
```

## Distillation Training Procedure

### Step 1: Generate Teacher Outputs

```python
def generate_distillation_data(teacher_model, dataloader, device):
    """Run teacher LSTM and save outputs for distillation."""
    teacher_model.eval()
    dataset = []
    
    with torch.no_grad():
        for batch in dataloader:
            d_en, style, lengths, mask = batch
            d = teacher_model.text_encoder(d_en, style, lengths, mask)
            teacher_model.lstm.flatten_parameters()
            lstm_out, _ = teacher_model.lstm(d)
            
            dataset.append({
                'input': d.cpu(),
                'target': lstm_out.cpu(),
                'mask': mask.cpu(),
            })
    
    return dataset
```

### Step 2: Train Student

```python
def train_student(student, distillation_data, epochs=100, lr=1e-4):
    """Train Transformer to match LSTM outputs."""
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')
    
    for epoch in range(epochs):
        for batch in distillation_data:
            x, target, mask = batch['input'], batch['target'], batch['mask']
            
            pred = student(x, mask)
            
            # Masked MSE loss (ignore padded positions)
            loss = criterion(pred, target)
            loss = (loss * ~mask.unsqueeze(-1)).sum() / (~mask).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Step 3: Fine-tune End-to-End (Optional)

After distillation, optionally fine-tune the full model (Transformer + duration_proj + F0/N predictors) on the original TTS loss to recover any quality degradation.

## Expected Benefits

| Metric | LSTM (Current) | Transformer (Expected) |
|--------|----------------|------------------------|
| TVM IR Generation | 10+ min (seq=128) | < 30 sec |
| IR Size | Millions of nodes | ~10K nodes |
| Compile Time | Hours | Minutes |
| GPU Inference | Memory-bound | Compute-efficient |
| CPU Inference | Sequential | Vectorized |

## Risks and Mitigations

### Risk 1: Prosody Quality Degradation
**Mitigation**: 
- Use intermediate layer matching (not just final output)
- Add perceptual loss on synthesized audio during fine-tuning
- Ensure sufficient training data coverage

### Risk 2: Attention Overhead for Short Sequences
**Mitigation**:
- For very short sequences (<32), LSTM might still be faster
- Consider hybrid: use LSTM for short, Transformer for long
- Flash Attention reduces overhead significantly

### Risk 3: Positional Encoding
**Mitigation**:
- LSTM has implicit positional information through recurrence
- Add sinusoidal or learned positional encodings to Transformer
- RoPE (Rotary Position Embedding) works well for variable-length sequences

## Implementation Roadmap

1. **Phase 1: Data Generation** (1 day)
   - Extract LSTM inputs/outputs from existing training data
   - Save as distillation dataset

2. **Phase 2: Student Training** (2-3 days)
   - Train Transformer student on distillation data
   - Hyperparameter search for model size

3. **Phase 3: Integration** (1 day)
   - Replace LSTM in ProsodyPredictor with trained Transformer
   - Verify TVM compilation succeeds with reasonable IR size

4. **Phase 4: Validation** (2 days)
   - Compare audio quality: LSTM vs Transformer
   - Benchmark inference latency
   - Measure MOS (Mean Opinion Score) if possible

## Alternative Approaches

### Option A: State Space Models (Mamba)
- Linear-time complexity, handles long sequences
- TVM support is limited as of 2024
- Requires custom operator implementation

### Option B: Linearized Attention
- Replace softmax attention with linear kernels
- Maintains O(N) complexity
- Less expressive than full attention

### Option C: Keep LSTM, Optimize Compilation
- Use symbolic shapes with dynamic loops (requires TVM improvements)
- Wait for better ONNX Loop support in Relax
- Pre-compile for fixed sequence lengths (trade flexibility for speed)

## Conclusion

Distilling the Kokoro ProsodyPredictor LSTM into a Transformer is a **high-value, moderate-effort** solution that addresses the TVM compilation bottleneck while maintaining model quality. The bidirectional nature of the existing LSTM makes this a natural fit for a Transformer encoder replacement.

## References

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton et al.
- [FastSpeech 2](https://arxiv.org/abs/2006.04558) - Transformer-based TTS without RNNs
- [TVM Relax LSTM Support](https://github.com/apache/tvm/pull/18346) - PR #18346
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - Original model
