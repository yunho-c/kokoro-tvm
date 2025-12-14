# Test full Kokoro Pipeline
from kokoro_tvm.pipeline import KokoroPipeline
import torch

def test_pipeline():
    print("Initializing KokoroPipeline...")
    # Assuming tvm_output contains all .so files
    pipeline = KokoroPipeline("tvm_output", "llvm")
    
    # Create inputs
    # Text: "Hello World" phonemes (mock)
    # length 10
    input_ids = torch.randint(1, 100, (1, 10), dtype=torch.long)
    ref_s = torch.randn(1, 256)
    
    print("Running forward...")
    audio = pipeline.forward(input_ids, ref_s)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Audio stats: min={audio.min()}, max={audio.max()}, mean={audio.mean()}")
    
    assert audio.shape[0] == 5120
    print("Test Passed!")

if __name__ == "__main__":
    test_pipeline()
