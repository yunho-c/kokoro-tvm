
import json
from huggingface_hub import hf_hub_download

def main():
    repo_id = 'hexgrad/Kokoro-82M'
    print(f"Downloading config from {repo_id}...")
    config_path = hf_hub_download(repo_id=repo_id, filename='config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    print("Config loaded.")
    print(json.dumps(config, indent=2))
    
    # Extract relevant params for Decoder
    hidden_dim = config.get('hidden_dim')
    style_dim = config.get('style_dim')
    n_mels = config.get('n_mels')
    istftnet_params = config.get('istftnet')
    
    print("\nDecoder Parameters:")
    print(f"hidden_dim (dim_in): {hidden_dim}")
    print(f"style_dim: {style_dim}")
    print(f"n_mels (dim_out): {n_mels}")
    print(f"istftnet params: {istftnet_params}")

if __name__ == "__main__":
    main()
