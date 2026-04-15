import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.data_loader import load_primary_data, load_secondary_path
from src.model import WaveBreakerPro as custom_model
from src.engine import train_anc_model
from src.acoustics import estimate_impulse_response

import warnings
warnings.filterwarnings('ignore')

def main():
    print("Initializing WaveBreaker ANC Pipeline...")

    # 1. Define Data Paths
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    processed_dir = os.path.join(data_dir, 'processed')

    # Ensure the processed folder exists
    os.makedirs(processed_dir, exist_ok=True)

    # Paths for our cached tensor files
    cache_ir_m1 = os.path.join(processed_dir, 'ir_m1.pt')
    cache_ir_m2 = os.path.join(processed_dir, 'ir_m2.pt')
    cache_x = os.path.join(processed_dir, 'X_chunks.pt')
    cache_y = os.path.join(processed_dir, 'Y_chunks.pt')

    # 2. Check for Cached Data
    if all(os.path.exists(p) for p in [cache_ir_m1, cache_ir_m2, cache_x, cache_y]):
        print("Found processed data! Loading cached tensors...")
        sec_path_ir_m1 = torch.load(cache_ir_m1)
        sec_path_ir_m2 = torch.load(cache_ir_m2)
        X_chunks = torch.load(cache_x)
        Y_chunks = torch.load(cache_y)

    else:
        print("No cache found. Processing raw audio files...")

        # Original paths
        primary_noise_path = os.path.join(data_dir, 'Primary_100kph.wav')
        sec_path_left = os.path.join(data_dir, 'Secondary_left.wav')
        sec_path_right = os.path.join(data_dir, 'Secondary_right.wav')

        # Load Raw Data
        print(f"Loading primary dataset from: {primary_noise_path}")
        X_vib_raw, Y_noise_raw, sample_rate = load_primary_data(primary_noise_path)

        print("Loading acoustic environment (Secondary Paths)...")
        sec_l, sec_r = load_secondary_path(sec_path_left, sec_path_right)

        # Process Secondary Paths
        print("Calculating Secondary Path Impulse Responses...")
        sec_path_ir_m1 = estimate_impulse_response(sec_l[:, 0], sec_l[:, 1])
        sec_path_ir_m2 = estimate_impulse_response(sec_r[:, 0], sec_r[:, 2])

        # Prepare Tensors
        print("Chunking audio sequences...")
        X_tensor = torch.tensor(X_vib_raw, dtype=torch.float32).t()
        Y_tensor = torch.tensor(Y_noise_raw, dtype=torch.float32).t()

        sequence_length = 500 # 0.03125 seconds (31.25 milliseconds)
        num_samples = X_tensor.shape[1]
        num_chunks = num_samples // sequence_length

        X_tensor = X_tensor[:, :num_chunks * sequence_length]
        Y_tensor = Y_tensor[:, :num_chunks * sequence_length]

        X_chunks = X_tensor.view(16, num_chunks, sequence_length).transpose(0, 1)
        Y_chunks = Y_tensor.view(2, num_chunks, sequence_length).transpose(0, 1)

        # Save to Cache
        print("Saving processed tensors to cache...")
        torch.save(sec_path_ir_m1, cache_ir_m1)
        torch.save(sec_path_ir_m2, cache_ir_m2)
        torch.save(X_chunks, cache_x)
        torch.save(Y_chunks, cache_y)

    # 3. Create PyTorch DataLoader
    print("Preparing PyTorch DataLoader...")
    dataset = TensorDataset(X_chunks, Y_chunks)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 4. Initialize the AI Model
    print("Building Neural Network Architecture...")
    model = custom_model(input_channels=16, output_channels=2)
    print("Successful Model Build:")
    print(model)

    # 5. Execute Training Loop
    # train_anc_model will automatically show the tqdm progress bar now!
    train_anc_model(model, dataloader, sec_path_ir_m1, sec_path_ir_m2, epochs=3)

    print("WaveBreaker execution complete.")


if __name__ == "__main__":
    main()