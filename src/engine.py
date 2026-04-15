import time
import torch
import torch.nn as nn
from tqdm import tqdm
from src.acoustics import apply_acoustic_path


def train_anc_model(model, dataloader, sec_path_ir_m1, sec_path_ir_m2, epochs=3):
    """
    Trains the WaveBreaker model using a custom physical loss function.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # We want to minimize the final sound power at the ear (Mean Squared Error)
    criterion = nn.MSELoss()

    model.train()
    print(f"--- Starting WaveBreaker Training Loop ({epochs} Epochs) ---")

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Start the stopwatch for the epoch
        start_time = time.time()

        # Wrap the dataloader in tqdm
        # leave=False: Makes the progress bar completely vanish after the epoch finishes!
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1:02d}/{epochs}", leave=False)

        for batch_idx, (X_vib, Y_primary_noise) in enumerate(progress_bar):
            optimizer.zero_grad()

            # 1. AI predicts the Anti-Noise y(n)
            y_anti_noise = model(X_vib)

            # 2. Simulate speakers S1 and S2 pushing sound through the air
            # anti_noise_m1 = S(z) * y(n)
            y_prime_m1 = apply_acoustic_path(y_anti_noise[:, 0:1, :], sec_path_ir_m1)
            y_prime_m2 = apply_acoustic_path(y_anti_noise[:, 1:2, :], sec_path_ir_m2)

            y_prime_combined = torch.cat([y_prime_m1, y_prime_m2], dim=1)

            # 3. Superposition: The physical combination of noise and anti-noise
            # e(n) = d(n) + y'(n)
            error_signal = Y_primary_noise + y_prime_combined

            # 4. Calculate loss and backpropagate
            loss = criterion(error_signal, torch.zeros_like(error_signal))
            loss.backward()
            optimizer.step()

            # Update tracking variables
            current_loss = loss.item()
            epoch_loss += current_loss

            # 5. Dynamically update the progress bar with the current batch loss
            progress_bar.set_postfix(Loss=f"{current_loss:.6f}")

        # Stop the stopwatch
        epoch_duration = time.time() - start_time

        # Print the final average loss and exact time for the epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"--> Epoch {epoch + 1:02d} | Avg Residual Loss: {avg_loss:.6f} | Time: {epoch_duration:.2f}s")