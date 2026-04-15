import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveBreakerANC(nn.Module):
    def __init__(self, input_channels=16, output_channels=2):
        super(WaveBreakerANC, self).__init__()

        # 1. Feature Extraction: Fast, local pattern recognition across 16 sensors
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        # 2. Temporal Modeling: Predicts the incoming waveform
        # GRUs are faster and lighter than LSTMs, crucial for <1ms latency
        self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)

        # 3. Output Generation: The anti-noise for S1 and S2 speakers
        self.fc = nn.Linear(64, output_channels)

    def forward(self, x):
        # x expected shape: (batch_size, channels, sequence_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Reshape for RNN: (batch_size, sequence_length, channels)
        x = x.transpose(1, 2)

        # Pass through GRU
        x, _ = self.gru(x)

        # Generate the anti-noise signals
        anti_noise = self.fc(x)

        # Reshape back to (batch_size, channels, sequence_length)
        return anti_noise.transpose(1, 2)


class WaveBreakerPro(nn.Module):
    def __init__(self, input_channels=16, output_channels=2, d_model=64, n_heads=4):
        super(WaveBreakerPro, self).__init__()

        # ---------------------------------------------------------
        # 1. Temporal Convolutional Block (TCN)
        # ---------------------------------------------------------
        self.kernel_size = 5
        self.dilation1 = 2
        self.dilation2 = 4

        # Calculate Causal Padding: (kernel_size - 1) * dilation
        # This ensures the convolution only looks at current and past samples, never future.
        self.pad1 = (self.kernel_size - 1) * self.dilation1
        self.pad2 = (self.kernel_size - 1) * self.dilation2

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32,
                               kernel_size=self.kernel_size, dilation=self.dilation1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model,
                               kernel_size=self.kernel_size, dilation=self.dilation2)

        # ---------------------------------------------------------
        # 2. Multi-Head Self-Attention
        # ---------------------------------------------------------
        # d_model must be divisible by n_heads (64 / 4 = 16 dimensions per head)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

        # ---------------------------------------------------------
        # 3. Dense / Fully Connected Layers
        # ---------------------------------------------------------
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(0.2)  # Helps prevent overfitting on the noise
        self.fc2 = nn.Linear(128, output_channels)

    def forward(self, x):
        # Initial shape: (batch_size, channels, sequence_length)

        # --- Phase 1: Temporal Convolutions ---
        # F.pad format for 1D: (pad_left, pad_right). We only pad the past!
        x = F.pad(x, (self.pad1, 0))
        x = torch.relu(self.conv1(x))

        x = F.pad(x, (self.pad2, 0))
        x = torch.relu(self.conv2(x))

        # --- Phase 2: Attention Mechanism ---
        # Attention requires shape: (batch_size, sequence_length, features/channels)
        x = x.transpose(1, 2)

        # Self-attention compares the sequence to itself (query=x, key=x, value=x)
        attn_out, _ = self.attention(x, x, x)

        # Add a residual connection and normalize to stabilize deep gradients
        x = self.layer_norm(x + attn_out)

        # --- Phase 3: Dense Mapping ---
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        anti_noise = self.fc2(x)  # Final shape: (batch_size, sequence_length, output_channels)

        # Return to expected PyTorch audio format: (batch_size, channels, sequence_length)
        return anti_noise.transpose(1, 2)