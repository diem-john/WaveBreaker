import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal  # <-- New import


def estimate_impulse_response(excitation, response, filter_length=512):
    """
    Estimates the secondary path impulse response using FFT for speed.
    The paper uses an FIR filter length of 512 for secondary path modelling.
    """
    # Use scipy's FFT correlation instead of numpy's time-domain correlation
    cross_corr = signal.correlate(response, excitation, mode='full', method='fft')
    auto_corr_center = len(excitation) - 1

    # Extract the causal part of the impulse response
    impulse_response = cross_corr[auto_corr_center: auto_corr_center + filter_length]

    # Normalize (and use .copy() to prevent PyTorch memory warnings from negative strides)
    impulse_response = impulse_response / np.max(np.abs(impulse_response))

    # Return as a PyTorch tensor for differentiable filtering later
    return torch.tensor(impulse_response.copy(), dtype=torch.float32)


def apply_acoustic_path(anti_noise, impulse_response):
    """
    Simulates the sound traveling through the air to the dummy's ear.
    """
    # Reshape for 1D Convolution: (batch, channels, length)
    weight = impulse_response.view(1, 1, -1)

    # Convolve the AI's output with the physical room acoustics
    actual_sound_at_ear = F.conv1d(anti_noise, weight, padding=len(impulse_response) - 1)

    # Crop to match the original anti_noise length
    return actual_sound_at_ear[:, :, :anti_noise.shape[2]]