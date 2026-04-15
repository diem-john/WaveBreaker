import soundfile as sf
import numpy as np


def load_primary_data(filepath, target_sr=4000):
    """
    Loads the 18-channel primary noise data.
    The paper originally sampled at 16 kHz but simulated at 4 kHz[cite: 38, 60].
    """
    data, sr = sf.read(filepath)

    # In 'Primary_xxkph' files:
    # Channels 0-15 (1-16) are vibration reference signals.
    # Channels 16-17 (17-18) are the sound pressure at mics M1 and M2.
    X_vib = data[:, :16]
    Y_noise = data[:, 16:18]

    return X_vib, Y_noise, sr


def load_secondary_path(left_path, right_path):
    """
    Loads the 3-channel secondary path modeling data.
    Channel 0: White noise excitation [cite: 55]
    Channel 1: Response at M1 [cite: 55]
    Channel 2: Response at M2 [cite: 55]
    """
    data_l, sr_l = sf.read(left_path)
    data_r, sr_r = sf.read(right_path)

    return data_l, data_r