import numpy as np
from scipy import signal

# Load the correct signature
signature = np.load("drone_signature.npy").astype(np.float32)
sig_norm = signature

FS = 10_000_000
NPERSEG = 4096
NOVERLAP = 2048

def match_drone(iq):
    # Compute spectrogram using SAME params as block generator
    f, t, Sxx = signal.spectrogram(
        iq,
        fs=FS,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        scaling="spectrum",
        mode="magnitude"
    )

    Sxx_db = 20 * np.log10(Sxx + 1e-12)

    # Shape MUST match signature: (2048, 1952)
    if Sxx_db.shape != sig_norm.shape:
        print("⚠ Shape mismatch:", Sxx_db.shape, "vs", sig_norm.shape)
        return 0.0

    # Normalize
    img_norm = (Sxx_db - Sxx_db.mean()) / (Sxx_db.std() + 1e-9)

    # Compute similarity
    sim = float(np.mean(img_norm * sig_norm))
    return sim
