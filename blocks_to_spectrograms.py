import numpy as np
import os
from scipy import signal

INPUT_DIR = "/media/emmadepp369/F0F4BF58F4BF2030/data"
OUTPUT_DIR = "/media/emmadepp369/F0F4BF58F4BF2030/blockspec"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 10_000_000
BLOCK = 2_000_000     # 2M samples per block (~0.2 sec)
NPERSEG = 2048
NOVERLAP = 1024

files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")])

for fname in files:
    full = os.path.join(INPUT_DIR, fname)
    print("Processing:", fname)
    iq = np.load(full, mmap_mode="r")

    total = iq.shape[0]
    blocks = total // BLOCK

    for i in range(blocks):
        s = i * BLOCK
        e = s + BLOCK
        chunk = iq[s:e]

        f, t, Sxx = signal.spectrogram(
            chunk, fs=FS, nperseg=NPERSEG,
            noverlap=NOVERLAP, scaling="spectrum", mode="magnitude"
        )

        Sxx_db = 20 * np.log10(Sxx + 1e-12)

        out = f"{fname}_block_{i:03d}.npy"
        np.save(os.path.join(OUTPUT_DIR, out), Sxx_db)
        print("Saved:", out)

print("DONE.")
