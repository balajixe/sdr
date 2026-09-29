import numpy as np
import os

INPUT_DIR = "/media/emmadepp369/F0F4BF58F4BF2030/block_spectrograms"
OUTPUT_SIG = "drone_signature.npy"

files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".npy"))

print("Total spectrogram blocks:", len(files))

acc = None
count = 0

for fname in files:
    spec = np.load(os.path.join(INPUT_DIR, fname))

    # Resize to 224×224 for consistency
    from cv2 import resize
    spec = resize(spec, (224,224))

    if acc is None:
        acc = np.zeros_like(spec)

    acc += spec
    count += 1

signature = acc / count
np.save(OUTPUT_SIG, signature)

print("DONE → drone_signature.npy saved.")
