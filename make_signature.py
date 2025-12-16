import numpy as np
import os

BLOCK_DIR = "/media/emmadepp369/F0F4BF58F4BF2030/blockspec"
OUTPUT_SIG = "drone_signature.npy"

files = sorted([f for f in os.listdir(BLOCK_DIR) if f.endswith(".npy")])

print("Total blocks:", len(files))

acc = None
count = 0

for f in files:
    x = np.load(os.path.join(BLOCK_DIR, f)).astype(np.float32)

    # Normalize each block
    x = (x - x.mean()) / (x.std() + 1e-9)

    if acc is None:
        acc = x
    else:
        acc += x

    count += 1

signature = acc / count
np.save(OUTPUT_SIG, signature)

print("✔ Signature saved:", OUTPUT_SIG)
