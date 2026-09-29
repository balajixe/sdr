import numpy as np
import os
import cv2

# Directory containing your spectrogram blocks
BLOCK_DIR = "block_spectrograms"

# Signature output file
OUTPUT_SIG = "drone_signature.npy"

TARGET = (224, 224)

files = sorted([f for f in os.listdir(BLOCK_DIR) if f.endswith(".npy")])

if len(files) == 0:
    raise SystemExit("ERROR: No spectrogram blocks found in block_spectrograms/")

print("Total blocks found:", len(files))

acc = None
count = 0

for f in files:
    path = os.path.join(BLOCK_DIR, f)
    spec = np.load(path)

    spec = cv2.resize(spec, TARGET).astype(np.float32)
    spec = (spec - spec.mean()) / (spec.std() + 1e-9)

    if acc is None:
        acc = spec
    else:
        acc += spec
    count += 1

signature = acc / count
signature = (signature - signature.mean()) / (signature.std() + 1e-9)

np.save(OUTPUT_SIG, signature)
print("✔ Signature saved →", OUTPUT_SIG)
