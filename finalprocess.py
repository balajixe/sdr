
import uhd
import uhd.usrp
import uhd.types
import numpy as np
import time
from scipy import signal
import cv2
import pandas as pd
from datetime import datetime
import os

# ============================================================
# LOAD DRONE SIGNATURE
# ============================================================
raw_sig = np.load("drone_signature.npy").astype(np.float32)

SIG_SIZE = (224, 224)
signature = (signature - signature.mean()) / (signature.std() + 1e-9)

print("✅ Signature loaded:", signature.shape)

# ============================================================
# USRP SETTINGS (OPTIMAL & SAFE)
# ============================================================
FS = 5e6                 # Optimal for 2.4 GHz drones
GAIN = 20                # Stable, avoids saturation
FREQ = 2.470e9
SAMPLES = 25000

usrp = uhd.usrp.MultiUSRP()
usrp.set_rx_rate(FS)
usrp.set_rx_freq(FREQ)
usrp.set_rx_gain(GAIN)

stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
streamer = usrp.get_rx_stream(stream_args)

metadata = uhd.types.RXMetadata()
buff = np.zeros(SAMPLES, dtype=np.complex64)

cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
cmd.stream_now = True
streamer.issue_stream_cmd(cmd)

print("🔥 DRONE DETECTOR STARTED")

# ============================================================
# EXCEL OUTPUT
# ============================================================
OUT_XLS = "results.xlsx"

if not os.path.exists(OUT_XLS):
    pd.DataFrame(columns=["timestamp", "similarity", "label"]).to_excel(
        OUT_XLS, index=False, engine="openpyxl"
    )

records = []

# ============================================================
# DETECTOR PARAMETERS
# ============================================================
CALIBRATION_FRAMES = 50       # First ~50 seconds must be noise-only
noise_sims = []
THRESH = None

REQUIRED_HITS = 3             # Temporal confirmation
hit_count = 0

# ============================================================
# FUNCTIONS
# ============================================================
def signal_present(iq, energy_thresh=0.02):
    """Reject silence / very low energy."""
    return np.mean(np.abs(iq)) > energy_thresh


def match_drone(iq):
    """Spectrogram + cosine similarity."""
    _, _, Sxx = signal.spectrogram(
        iq,
        fs=FS,
        nperseg=512,
        noverlap=384,
        scaling="spectrum",
        mode="magnitude"
    )

    Sxx_db = 20 * np.log10(Sxx + 1e-12)
    img = cv2.resize(Sxx_db, SIG_SIZE).astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-9)

    # Cosine similarity (stable against gain changes)
    sim = np.sum(img * signature) / (
        np.linalg.norm(img) * np.linalg.norm(signature) + 1e-9
    )
    return sim


# ============================================================
# MAIN LOOP
# ============================================================
while True:
    num = streamer.recv(buff, metadata, timeout=1.0)

    if num == 0:
        continue

    if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
        print("⚠ USRP Error:", metadata.strerror())
        continue

    iq = buff[:num]

    # ---------------- NO SIGNAL CHECK ----------------
    if not signal_present(iq):
        label = "NO SIGNAL"
        sim = 0.0
        hit_count = 0
        print("⚪ NO SIGNAL")

    else:
        sim = match_drone(iq)

        # ----------- CALIBRATION PHASE ----------------
        if THRESH is None:
            noise_sims.append(sim)
            print(f"🔧 Calibrating noise {len(noise_sims)}/{CALIBRATION_FRAMES}")

            if len(noise_sims) >= CALIBRATION_FRAMES:
                noise_mean = np.mean(noise_sims)
                noise_std = np.std(noise_sims)
                THRESH = noise_mean + 5 * noise_std
                print(f"✅ Dynamic threshold set: {THRESH:.3f}")

            time.sleep(1.0)
            continue

        # --------------- DETECTION -------------------
        if sim > THRESH:
            hit_count += 1
        else:
            hit_count = 0

        if hit_count >= REQUIRED_HITS:
            label = "DRONE"
            print(f"✔ DRONE CONFIRMED | sim={sim:.3f}")
        else:
            label = "NON-DRONE"
            print(f"✖ NON-DRONE | sim={sim:.3f}")

    # ---------------- EXCEL LOG ---------------------
    records.append({
        "timestamp": datetime.utcnow().isoformat(),
        "similarity": float(sim),
        "label": label
    })

    if len(records) % 10 == 0:
        pd.DataFrame(records).to_excel(
            OUT_XLS, index=False, engine="openpyxl"
        )

    time.sleep(1.0)