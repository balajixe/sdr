import uhd
import numpy as np
import time
from scipy import signal
import cv2
import pandas as pd
from datetime import datetime
import os

# ------------------------------------------------------------
# LOAD SIGNATURE
# ------------------------------------------------------------
raw_sig = np.load("drone_signature.npy").astype(np.float32)

SIG_SIZE = (224, 224)
signature = cv2.resize(raw_sig, SIG_SIZE).astype(np.float32)
signature = (signature - signature.mean()) / (signature.std() + 1e-9)

print("Loaded signature:", signature.shape)

# ------------------------------------------------------------
# USRP SETTINGS
# ------------------------------------------------------------
FS = 7e6
GAIN = 35
FREQ = 2.472e9
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

print("\n🔥 UNIVERSAL DRONE DETECTOR STARTED\n")

# ------------------------------------------------------------
# EXCEL FILE SETUP
# ------------------------------------------------------------
excel_file = "drone_detection_log.xlsx"

if not os.path.exists(excel_file):
    df_init = pd.DataFrame(columns=[
        "Timestamp",
        "Frequency_Hz",
        "Similarity",
        "Result"
    ])
    df_init.to_excel(excel_file, index=False)

# ------------------------------------------------------------
# MATCH FUNCTION
# ------------------------------------------------------------
def match_drone(iq):
    f, t, Sxx = signal.spectrogram(
        iq,
        fs=FS,
        nperseg=1024,
        noverlap=512,
        scaling="spectrum",
        mode="magnitude"
    )

    Sxx_db = 20 * np.log10(Sxx + 1e-12)
    img = cv2.resize(Sxx_db, SIG_SIZE).astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-9)

    sim = np.mean(img * signature)
    return sim

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
THRESH = 0.21

while True:
    num = streamer.recv(buff, metadata, timeout=1.0)

    if num == 0:
        print("⚠ No samples")
        continue

    iq = buff[:num]
    sim = match_drone(iq)

    result = "DRONE" if sim > THRESH else "NON-DRONE"

    print(f"{result} | similarity={sim:.3f}")

    # ---------------- EXCEL LOGGING ----------------
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Frequency_Hz": FREQ,
        "Similarity": round(sim, 4),
        "Result": result
    }

    df = pd.read_excel(excel_file)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(excel_file, index=False)

    time.sleep(1.0)
