import uhd
import numpy as np
import time
from scipy import signal
import cv2   # for resizing

# ------------------------------------------------------------
# LOAD SIGNATURE (ANY SHAPE)
# ------------------------------------------------------------
raw_sig = np.load("drone_signature.npy").astype(np.float32)

# Resize signature to fixed size for stable similarity
SIG_SIZE = (224, 224)
signature = cv2.resize(raw_sig, SIG_SIZE).astype(np.float32)

# Normalize
signature = (signature - signature.mean()) / (signature.std() + 1e-9)

print("Loaded signature, resized to:", signature.shape)

# ------------------------------------------------------------
# USRP SETTINGS
# ------------------------------------------------------------
FS =10e6             # Works even on USB2
GAIN = 40
FREQ = 2.466e9
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
# MATCH DRONE FUNCTION (UNIVERSAL)
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

    # Resize live spectrogram to match signature
    img = cv2.resize(Sxx_db, SIG_SIZE).astype(np.float32)

    # Normalize
    img = (img - img.mean()) / (img.std() + 1e-9)

    # Similarity score
    sim = np.mean(img * signature)

    return sim

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
THRESH = 0.15 # tune after testing

while True:
    num = streamer.recv(buff, metadata, timeout=1.0)

    if num == 0:
        print("⚠ No samples")
        continue

    iq = buff[:num]

    sim = match_drone(iq)

    if sim > THRESH:
        print(f"✔ DRONE DETECTED | similarity={sim:.3f}")
    else:
        print(f"✖ NON-DRONE | similarity={sim:.3f}")

    time.sleep(1.0)
