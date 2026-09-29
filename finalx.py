import uhd
import numpy as np
import time
from scipy import signal
import cv2

# ------------------------------------------------------------
# LOAD SIGNATURE
# ------------------------------------------------------------
raw_sig = np.load("drone_signature.npy").astype(np.float32)

SIG_SIZE = (224, 224)
signature = cv2.resize(raw_sig, SIG_SIZE).astype(np.float32)
signature = (signature - signature.mean()) / (signature.std() + 1e-9)

print("Loaded signature, resized to:", signature.shape)

# ------------------------------------------------------------
# USRP SETTINGS
# ------------------------------------------------------------
FS = 25e6
GAIN = 40
FREQ = 2.410e9
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
# MATCH DRONE FUNCTION
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
# YOUR COUNT-BASED LOGIC
# ------------------------------------------------------------
THRESH = 0.18
WINDOW_SIZE = 10
DETECT_COUNT = 3     # >3 ones → drone detected
binary_window = []

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
while True:

    num = streamer.recv(buff, metadata, timeout=1.0)

    if num == 0:
        print("⚠ No samples")
        continue

    iq = buff[:num]

    # USRP similarity detection
    sim = match_drone(iq)

    # Convert to binary
    current_binary = 1 if sim > THRESH else 0

    # Fill first 10 seconds
    if len(binary_window) < WINDOW_SIZE:
        binary_window.append(current_binary)
        print(f"Collecting ({len(binary_window)}/10) | USRP={current_binary}")
        time.sleep(1.0)
        continue

    # Count number of 1s
    one_count = binary_window.count(1)

    # Final decision (YOUR RULE)
    if one_count > DETECT_COUNT:
        final_output = 1
        print(f"🚁 DRONE DETECTED | FINAL=1 | COUNT={one_count}/10")
    else:
        final_output = 0
        print(f"❌ NO DRONE | FINAL=0 | COUNT={one_count}/10")

    # Slide window
    binary_window.pop(0)
    binary_window.append(current_binary)

    time.sleep(1.0)

