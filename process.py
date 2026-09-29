import uhd
import numpy as np
import time
from dedector import match_drone

############################################
# USRP SETUP
############################################

# Create USRP device
usrp = uhd.usrp.MultiUSRP()

# Select RX channel 0 using SubdevSpec (required for UHD 4.9)
spec = uhd.usrp.SubdevSpec("A:A")
usrp.set_rx_subdev_spec(spec)

# Configure RF parameters
usrp.set_rx_rate(20e6)          # Very stable sample rate
usrp.set_rx_gain(35)
usrp.set_rx_freq(2.472e9)       # DJI / WiFi band

# Stream arguments (positional args only!)
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
streamer = usrp.get_rx_stream(stream_args)

# Start continuous streaming (YOUR UHD supports start_cont)
cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
cmd.stream_now = True
streamer.issue_stream_cmd(cmd)

# Buffers
N = 2048
buff = np.zeros(N, dtype=np.complex64)
metadata = uhd.types.RXMetadata()

# Drone threshold
THRESHOLD = 0.25

print("\n🚁 DJI Drone Detector Running…")
print(f"Listening at {usrp.get_rx_freq()/1e9:.3f} GHz\n")

############################################
# MAIN LOOP
############################################

while True:
    # UHD 4.9 receive function
    num_rx = streamer.recv(buff, metadata, timeout=1.0)

    if num_rx <= 0:
        print("⚠ No samples received (timeout)")
        continue

    # Run your drone signature matcher
    sim = match_drone(buff)

    # Detection output
    if sim > THRESHOLD:
        print(f"✔ DRONE DETECTED | similarity={sim:.3f} | freq={usrp.get_rx_freq()/1e9:.3f} GHz")
    else:
        print(f"✖ NON-DRONE | similarity={sim:.3f}")

    time.sleep(1.0)
