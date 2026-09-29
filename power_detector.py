import uhd
import numpy as np
import time

FS = 5_000_000           # 5 MHz sample rate
FREQ = 2.472e9           # DJI Phantom 4 Pro frequency you saw in GQRX
GAIN = 40                # Start with moderate gain
N = 200000               # ~40 ms of samples

usrp = uhd.usrp.MultiUSRP()
usrp.set_rx_rate(FS)
usrp.set_rx_freq(FREQ)
usrp.set_rx_gain(GAIN)

# UHD 4.9 correct StreamArgs
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
streamer = usrp.get_rx_stream(stream_args)

buff = np.zeros(N, dtype=np.complex64)
metadata = uhd.types.RXMetadata()

print("\n🔥 SIMPLE DRONE POWER DETECTOR RUNNING")
print(f"Listening at {FREQ/1e9} GHz, FS={FS/1e6} MHz\n")

THRESHOLD = -30   # TEMPORARY — we will tune it

def get_power(iq):
    return 10 * np.log10(np.mean(np.abs(iq)**2) + 1e-12)

# Start streaming
cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
cmd.stream_now = True
streamer.issue_stream_cmd(cmd)

while True:
    num_rx = streamer.recv(buff, metadata, timeout=1.0)

    if num_rx == 0:
        print("⚠ No samples received")
        continue

    p = get_power(buff[:num_rx])

    if p > THRESHOLD:
        print(f"✔ DRONE DETECTED | power={p:.1f} dB")
    else:
        print(f"✖ NON-DRONE | power={p:.1f} dB")

    time.sleep(0.5)
