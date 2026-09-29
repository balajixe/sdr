import uhd
import numpy as np
import time
import os

# ===================================================
# USER SETTINGS
# ===================================================
save_path = "/media/emmadepp369/F0F4BF58F4BF2030/data"
os.makedirs(save_path, exist_ok=True)

center_freq = 2.43e9
sample_rate = 10e6          # 10 MS/s
gain = 40
chunk_duration = 10         # seconds per file
total_duration = 600        # 10 minutes (600 sec)
# ===================================================

samples_per_chunk = int(sample_rate * chunk_duration)
num_chunks = total_duration // chunk_duration

buffer_size = 4096
buffer = np.zeros(buffer_size, dtype=np.complex64)

usrp = uhd.usrp.MultiUSRP()
usrp.set_rx_rate(sample_rate)
usrp.set_rx_freq(uhd.types.TuneRequest(center_freq))
usrp.set_rx_gain(gain)

stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
rx_stream = usrp.get_rx_stream(stream_args)
md = uhd.types.RXMetadata()

print(f"Recording {total_duration/60} minutes in {num_chunks} chunks...")

for chunk_id in range(num_chunks):

    print(f"\nStarting chunk {chunk_id+1}/{num_chunks}...")

    # Output file for this chunk
    filename = os.path.join(save_path, f"drone_chunk_{chunk_id:03d}.npy")
    
    # Preallocate ONLY the chunk (fits in RAM)
    chunk_data = np.zeros(samples_per_chunk, dtype=np.complex64)

    # Start streaming for this chunk
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    cmd.num_samps = samples_per_chunk
    cmd.stream_now = True
    rx_stream.issue_stream_cmd(cmd)

    collected = 0

    while collected < samples_per_chunk:
        samps_to_recv = min(buffer_size, samples_per_chunk - collected)
        num_rx = rx_stream.recv(buffer[:samps_to_recv], md, timeout=2.0)

        if num_rx > 0:
            chunk_data[collected:collected+num_rx] = buffer[:num_rx]
            collected += num_rx

    print(f" → Saved {collected} samples to {filename}")
    np.save(filename, chunk_data)

print("\n✔ Finished recording all chunks.")
print("Files saved in:", save_path)
