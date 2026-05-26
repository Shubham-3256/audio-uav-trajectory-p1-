import os
import sys
import argparse
import shutil
import numpy as np
import soundfile as sf
from collections import defaultdict

def _get_reader():
    try:
        from rosbags.rosbag1 import Reader
        return Reader
    except ImportError:
        print("\nERROR: rosbags is not installed.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Synchronize raw onboard ROSbag audio with external unzipped ground-truth poses using timeline resampling.")
    parser.add_argument("--bag", type=str, required=True, help="Path to the raw ROS .bag file containing audio")
    parser.add_argument("--gt", type=str, required=True, help="Path to the unzipped ground_truth directory containing .npy files")
    parser.add_argument("--out", type=str, default="dataset", help="Output dataset directory")
    parser.add_argument("--seq", type=str, default="seq_001", help="Name of the sequence folder (e.g. seq_001)")
    parser.add_argument("--sr", type=int, default=41800, help="Target audio sample rate (default: 41800)")
    args = parser.parse_args()

    bag_path = args.bag
    gt_source_dir = args.gt
    out_dir = args.out
    seq_name = args.seq
    target_sr = args.sr

    # The raw audio in the MMAUD bag file is recorded at exactly 6000 Hz
    bag_sr = 6000

    print("\n========== ONBOARD BAG & EXTERNAL GT SYNCHRONIZER ==========")
    print(f"ROS Bag File    : {bag_path}")
    print(f"Ground Truth Dir: {gt_source_dir}")
    print(f"Target Sequence : {out_dir}/.../{seq_name}")
    print(f"True Bag Rate   : {bag_sr} Hz  -->  Target Rate: {target_sr} Hz")

    if not os.path.exists(bag_path):
        print(f"Error: ROS bag file '{bag_path}' not found.")
        return

    if not os.path.exists(gt_source_dir):
        print(f"Error: Ground-truth directory '{gt_source_dir}' not found.")
        return

    # 1. Read and Concatenate Audio Channels from ROSbag
    print("\n[1/3] Extracting raw audio streams from ROSbag...")
    Reader = _get_reader()
    
    AUDIO_TOPICS = [
        "/audio1/audio",
        "/audio2/audio",
        "/audio3/audio",
        "/audio4/audio",
    ]
    TOPIC_TO_CHANNEL = {t: i for i, t in enumerate(AUDIO_TOPICS)}
    
    audio_buffers = defaultdict(list)
    t_audio_start = None

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in AUDIO_TOPICS]
        if not connections:
            print("Error: No audio topics found in this ROSbag.")
            return

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            try:
                # Record the absolute timestamp of the first audio packet
                if t_audio_start is None and conn.topic == "/audio1/audio":
                    t_audio_start = timestamp * 1e-9 # convert nanoseconds to seconds

                # Skip the 4-byte ROS array length prefix in on-wire data
                pcm_bytes = rawdata[4:]
                samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                ch = TOPIC_TO_CHANNEL[conn.topic]
                audio_buffers[ch].append(samples)
            except Exception:
                continue

    if not audio_buffers or t_audio_start is None:
        print("Error: No audio data could be read from ROSbag.")
        return

    # Check channels lengths and concatenate
    channel_data = []
    min_len = float('inf')
    for ch in range(4):
        frames = audio_buffers.get(ch)
        if not frames:
            print(f"Error: Missing data for audio channel {ch+1}")
            return
        audio_stream = np.concatenate(frames)
        channel_data.append(audio_stream)
        min_len = min(min_len, len(audio_stream))
        print(f"  Channel {ch+1} extracted: {len(audio_stream):,} samples")

    # Stack channels
    audio_4ch = np.column_stack([c[:min_len] for c in channel_data]) # Shape: (samples, 4)
    print(f"Combined 4-channel audio shape: {audio_4ch.shape}")
    print(f"Audio Recording Start Unix Time: {t_audio_start:.6f} seconds")

    # 2. Index the Ground-Truth .npy Files
    print("\n[2/3] Loading and indexing timestamped poses...")
    pose_files = [f for f in os.listdir(gt_source_dir) if f.endswith(".npy")]
    if not pose_files:
        print("Error: No .npy pose files found in ground_truth directory!")
        return

    pose_records = []
    for f in pose_files:
        try:
            ts = float(os.path.splitext(f)[0])
            pose_records.append((ts, f))
        except ValueError:
            continue

    # Sort temporally
    pose_records.sort(key=lambda x: x[0])
    pose_timestamps = np.array([x[0] for x in pose_records])
    print(f"Indexed {len(pose_timestamps):,} Leica ground-truth pose files.")
    print(f"GT Tracking Start Unix Time    : {pose_timestamps[0]:.6f} seconds")
    print(f"GT Tracking End Unix Time      : {pose_timestamps[-1]:.6f} seconds")

    # 3. Synchronize and Segment
    print("\n[3/3] Synchronizing audio windows and Leica poses...")
    
    audio_npy_dir = os.path.join(out_dir, "audio_npy", seq_name)
    gt_dir = os.path.join(out_dir, "gt", seq_name)
    pseudo_dir = os.path.join(out_dir, "pseudo_label", seq_name)

    # Overwrite targeted sequence folders
    for d in [audio_npy_dir, gt_dir, pseudo_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    window_ms = 400.0
    overlap = 0.75
    
    # Timeline window sizes in terms of raw bag samples (6000 Hz)
    win_samples_bag = int(window_ms * 1e-3 * bag_sr) # 2,400 samples
    hop_samples_bag = int(win_samples_bag * (1 - overlap)) # 600 samples

    # Target window size expected by neural network (41800 Hz)
    win_samples_target = int(window_ms * 1e-3 * target_sr) # 16,720 samples

    n_samples = audio_4ch.shape[0]
    start = 0
    idx = 2
    saved_count = 0

    # Ensure reproducibility
    np.random.seed(42)

    # Max gap in seconds between audio window center and closest ground truth coordinates
    max_alignment_gap = 0.3 

    # Linear interpolation timeline parameters
    x_old = np.linspace(0, 1, win_samples_bag)
    x_new = np.linspace(0, 1, win_samples_target)

    while start + win_samples_bag <= n_samples:
        # Relative center time of the window in seconds from audio start
        rel_center_t = (start + win_samples_bag / 2) / bag_sr
        
        # Absolute Unix timestamp of this audio window
        abs_center_ts = t_audio_start + rel_center_t
        
        # Find closest pose index
        nearest_idx = np.abs(pose_timestamps - abs_center_ts).argmin()
        nearest_ts = pose_timestamps[nearest_idx]
        nearest_file = pose_records[nearest_idx][1]
        
        # Check if the closest pose is temporally aligned
        if np.abs(nearest_ts - abs_center_ts) <= max_alignment_gap:
            # Load the 3D position [x, y, z] from .npy
            gt_coord = np.load(os.path.join(gt_source_dir, nearest_file)).astype(np.float32)
            
            # Squeeze raw 4-channel audio segment (shape: 2400, 4)
            audio_segment_raw = audio_4ch[start : start + win_samples_bag, :]
            
            # Resample raw 6,000 Hz audio segment to expected target rate (41,800 Hz)
            audio_segment = np.zeros((win_samples_target, 4), dtype=np.float32)
            for ch in range(4):
                audio_segment[:, ch] = np.interp(x_new, x_old, audio_segment_raw[:, ch])
            
            # Simulate teacher network pseudo-labels (mean error ~0.5m)
            noise = np.random.normal(0, 0.25, size=3).astype(np.float32)
            pseudo_coord = gt_coord + noise

            # Save files stepped by 2 to support temporal sequence history loading
            np.save(os.path.join(audio_npy_dir, f"{idx}.npy"), audio_segment.astype(np.float32))
            np.save(os.path.join(gt_dir, f"{idx}.npy"), gt_coord)
            np.save(os.path.join(pseudo_dir, f"{idx}.npy"), pseudo_coord)
            
            idx += 2
            saved_count += 1
            
        start += hop_samples_bag

    print(f"\n========== SYNCHRONIZATION COMPLETE! ==========")
    print(f"Total synchronized segments written: {saved_count}")
    print(f"Output files stored in: {out_dir}")
    print("\nNext: Run 'python create_splits.py' to generate your train/val split lists!")

if __name__ == "__main__":
    main()
