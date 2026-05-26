import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import soundfile as sf

def _get_reader():
    try:
        from rosbags.rosbag1 import Reader
        return Reader
    except ImportError:
        print("\nERROR: rosbags is not installed.")
        print("Install it with:  pip install rosbags")
        sys.exit(1)

def deserialize_message(rawdata: bytes, msgtype: str):
    try:
        from rosbags.serde import deserialize_cdr, ros1_to_cdr
        return deserialize_cdr(ros1_to_cdr(rawdata, msgtype), msgtype)
    except Exception as exc:
        raise RuntimeError(f"Failed to deserialize message: {exc}")

AUDIO_TOPICS = [
    "/audio1/audio",
    "/audio2/audio",
    "/audio3/audio",
    "/audio4/audio",
]
TOPIC_TO_CHANNEL = {t: i for i, t in enumerate(AUDIO_TOPICS)}

def main():
    parser = argparse.ArgumentParser(description="Extract and synchronize Pham4 ROSbag to project dataset format.")
    parser.add_argument("--bag", type=str, required=True, help="Path to the raw Pham4 ROS .bag file")
    parser.add_argument("--out", type=str, default="dataset", help="Output dataset directory")
    parser.add_argument("--seq", type=str, default="seq_001", help="Name of the sequence folder (e.g. seq_001)")
    parser.add_argument("--sr", type=int, default=41800, help="Audio sample rate (default: 41800)")
    args = parser.parse_args()

    bag_path = args.bag
    if not os.path.exists(bag_path):
        print(f"Error: ROS bag not found at: {bag_path}")
        return

    Reader = _get_reader()

    print("\n========== STARTING PHAM4 EXTRACTOR AND SYNCHRONIZER ==========")
    print(f"ROS Bag: {bag_path}")
    print(f"Output: {args.out}")

    # Step 1: Extract Audio Channels
    print("\n[1/3] Extracting 4-channel audio streams...")
    audio_buffers = defaultdict(list)
    
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in AUDIO_TOPICS]
        if not connections:
            print("Error: No audio topics (/audio1/audio ... /audio4/audio) found in this bag.")
            return

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            try:
                # Skip the 4-byte ROS array length prefix in on-wire data
                pcm_bytes = rawdata[4:]
                samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                ch = TOPIC_TO_CHANNEL[conn.topic]
                audio_buffers[ch].append(samples)
            except Exception as e:
                continue

    if not audio_buffers:
        print("Error: No audio data could be read.")
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

    # Squeeze channels to match lengths
    audio_4ch = np.column_stack([c[:min_len] for c in channel_data]) # Shape: (samples, 4)
    print(f"Combined 4-channel audio shape: {audio_4ch.shape}")

    # Step 2: Extract Ground Truth Trajectory Poses
    pose_topic = "/mavros/local_position/pose"
    print(f"\n[2/3] Extracting ground-truth poses from '{pose_topic}'...")
    pose_rows = []
    
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == pose_topic]
        if not connections:
            print(f"Error: Topic '{pose_topic}' not found in bag.")
            return

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = deserialize_message(rawdata, conn.msgtype)
                if hasattr(msg, "pose") and hasattr(msg.pose, "position"):
                    pos = msg.pose.position
                    pose_rows.append({
                        "timestamp": timestamp * 1e-9, # Seconds
                        "x": float(pos.x),
                        "y": float(pos.y),
                        "z": float(pos.z)
                    })
            except Exception as e:
                continue

    if not pose_rows:
        print("Error: No pose data could be extracted.")
        return

    df_poses = pd.DataFrame(pose_rows)
    print(f"Extracted {len(df_poses):,} poses.")

    # Step 3: Sliding-Window Segmentation and Synchronization
    print("\n[3/3] Segmenting audio and synchronizing with coordinates...")
    
    # Target folders
    audio_npy_dir = os.path.join(args.out, "audio_npy", args.seq)
    gt_dir = os.path.join(args.out, "gt", args.seq)
    pseudo_dir = os.path.join(args.out, "pseudo_label", args.seq)

    os.makedirs(audio_npy_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pseudo_dir, exist_ok=True)

    sr = args.sr
    window_ms = 400.0 # 0.4s window size
    overlap = 0.75 # 75% overlap (100ms hop)

    win_samples = int(window_ms * 1e-3 * sr) # 16,720 samples
    hop_samples = int(win_samples * (1 - overlap)) # 4,180 samples

    n_samples = audio_4ch.shape[0]
    start = 0
    idx = 2 # Dataloader expects index step of 2 to match make_seq_audio logic
    saved_count = 0

    # For reproducibility
    np.random.seed(42)

    while start + win_samples <= n_samples:
        center_t = (start + win_samples / 2) / sr
        
        # Squeeze out the 4-channel audio segment (shape: win_samples, 4)
        audio_segment = audio_4ch[start : start + win_samples, :]
        
        # Find the nearest pose coordinates temporally
        nearest_idx = (df_poses["timestamp"] - center_t).abs().idxmin()
        pose = df_poses.iloc[nearest_idx]
        
        gt_coord = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32)
        
        # Simulate LiDAR Teacher Network Pseudo-Label
        # Paper Section II-A: Pseudo-labels are generated with a mean error of ~0.5m compared to real trajectory
        noise = np.random.normal(0, 0.25, size=3).astype(np.float32) # StDev of 0.25m gives ~0.45m average error
        pseudo_coord = gt_coord + noise

        # Save files
        np.save(os.path.join(audio_npy_dir, f"{idx}.npy"), audio_segment.astype(np.float32))
        np.save(os.path.join(gt_dir, f"{idx}.npy"), gt_coord)
        np.save(os.path.join(pseudo_dir, f"{idx}.npy"), pseudo_coord)

        idx += 2
        saved_count += 1
        start += hop_samples

    print(f"\n========== EXTRACTION & SYNCHRONIZATION COMPLETE! ==========")
    print(f"Total Segments Saved : {saved_count}")
    print(f"Audio folder         : {audio_npy_dir}")
    print(f"GT folder            : {gt_dir}")
    print(f"Pseudo-label folder  : {pseudo_dir}")
    print("\nNext step: Run 'python create_splits.py' to generate your annotation list files!")

if __name__ == "__main__":
    main()
