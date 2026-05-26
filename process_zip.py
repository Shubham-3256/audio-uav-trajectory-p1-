import os
import sys
import argparse
import shutil
import numpy as np
import soundfile as sf

def main():
    parser = argparse.ArgumentParser(description="Process and synchronize unzipped raw drone datasets directly from their dataset/ folder.")
    parser.add_argument("--drone", type=str, default="Mavic2", help="Name of the drone folder inside dataset/ (e.g. pham4, Mavic2, Mavic3, Avata, M300)")
    parser.add_argument("--seq", type=str, default="seq_002", help="Sequence folder name inside dataset/ (e.g., seq_002)")
    parser.add_argument("--sr", type=int, default=41800, help="Audio sample rate (default: 41800)")
    args = parser.parse_args()

    drone_name = args.drone
    seq_name = args.seq
    sr = args.sr

    # Base paths (looking directly under dataset/<drone_name>/)
    drone_dir = os.path.join("dataset", drone_name)
    audio_dir = os.path.join(drone_dir, "audio")
    gt_source_dir = os.path.join(drone_dir, "ground_truth")

    print("\n========== AUTOMATED UNZIPPED DATASET IMPORT UTILITY ==========")
    print(f"Target Drone     : {drone_name}")
    print(f"Expected Audio   : {audio_dir}/ch1.wav ... ch4.wav")
    print(f"Expected Poses   : {gt_source_dir}/*.npy files")
    print(f"Target Sequence  : dataset/.../{seq_name}")

    # Check directories
    if not os.path.exists(audio_dir):
        print(f"\n[Error] Audio directory not found: {audio_dir}")
        print(f"Please paste your 4 WAV files under: {audio_dir}/")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(gt_source_dir, exist_ok=True)
        return

    if not os.path.exists(gt_source_dir):
        print(f"\n[Error] Ground truth directory not found: {gt_source_dir}")
        print(f"Please paste your unzipped ground-truth .npy files under: {gt_source_dir}/")
        os.makedirs(gt_source_dir, exist_ok=True)
        return

    # 1. Load Audio Channels
    print("\n[1/3] Loading audio channels...")
    ch_data = []
    min_len = float('inf')
    for ch in range(1, 5):
        wav_path = os.path.join(audio_dir, f"ch{ch}.wav")
        if not os.path.exists(wav_path):
            print(f"Error: Missing audio channel: {wav_path}")
            return
        data, wav_sr = sf.read(wav_path)
        ch_data.append(data)
        min_len = min(min_len, len(data))
        sr = wav_sr
        print(f"  Channel {ch} loaded: {len(data):,} samples at {sr} Hz")

    # Stack channels
    audio_4ch = np.column_stack([c[:min_len] for c in ch_data])
    print(f"Combined 4-channel audio shape: {audio_4ch.shape}")

    # 2. Load and Sort Ground-Truth Files
    print("\n[2/3] Loading and indexing timestamped poses...")
    pose_files = [f for f in os.listdir(gt_source_dir) if f.endswith(".npy")]
    if not pose_files:
        print("Error: No .npy pose files found in ground_truth directory!")
        return

    pose_records = []
    for f in pose_files:
        try:
            # Extract timestamp from filename (e.g. 1692847902.611685.npy -> 1692847902.611685)
            ts = float(os.path.splitext(f)[0])
            pose_records.append((ts, f))
        except ValueError:
            continue

    # Sort temporally
    pose_records.sort(key=lambda x: x[0])
    pose_timestamps = np.array([x[0] for x in pose_records])
    print(f"Indexed {len(pose_timestamps):,} timestamped coordinate files.")

    # 3. Synchronize and Segment
    print("\n[3/3] Segmenting and synchronizing...")
    
    out_dir = "dataset"
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
    
    win_samples = int(window_ms * 1e-3 * sr) # 16,720 samples
    hop_samples = int(win_samples * (1 - overlap)) # 4,180 samples

    t_0 = pose_timestamps[0]
    n_samples = audio_4ch.shape[0]
    start = 0
    idx = 2
    saved_count = 0

    # Ensure reproducibility
    np.random.seed(42)

    while start + win_samples <= n_samples:
        # Relative center time of the window
        rel_center_t = (start + win_samples / 2) / sr
        
        # Absolute center timestamp
        abs_center_ts = t_0 + rel_center_t
        
        # Find closest pose index
        nearest_idx = np.abs(pose_timestamps - abs_center_ts).argmin()
        nearest_file = pose_records[nearest_idx][1]
        
        # Load the 3D position [x, y, z]
        gt_coord = np.load(os.path.join(gt_source_dir, nearest_file)).astype(np.float32)
        
        # Squeeze 4-channel audio segment
        audio_segment = audio_4ch[start : start + win_samples, :]
        
        # Simulate teacher network pseudo-labels (mean error ~0.5m)
        noise = np.random.normal(0, 0.25, size=3).astype(np.float32)
        pseudo_coord = gt_coord + noise

        # Save files stepped by 2 to support temporal sequence history loading
        np.save(os.path.join(audio_npy_dir, f"{idx}.npy"), audio_segment.astype(np.float32))
        np.save(os.path.join(gt_dir, f"{idx}.npy"), gt_coord)
        np.save(os.path.join(pseudo_dir, f"{idx}.npy"), pseudo_coord)
        
        idx += 2
        saved_count += 1
        start += hop_samples

    print(f"\n========== SYNCHRONIZATION COMPLETE! ==========")
    print(f"Total synchronized segments written: {saved_count}")
    print(f"Output files stored in: {out_dir}")
    print("\nNext: Run 'python create_splits.py' to generate your train/val split lists!")

if __name__ == "__main__":
    main()
