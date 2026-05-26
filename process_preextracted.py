import os
import argparse
import shutil
import numpy as np
import pandas as pd
import soundfile as sf

def main():
    parser = argparse.ArgumentParser(description="Segment and align pre-extracted Pham4 WAV tracks and CSV labels placed inside dataset/.")
    parser.add_argument("--audio-dir", type=str, default="dataset/audio/Pham4", help="Directory containing ch1.wav ... ch4.wav")
    parser.add_argument("--labels", type=str, default="dataset/features/Pham4/labels.csv", help="Path to aligned labels.csv")
    parser.add_argument("--out", type=str, default="dataset", help="Output dataset directory")
    parser.add_argument("--seq", type=str, default="seq_001", help="Name of the sequence folder (e.g. seq_001)")
    args = parser.parse_args()

    audio_dir = args.audio_dir
    labels_path = args.labels
    out_dir = args.out
    seq_name = args.seq

    print("\n========== LOCAL PRE-EXTRACTED DATASET SEGMENTER ==========")
    print(f"Audio directory : {audio_dir}")
    print(f"Labels CSV path : {labels_path}")
    print(f"Target folder   : {out_dir}/.../{seq_name}")

    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found.")
        print("Please copy your extracted 'audio' folder directly into the 'dataset/' folder in your workspace!")
        return

    if not os.path.exists(labels_path):
        print(f"Error: Labels CSV '{labels_path}' not found.")
        print("Please copy your extracted 'features' folder directly into the 'dataset/' folder in your workspace!")
        return

    # 1. Load the 4 WAV channels
    print("\n[1/3] Loading audio channels...")
    ch_data = []
    min_len = float('inf')
    sr = None
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

    # 2. Load Labels
    print("\n[2/3] Loading synchronized pose labels...")
    df_labels = pd.read_csv(labels_path)
    print(f"Loaded {len(df_labels)} pose labels.")

    # 3. Create Target Folders
    audio_npy_dir = os.path.join(out_dir, "audio_npy", seq_name)
    gt_dir = os.path.join(out_dir, "gt", seq_name)
    pseudo_dir = os.path.join(out_dir, "pseudo_label", seq_name)

    # Overwrite targeted sequence folders
    for d in [audio_npy_dir, gt_dir, pseudo_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 4. Slide window and segment
    window_ms = 400.0
    overlap = 0.75
    
    win_samples = int(window_ms * 1e-3 * sr) # 16,720 samples
    hop_samples = int(win_samples * (1 - overlap)) # 4,180 samples

    # Ensure reproducibility
    np.random.seed(42)
    saved_count = 0

    print("\n[3/3] Segmenting and synchronizing...")
    for k, row in df_labels.iterrows():
        start = int(row['window_idx'] * hop_samples)
        
        # Guard bound
        if start + win_samples > min_len:
            break
            
        audio_segment = audio_4ch[start : start + win_samples, :]
        gt_coord = np.array([row['x'], row['y'], row['z']], dtype=np.float32)
        
        # Simulate teacher network pseudo-labels (mean error ~0.5m)
        noise = np.random.normal(0, 0.25, size=3).astype(np.float32)
        pseudo_coord = gt_coord + noise

        # Save files stepped by 2 to support temporal sequence history loading
        idx = int(2 * row['window_idx'])
        
        np.save(os.path.join(audio_npy_dir, f"{idx}.npy"), audio_segment.astype(np.float32))
        np.save(os.path.join(gt_dir, f"{idx}.npy"), gt_coord)
        np.save(os.path.join(pseudo_dir, f"{idx}.npy"), pseudo_coord)
        
        saved_count += 1

    print(f"\n========== SYNCHRONIZATION COMPLETE! ==========")
    print(f"Total synchronized segments written: {saved_count}")
    print(f"Output files stored in: {out_dir}")
    print("\nNext: Run 'python create_splits.py' to generate your train/val split lists!")

if __name__ == "__main__":
    main()
