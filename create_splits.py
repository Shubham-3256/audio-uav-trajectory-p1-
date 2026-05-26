import os
import argparse
import random
import numpy as np

def generate_mock_dataset(dataset_dir):
    print("Generating mock/toy dataset for dry-run verification...")
    audio_dir = os.path.join(dataset_dir, "audio_npy", "seq_mock")
    gt_dir = os.path.join(dataset_dir, "gt", "seq_mock")
    pseudo_dir = os.path.join(dataset_dir, "pseudo_label", "seq_mock")

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pseudo_dir, exist_ok=True)

    # We need indices 2, 4, 6, 8, ..., 50
    # Files >= 10 will have a complete history of (index - 2*f) for f in 1..4
    indices = list(range(2, 52, 2))
    
    # Let's simulate a 3D helical trajectory
    t = np.linspace(0, 4*np.pi, len(indices))
    xs = 5.0 * np.cos(t)
    ys = 5.0 * np.sin(t)
    zs = 2.0 * t

    for i, idx in enumerate(indices):
        # Audio segment: 4 channels, let's say 4000 samples each
        # Shape: (4000, 4)
        audio_data = np.random.randn(4000, 4).astype(np.float32)
        
        # Ground Truth coordinate: (3,)
        gt_coord = np.array([xs[i], ys[i], zs[i]], dtype=np.float32)
        
        # Pseudo Label: GT + small noise (mean error ~0.5m)
        pseudo_coord = gt_coord + np.random.normal(0, 0.2, size=3).astype(np.float32)

        np.save(os.path.join(audio_dir, f"{idx}.npy"), audio_data)
        np.save(os.path.join(gt_dir, f"{idx}.npy"), gt_coord)
        np.save(os.path.join(pseudo_dir, f"{idx}.npy"), pseudo_coord)

    print(f"Mock data created inside {dataset_dir} successfully!")

def main():
    parser = argparse.ArgumentParser(description="Create train/val split annotation files.")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train/val split ratio (default: 0.8)")
    parser.add_argument("--mock", action="store_true", help="Generate a mock dataset for testing first")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")

    if args.mock:
        generate_mock_dataset(dataset_dir)

    audio_path = os.path.join(dataset_dir, "audio_npy")
    gt_path = os.path.join(dataset_dir, "gt")
    pseudo_path = os.path.join(dataset_dir, "pseudo_label")

    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} does not exist. Run with --mock to generate test data.")
        return

    # Scan for valid .npy files that have counterparts in gt and pseudo_label
    valid_files = []
    
    # We walk through audio_npy
    for root, _, files in os.walk(audio_path):
        for f in files:
            if f.endswith(".npy"):
                # Relative path from audio_path
                rel_path = os.path.relpath(os.path.join(root, f), audio_path)
                
                # Check counterparts
                gt_file = os.path.join(gt_path, rel_path)
                pseudo_file = os.path.join(pseudo_path, rel_path)
                
                if os.path.exists(gt_file) and os.path.exists(pseudo_file):
                    # We also need to make sure we can load the history (for indices >= 10)
                    parts = rel_path.replace("\\", "/").split("/")
                    index_str = parts[-1][:-4]
                    try:
                        idx_val = int(index_str)
                        # Dataloader expects index - 2*f for f in 1..4
                        # So idx_val must be at least 10 (or whatever start sequence contains the previous 4 steps)
                        # We verify if previous files exist
                        has_history = True
                        for step in range(1, 5):
                            prev_filename = f"{parts[0]}/{idx_val - 2*step}.npy"
                            if not os.path.exists(os.path.join(audio_path, prev_filename)):
                                has_history = False
                                break
                        
                        if has_history:
                            valid_files.append(rel_path.replace("\\", "/"))
                    except ValueError:
                        # Non-integer index, we'll append it anyway if no sequential history is required, 
                        # but standard tutorial loader expects sequential integer files.
                        pass

    if len(valid_files) == 0:
        print("No valid files with history found. Ensure files are named as sequential integers (e.g. 2.npy, 4.npy, 6.npy...) starting from 10.")
        return

    print(f"Found {len(valid_files)} valid training samples.")

    # Shuffle and split
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * args.ratio)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]

    # Write annotation files
    # The dataloader reads files as relative paths with extension, e.g. "seq_mock/10.npy\n"
    train_anno_path = os.path.join(dataset_dir, "train_anno.txt")
    val_anno_path = os.path.join(dataset_dir, "val_anno.txt")

    with open(train_anno_path, "w", encoding="utf-8") as f:
        for item in sorted(train_files):
            f.write(f"{item}\n")

    with open(val_anno_path, "w", encoding="utf-8") as f:
        for item in sorted(val_files):
            f.write(f"{item}\n")

    print(f"Splits successfully written:")
    print(f" - Train Annotation ({len(train_files)} files): {train_anno_path}")
    print(f" - Val Annotation ({len(val_files)} files): {val_anno_path}")

if __name__ == "__main__":
    main()
