import os
import sys
import subprocess
import glob

def find_case_insensitive_dir(parent, target_base):
    """Find a directory case-insensitively (e.g. Pham4 vs pham4)."""
    if not os.path.exists(parent):
        return None
    for entry in os.listdir(parent):
        if entry.lower() == target_base.lower():
            full_path = os.path.join(parent, entry)
            if os.path.isdir(full_path):
                return full_path
    return None

def main():
    print("\n============================================================")
    print("             AAUTE AUTOMATED END-TO-END PIPELINE            ")
    print("============================================================\n")

    dataset_dir = "dataset"
    
    # 1. Scan for .bag files inside the dataset/ folder
    bag_files = glob.glob(os.path.join(dataset_dir, "*.bag"))
    
    if not bag_files:
        print(f"[Error] No ROS bag (.bag) files found inside the '{dataset_dir}/' folder.")
        print(f"--> Please drop your 'Pham4.bag' (or any other bag file) directly inside the '{dataset_dir}/' folder.")
        print("    Example path: dataset/Pham4.bag\n")
        return

    print(f"[Pipeline] Found {len(bag_files)} ROS bag file(s) inside '{dataset_dir}/':")
    for b in bag_files:
        print(f"  - {b}")

    # Check if synchronized files already exist to make extraction optional and instant
    check_path = os.path.join(dataset_dir, "audio_npy", "seq_001")
    exists_already = os.path.exists(check_path) and len(glob.glob(os.path.join(check_path, "*.npy"))) > 0
    
    if exists_already:
        print("\n[Pipeline] Detected existing synchronized flight segments in 'dataset/audio_npy/seq_001/'!")
        print("[Pipeline] Skipping time-consuming ROSbag extraction and proceeding directly to training!\n")
        skip_extraction = True
    else:
        print("\n[Pipeline] No pre-existing synchronized segments found. Wiping folders for a fresh extraction...")
        skip_extraction = False
        # Clean up any existing processed directories to prevent contamination
        for folder in ["audio_npy", "gt", "pseudo_label"]:
            p = os.path.join(dataset_dir, folder)
            if os.path.exists(p):
                print(f"[Pipeline] Cleaning old processed folder: {p}")
                import shutil
                shutil.rmtree(p)

    # 2. Extract and Segment each bag file by aligning it with its unzipped ground-truth
    if not skip_extraction:
        for idx, bag_path in enumerate(bag_files, 1):
            seq_name = f"seq_{idx:03d}"
            drone_name = os.path.splitext(os.path.basename(bag_path))[0]
            
            # Look for matching unzipped directory case-insensitively
            drone_dir = find_case_insensitive_dir(dataset_dir, drone_name)
            if drone_dir is None:
                print(f"\n[Pipeline] [Error] Corresponding unzipped directory for '{drone_name}' not found!")
                print(f"--> Please unzip your Point Cloud ZIP file for '{drone_name}' directly inside the '{dataset_dir}/' folder")
                print(f"    so that the folder 'dataset/{drone_name}/ground_truth' containing .npy files exists!\n")
                return
                
            gt_dir = find_case_insensitive_dir(drone_dir, "ground_truth")
            if gt_dir is None:
                print(f"\n[Pipeline] [Error] 'ground_truth' folder not found inside '{drone_dir}'!")
                print(f"--> Make sure you extract the ZIP file so that the path 'dataset/{drone_name}/ground_truth/' exists!\n")
                return

            print(f"\n[Pipeline] [Step 1/4] Synchronizing Audio from '{bag_path}' & Leica poses from '{gt_dir}' as '{seq_name}'...")
            
            # Run process_bag_zip.py
            cmd = [
                sys.executable, "process_bag_zip.py",
                "--bag", bag_path,
                "--gt", gt_dir,
                "--out", dataset_dir,
                "--seq", seq_name
            ]
            
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"[Pipeline] Error occurred while synchronizing {bag_path}. Aborting pipeline.")
                return
    else:
        print("[Pipeline] [Step 1/4] Audio and pose segment directories already exist. Skipping extraction.")

    # 3. Generate Annotation Splits
    if not skip_extraction:
        print("\n[Pipeline] [Step 2/4] Generating train/validation annotations splits...")
        res = subprocess.run([sys.executable, "create_splits.py"])
        if res.returncode != 0:
            print("[Pipeline] Error occurred while creating splits. Aborting pipeline.")
            return
    else:
        print("[Pipeline] [Step 2/4] Splits already exist. Skipping splits generation.")

    # 4. Kick off GPU Neural Network Training
    print("\n[Pipeline] [Step 3/4] Launching GPU deep learning training loop (100 epochs)...")
    res = subprocess.run([sys.executable, "train.py", "--epochs", "100", "--batch-size", "32", "--lr", "1e-4", "--alpha", "0.0"])
    if res.returncode != 0:
        print("[Pipeline] Error occurred during training. Aborting pipeline.")
        return

    # 5. Run Trajectory Evaluation and GP Trajectory Smoothing
    print("\n[Pipeline] [Step 4/4] Evaluating trained model and applying GP Smoothing...")
    res = subprocess.run([sys.executable, "evaluate.py"])
    if res.returncode != 0:
        print("[Pipeline] Error occurred during evaluation.")
        return

    print("\n============================================================")
    print("            PIPELINE COMPLETED SUCCESSFULLY! 🎉             ")
    print("============================================================")
    print("  - Segmented data is stored in: dataset/audio_npy/, dataset/gt/")
    print("  - Annotation splits are at   : dataset/train_anno.txt, dataset/val_anno.txt")
    print("  - Best trained model is at   : output/model_best.pth")
    print("  - 3D Trajectory comparison   : output/trajectory_comparison.png")
    print("============================================================\n")

if __name__ == "__main__":
    main()
