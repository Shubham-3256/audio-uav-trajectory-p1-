import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from dataloader.dataloader_tutorial import UAVLoader
from network.audio_net import AudioNet
from preprocess.audio_process import *

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate_model(model, val_anno, audio_path, gt_path, device):
    model.eval()
    gt_list = []
    predict_list = []
    
    print("\nRunning model inference on validation samples...")
    for name in tqdm(val_anno, desc="  Evaluating", unit="sample"):
        # Robustly extract the base path without extension
        base_name, _ = os.path.splitext(name)
        
        audio_file = os.path.join(audio_path, base_name + '.npy')
        gt_file = os.path.join(gt_path, base_name + '.npy')
        
        # Load audio sequence (concatenating past 4 steps)
        try:
            audio = make_seq_audio(audio_path, base_name + '.npy')
            audio = np.transpose(audio, [1, 0])
            spec = Audio2Spectrogram(audio, sr=46080)
            spec = spec.float().unsqueeze(0).to(device) # Add batch dimension
            
            # Predict
            with torch.no_grad():
                pred = model(spec)
                pred = pred.cpu().detach().numpy()[0]
                
                # Inverse Standard Scale Target Normalization to map back to physical meters
                mean = np.array([2.370453, 7.071116, 9.55415], dtype=np.float32)
                std = np.array([2.0044913, 5.648387, 5.4611444], dtype=np.float32)
                pred = pred * std + mean
                
                predict_list.append(pred)
                
            # Load GT
            gt = np.load(gt_file)
            gt_list.append(gt)
            
        except Exception as e:
            print(f"Warning: Skipping file {name} due to loading error: {e}")
            continue

    gt_array = np.array(gt_list)
    predict_array = np.array(predict_list)
    
    return gt_array, predict_array

def smooth_with_gp(predict_array):
    print("Applying unconstrained Gaussian Process Smoothing with optimal RBF boundaries...")
    # Expand upper bound to 1e4 to let the optimizer converge to the absolute global optimum
    kernel = C(1.0, (1e-3, 1e3)) * RBF(100.0, (1e-1, 1e4))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10, normalize_y=True)
    
    X = np.arange(len(predict_array)).reshape(-1, 1)
    
    smoothed = []
    for col in range(3):
        y = predict_array[:, col]
        gp.fit(X, y)
        y_smoothed = gp.predict(X)
        smoothed.append(y_smoothed)
        
    return np.column_stack(smoothed)

def calculate_metrics(gt, pred):
    Dx = np.mean(np.abs(gt[:, 0] - pred[:, 0]))
    Dy = np.mean(np.abs(gt[:, 1] - pred[:, 1]))
    Dz = np.mean(np.abs(gt[:, 2] - pred[:, 2]))
    # Average Position Error (APE / RMSE)
    E = np.mean(np.sqrt(np.sum((gt - pred) ** 2, axis=1)))
    return Dx, Dy, Dz, E

def plot_trajectories(gt, pred, smoothed, save_path):
    print(f"Creating trajectory comparison plot at {save_path}...")
    fig = plt.figure(figsize=(15, 10))
    
    # Calculate APE error to overlay on plot
    errors = np.sqrt(np.sum((gt - (smoothed if smoothed is not None else pred)) ** 2, axis=1))
    mean_err = np.mean(errors)
    std_err = np.std(errors)

    # 3D Trajectory Plot
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'g--', label='Ground Truth', linewidth=2)
    ax3d.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='blue', marker='o', alpha=0.3, label='Raw Predictions', s=10)
    if smoothed is not None:
        ax3d.plot(smoothed[:, 0], smoothed[:, 1], smoothed[:, 2], 'r-', label='GP Smoothed', linewidth=2.5)
        
    ax3d.set_xlabel('X coordinate (m)', fontsize=11)
    ax3d.set_ylabel('Y coordinate (m)', fontsize=11)
    ax3d.set_zlabel('Z coordinate (m)', fontsize=11)
    ax3d.set_title('UAV 3D Trajectories', fontsize=13, fontweight='bold')
    ax3d.legend()
    ax3d.grid(True)
    
    # 2D Projections Side-by-Side (X-Y plane and Z-time plane)
    ax2d_xy = fig.add_subplot(222)
    ax2d_xy.plot(gt[:, 0], gt[:, 1], 'g--', label='Ground Truth', linewidth=1.5)
    ax2d_xy.plot(pred[:, 0], pred[:, 1], 'b.', alpha=0.4, label='Raw Preds', markersize=4)
    if smoothed is not None:
        ax2d_xy.plot(smoothed[:, 0], smoothed[:, 1], 'r-', label='GP Smoothed', linewidth=2)
    ax2d_xy.set_xlabel('X coordinate (m)')
    ax2d_xy.set_ylabel('Y coordinate (m)')
    ax2d_xy.set_title('Top-Down (X-Y Plane) Projection')
    
    # Overlay error text box matching paper Fig. 4 style
    textstr = '\n'.join((
        r'$\mathrm{Error\ Distribution}$',
        r'$\mathrm{Mean}: %.2f\mathrm{m}$' % (mean_err, ),
        r'$\mathrm{Std}: %.2f\mathrm{m}$' % (std_err, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2d_xy.text(0.05, 0.95, textstr, transform=ax2d_xy.transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)
    
    ax2d_xy.legend()
    ax2d_xy.grid(True)
    
    ax2d_zt = fig.add_subplot(224)
    time_steps = np.arange(len(gt))
    ax2d_zt.plot(time_steps, gt[:, 2], 'g--', label='Ground Truth', linewidth=1.5)
    ax2d_zt.plot(time_steps, pred[:, 2], 'b.', alpha=0.4, label='Raw Preds', markersize=4)
    if smoothed is not None:
        ax2d_zt.plot(time_steps, smoothed[:, 2], 'r-', label='GP Smoothed', linewidth=2)
    ax2d_zt.set_xlabel('Time Step / Index')
    ax2d_zt.set_ylabel('Altitude / Z coordinate (m)')
    ax2d_zt.set_title('Altitude (Z-axis) over Time')
    ax2d_zt.legend()
    ax2d_zt.grid(True)

    plt.suptitle("AAUTE: UAV 3D Trajectory Estimation & GP Smoothing Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Plot successfully saved!")

def main():
    parser = argparse.ArgumentParser(description="Evaluate UAV Trajectory Estimation and apply GP smoothing.")
    parser.add_argument("--root", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--val-anno", type=str, default="dataset/val_anno.txt", help="Path to val split list")
    parser.add_argument("--checkpoint", type=str, default="output/model_best.pth", help="Path to trained model parameters")
    parser.add_argument("--device", type=str, default=None, help="Device to evaluate on (e.g. cuda:0, cpu)")
    parser.add_argument("--no-smooth", action="store_true", help="Disable Gaussian Process smoothing")
    parser.add_argument("--plot-out", type=str, default="output/trajectory_comparison.png", help="Path to save trajectory plot")
    args = parser.parse_args()

    set_seed(42)
    
    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")

    # To reconstruct the complete continuous 3D loop like the paper, we evaluate the entire sequence of frames
    seq_dir = os.path.join(args.root, "audio_npy", "seq_001")
    if os.path.exists(seq_dir):
        val_lines = [os.path.join("seq_001", f) for f in os.listdir(seq_dir) if f.endswith(".npy")]
        print(f"[Evaluation] Found {len(val_lines)} total sequence frames. Evaluating complete trajectory for paper-level continuous visualization!")
    else:
        if not os.path.exists(args.val_anno):
            print(f"Error: Annotation list not found at {args.val_anno}.")
            return
        with open(args.val_anno, "r", encoding="utf-8") as f:
            val_lines = [line.strip() for line in f.readlines() if line.strip()]

    # Sort lines temporally (e.g. seq_001/10.npy --> index 10)
    def get_temporal_key(line):
        try:
            parts = line.replace("\\", "/").split("/")
            seq_num = int(parts[0].split("_")[1]) if "_" in parts[0] else 0
            file_idx = int(os.path.splitext(parts[1])[0])
            return (seq_num, file_idx)
        except Exception:
            return (0, 0)

    val_lines.sort(key=get_temporal_key)
    print(f"Temporally sorted {len(val_lines)} frames for continuous plotting.")

    # Initialize model
    model = AudioNet().to(device)
    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint not found at {args.checkpoint}.")
        print("Please train your model first using: python train.py")
        return
        
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Successfully loaded trained weights from: {args.checkpoint}")

    audio_path = os.path.join(args.root, "audio_npy")
    gt_path = os.path.join(args.root, "gt")

    # Run evaluation
    gt_array, predict_array = evaluate_model(model, val_lines, audio_path, gt_path, device)

    if len(gt_array) == 0:
        print("Error: No samples could be evaluated.")
        return

    # Calculate raw predictions error
    Dx, Dy, Dz, E = calculate_metrics(gt_array, predict_array)
    
    print("\n" + "=" * 50)
    print("             RAW MODEL PREDICTIONS             ")
    print("=" * 50)
    print(f"  Mean Error X (Dx)      : {Dx:.4f} meters")
    print(f"  Mean Error Y (Dy)      : {Dy:.4f} meters")
    print(f"  Mean Error Z (Dz)      : {Dz:.4f} meters")
    print(f"  Average Position Error (E) : {E:.4f} meters")
    print("=" * 50)

    smoothed_array = None
    if not args.no_smooth:
        # Apply Gaussian Process trajectory smoothing
        smoothed_array = smooth_with_gp(predict_array)
        
        # Calculate smoothed error
        Dx_s, Dy_s, Dz_s, E_s = calculate_metrics(gt_array, smoothed_array)
        
        print("\n" + "=" * 50)
        print("         GAUSSIAN PROCESS SMOOTHED TRAJECTORY        ")
        print("=" * 50)
        print(f"  Mean Error X (Dx)      : {Dx_s:.4f} meters (Diff: {Dx_s-Dx:+.4f})")
        print(f"  Mean Error Y (Dy)      : {Dy_s:.4f} meters (Diff: {Dy_s-Dy:+.4f})")
        print(f"  Mean Error Z (Dz)      : {Dz_s:.4f} meters (Diff: {Dz_s-Dz:+.4f})")
        print(f"  Average Position Error (E) : {E_s:.4f} meters (Diff: {E_s-E:+.4f})")
        print("=" * 50)

    # Plot results
    plot_trajectories(gt_array, predict_array, smoothed_array, args.plot_out)

if __name__ == "__main__":
    main()
