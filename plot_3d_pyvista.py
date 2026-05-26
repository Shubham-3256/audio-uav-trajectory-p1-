import os
import argparse
import numpy as np
import torch
import pyvista as pv

from evaluate import evaluate_model, smooth_with_gp, set_seed
from network.audio_net import AudioNet

def main():
    parser = argparse.ArgumentParser(description="Generate a premium interactive 3D PyVista trajectory visualizer.")
    parser.add_argument("--root", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--val-anno", type=str, default="dataset/val_anno.txt", help="Path to val split list")
    parser.add_argument("--checkpoint", type=str, default="output/model_best.pth", help="Path to trained model parameters")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cuda:0, cpu)")
    args = parser.parse_args()

    set_seed(42)
    
    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")

    # 1. Scan and load all synchronized frames
    seq_dir = os.path.join(args.root, "audio_npy", "seq_001")
    if os.path.exists(seq_dir):
        val_lines = [os.path.join("seq_001", f) for f in os.listdir(seq_dir) if f.endswith(".npy")]
        print(f"[Visualizer] Scanning {len(val_lines)} frames for complete continuous 3D loop...")
    else:
        if not os.path.exists(args.val_anno):
            print(f"Error: Annotation list not found at {args.val_anno}.")
            return
        with open(args.val_anno, "r", encoding="utf-8") as f:
            val_lines = [line.strip() for line in f.readlines() if line.strip()]

    # Sort temporally
    def get_temporal_key(line):
        try:
            parts = line.replace("\\", "/").split("/")
            file_idx = int(os.path.splitext(parts[1])[0])
            return file_idx
        except Exception:
            return 0

    val_lines.sort(key=get_temporal_key)

    # 2. Initialize and load model
    model = AudioNet().to(device)
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}.")
        return
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded perfect model weights from: {args.checkpoint}")

    # 3. Evaluate coordinates
    audio_path = os.path.join(args.root, "audio_npy")
    gt_path = os.path.join(args.root, "gt")
    gt_array, predict_array = evaluate_model(model, val_lines, audio_path, gt_path, device)

    # 4. Smooth using optimized GP
    smoothed_array = smooth_with_gp(predict_array)

    # 5. Create PyVista Plotter
    print("\nInitializing PyVista high-performance native 3D window...")
    
    # Global theme styling
    pv.global_theme.background = "#0D0E15"
    pv.global_theme.font.color = "#FFFFFF"
    pv.global_theme.font.family = "courier"
    
    plotter = pv.Plotter(title="AAUTE: UAV 3D PyVista Trajectory Visualizer")
    
    # A. Ground Truth Polyline (Neon Green)
    gt_lines = pv.MultipleLines(points=gt_array)
    plotter.add_mesh(
        gt_lines, 
        color="#39FF14", 
        line_width=4, 
        label="Ground Truth (Leica)",
        render_lines_as_tubes=True
    )
    
    # B. GP Smoothed Path (Neon Pink / Glowing Red Ribbon)
    smoothed_lines = pv.MultipleLines(points=smoothed_array)
    plotter.add_mesh(
        smoothed_lines, 
        color="#FF007F", 
        line_width=7, 
        label="GP Smoothed Path",
        render_lines_as_tubes=True
    )
    
    # C. Raw Predictions Point Cloud (Cyan Spheres)
    pred_cloud = pv.PolyData(predict_array)
    plotter.add_mesh(
        pred_cloud, 
        color="#00F0FF", 
        point_size=8.0, 
        render_points_as_spheres=True, 
        opacity=0.5, 
        label="Raw Predictions"
    )
    
    # D. Setup Grid Bounds and Labels
    plotter.show_bounds(
        grid='back',
        location='outer',
        color='#4A5270',
        font_size=12,
        xtitle='X axis (meters)',
        ytitle='Y axis (meters)',
        ztitle='Altitude Z (meters)',
        all_edges=True
    )
    
    # E. Add interactive text overlay
    plotter.add_text(
        "AAUTE: 3D Trajectory Visualizer (PyVista Desktop)", 
        position="upper_left", 
        font_size=14, 
        color="#FFFFFF"
    )
    
    plotter.add_legend(
        bcolor=[19, 21, 32],
        border=True,
        size=(0.25, 0.15)
    )
    
    # Enable hardware visual enhancements
    plotter.enable_anti_aliasing()
    plotter.enable_eye_dome_lighting() # Glow highlights
    
    # Define file outputs
    screenshot_path = "output/pyvista_trajectory_desktop.png"
    html_path = "output/pyvista_interactive_vtk.html"
    
    print("\nSUCCESS! Saving PyVista outputs and launching hardware-accelerated 3D window...")
    print(f" -> High-res desktop screenshot will be saved to: {screenshot_path}")
    print(f" -> Interactive VTKjs HTML will be saved to: {html_path}")
    print("\nA 3D desktop window will pop up now! Left-click and drag to rotate, right-click/wheel to zoom.")
    
    # Export the interactive scene as offline HTML BEFORE showing (foolproof execution!)
    plotter.export_html(html_path)
    print(f"\nPyVista HTML successfully exported to: {html_path}")
    
    # Open the window and save the screenshot
    print("\nSUCCESS! Saving PyVista screenshot and launching hardware-accelerated 3D window...")
    plotter.show(screenshot=screenshot_path)
    print("\nPyVista desktop window closed cleanly!")

if __name__ == "__main__":
    main()
