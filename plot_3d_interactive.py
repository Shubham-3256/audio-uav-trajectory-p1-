import os
import argparse
import numpy as np
import torch
import plotly.graph_objects as go

from evaluate import evaluate_model, smooth_with_gp, set_seed
from network.audio_net import AudioNet

def main():
    parser = argparse.ArgumentParser(description="Generate a premium interactive 3D trajectory visualizer.")
    parser.add_argument("--root", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--val-anno", type=str, default="dataset/val_anno.txt", help="Path to val split list")
    parser.add_argument("--checkpoint", type=str, default="output/model_best.pth", help="Path to trained model parameters")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cuda:0, cpu)")
    parser.add_argument("--out-html", type=str, default="output/interactive_trajectory_3d.html", help="Path to save HTML plot")
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
    timestamps = np.array([get_temporal_key(line) for line in val_lines])

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

    # 5. Create stunning interactive Plotly figure
    print("\nGenerating gorgeous, premium 3D Plotly visualizer...")
    fig = go.Figure()

    # A. Ground Truth Trace (Neon Green Dashed Line)
    fig.add_trace(go.Scatter3d(
        x=gt_array[:, 0],
        y=gt_array[:, 1],
        z=gt_array[:, 2],
        mode='lines',
        name='Ground Truth (Leica)',
        line=dict(
            color='#39FF14',
            width=5,
            dash='dash'
        ),
        hovertemplate=
        '<b>Ground Truth</b><br>' +
        'X: %{x:.2f}m<br>' +
        'Y: %{y:.2f}m<br>' +
        'Z (Alt): %{z:.2f}m<br>' +
        'Frame: %{text}<extra></extra>',
        text=[f"#{t}" for t in timestamps]
    ))

    # B. Raw Predictions Trace (Semi-transparent Blue Spheres)
    fig.add_trace(go.Scatter3d(
        x=predict_array[:, 0],
        y=predict_array[:, 1],
        z=predict_array[:, 2],
        mode='markers',
        name='Raw Predictions',
        marker=dict(
            size=3.5,
            color='#00F0FF',
            opacity=0.4
        ),
        hovertemplate=
        '<b>Raw Prediction</b><br>' +
        'X: %{x:.2f}m<br>' +
        'Y: %{y:.2f}m<br>' +
        'Z (Alt): %{z:.2f}m<br>' +
        'Frame: %{text}<extra></extra>',
        text=[f"#{t}" for t in timestamps]
    ))

    # C. GP Smoothed Trace (Neon Glowing Red Solid Ribbon)
    fig.add_trace(go.Scatter3d(
        x=smoothed_array[:, 0],
        y=smoothed_array[:, 1],
        z=smoothed_array[:, 2],
        mode='lines',
        name='GP Smoothed Path',
        line=dict(
            color='#FF007F',
            width=8
        ),
        hovertemplate=
        '<b>GP Smoothed</b><br>' +
        'X: %{x:.2f}m<br>' +
        'Y: %{y:.2f}m<br>' +
        'Z (Alt): %{z:.2f}m<br>' +
        'Frame: %{text}<extra></extra>',
        text=[f"#{t}" for t in timestamps]
    ))

    # D. Style Layout with Premium Dark Theme
    fig.update_layout(
        title={
            'text': "🌌 AAUTE: UAV 3D Interactive Trajectory Flight Visualizer",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 22, 'color': '#ffffff', 'family': 'Outfit, sans-serif'}
        },
        paper_bgcolor='#0D0E15',
        plot_bgcolor='#0D0E15',
        scene=dict(
            xaxis=dict(
                title=dict(text='X Coordinate (meters)', font=dict(color='#ffffff', size=12)),
                backgroundcolor='#131520',
                gridcolor='#252A40',
                showbackground=True,
                tickfont=dict(color='#A0AABF')
            ),
            yaxis=dict(
                title=dict(text='Y Coordinate (meters)', font=dict(color='#ffffff', size=12)),
                backgroundcolor='#131520',
                gridcolor='#252A40',
                showbackground=True,
                tickfont=dict(color='#A0AABF')
            ),
            zaxis=dict(
                title=dict(text='Altitude / Z (meters)', font=dict(color='#ffffff', size=12)),
                backgroundcolor='#131520',
                gridcolor='#252A40',
                showbackground=True,
                tickfont=dict(color='#A0AABF')
            ),
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(19, 21, 32, 0.8)',
            bordercolor='#252A40',
            borderwidth=1,
            font=dict(color='#ffffff', size=12)
        ),
        margin=dict(l=0, r=0, b=0, t=60)
    )

    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    fig.write_html(args.out_html)
    print(f"\nSUCCESS! Stunning 3D Interactive Visualizer saved to:\n   --> {args.out_html}")
    print("Double-click this HTML file in your folder to open it in your web browser, spin it, and explore the flight loop!")

if __name__ == "__main__":
    main()
