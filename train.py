import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataloader_tutorial import UAVLoader
from network.audio_net import AudioNet
from utils import loss

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_epoch(model, dataloader, optimizer, loss_fn, alpha, device):
    model.train()
    total_loss = 0.0
    
    progress = tqdm(dataloader, desc="  Training", leave=False, unit="batch")
    for data in progress:
        spec, gt, pseudo_label = [d.to(device) for d in data]
        
        optimizer.zero_grad()
        predictions = model(spec)
        
        # Calculate loss: total_loss = alpha * L_gt + (1 - alpha) * L_pseudo
        batch_loss = loss_fn(predictions, gt, pseudo_label, alpha)
        
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        progress.set_postfix(loss=batch_loss.item())
        
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data in dataloader:
            spec, gt, _ = [d.to(device) for d in data]
            predictions = model(spec)
            
            # Ground truth validation loss (standard L1 loss)
            batch_loss = loss_fn(predictions, gt)
            total_loss += batch_loss.item()
            
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="Train Audio Array UAV Trajectory Estimator.")
    parser.add_argument("--root", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--train-anno", type=str, default="dataset/train_anno.txt", help="Path to train split list")
    parser.add_argument("--val-anno", type=str, default="dataset/val_anno.txt", help="Path to val split list")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Self-supervised weight alpha (0=LiDAR Pseudo-Labels, 1=Leica Supervised)")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (e.g. cuda:0, cpu)")
    parser.add_argument("--save-dir", type=str, default="output", help="Directory to save model checkpoints")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout rate (default: 0.2)")
    parser.add_argument("--dry-run", action="store_true", help="Quick run with 1 epoch and small batch limit to verify correctness")
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    set_seed(42)

    # Load annotations
    if not os.path.exists(args.train_anno) or not os.path.exists(args.val_anno):
        print(f"Error: Annotation splits not found at {args.train_anno} or {args.val_anno}.")
        print("Please run: python create_splits.py first (or with --mock) to set up annotations.")
        return

    with open(args.train_anno, "r", encoding="utf-8") as f:
        train_lines = [line.strip() for line in f.readlines() if line.strip()]
        
    with open(args.val_anno, "r", encoding="utf-8") as f:
        val_lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Loaded annotations: {len(train_lines)} train, {len(val_lines)} validation samples.")

    # Datasets & Dataloaders
    # Note: original loader expects root_path as a parent directory and joins folder paths
    uav_train = UAVLoader(train_lines, args.root, dark_aug=1)
    uav_val = UAVLoader(val_lines, args.root, dark_aug=1, testing=1)

    # Adjust workers and batches for dry-runs or Windows environments
    num_workers = 0 if os.name == 'nt' else 4
    batch_size = 2 if args.dry_run else args.batch_size
    epochs = 1 if args.dry_run else args.epochs

    # Drop last batch only if we have enough files
    drop_last = True if len(train_lines) >= batch_size else False

    train_loader = DataLoader(uav_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last)
    val_loader = DataLoader(uav_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    # Model definition
    model = AudioNet(dropout_rate=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    loss_train = loss.regression_loss_w_pseudo
    loss_val = loss.regression_loss

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "model_best.pth")

    best_val_loss = float('inf')

    print(f"\n--- Starting Training (Alpha: {args.alpha}, Epochs: {epochs}, Batch size: {batch_size}) ---")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_train, args.alpha, device)
        val_loss = validate(model, val_loader, loss_val, device)
        
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path}")

    print("\nTraining completed successfully!")
    print(f"Best Validation Loss achieved: {best_val_loss:.5f}")

if __name__ == "__main__":
    main()
