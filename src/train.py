"""
train.py

Stage 1 baseline training: U-Net generator only (no discriminator yet).
Loss = L1 (pixel accuracy) + SSIM (structural similarity).

Run this in Colab with a GPU. Checkpoints save to Google Drive so they
survive session disconnects.

Usage:
    python src/train.py --processed_dir data/processed --checkpoint_dir /content/drive/MyDrive/ir-colorization-checkpoints --epochs 25
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import IRColorizationDataset
from model import UNetGenerator


def ssim_loss(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """Simple single-scale SSIM implemented with average pooling, so we don't
    need an extra dependency. Returns (1 - SSIM) as a loss (0 = identical)."""
    pred = (pred + 1) / 2   # tanh output [-1,1] -> [0,1] for a stable SSIM range
    target = (target + 1) / 2

    pad = window_size // 2
    mu_pred = nn.functional.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_target = nn.functional.avg_pool2d(target, window_size, stride=1, padding=pad)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = nn.functional.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_pred_sq
    sigma_target_sq = nn.functional.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_target_sq
    sigma_pred_target = nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )
    return 1 - ssim_map.mean()


def to_neg1_1(x):
    """Dataset gives RGB in [0, 1]; generator uses tanh so targets need [-1, 1]."""
    return x * 2 - 1


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    full_dataset = IRColorizationDataset(args.processed_dir)
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Model ---
    model = UNetGenerator(in_channels=4, out_channels=3, base_filters=args.base_filters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    l1_fn = nn.L1Loss()

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for ir, rgb, _idx in train_loader:
            ir, rgb = ir.to(device), rgb.to(device)
            rgb_target = to_neg1_1(rgb)

            optimizer.zero_grad()
            pred = model(ir)

            loss_l1 = l1_fn(pred, rgb_target)
            loss_ssim = ssim_loss(pred, rgb_target)
            loss = args.l1_weight * loss_l1 + args.ssim_weight * loss_ssim

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * ir.size(0)

        train_loss = running_loss / len(train_ds)

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for ir, rgb, _idx in val_loader:
                ir, rgb = ir.to(device), rgb.to(device)
                rgb_target = to_neg1_1(rgb)
                pred = model(ir)
                loss_l1 = l1_fn(pred, rgb_target)
                loss_ssim = ssim_loss(pred, rgb_target)
                loss = args.l1_weight * loss_l1 + args.ssim_weight * loss_ssim
                val_running_loss += loss.item() * ir.size(0)
        val_loss = val_running_loss / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | {elapsed:.1f}s")

        # Save checkpoint every N epochs, and whenever val loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth"))

    np.save(os.path.join(args.output_dir, "train_history.npy"), history)
    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_filters", type=int, default=64)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.5)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    train(args)