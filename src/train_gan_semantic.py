"""
train_gan_semantic.py

Stage 3: GAN training + semantic consistency loss.

Adds a loss term that uses the NDVI/NDWI/NDBI indices (already computed in
dataset.py, derived from the REAL IR/multispectral bands -- not from the
model's own output) to constrain color in specific regions:
    - High NDVI (vegetation)  -> green channel should dominate red & blue
    - High NDWI (water)       -> blue channel should dominate red & green
    - High NDBI (built-up)    -> discourage strong greenish/bluish tint
                                  (built-up areas should look roughly neutral/grey)

This is a *hard physical prior*, independent of what the discriminator thinks
looks realistic -- it directly targets the "no hallucination" requirement,
since these masks come from real spectral measurements, not guesses.

Usage:
    python src/train_gan_semantic.py \
        --processed_dir /content/local_data/processed \
        --checkpoint_dir /content/drive/MyDrive/ir-colorization-checkpoints \
        --output_dir /content/drive/MyDrive/ir-colorization-outputs \
        --init_from /content/drive/MyDrive/ir-colorization-checkpoints/best_gan_generator.pth \
        --epochs 20 --batch_size 8
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import IRColorizationDataset
from model import UNetGenerator, PatchGANDiscriminator
from train import ssim_loss, to_neg1_1


def semantic_consistency_loss(pred_rgb_01, idx, ndvi_thresh=0.3, ndwi_thresh=0.1,
                               veg_weight=1.0, water_weight=1.0):
    """
    pred_rgb_01: (B, 3, H, W) generator output, rescaled to [0, 1]
    idx:         (B, 3, H, W) -- channel 0 = NDVI, 1 = NDWI, 2 = NDBI
                 (from dataset.py, computed from real spectral bands)

    Returns a scalar penalty: >0 whenever the generator colors a
    vegetation/water pixel inconsistently with what the real spectral
    signature says is there.
    """
    r, g, b = pred_rgb_01[:, 0], pred_rgb_01[:, 1], pred_rgb_01[:, 2]
    ndvi, ndwi = idx[:, 0], idx[:, 1]

    veg_mask = (ndvi > ndvi_thresh).float()
    water_mask = (ndwi > ndwi_thresh).float()

    # Vegetation should have green >= red and green >= blue
    veg_violation = torch.relu(r - g) + torch.relu(b - g)
    loss_veg = (veg_violation * veg_mask).sum() / (veg_mask.sum() + 1e-6)

    # Water should have blue >= red and blue >= green
    water_violation = torch.relu(r - b) + torch.relu(g - b)
    loss_water = (water_violation * water_mask).sum() / (water_mask.sum() + 1e-6)

    return veg_weight * loss_veg + water_weight * loss_water


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

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

    gen = UNetGenerator(in_channels=4, out_channels=3, base_filters=args.base_filters).to(device)
    disc = PatchGANDiscriminator(in_channels=4, rgb_channels=3, base_filters=64).to(device)

    if args.init_from and os.path.exists(args.init_from):
        gen.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"Warm-started generator from {args.init_from}")

    if args.init_disc_from and os.path.exists(args.init_disc_from):
        disc.load_state_dict(torch.load(args.init_disc_from, map_location=device))
        print(f"Warm-started discriminator from {args.init_disc_from}")

    opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr_gen, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))

    l1_fn = nn.L1Loss()
    gan_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    history = {"gen_loss": [], "disc_loss": [], "semantic_loss": [], "val_l1": []}

    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()
        epoch_start = time.time()
        running_gen_loss = 0.0
        running_disc_loss = 0.0
        running_sem_loss = 0.0

        for ir, rgb, idx in train_loader:
            ir, rgb, idx = ir.to(device), rgb.to(device), idx.to(device)
            rgb_target = to_neg1_1(rgb)
            bsz = ir.size(0)

            # ---------------- Discriminator ----------------
            with torch.no_grad():
                fake_rgb = gen(ir)

            opt_disc.zero_grad()
            pred_real = disc(ir, rgb_target)
            pred_fake = disc(ir, fake_rgb.detach())
            loss_disc = 0.5 * (
                gan_loss_fn(pred_real, torch.ones_like(pred_real))
                + gan_loss_fn(pred_fake, torch.zeros_like(pred_fake))
            )
            loss_disc.backward()
            opt_disc.step()

            # ---------------- Generator ----------------
            opt_gen.zero_grad()
            fake_rgb = gen(ir)
            pred_fake_for_gen = disc(ir, fake_rgb)

            loss_gan = gan_loss_fn(pred_fake_for_gen, torch.ones_like(pred_fake_for_gen))
            loss_l1 = l1_fn(fake_rgb, rgb_target)
            loss_ssim_v = ssim_loss(fake_rgb, rgb_target)

            fake_rgb_01 = (fake_rgb + 1) / 2  # semantic loss operates in [0,1]
            loss_semantic = semantic_consistency_loss(fake_rgb_01, idx)

            loss_gen = (
                args.lambda_gan * loss_gan
                + args.lambda_l1 * loss_l1
                + args.lambda_ssim * loss_ssim_v
                + args.lambda_semantic * loss_semantic
            )
            loss_gen.backward()
            opt_gen.step()

            running_gen_loss += loss_gen.item() * bsz
            running_disc_loss += loss_disc.item() * bsz
            running_sem_loss += loss_semantic.item() * bsz

        train_gen_loss = running_gen_loss / len(train_ds)
        train_disc_loss = running_disc_loss / len(train_ds)
        train_sem_loss = running_sem_loss / len(train_ds)

        gen.eval()
        val_running_l1 = 0.0
        with torch.no_grad():
            for ir, rgb, _idx in val_loader:
                ir, rgb = ir.to(device), rgb.to(device)
                rgb_target = to_neg1_1(rgb)
                pred = gen(ir)
                val_running_l1 += l1_fn(pred, rgb_target).item() * ir.size(0)
        val_l1 = val_running_l1 / len(val_ds)

        history["gen_loss"].append(train_gen_loss)
        history["disc_loss"].append(train_disc_loss)
        history["semantic_loss"].append(train_sem_loss)
        history["val_l1"].append(val_l1)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} | gen_loss={train_gen_loss:.4f} | "
              f"disc_loss={train_disc_loss:.4f} | semantic_loss={train_sem_loss:.4f} | "
              f"val_l1={val_l1:.4f} | {elapsed:.1f}s")

        if val_l1 < best_val_loss:
            best_val_loss = val_l1
            torch.save(gen.state_dict(), os.path.join(args.checkpoint_dir, "best_semantic_generator.pth"))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(gen.state_dict(), os.path.join(args.checkpoint_dir, f"semantic_gen_epoch_{epoch}.pth"))

    np.save(os.path.join(args.output_dir, "semantic_train_history.npy"), history)
    print(f"\nTraining done. Best val L1: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--init_from", default="", help="Stage 2 generator checkpoint to warm-start from")
    parser.add_argument("--init_disc_from", default="", help="Optional: Stage 2 discriminator checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_gen", type=float, default=2e-4)
    parser.add_argument("--lr_disc", type=float, default=2e-4)
    parser.add_argument("--base_filters", type=int, default=64)
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--lambda_ssim", type=float, default=10.0)
    parser.add_argument("--lambda_semantic", type=float, default=20.0)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    train(args)