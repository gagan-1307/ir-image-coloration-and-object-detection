"""
train_gan.py

Stage 2: Full adversarial (Pix2Pix-style) training.
Adds the PatchGAN discriminator on top of the Stage 1 baseline.

Generator loss = L1 + SSIM + adversarial (fool the discriminator)
Discriminator loss = standard GAN BCE loss (real vs fake)

Can warm-start the generator from the Stage 1 checkpoint (recommended --
converges faster and more stably than starting from scratch).

Usage:
    python src/train_gan.py \
        --processed_dir /content/local_data/processed \
        --checkpoint_dir /content/drive/MyDrive/ir-colorization-checkpoints \
        --output_dir /content/drive/MyDrive/ir-colorization-outputs \
        --init_from /content/drive/MyDrive/ir-colorization-checkpoints/best_model.pth \
        --epochs 30 --batch_size 8
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
from train import ssim_loss, to_neg1_1  # reuse Stage 1's SSIM implementation


def train_gan(args):
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

    # --- Models ---
    gen = UNetGenerator(in_channels=4, out_channels=3, base_filters=args.base_filters).to(device)
    disc = PatchGANDiscriminator(in_channels=4, rgb_channels=3, base_filters=64).to(device)

    if args.init_from and os.path.exists(args.init_from):
        gen.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"Warm-started generator from {args.init_from}")
    else:
        print("Training generator from scratch (no warm-start checkpoint given/found).")

    opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr_gen, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))

    l1_fn = nn.L1Loss()
    gan_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    history = {"gen_loss": [], "disc_loss": [], "val_l1": []}

    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()
        epoch_start = time.time()
        running_gen_loss = 0.0
        running_disc_loss = 0.0

        for ir, rgb, _idx in train_loader:
            ir, rgb = ir.to(device), rgb.to(device)
            rgb_target = to_neg1_1(rgb)
            bsz = ir.size(0)

            # ---------------- Train Discriminator ----------------
            with torch.no_grad():
                fake_rgb = gen(ir)

            opt_disc.zero_grad()
            pred_real = disc(ir, rgb_target)
            pred_fake = disc(ir, fake_rgb.detach())

            real_labels = torch.ones_like(pred_real)
            fake_labels = torch.zeros_like(pred_fake)

            loss_disc_real = gan_loss_fn(pred_real, real_labels)
            loss_disc_fake = gan_loss_fn(pred_fake, fake_labels)
            loss_disc = 0.5 * (loss_disc_real + loss_disc_fake)

            loss_disc.backward()
            opt_disc.step()

            # ---------------- Train Generator ----------------
            opt_gen.zero_grad()
            fake_rgb = gen(ir)
            pred_fake_for_gen = disc(ir, fake_rgb)

            loss_gan = gan_loss_fn(pred_fake_for_gen, torch.ones_like(pred_fake_for_gen))
            loss_l1 = l1_fn(fake_rgb, rgb_target)
            loss_ssim = ssim_loss(fake_rgb, rgb_target)

            loss_gen = (
                args.lambda_gan * loss_gan
                + args.lambda_l1 * loss_l1
                + args.lambda_ssim * loss_ssim
            )
            loss_gen.backward()
            opt_gen.step()

            running_gen_loss += loss_gen.item() * bsz
            running_disc_loss += loss_disc.item() * bsz

        train_gen_loss = running_gen_loss / len(train_ds)
        train_disc_loss = running_disc_loss / len(train_ds)

        # --- Validation (L1 only, as a simple tracked metric) ---
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
        history["val_l1"].append(val_l1)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} | gen_loss={train_gen_loss:.4f} | "
              f"disc_loss={train_disc_loss:.4f} | val_l1={val_l1:.4f} | {elapsed:.1f}s")

        if val_l1 < best_val_loss:
            best_val_loss = val_l1
            torch.save(gen.state_dict(), os.path.join(args.checkpoint_dir, "best_gan_generator.pth"))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(gen.state_dict(), os.path.join(args.checkpoint_dir, f"gan_gen_epoch_{epoch}.pth"))
            torch.save(disc.state_dict(), os.path.join(args.checkpoint_dir, f"gan_disc_epoch_{epoch}.pth"))

    np.save(os.path.join(args.output_dir, "gan_train_history.npy"), history)
    print(f"\nGAN training done. Best val L1: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--init_from", default="", help="Path to Stage 1 generator checkpoint to warm-start from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_gen", type=float, default=2e-4)
    parser.add_argument("--lr_disc", type=float, default=2e-4)
    parser.add_argument("--base_filters", type=int, default=64)
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--lambda_ssim", type=float, default=10.0)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    train_gan(args)