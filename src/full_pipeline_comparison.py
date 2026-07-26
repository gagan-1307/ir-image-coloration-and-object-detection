"""
full_pipeline_comparison.py

The single "tell the whole story" figure: for a few validation tiles, shows
    Thermal IR input | Stage 1 (U-Net) | Stage 2 (GAN) | Stage 3 (Semantic) | Real RGB
side by side, so progressive improvement across your staged approach is
visible in one image -- ideal for a presentation slide.

Usage:
    python src/full_pipeline_comparison.py \
        --processed_dir /content/local_data/processed \
        --ckpt_stage1 /content/drive/MyDrive/ir-colorization-checkpoints/best_model.pth \
        --ckpt_stage2 /content/drive/MyDrive/ir-colorization-checkpoints/best_gan_generator.pth \
        --ckpt_stage3 /content/drive/MyDrive/ir-colorization-checkpoints/best_semantic_generator.pth \
        --output /content/drive/MyDrive/ir-colorization-outputs/full_pipeline_comparison.png \
        --n_samples 4
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from dataset import IRColorizationDataset
from model import UNetGenerator


def denorm(x):
    return (x + 1) / 2


def load_model(checkpoint_path, device):
    model = UNetGenerator(in_channels=4, out_channels=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = IRColorizationDataset(args.processed_dir)
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    _train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    stage1 = load_model(args.ckpt_stage1, device)
    stage2 = load_model(args.ckpt_stage2, device)
    stage3 = load_model(args.ckpt_stage3, device)

    n_samples = min(args.n_samples, len(val_ds))
    idxs = np.random.RandomState(1).choice(len(val_ds), n_samples, replace=False)

    col_titles = ["IR Input (thermal)", "Stage 1: U-Net", "Stage 2: GAN",
                  "Stage 3: Semantic GAN", "Real RGB (ground truth)"]

    fig, axes = plt.subplots(n_samples, 5, figsize=(15, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 5)

    with torch.no_grad():
        for row, i in enumerate(idxs):
            ir, rgb, _idx = val_ds[i]
            ir_batch = ir.unsqueeze(0).to(device)

            thermal = ir[0].numpy()
            pred1 = denorm(stage1(ir_batch).squeeze(0).cpu()).permute(1, 2, 0).numpy()
            pred2 = denorm(stage2(ir_batch).squeeze(0).cpu()).permute(1, 2, 0).numpy()
            pred3 = denorm(stage3(ir_batch).squeeze(0).cpu()).permute(1, 2, 0).numpy()
            real = rgb.permute(1, 2, 0).numpy()

            images = [thermal, np.clip(pred1, 0, 1), np.clip(pred2, 0, 1),
                      np.clip(pred3, 0, 1), real]
            cmaps = ["gray", None, None, None, None]

            for col, (img, cmap) in enumerate(zip(images, cmaps)):
                axes[row, col].imshow(img, cmap=cmap)
                if row == 0:
                    axes[row, col].set_title(col_titles[col], fontsize=11)
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=110)
    print(f"Saved full pipeline comparison to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--ckpt_stage1", required=True)
    parser.add_argument("--ckpt_stage2", required=True)
    parser.add_argument("--ckpt_stage3", required=True)
    parser.add_argument("--output", default="outputs/full_pipeline_comparison.png")
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    run(args)