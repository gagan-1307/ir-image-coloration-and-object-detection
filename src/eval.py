"""
eval.py

Formal evaluation of a trained generator checkpoint on the held-out
validation set: PSNR, SSIM, and FID.

- PSNR / SSIM: computed per-tile (generated vs real RGB), then averaged.
- FID: computed once, comparing the full distribution of generated tiles
  against the full distribution of real tiles (via pytorch-fid, which uses
  a pretrained InceptionV3).

Usage:
    python src/eval.py \
        --processed_dir /content/local_data/processed \
        --checkpoint /content/drive/MyDrive/ir-colorization-checkpoints/best_gan_generator.pth \
        --output_dir /content/drive/MyDrive/ir-colorization-outputs \
        --tag gan_stage2
"""

import os
import argparse
import json
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

from dataset import IRColorizationDataset
from model import UNetGenerator


def denorm(x):
    """[-1, 1] -> [0, 1]."""
    return (x + 1) / 2


def to_uint8_hwc(tensor_chw):
    """(3, H, W) float [0,1] tensor -> (H, W, 3) uint8 numpy, for saving/SSIM."""
    arr = tensor_chw.permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data: same val split as training (seed=42), so this is a fair,
    # unseen-during-training evaluation set ---
    full_dataset = IRColorizationDataset(args.processed_dir)
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    _train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Evaluating on {len(val_ds)} held-out validation tiles.")

    # --- Model ---
    model = UNetGenerator(in_channels=4, out_channels=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # --- Temp folders for FID (pytorch-fid needs image files on disk) ---
    fid_real_dir = os.path.join(args.output_dir, f"_fid_real_{args.tag}")
    fid_fake_dir = os.path.join(args.output_dir, f"_fid_fake_{args.tag}")
    for d in (fid_real_dir, fid_fake_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    psnr_scores = []
    ssim_scores = []

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (ir, rgb, _idx) in enumerate(val_loader):
            ir = ir.to(device)
            pred = model(ir).squeeze(0).cpu()
            pred_img = to_uint8_hwc(denorm(pred))
            real_img = to_uint8_hwc(rgb.squeeze(0))

            # PSNR / SSIM (per-tile)
            psnr_scores.append(peak_signal_noise_ratio(real_img, pred_img, data_range=255))
            ssim_scores.append(
                structural_similarity(real_img, pred_img, channel_axis=2, data_range=255)
            )

            # Save for FID
            Image.fromarray(pred_img).save(os.path.join(fid_fake_dir, f"{i:05d}.png"))
            Image.fromarray(real_img).save(os.path.join(fid_real_dir, f"{i:05d}.png"))

    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))
    print(f"\nMean PSNR: {mean_psnr:.2f} dB")
    print(f"Mean SSIM: {mean_ssim:.4f}")

    # --- FID ---
    print("\nComputing FID (this loads InceptionV3, may take a minute)...")
    from pytorch_fid.fid_score import calculate_fid_given_paths

    fid_value = calculate_fid_given_paths(
        [fid_real_dir, fid_fake_dir],
        batch_size=min(50, len(val_ds)),
        device=device,
        dims=2048,
    )
    print(f"FID: {fid_value:.2f}")

    # --- Save results ---
    results = {
        "checkpoint": args.checkpoint,
        "n_tiles_evaluated": len(val_ds),
        "mean_psnr_db": mean_psnr,
        "mean_ssim": mean_ssim,
        "fid": fid_value,
    }
    results_path = os.path.join(args.output_dir, f"eval_results_{args.tag}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Cleanup temp image folders (keep the json, not the raw PNGs)
    shutil.rmtree(fid_real_dir)
    shutil.rmtree(fid_fake_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--tag", default="eval", help="Label for this run, used in output filenames")
    args = parser.parse_args()

    evaluate(args)