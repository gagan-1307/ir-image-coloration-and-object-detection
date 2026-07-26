"""
inference.py

Standalone inference: load one checkpoint, run it on a single tile
(or all tiles in a folder), report per-tile inference time.

This is the "productized" entry point -- what a downstream user/system
would actually call, separate from all the training scripts.

Usage (single tile from your processed .npz set):
    python src/inference.py \
        --checkpoint /content/drive/MyDrive/ir-colorization-checkpoints/best_semantic_generator.pth \
        --input_npz /content/local_data/processed/scene_01_tile0005.npz \
        --output_path outputs/single_inference.png

Usage (benchmark timing over N random tiles):
    python src/inference.py \
        --checkpoint /content/drive/MyDrive/ir-colorization-checkpoints/best_semantic_generator.pth \
        --processed_dir /content/local_data/processed \
        --benchmark_n 20
"""

import os
import argparse
import time
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import UNetGenerator


def denorm(x):
    return (x + 1) / 2


def load_model(checkpoint_path, device):
    model = UNetGenerator(in_channels=4, out_channels=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_single(model, npz_path, output_path, device):
    data = np.load(npz_path)
    ir = torch.from_numpy(data["ir"]).float().unsqueeze(0).to(device)  # (1, 4, H, W)
    real_rgb = data["rgb"]  # (3, H, W), for display only

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    with torch.no_grad():
        pred = model(ir)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed_ms = (time.time() - start) * 1000

    pred_img = denorm(pred.squeeze(0).cpu()).permute(1, 2, 0).numpy()
    pred_img = np.clip(pred_img, 0, 1)
    real_img = np.transpose(real_rgb, (1, 2, 0))
    thermal_img = data["ir"][0]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(thermal_img, cmap="gray")
    axes[0].set_title("IR input")
    axes[0].axis("off")
    axes[1].imshow(pred_img)
    axes[1].set_title(f"Colorized output\n({elapsed_ms:.1f} ms)")
    axes[1].axis("off")
    axes[2].imshow(real_img)
    axes[2].set_title("Real RGB (reference)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=110)

    print(f"Inference time: {elapsed_ms:.2f} ms")
    print(f"Saved result to {output_path}")
    return elapsed_ms


def run_benchmark(model, processed_dir, n, device):
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    sample_files = np.random.choice(files, min(n, len(files)), replace=False)

    times = []
    for f in sample_files:
        data = np.load(f)
        ir = torch.from_numpy(data["ir"]).float().unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(ir)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)

    times = np.array(times)
    print(f"Benchmarked on {len(times)} tiles ({device}):")
    print(f"  Mean:   {times.mean():.2f} ms/tile")
    print(f"  Median: {np.median(times):.2f} ms/tile")
    print(f"  Min:    {times.min():.2f} ms/tile")
    print(f"  Max:    {times.max():.2f} ms/tile")
    print(f"  Throughput: {1000 / times.mean():.2f} tiles/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_npz", default=None, help="Single .npz tile to run inference on")
    parser.add_argument("--output_path", default="outputs/single_inference.png")
    parser.add_argument("--processed_dir", default=None, help="Folder of .npz tiles, for benchmarking")
    parser.add_argument("--benchmark_n", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, device)

    if args.input_npz:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        run_single(model, args.input_npz, args.output_path, device)
    elif args.processed_dir:
        run_benchmark(model, args.processed_dir, args.benchmark_n, device)
    else:
        print("Provide either --input_npz (single tile) or --processed_dir (benchmark mode).")