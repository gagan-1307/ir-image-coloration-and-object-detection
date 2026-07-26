"""
downstream_detection_test.py

Downstream task test: does colorization actually help object detection?

Runs a pretrained YOLOv8 detector (no training -- inference only) on three
versions of the same tile:
    1. Raw IR (thermal channel replicated to 3-channel grayscale)
    2. Your model's colorized output
    3. Real RGB (ground truth, upper bound reference)

Compares number of detections and average confidence across all three,
and saves an annotated side-by-side image grid.

Note: YOLOv8 here is pretrained on COCO (everyday objects: cars, people,
trucks, etc.), not on satellite-specific classes. It won't reliably detect
"building" or "road" by name -- but it's still a valid, standard proxy for
"how much usable visual/structural information does a detector's backbone
extract from this image," which is the point of this test: more confident,
more numerous detections indicate the image carries more recognizable
structure for downstream CV pipelines in general.

Usage:
    python src/downstream_detection_test.py \
        --processed_dir /content/local_data/processed \
        --checkpoint /content/drive/MyDrive/ir-colorization-checkpoints/best_semantic_generator.pth \
        --output_dir /content/drive/MyDrive/ir-colorization-outputs \
        --n_samples 6
"""

import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from dataset import IRColorizationDataset
from model import UNetGenerator


def denorm(x):
    return (x + 1) / 2


def to_uint8_hwc(tensor_chw):
    arr = tensor_chw.permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")

    full_dataset = IRColorizationDataset(args.processed_dir)
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    _train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    model = UNetGenerator(in_channels=4, out_channels=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    n_samples = min(args.n_samples, len(val_ds))
    idxs = np.random.RandomState(0).choice(len(val_ds), n_samples, replace=False)

    results_summary = {"raw_ir": [], "colorized": [], "real_rgb": []}
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 3)

    for row, i in enumerate(idxs):
        ir, rgb, _idx = val_ds[i]

        # Version 1: raw IR (thermal channel replicated to 3-channel)
        thermal = ir[0].numpy()
        raw_ir_img = np.stack([thermal] * 3, axis=-1)
        raw_ir_img = (np.clip(raw_ir_img, 0, 1) * 255).astype(np.uint8)

        # Version 2: model's colorized output
        with torch.no_grad():
            pred = model(ir.unsqueeze(0).to(device)).squeeze(0).cpu()
        colorized_img = to_uint8_hwc(denorm(pred))

        # Version 3: real RGB
        real_img = to_uint8_hwc(rgb)

        for col, (name, img) in enumerate([
            ("raw_ir", raw_ir_img), ("colorized", colorized_img), ("real_rgb", real_img)
        ]):
            img = np.ascontiguousarray(img)
            det = yolo.predict(img, verbose=False, conf=0.15)[0]
            n_det = len(det.boxes)
            mean_conf = float(det.boxes.conf.mean()) if n_det > 0 else 0.0
            results_summary[name].append({"n_detections": n_det, "mean_conf": mean_conf})

            annotated = det.plot()  # BGR annotated image with boxes drawn
            axes[row, col].imshow(annotated[:, :, ::-1])  # BGR -> RGB for matplotlib
            axes[row, col].set_title(f"{name} ({n_det} det, conf={mean_conf:.2f})", fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "downstream_detection_grid.png")
    plt.savefig(fig_path, dpi=100)
    print(f"Saved detection comparison grid to {fig_path}")

    # Aggregate summary
    summary = {}
    for name, entries in results_summary.items():
        avg_n_det = float(np.mean([e["n_detections"] for e in entries]))
        avg_conf = float(np.mean([e["mean_conf"] for e in entries]))
        summary[name] = {"avg_n_detections": avg_n_det, "avg_confidence": avg_conf}
        print(f"{name:12s} | avg detections: {avg_n_det:.2f} | avg confidence: {avg_conf:.3f}")

    results_path = os.path.join(args.output_dir, "downstream_detection_summary.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--n_samples", type=int, default=6)
    args = parser.parse_args()

    run_test(args)