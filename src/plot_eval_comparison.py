"""
plot_eval_comparison.py

Loads two (or more) eval_results_*.json files (produced by eval.py) and
plots a side-by-side bar chart comparing PSNR, SSIM, and FID.

Usage:
    python src/plot_eval_comparison.py \
        --results outputs/eval_results_stage1_baseline.json outputs/eval_results_gan_stage2.json \
        --labels "Stage 1 (U-Net)" "Stage 2 (GAN)" \
        --output outputs/eval_comparison.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(result_paths, labels, output_path):
    results = []
    for p in result_paths:
        with open(p) as f:
            results.append(json.load(f))

    metrics = ["mean_psnr_db", "mean_ssim", "fid"]
    titles = ["PSNR (dB) \u2191 higher better", "SSIM \u2191 higher better", "FID \u2193 lower better"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, metric, title in zip(axes, metrics, titles):
        values = [r[metric] for r in results]
        bars = ax.bar(labels, values, color=colors[: len(labels)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    print(f"Saved comparison chart to {output_path}")

    print("\n--- Summary ---")
    print(f"{'Model':<20}{'PSNR (dB)':<12}{'SSIM':<10}{'FID':<10}")
    for label, r in zip(labels, results):
        print(f"{label:<20}{r['mean_psnr_db']:<12.2f}{r['mean_ssim']:<10.4f}{r['fid']:<10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/eval_comparison.png")
    args = parser.parse_args()

    assert len(args.results) == len(args.labels), "Need one label per result file"
    plot_comparison(args.results, args.labels, args.output)