"""
dataset.py

Handles loading Landsat 8/9 scenes, computing spectral indices (NDVI/NDWI/NDBI),
tiling large scenes into fixed-size patches, and serving them as a PyTorch Dataset.

Expected raw data layout (see README "Data acquisition" section for how to get this):

    data/raw/<scene_id>/
        B2.TIF   # Blue
        B3.TIF   # Green
        B4.TIF   # Red
        B5.TIF   # NIR
        B6.TIF   # SWIR1
        B7.TIF   # SWIR2
        B8.TIF   # Panchromatic (15m)
        B10.TIF  # Thermal (TIRS)

Each <scene_id> folder = one Landsat scene download (all bands from the same
product, so they're already co-registered to the same grid).
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
from torch.utils.data import Dataset


BAND_FILES = {
    "blue": "B2.TIF",
    "green": "B3.TIF",
    "red": "B4.TIF",
    "nir": "B5.TIF",
    "swir1": "B6.TIF",
    "swir2": "B7.TIF",
    "pan": "B8.TIF",
    "thermal": "B10.TIF",
}


def read_band(path, target_shape=None):
    """Read a single-band GeoTIFF as a float32 numpy array, optionally
    resampling to a target (H, W) shape so all bands share one grid."""
    with rasterio.open(path) as src:
        if target_shape is None:
            arr = src.read(1).astype(np.float32)
        else:
            arr = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.bilinear,
            ).astype(np.float32)
    return arr


def normalize(arr, low_pct=2, high_pct=98):
    """Percentile-based normalization to [0, 1]. Robust to outlier pixels
    (common in satellite imagery from sensor noise/saturation)."""
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def compute_indices(red, green, nir, swir1):
    """Compute NDVI (vegetation), NDWI (water), NDBI (built-up) masks.
    These are used later as a semantic constraint during training, so the
    model doesn't hallucinate colors that contradict the actual land cover.
    """
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir1 - nir) / (swir1 + nir + eps)
    return ndvi, ndwi, ndbi


def load_scene(scene_dir):
    """Load all bands for one scene, resampled to the thermal band's
    native shape (coarsest), and return a dict of normalized arrays
    + computed indices."""
    thermal_path = os.path.join(scene_dir, BAND_FILES["thermal"])
    with rasterio.open(thermal_path) as src:
        target_shape = (src.height, src.width)

    bands = {}
    for key, fname in BAND_FILES.items():
        path = os.path.join(scene_dir, fname)
        bands[key] = read_band(path, target_shape=target_shape)

    ndvi, ndwi, ndbi = compute_indices(
        bands["red"], bands["green"], bands["nir"], bands["swir1"]
    )

    # IR input stack: thermal + NIR + SWIR1 + SWIR2 (4 channels)
    ir_stack = np.stack(
        [
            normalize(bands["thermal"]),
            normalize(bands["nir"]),
            normalize(bands["swir1"]),
            normalize(bands["swir2"]),
        ],
        axis=0,
    )

    # RGB target (3 channels)
    rgb_stack = np.stack(
        [
            normalize(bands["red"]),
            normalize(bands["green"]),
            normalize(bands["blue"]),
        ],
        axis=0,
    )

    index_stack = np.stack([ndvi, ndwi, ndbi], axis=0).astype(np.float32)

    return ir_stack, rgb_stack, index_stack


def tile_array(arr, tile_size=256, stride=256):
    """Split a (C, H, W) array into a list of (C, tile_size, tile_size) tiles.
    Drops any partial tiles at the edges for simplicity."""
    c, h, w = arr.shape
    tiles = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tiles.append(arr[:, y : y + tile_size, x : x + tile_size])
    return tiles


def build_tile_dataset(raw_data_dir, processed_dir, tile_size=256, stride=256):
    """Walk every scene folder under raw_data_dir, tile it, and save each
    tile as a .npz file under processed_dir. Run this ONCE as a preprocessing
    step before training (see train.py)."""
    os.makedirs(processed_dir, exist_ok=True)
    scene_dirs = sorted(
        d for d in glob.glob(os.path.join(raw_data_dir, "*")) if os.path.isdir(d)
    )

    if not scene_dirs:
        print(f"No scene folders found in {raw_data_dir}. "
              f"Download Landsat scenes first (see README).")
        return

    tile_count = 0
    for scene_dir in scene_dirs:
        scene_id = os.path.basename(scene_dir)
        try:
            ir_stack, rgb_stack, index_stack = load_scene(scene_dir)
        except Exception as e:
            print(f"Skipping {scene_id}: {e}")
            continue

        ir_tiles = tile_array(ir_stack, tile_size, stride)
        rgb_tiles = tile_array(rgb_stack, tile_size, stride)
        idx_tiles = tile_array(index_stack, tile_size, stride)

        for i, (ir_t, rgb_t, idx_t) in enumerate(zip(ir_tiles, rgb_tiles, idx_tiles)):
            out_path = os.path.join(processed_dir, f"{scene_id}_tile{i:04d}.npz")
            np.savez_compressed(out_path, ir=ir_t, rgb=rgb_t, idx=idx_t)
            tile_count += 1

        print(f"{scene_id}: {len(ir_tiles)} tiles saved")

    print(f"Done. {tile_count} total tiles saved to {processed_dir}")


class IRColorizationDataset(Dataset):
    """PyTorch Dataset that loads pre-tiled .npz files produced by
    build_tile_dataset()."""

    def __init__(self, processed_dir, file_list=None):
        self.processed_dir = processed_dir
        if file_list is not None:
            self.files = file_list
        else:
            self.files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npz tiles found in {processed_dir}. "
                f"Run build_tile_dataset() first."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        ir = torch.from_numpy(data["ir"]).float()      # (4, H, W)
        rgb = torch.from_numpy(data["rgb"]).float()     # (3, H, W)
        indices = torch.from_numpy(data["idx"]).float() # (3, H, W) NDVI/NDWI/NDBI
        return ir, rgb, indices


if __name__ == "__main__":
    # Quick manual test / preprocessing entry point.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    build_tile_dataset(args.raw_dir, args.processed_dir, args.tile_size, args.stride)