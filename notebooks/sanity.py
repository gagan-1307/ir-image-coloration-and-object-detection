import numpy as np
import matplotlib.pyplot as plt
import glob

files = sorted(glob.glob("data/processed/*.npz"))
print(f"Total tiles: {len(files)}")

# Pick a few random tiles to inspect
sample_files = np.random.choice(files, 4, replace=False)

fig, axes = plt.subplots(4, 2, figsize=(6, 12))
for i, f in enumerate(sample_files):
    data = np.load(f)
    ir = data["ir"]     # (4, H, W): thermal, nir, swir1, swir2
    rgb = data["rgb"]   # (3, H, W): red, green, blue

    # Show thermal channel as grayscale, and RGB as color
    axes[i, 0].imshow(ir[0], cmap="gray")
    axes[i, 0].set_title("Thermal (IR input)")
    axes[i, 0].axis("off")

    rgb_img = np.transpose(rgb, (1, 2, 0))  # (H, W, 3) for matplotlib
    axes[i, 1].imshow(rgb_img)
    axes[i, 1].set_title("RGB target")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig("outputs/sanity_check.png", dpi=100)
plt.show()
print("Saved to outputs/sanity_check.png")