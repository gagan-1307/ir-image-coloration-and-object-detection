# IR Image Colorization & Enhancement

Deep learning pipeline that takes single/multi-band infrared satellite imagery
(Landsat 8/9) and outputs a sharpened, realistically colorized RGB image.

## Project Structure
```
project/
├── data/           # raw + processed tiles (not committed to git)
├── notebooks/       # Colab exploration notebooks
├── src/            # dataset.py, model.py, train.py, eval.py
├── checkpoints/    # saved model weights (not committed to git)
└── outputs/        # generated images, comparison grids
```

## Setup
```bash
pip install -r requirements.txt
```

## Status
- [ ] Phase 0: Repo & environment setup
- [ ] Phase 1: Data acquisition
- [ ] Phase 2: Preprocessing
- [ ] Phase 3: Baseline U-Net
- [ ] Phase 4: GAN (PatchGAN discriminator)
- [ ] Phase 5: Semantic constraint loss
- [ ] Phase 6: Sharpening/enhancement module
- [ ] Phase 7: Evaluation
- [ ] Phase 8: Demo / packaging
