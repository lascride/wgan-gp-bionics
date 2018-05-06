Structural Bionics and its Applications Based on GAN
=====================================

Code for wgan-gp and iGAN in tensorflow, and their applications in Structural Bionics.


## Prerequisites

- Python, NumPy, TensorFlow, SciPy, Matplotlib
- A recent NVIDIA GPU

## Training Models
- `python gan_bionics.py`:  Train wgan-gp on bamboo cross-section dataset.

## Optimization With User Constraints
- `python -u opt.py --restore_index 179999 --input_color_name blank --ITERS 100 --BATCH_SIZE 64  --input_edge_name raw_edge5
`:  Optimization with user constraints(color,sketch).