Structural Bionics and its Applications Based on GAN
=====================================

Code for wgan-gp and iGAN in tensorflow, and their applications in Structural Bionics.


## Prerequisites

- Python, NumPy, TensorFlow, SciPy, Matplotlib
- A recent NVIDIA GPU

## Instruction

- gan_bionics.py: Step 1. Learning Approximate Manifold.
- opt.py: Step 2: Solving User Constraints.

- gan_bionics_light.py: Using basic tensorflow functions.


- eigener.py: Training a CNN to predict eigenvalue.
- gan_eigen_alter.py: Learning joint distribution (adding eigenvalue) with GAN.
- opt_eigen.py: Solving user constraints including eigenvalue.

- LGAN_bionics.py: A tensorflow implementation to train LGAN.


- tf_hog.py: Calculating HOG Feature.



## Training Models
- `python gan_bionics.py`:  Train wgan-gp on bamboo cross-section dataset.

## Optimization With User Constraints
- `python -u opt.py --restore_index 249999 --input_color_name blank --ITERS 100 --BATCH_SIZE 64  --input_edge_name raw_edge5
`:  Optimization with user constraints(color,sketch).

