# Assignment 3 - Variational Autoencoders (VAEs)

This repository contains the code, documentation, and resources for Assignment 3 of CS552 - Generative AI.

## Course Information

**Course:** CS552 - Generative AI  

## Repository Structure

```
Assignment-3/
├── code.ipynb          # Main notebook with theory questions and coding tasks
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules (excludes data/ folder)
├── 032---Assignment-3---Variational-Autoencoders.pdf
└── data/               # Dataset files (gitignored)
```

## Assignment 3: Variational Autoencoders (VAEs)

This assignment focuses on implementing and understanding Variational Autoencoders (VAEs) for image generation.

### Contents

- **Theory Questions**: Four comprehensive questions covering:
  - KL Divergence in VAE loss function
  - Reparameterization trick
  - Probabilistic vs fixed latent space
  - KL Divergence and smooth latent space

- **Coding Tasks**:
  1. **Convolutional VAE on CIFAR-10**: Implementation of a convolutional VAE architecture for the CIFAR-10 dataset
  2. **Latent Space Interpolation**: Visualizing smooth transitions between images in the latent space
  3. **New Dataset Application**: Applying the VAE to CelebA dataset and analyzing generated samples

### Key Features

- Convolutional encoder and decoder architectures
- Reparameterization trick implementation
- KL divergence regularization
- Latent space visualization and interpolation
- Image generation from different latent space regions

## Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Installation

1. Navigate to the Assignment 3 directory:
```bash
cd Assignment-3
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `code.ipynb` and run the cells.

## Dependencies

- `torch` >= 2.0.0 - PyTorch deep learning framework
- `torchvision` >= 0.15.0 - Computer vision utilities for PyTorch
- `matplotlib` >= 3.7.0 - Plotting and visualization
- `numpy` >= 1.24.0 - Numerical computing
- `jupyter` >= 1.0.0 - Jupyter notebook environment
- `notebook` >= 6.5.0 - Jupyter notebook server

## Data

Dataset files are stored in the `data/` directory. This directory is excluded from version control via `.gitignore` to keep the repository size manageable.

For CIFAR-10, the dataset will be automatically downloaded when running the notebook.

For CelebA or other custom datasets, place them in the `data/` directory.

## Notes

- The notebook is designed to run on GPU if available, but will fall back to CPU
- Training times vary based on hardware and dataset size
- Model checkpoints and generated images are saved during execution
