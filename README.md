# Diffusion-models-for-single-cell-gene-expression-data
We train a denoising diffusion probabilistic model on single-cell RNA-seq data (PBMC3k) to learn a generative model of gene-expression states, and show that generated cells recover realistic cell-type structure and marker-gene profiles.
# PBMC3k Single-Cell RNA-seq Analysis

This repository provides an exploratory analysis of the **PBMC3k** single-cell RNA-seq dataset using **Scanpy (v1.11.5)**.  
The goal is to understand the structure of scRNA-seq data—from raw UMI counts to low-dimensional embeddings and clustering—using a transparent and standard Scanpy workflow.

Some analyses based on this dataset are conducted in **`data_import.ipynb`**.

---

## Dataset: PBMC3k

**PBMC3k** is a canonical benchmark dataset for single-cell RNA sequencing. 

- **Source**: 10x Genomics
- **Cells**: ~3,000 peripheral blood mononuclear cells (PBMCs)
- **Technology**: UMI-based scRNA-seq

The dataset contains 2700 cells and 32738 genes, stored in a Scanpy AnnData object.

PBMCs include several major immune cell populations:
- T cells
- B cells
- NK cells
- Monocytes
- Platelets

Due to its moderate size and well-characterized biology, PBMC3k is widely used for:
- Method development
- Educational purposes
- Benchmarking single-cell analysis pipelines


## Preprocessing and Latent Representation

Raw scRNA-seq measurements are high-dimensional, sparse, and noisy.  
To obtain a suitable continuous representation, we apply the following preprocessing steps:

1. **Quality control (QC)**
   - Remove cells with high mitochondrial gene expression
   - Filter genes expressed in very few cells

2. **Normalization**
   - Library-size normalization
   - Logarithmic transformation

3. **Feature selection**
   - Selection of highly variable genes (HVGs)

4. **Dimensionality reduction**
   - Principal Component Analysis (PCA)

After preprocessing, each cell is represented by a continuous latent vector:

$$
\mathbf{z}_i \in \mathbb{R}^d
$$

which captures major biological variation across the population.

This latent space serves as the input domain for diffusion modeling.

---

## Latent Diffusion Model

We adopt a **Denoising Diffusion Probabilistic Model (DDPM)** operating on latent vectors.

### Forward (Noising) Process

The forward diffusion process gradually perturbs the latent representation by adding Gaussian noise:

$$
\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},
\quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

This process progressively destroys biological structure in a controlled manner.

---

### Reverse (Denoising) Process

A neural network is trained to predict the injected noise given a noisy latent vector and diffusion time:

$$
\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t) \approx \boldsymbol{\epsilon}
$$

The training objective minimizes the mean squared error:

$$
\mathcal{L} = \mathbb{E}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)\|^2\right]
$$

Sampling is performed by iteratively reversing the diffusion process, starting from isotropic Gaussian noise.

---

## Model Architecture

The noise-prediction network is implemented as a **multi-layer perceptron (MLP)** with:

- Input: latent vector \( \mathbf{z}_t \in \mathbb{R}^d \)
- Sinusoidal embedding of diffusion time
- Several fully connected layers with nonlinear activations
- Output: predicted noise vector in latent space

Given the low dimensionality of the latent representation, this architecture is sufficient to model the diffusion dynamics without over-parameterization.

---

## Training Procedure

- Latent vectors are sampled uniformly from the dataset
- Diffusion timesteps are sampled uniformly
- The model is optimized using Adam
- Training minimizes the standard DDPM noise-prediction loss

GPU acceleration is supported via PyTorch.

---

## Sampling and Generation

After training, the model generates new latent cell states by:

1. Sampling from a standard Gaussian prior
2. Iteratively applying the learned reverse diffusion steps

The generated latent vectors can be:
- Compared against real data in latent space
- Visualized using UMAP or PCA
- Used for downstream biological or statistical analysis

---

## Interpretation and Use Cases

The latent diffusion model learns a smooth probability distribution over cell states, which can be interpreted as:

- A continuous landscape of cellular phenotypes
- A generative model for synthetic cell populations
- A framework for studying transitions and interpolations between cell types

This approach naturally aligns with perspectives from statistical physics and generative modeling.

---

## Implementation Notes

- Diffusion is performed in latent space, not gene space
- No assumptions are made about discrete count distributions
- The pipeline is modular: preprocessing and diffusion components are decoupled
- The framework can be extended to conditional diffusion using cell-type annotations

---
## Main Script (`main.py`)

The `main.py` script serves as an **end-to-end executable entry point** for the latent diffusion pipeline.

Its responsibilities include:

1. **Data loading**
   - Load the PBMC3k dataset via Scanpy (or a user-provided AnnData object)

2. **Preprocessing**
   - Apply quality control, normalization, highly variable gene selection, and PCA
   - Construct a continuous latent representation of cell states

3. **Model construction**
   - Initialize a DDPM-style latent diffusion model
   - Configure diffusion schedule and network architecture

4. **Training**
   - Sample mini-batches of latent vectors
   - Optimize the diffusion objective using stochastic gradient descent (Adam)
   - Monitor training loss

5. **Sampling**
   - Generate new latent cell states by reverse diffusion
   - Save generated samples for downstream analysis and visualization

The script is designed to be:
- **Modular**: core logic is delegated to preprocessing and model modules
- **Reproducible**: all hyperparameters are defined explicitly
- **Extensible**: can be adapted to other datasets or conditional diffusion settings

Running `main.py` reproduces the full pipeline from raw data to generated latent samples with minimal user intervention.


## Summary

This implementation demonstrates how diffusion models can be effectively applied to single-cell data by operating on carefully constructed latent representations.  
The resulting model provides a flexible and interpretable generative framework for analyzing cell populations.




