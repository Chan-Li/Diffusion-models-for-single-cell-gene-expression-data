"""
Train a latent diffusion model on PBMC3k PCA embeddings.

Pipeline:
1) Load PBMC3k with Scanpy
2) Preprocess data (QC, normalization, HVG, PCA)
3) Train a DDPM-style diffusion model in latent space
4) Sample new latent vectors

Usage:
    python train_latent_diffusion.py
"""

import torch
import scanpy as sc

from preprocessing import preprocess_and_pca, PreprocessConfig
from model import build_latent_ddpm


def main():
    # -------------------------------------------------
    # 1. Load dataset
    # -------------------------------------------------
    print("Loading PBMC3k dataset...")
    adata = sc.datasets.pbmc3k()

    # -------------------------------------------------
    # 2. Preprocess + PCA
    # -------------------------------------------------
    print("Preprocessing data and computing PCA latent...")
    cfg = PreprocessConfig(
        n_pcs=50,
        max_pct_counts_mt=5.0,
        max_genes_by_counts=2500,
    )

    adata_p, Z = preprocess_and_pca(adata, cfg)

    print("Processed AnnData:")
    print(adata_p)
    print("Latent shape:", Z.shape)  # (n_cells, 50)

    # -------------------------------------------------
    # 3. Prepare data for PyTorch
    # -------------------------------------------------
    Z_torch = torch.tensor(Z, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Z_torch = Z_torch.to(device)

    # -------------------------------------------------
    # 4. Build diffusion model
    # -------------------------------------------------
    print("Building diffusion model...")
    model = build_latent_ddpm(
        z_dim=Z_torch.shape[1],
        T=500,                 # diffusion steps
        schedule="cosine",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------
    print("Starting training...")
    n_steps = 2000
    batch_size = 128

    for step in range(n_steps):
        # Sample a random minibatch of cells
        idx = torch.randint(0, Z_torch.shape[0], (batch_size,))
        batch = Z_torch[idx]

        loss = model.training_loss(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step:04d} | loss = {loss.item():.6f}")

    print("Training finished.")

    # -------------------------------------------------
    # 6. Sampling
    # -------------------------------------------------
    print("Sampling new latent vectors...")
    with torch.no_grad():
        Z_gen = model.sample(
            n=1000,
            z_dim=Z_torch.shape[1],
            device=device,
        )

    print("Generated latent shape:", Z_gen.shape)

    # Optional: save generated latents
    torch.save(Z_gen.cpu(), "pbmc3k_latent_generated.pt")
    print("Saved generated latents to pbmc3k_latent_generated.pt")


if __name__ == "__main__":
    main()
