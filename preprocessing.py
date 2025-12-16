# scrna_preprocess.py
"""
Preprocess scRNA-seq AnnData and extract PCA latent space (no skmisc required).

Usage:
    import scanpy as sc
    from scrna_preprocess import preprocess_and_pca, PreprocessConfig

    adata = sc.datasets.pbmc3k()  # or your loaded AnnData
    adata_p, Z = preprocess_and_pca(adata, PreprocessConfig(n_pcs=50))
    # Z: (n_cells, n_pcs) numpy array
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData


@dataclass
class PreprocessConfig:
    # QC / filtering
    min_cells_per_gene: int = 3
    max_genes_by_counts: int = 2500
    max_pct_counts_mt: float = 5.0

    # Normalization / HVG
    target_sum: float = 1e4
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat"  # NOTE: seurat_v3 requires skmisc; we avoid it.

    # PCA
    n_pcs: int = 50
    scale_max_value: float = 10.0
    pca_solver: str = "arpack"

    # Optional neighbors graph (for UMAP/leiden later)
    build_neighbors: bool = False
    n_neighbors: int = 10


def preprocess_and_pca(
    adata: AnnData,
    config: Optional[PreprocessConfig] = None,
    *,
    copy: bool = True,
) -> Tuple[AnnData, np.ndarray]:
    """
    Preprocess AnnData and return (processed_adata, PCA_latent).

    Steps:
      1) Compute QC metrics (+ mito genes)
      2) Filter cells/genes
      3) Normalize + log1p
      4) Select HVGs (seurat flavor, no skmisc)
      5) Scale + PCA
      6) (Optional) build neighbors graph

    Returns:
      processed_adata: AnnData with adata.obsm["X_pca"]
      Z: numpy array (n_cells, n_pcs) from X_pca
    """
    if config is None:
        config = PreprocessConfig()

    ad = adata.copy() if copy else adata

    # ---- 1) QC metrics (mitochondrial genes) ----
    # Human convention: "MT-". If you ever use mouse data, replace with "mt-".
    ad.var["mt"] = ad.var_names.str.startswith("MT-")

    sc.pp.calculate_qc_metrics(
        ad,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # ---- 2) Filter cells / genes ----
    if config.max_genes_by_counts is not None:
        ad = ad[ad.obs["n_genes_by_counts"] < config.max_genes_by_counts, :].copy()

    if config.max_pct_counts_mt is not None:
        ad = ad[ad.obs["pct_counts_mt"] < config.max_pct_counts_mt, :].copy()

    if config.min_cells_per_gene and config.min_cells_per_gene > 0:
        sc.pp.filter_genes(ad, min_cells=config.min_cells_per_gene)

    # ---- 3) Normalize + log ----
    sc.pp.normalize_total(ad, target_sum=config.target_sum)
    sc.pp.log1p(ad)

    # Save log-normalized (pre-HVG) as raw for later marker plots
    ad.raw = ad

    # ---- 4) HVG selection (no skmisc) ----
    sc.pp.highly_variable_genes(
        ad,
        n_top_genes=config.n_top_genes,
        flavor=config.hvg_flavor,
    )
    ad = ad[:, ad.var["highly_variable"]].copy()

    # ---- 5) Scale + PCA ----
    sc.pp.scale(ad, max_value=config.scale_max_value)
    sc.tl.pca(ad, svd_solver=config.pca_solver)

    if "X_pca" not in ad.obsm:
        raise RuntimeError('PCA failed: ad.obsm["X_pca"] not found.')

    Z = np.asarray(ad.obsm["X_pca"][:, : config.n_pcs])

    # ---- 6) Optional neighbors graph ----
    if config.build_neighbors:
        sc.pp.neighbors(ad, n_neighbors=config.n_neighbors, n_pcs=min(config.n_pcs, Z.shape[1]))

    return ad, Z
