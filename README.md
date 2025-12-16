# Diffusion-models-for-single-cell-gene-expression-data
We train a denoising diffusion probabilistic model on single-cell RNA-seq data (PBMC3k) to learn a generative model of gene-expression states, and show that generated cells recover realistic cell-type structure and marker-gene profiles.
# PBMC3k Single-Cell RNA-seq Analysis

This repository provides an exploratory analysis of the **PBMC3k** single-cell RNA-seq dataset using **Scanpy (v1.11.5)**.  
The goal is to understand the structure of scRNA-seq data—from raw UMI counts to low-dimensional embeddings and clustering—using a transparent and standard Scanpy workflow.

All analyses and experiments are conducted in **`data_import.ipynb`**.

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



