

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: schedules + embeddings
# -----------------------------

def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal (2021), adapted to discrete timesteps.
    Returns betas of shape (T,).
    """
    # alpha_bar(t) = cos^2( (t/T + s) / (1+s) * pi/2 )
    steps = torch.arange(T + 1, dtype=torch.float64)
    t = steps / T
    alpha_bar = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]

    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clamp(betas, 1e-8, 0.999)  # numerical safety
    return betas.to(torch.float32)


def _linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """
    Linear schedule from beta_start to beta_end.
    Returns betas of shape (T,).
    """
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for discrete timesteps t in [0, T-1].

    Produces a vector emb(t) in R^{dim}. This is similar to transformer positional encoding.
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even for sinusoidal embedding.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: int64 tensor of shape (B,) with values in [0, T-1]
        returns: float tensor of shape (B, dim)
        """
        half = self.dim // 2
        # frequencies: exp(-log(10000) * k / (half-1))
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )  # (half,)

        # t is (B,). Convert to float and outer product -> (B, half)
        args = t.to(torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)
        return emb


# -----------------------------
# Epsilon model: MLP for latent vectors
# -----------------------------

class EpsMLP(nn.Module):
    """
    A simple MLP epsilon-predictor for latent diffusion.
    Input:  z_t (B, d), timestep t (B,)
    Output: eps_hat (B, d)
    """
    def __init__(
        self,
        z_dim: int,
        time_emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

        # Project time embedding to same scale as hidden representation
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
        )

        # First layer takes z plus time-conditioned bias
        layers = []
        in_dim = z_dim
        for layer_idx in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, z_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z_t: (B, z_dim)
        t:   (B,) int64
        """
        # Compute time embedding and project to hidden_dim
        te = self.time_proj(self.time_embed(t))  # (B, hidden_dim)

        # A simple conditioning trick:
        # Push z_t through MLP, but inject time embedding as an additive bias at input stage.
        # (Works well for small latent vectors; for larger models you could use FiLM.)
        h = self.net[0](z_t)  # first Linear
        h = h + te            # time conditioning
        # Continue remaining layers (starting from activation after first Linear)
        h = F.silu(h)

        # Manually run remaining layers in self.net (we already used the first Linear).
        # self.net = [Linear, SiLU, (Dropout), Linear, SiLU, ...]
        # We consumed index 0 (Linear). Next index is 1 (SiLU) which we replaced with F.silu above.
        # So we start from index 2.
        for layer in self.net[2:]:
            h = layer(h)

        return self.out(h)


# -----------------------------
# DDPM wrapper for training + sampling
# -----------------------------

@dataclass
class DiffusionConfig:
    T: int = 1000
    schedule: Literal["cosine", "linear"] = "cosine"

    # linear schedule params (used only if schedule="linear")
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # sampling params
    clip_denoised: bool = False  # if True, clamp predicted z0 to a range (often unnecessary for PCA)
    z0_clip_range: float = 5.0   # used when clip_denoised=True


class LatentDDPM(nn.Module):
    """
    DDPM diffusion process for latent vectors.
    Provides:
      - training_loss(z0): returns scalar loss
      - sample(n, z_dim): generate new latent vectors
    """
    def __init__(self, eps_model: nn.Module, cfg: Optional[DiffusionConfig] = None):
        super().__init__()
        self.eps_model = eps_model
        self.cfg = cfg or DiffusionConfig()

        betas = self._make_betas(self.cfg)
        # Register as buffers so they move with .to(device) and are saved in state_dict
        self.register_buffer("betas", betas)  # (T,)

        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)  # (T,)

        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)  # (T,)

        # Precompute sqrt terms used frequently
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        # For reverse step posterior variance (DDPM):
        # posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=betas.device), alpha_bar[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("posterior_var", posterior_var.clamp(min=1e-20))

    @staticmethod
    def _make_betas(cfg: DiffusionConfig) -> torch.Tensor:
        if cfg.schedule == "cosine":
            return _cosine_beta_schedule(cfg.T)
        if cfg.schedule == "linear":
            return _linear_beta_schedule(cfg.T, cfg.beta_start, cfg.beta_end)
        raise ValueError(f"Unknown schedule: {cfg.schedule}")

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample z_t from q(z_t | z0):
            z_t = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * eps
        """
        if eps is None:
            eps = torch.randn_like(z0)

        # Gather per-sample coefficients for given t
        # sqrt_alpha_bar[t] has shape (T,), we need (B, 1)
        s1 = self.sqrt_alpha_bar[t].unsqueeze(1)
        s2 = self.sqrt_one_minus_alpha_bar[t].unsqueeze(1)
        return s1 * z0 + s2 * eps

    def training_loss(self, z0: torch.Tensor) -> torch.Tensor:
        """
        Compute DDPM MSE loss for a batch of clean latent vectors z0.

        z0: (B, d)
        """
        B = z0.shape[0]
        device = z0.device

        # Sample timesteps uniformly from [0, T-1]
        t = torch.randint(0, self.cfg.T, (B,), device=device, dtype=torch.long)

        # Sample noise and construct z_t
        eps = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, eps=eps)

        # Predict noise
        eps_hat = self.eps_model(z_t, t)

        # Standard DDPM objective
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def p_sample_step(self, z_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        One reverse step: sample z_{t-1} from p_theta(z_{t-1} | z_t)

        DDPM reverse mean:
            mu_theta = 1/sqrt(alpha_t) * ( z_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta(z_t, t) )

        Then add noise with variance posterior_var[t] if t > 0.
        """
        B, d = z_t.shape
        device = z_t.device
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]

        eps_hat = self.eps_model(z_t, t_batch)

        # Mean of reverse distribution
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mu = coef1 * (z_t - coef2 * eps_hat)

        if t == 0:
            return mu  # final step: no noise

        # Sample noise for stochasticity
        noise = torch.randn_like(z_t)
        var = self.posterior_var[t]
        return mu + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n: int, z_dim: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate n samples in R^{z_dim} by reverse diffusion.

        Returns:
            z0_hat: (n, z_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        # Start from pure noise
        z_t = torch.randn(n, z_dim, device=device)

        # Reverse diffusion loop: T-1 -> 0
        for t in reversed(range(self.cfg.T)):
            z_t = self.p_sample_step(z_t, t)

            # Optional: clamp the denoised estimate to avoid rare explosions
            if self.cfg.clip_denoised and t == 0:
                z_t = z_t.clamp(-self.cfg.z0_clip_range, self.cfg.z0_clip_range)

        return z_t


# -----------------------------
# Convenience: build a ready-to-train model
# -----------------------------

def build_latent_ddpm(
    z_dim: int,
    *,
    T: int = 1000,
    schedule: Literal["cosine", "linear"] = "cosine",
    time_emb_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.0,
) -> LatentDDPM:
    """
    Factory to build a LatentDDPM with a default EpsMLP.
    """
    eps_model = EpsMLP(
        z_dim=z_dim,
        time_emb_dim=time_emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    cfg = DiffusionConfig(T=T, schedule=schedule)
    return LatentDDPM(eps_model=eps_model, cfg=cfg)
