import torch
import torch.nn as nn
import math


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding as in standard DDPM."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch,) timesteps
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CorrectionNetwork(nn.Module):
    """Lightweight correction network δ_θ(y_t, t, x, μ_GP, Σ_GP).

    A 2-3 layer MLP with ~10K parameters.
    Takes: noisy sample, timestep embedding, GP prior info, and conditioning.
    """
    def __init__(self, d_y=16, d_x=8, hidden_dim=128, num_layers=3):
        super().__init__()
        self.time_embed = SinusoidalTimestepEmbedding(dim=32)

        # Input: y_t (d_y) + time_emb (32) + μ_GP (d_y) + Σ_GP flat (d_y*d_y) + x (d_x)
        gp_flat_dim = d_y + d_y * d_y
        input_dim = d_y + 32 + gp_flat_dim + d_x

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [d_y]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, y_t, t, mu_GP, Sigma_GP, x):
        """Forward pass.

        Args:
            y_t:      (batch, d_y) noisy sample
            t:        (batch,) timesteps
            mu_GP:    (batch, d_y) GP posterior mean
            Sigma_GP: (batch, d_y, d_y) GP posterior covariance
            x:        (batch, d_x) conditioning input

        Returns:
            delta:    (batch, d_y) correction term
        """
        t_emb = self.time_embed(t)
        batch = y_t.shape[0]
        # Flatten Sigma_GP
        Sigma_flat = Sigma_GP.reshape(batch, -1)
        # Concatenate all inputs
        inp = torch.cat([y_t, t_emb, mu_GP, Sigma_flat, x], dim=-1)
        return self.net(inp)
