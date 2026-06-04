import torch
import torch.nn as nn
import math


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule as in Nichol & Dhariwal (2021)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    f = torch.cos((t / timesteps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0.0001, 0.02)


class GaussianDiffusionGP(nn.Module):
    """GP-guided diffusion process.

    Implements the forward and reverse processes as described in the
    GP-Diff framework.
    """
    def __init__(self, correction_net, timesteps=200, d_y=16):
        super().__init__()
        self.correction_net = correction_net
        self.timesteps = timesteps
        self.d_y = d_y

        # Register betas, alphas, alphas_cumprod
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

        # Gamma schedule: linear from 0 to 1 over timesteps
        gamma = torch.linspace(0, 1, timesteps + 1)[1:]  # [1...T]
        self.register_buffer('gamma', gamma)

    def q_sample(self, y_0, mu_GP, Sigma_GP, t, noise=None):
        """Forward process: sample y_t ~ q(y_t | y_0).

        q(y_t | y_0) = N( sqrt(ᾱ_t) y_0 + (1 - sqrt(ᾱ_t)) μ_GP,
                           γ_t Σ_GP + (1 - γ_t) I )
        """
        if noise is None:
            noise = torch.randn_like(y_0)

        batch = y_0.shape[0]
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        gamma_val = self.gamma[t].view(-1, 1)  # (batch, 1)

        mean = sqrt_alpha_bar * y_0 + (1 - sqrt_alpha_bar) * mu_GP
        try:
            L_GP = torch.linalg.cholesky(Sigma_GP)  # (batch, d, d)
            # y_t = mean + √γ_t · L_GP @ noise + √(1-γ_t) · noise
            noise_3d = noise.unsqueeze(-1)  # (batch, d, 1)
            gp_part = (L_GP @ noise_3d).squeeze(-1)  # (batch, d)
            y_t = mean + torch.sqrt(gamma_val) * gp_part + torch.sqrt(1 - gamma_val) * noise
        except:
            # Fallback: diagonal approximation
            diag = torch.diagonal(Sigma_GP, dim1=-2, dim2=-1).clamp(min=1e-6)
            std = torch.sqrt(gamma_val * diag + (1 - gamma_val))
            y_t = mean + std * noise

        return y_t

    def p_sample(self, y_t, mu_GP, Sigma_GP, x, t):
        """Reverse process: sample y_{t-1} ~ p_θ(y_{t-1} | y_t).

        ε_θ = (y_t - √ᾱ_t μ_GP) / √(1-ᾱ_t) + √(1-ᾱ_t) δ_θ
        y_{t-1} = 1/√α_t (y_t - (1-α_t)/√(1-ᾱ_t) ε_θ) + σ_t z
        """
        batch = y_t.shape[0]
        t_tensor = t.expand(batch)

        # Compute correction
        delta = self.correction_net(y_t, t_tensor.float(), mu_GP, Sigma_GP, x)

        # Noise prediction (Theorem 2)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        eps_theta = (y_t - sqrt_alpha_bar * mu_GP) / sqrt_one_minus + sqrt_one_minus * delta

        # Reverse step
        alpha = self.alphas[t].view(-1, 1)
        beta = self.betas[t].view(-1, 1)
        alpha_bar_prev = self.alphas_cumprod[t - 1].view(-1, 1) if t.min() > 0 else torch.ones(batch, 1, device=y_t.device)
        sigma = torch.sqrt(beta * (1 - alpha_bar_prev) / (1 - sqrt_alpha_bar ** 2))

        pred_mean = (y_t - (1 - alpha) / sqrt_one_minus * eps_theta) / torch.sqrt(alpha)
        noise = torch.randn_like(y_t) if t.min() > 0 else 0

        return pred_mean + sigma * noise

    def sample(self, mu_GP, Sigma_GP, x, num_samples=1):
        """Generate samples from the reverse process.

        Args:
            mu_GP:    (1, d_y) GP posterior mean
            Sigma_GP: (1, d_y, d_y) GP posterior covariance
            x:        (1, d_x) conditioning
            num_samples: number of samples

        Returns:
            samples: (num_samples, d_y) generated trajectories
        """
        device = mu_GP.device
        samples = []
        for _ in range(num_samples):
            # Sample y_T ~ N(μ_GP, Σ_GP)
            try:
                L_GP = torch.linalg.cholesky(Sigma_GP)
                y = mu_GP + (L_GP @ torch.randn(self.d_y, 1, device=device)).squeeze(-1)
            except:
                y = mu_GP + torch.sqrt(torch.diagonal(Sigma_GP, dim1=-2, dim2=-1).clamp(min=1e-6)) * torch.randn(1, self.d_y, device=device)
                y = y.squeeze(0)

            # Reverse diffusion
            for t in reversed(range(self.timesteps)):
                t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                y = self.p_sample(y, mu_GP, Sigma_GP, x, t_tensor)

            samples.append(y.squeeze(0))

        return torch.stack(samples)
