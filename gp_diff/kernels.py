import torch
import torch.nn as nn
import math

class SEKernel(nn.Module):
    """Squared Exponential (RBF) kernel."""
    def __init__(self, output_scale=1.0, lengthscale=1.0):
        super().__init__()
        self.log_output_scale = nn.Parameter(torch.log(torch.tensor(output_scale)))
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        ls = torch.exp(self.log_lengthscale)
        sq_dist = torch.cdist(x1 / ls, x2 / ls) ** 2
        return torch.exp(self.log_output_scale) * torch.exp(-0.5 * sq_dist)


class PeriodicKernel(nn.Module):
    """Periodic kernel for seasonal/cyclic patterns."""
    def __init__(self, output_scale=1.0, lengthscale=1.0, period=1.0):
        super().__init__()
        self.log_output_scale = nn.Parameter(torch.log(torch.tensor(output_scale)))
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
        self.log_period = nn.Parameter(torch.log(torch.tensor(period)))

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        ls = torch.exp(self.log_lengthscale)
        p = torch.exp(self.log_period)
        # Pairwise difference matrix
        dist = torch.cdist(x1, x2)
        sin_term = torch.sin(math.pi * dist / p) ** 2
        return torch.exp(self.log_output_scale) * torch.exp(-2 * sin_term / ls ** 2)


class LinearKernel(nn.Module):
    """Linear kernel for long-term trends."""
    def __init__(self, variance_b=1.0, variance_v=1.0):
        super().__init__()
        self.log_var_b = nn.Parameter(torch.log(torch.tensor(variance_b)))
        self.log_var_v = nn.Parameter(torch.log(torch.tensor(variance_v)))

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        var_b = torch.exp(self.log_var_b)
        var_v = torch.exp(self.log_var_v)
        return var_b + var_v * (x1 @ x2.T)


class MaternKernel(nn.Module):
    """Matérn kernel with controlled smoothness ν = 3/2."""
    def __init__(self, output_scale=1.0, lengthscale=1.0, nu=1.5):
        super().__init__()
        self.log_output_scale = nn.Parameter(torch.log(torch.tensor(output_scale)))
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
        self.nu = nu

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        ls = torch.exp(self.log_lengthscale)
        dist = torch.cdist(x1 / ls, x2 / ls)
        if self.nu == 0.5:
            return torch.exp(self.log_output_scale) * torch.exp(-dist)
        elif self.nu == 1.5:
            sqrt3 = math.sqrt(3)
            r = sqrt3 * dist
            return torch.exp(self.log_output_scale) * (1 + r) * torch.exp(-r)
        elif self.nu == 2.5:
            sqrt5 = math.sqrt(5)
            r = sqrt5 * dist
            return torch.exp(self.log_output_scale) * (1 + r + r**2 / 3) * torch.exp(-r)
        else:
            raise ValueError(f"Unsupported nu={self.nu}. Use 0.5, 1.5, or 2.5.")


class GP_CPS_Kernel(nn.Module):
    """Combined GP-CPS kernel: weighted sum of SE, Periodic, Linear, Matern.

    k_CPS(τ) = Σ w_j · k_j(τ)
    """
    def __init__(self, d_in=1):
        super().__init__()
        self.se = SEKernel(output_scale=1.0, lengthscale=1.0)
        self.per = PeriodicKernel(output_scale=0.5, lengthscale=1.0, period=2.0)
        self.lin = LinearKernel(variance_b=0.1, variance_v=0.1)
        self.mat = MaternKernel(output_scale=0.5, lengthscale=1.0, nu=1.5)
        # Learnable log-weights for each kernel component
        self.log_weights = nn.Parameter(torch.zeros(4))

    def forward(self, x1, x2=None):
        w = torch.softmax(self.log_weights, dim=0)
        K = (w[0] * self.se(x1, x2) +
             w[1] * self.per(x1, x2) +
             w[2] * self.lin(x1, x2) +
             w[3] * self.mat(x1, x2))
        # Add jitter for numerical stability
        if x2 is None or x1 is x2:
            K = K + 1e-6 * torch.eye(K.shape[0], device=K.device)
        return K
