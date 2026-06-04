import torch
import torch.nn as nn
from .kernels import GP_CPS_Kernel


class GPPrior(nn.Module):
    """GP prior module: computes μ_GP and Σ_GP from input data.

    Uses the GP-CPS kernel with learnable hyperparameters.
    """
    def __init__(self, d_in=1, noise_variance=1e-3):
        super().__init__()
        self.kernel = GP_CPS_Kernel(d_in=d_in)
        self.log_noise = nn.Parameter(torch.log(torch.tensor(noise_variance)))
        # Learnable mean function parameter (constant mean)
        self.mean = nn.Parameter(torch.zeros(1))

    def compute_posterior(self, x_train, y_train, x_test):
        """Compute GP posterior at test points given training data.

        Args:
            x_train: (N, d) training inputs
            y_train: (N, 1) training targets
            x_test:  (M, d) test inputs

        Returns:
            mu_GP:    (M, 1) posterior mean
            Sigma_GP: (M, M) posterior covariance
        """
        noise = torch.exp(self.log_noise)
        K_xx = self.kernel(x_train, x_train) + noise * torch.eye(x_train.shape[0], device=x_train.device)
        K_xs = self.kernel(x_train, x_test)
        K_ss = self.kernel(x_test, x_test)

        # Solve: (K_xx + σ²I) \ K_xs
        L = torch.linalg.cholesky(K_xx)
        alpha = torch.cholesky_solve(K_xs, L)  # (N, M)

        # Posterior mean: μ(x*) + K(x*,x) (K(x,x)+σ²I)^{-1} (y - μ(x))
        mu_train = self.mean.expand(x_train.shape[0], 1)
        mu_test = self.mean.expand(x_test.shape[0], 1)
        mu_GP = mu_test + K_xs.T @ torch.cholesky_solve(y_train - mu_train, L)

        # Posterior covariance: K(x*,x*) - K(x*,x) (K(x,x)+σ²I)^{-1} K(x,x*)
        Sigma_GP = K_ss - K_xs.T @ torch.cholesky_solve(K_xs, L)

        # Ensure symmetry
        Sigma_GP = (Sigma_GP + Sigma_GP.T) / 2
        return mu_GP, Sigma_GP

    def forward(self, x_train, y_train, x_test):
        return self.compute_posterior(x_train, y_train, x_test)
