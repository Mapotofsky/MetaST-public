from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class RidgeRegressor(nn.Module):
    """
    https://github.com/salesforce/DeepTime
    """
    def __init__(self, lambda_init: Optional[float] = 0.):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))

    def forward(self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)  # Last column represents bias term, initialized to 1
        # X: [batch_size, n_samples, n_dim + 1]
        # Y: [batch_size, n_samples, C]

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.mT, X)  # [batch_size, n_dim + 1, n_dim + 1]
            # Add regularization term to avoid potential singular matrices during computation
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)  # A = X^T * X + rI
            B = torch.bmm(X.mT, Y)  # [batch_size, n_dim + 1, C]
            weights = torch.linalg.solve(A, B)  # [batch_size, n_dim + 1, C]
        else:
            # When sample size is smaller than feature size, use Woodbury formula to solve weights via left inverse
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)  # A = X * X^T + rI
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))  # [batch_size, n_dim + 1, C]

        # w: [batch_size, n_dim, C], b: [batch_size, 1, C]
        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)  # softplus(x) = log(1 + e^x)
