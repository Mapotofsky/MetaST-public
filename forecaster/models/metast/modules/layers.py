from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.metast.modules.feature_transforms import GaussianFourierFeatureTransform


def get_activation(activation):
    if callable(activation):
        return activation()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", "tanh", or a callable'
    )


class MLPLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 dropout: Optional[float] = 0.1, activation: Optional[str] = "relu"):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        out = self._layer(x)
        return self.norm(out)

    def _layer(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.linear(x)))


class MLP(nn.Module):
    def __init__(self, layer_num: int, input_dims: int, hidden_dims: int, output_dims: int,
                 n_fourier_feats: int, scales: float,
                 dropout: Optional[float] = 0.1, activation: Optional[str] = "relu"):
        super().__init__()
        if n_fourier_feats == 0:
            self.features = MLPLayer(input_dims, hidden_dims, dropout=dropout, activation=activation)
        else:
            self.features = nn.Sequential(
                GaussianFourierFeatureTransform(input_dims, n_fourier_feats, scales),
                MLPLayer(n_fourier_feats, hidden_dims, dropout=dropout, activation=activation)
            )
        layers = [MLPLayer(hidden_dims, hidden_dims, dropout=dropout, activation=activation) for _ in range(layer_num - 2)] + \
                 [MLPLayer(hidden_dims, output_dims, dropout=dropout, activation=activation)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.layers(x)
