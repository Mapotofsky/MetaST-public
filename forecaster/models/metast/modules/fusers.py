import torch
import torch.nn as nn
from einops import rearrange

from models.metast.modules.layers import MLP, get_activation


class ConditionalFuser(nn.Module):
    def __init__(self, hidden_dims, dropout=0.1, x_len=None, y_len=None):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.x_len = x_len
        self.y_len = y_len

    def forward(self, time_reprs, anchor_vectors):
        """
        Args:
            time_reprs: [B, x_len + y_len, hidden_dims]
            anchor_vectors: [B, x_len, hidden_dims]

        Returns:
            conditioned_reprs: [B, x_len + y_len, hidden_dims]
        """
        raise NotImplementedError


class ConcatConditionalFuser(ConditionalFuser):
    def __init__(self, hidden_dims, dropout=0.1, x_len=None, y_len=None, mlp_layer_num=2, activation="relu"):
        super().__init__(hidden_dims, dropout, x_len, y_len)
        self.anchor_projection = nn.Sequential(
            nn.Linear(x_len, x_len + y_len),
            get_activation(activation),
            nn.Dropout(dropout)
        )  # x_len -> total_len
        self.fusion_mlp = MLP(mlp_layer_num, hidden_dims * 2, hidden_dims, hidden_dims,
                              0, None, dropout, activation=activation)

    def forward(self, time_reprs, anchor_vectors):
        projected_anchors = self.anchor_projection(anchor_vectors.mT).mT  # [B, total_len, hidden_dims]
        combined = torch.cat([time_reprs, projected_anchors], dim=-1)  # [B, total_len, 2 * hidden_dims]
        conditioned_reprs = self.fusion_mlp(combined)  # [B, total_len, hidden_dims]

        return conditioned_reprs


class GRUConditionalFuser(ConditionalFuser):
    def __init__(self, hidden_dims, dropout=0.1, x_len=None, y_len=None, gru_layers=1, bidirectional=False):
        super().__init__(hidden_dims, dropout, x_len, y_len)
        D = 2 if bidirectional else 1
        # anchor_vector -> h_0
        self.anchor_projection = nn.Sequential(
            nn.Linear(x_len, D * gru_layers),
            nn.ReLU(),
            nn.Dropout(dropout)
        )  # x_len -> total_len
        self.gru = nn.GRU(hidden_dims, hidden_dims, gru_layers,
                          batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, time_reprs, anchor_vectors):
        projected_anchors = self.anchor_projection(anchor_vectors.mT)  # [B, hidden_dims, D * gru_layers]
        h_0 = projected_anchors.permute(2, 0, 1)  # [D * gru_layers, B, hidden_dims]
        conditioned_reprs, _ = self.gru(time_reprs, h_0)  # [B, total_len, hidden_dims]

        return conditioned_reprs


class AttentionConditionalFuser(ConditionalFuser):
    def __init__(self, hidden_dims, dropout=0.1, x_len=None, y_len=None, num_heads=8):
        super().__init__(hidden_dims, dropout, x_len, y_len)
        self.num_heads = num_heads

        # K, V -> [B, x_len, hidden_dims], Q -> [B, total_len, hidden_dims]
        # output -> [B, total_len, hidden_dims]
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dims, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dims)
        self.norm2 = nn.LayerNorm(hidden_dims)

    def forward(self, time_reprs, anchor_vectors):
        attn_output, _ = self.multihead_attn(time_reprs, anchor_vectors, anchor_vectors)
        attn_output = self.norm1(time_reprs + attn_output)
        ffn_output = self.ffn(attn_output)
        conditioned_reprs = self.norm2(attn_output + ffn_output)

        return conditioned_reprs


class FiLMConditionalFuser(ConditionalFuser):
    def __init__(self, hidden_dims, dropout=0.1, x_len=None, y_len=None):
        super().__init__(hidden_dims, dropout, x_len, y_len)
        self.gamma = nn.Linear(x_len, x_len + y_len)
        self.beta = nn.Linear(x_len, x_len + y_len)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(hidden_dims)

    def forward(self, time_reprs, anchor_vectors):
        gamma = self.gamma(anchor_vectors.mT).mT  # [B, total_len, hidden_dims]
        beta = self.beta(anchor_vectors.mT).mT
        film_output = gamma * time_reprs + beta
        ffn_output = self.ffn(film_output)
        conditioned_reprs = self.norm(film_output + ffn_output)

        return conditioned_reprs
