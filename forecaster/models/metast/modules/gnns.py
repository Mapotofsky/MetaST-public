import torch
import torch.nn as nn
import math

from models.metast.modules.layers import get_activation, MLP


class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 combination="gcn", use_bias=True,
                 mlp_layer_num=2, learn_eps=False, mlp_dropout=0.1, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = get_activation(activation)

        if combination == "gcn":
            self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
            self.bias = nn.Parameter(torch.FloatTensor(out_channels)) if use_bias else None

            def combine_weight(x, agg):
                out = torch.einsum("btif,fo->btio", agg, self.weight)
                if self.bias is not None:
                    out = out + self.bias
                return self.activation(out)
            self._combine = combine_weight

            self.reset_parameters()

        elif combination == "gin":
            self.eps = nn.Parameter(torch.zeros(1), requires_grad=learn_eps)
            self.mlp = MLP(mlp_layer_num, in_channels, out_channels, out_channels, 0,
                           None, mlp_dropout, activation=activation)

            def combine_mlp(x, agg):
                return self.mlp(self.eps * x + agg)
            self._combine = combine_mlp

        else:
            raise NotImplementedError(f"Combination '{combination}' not supported.")

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        # x: [batch, time, nodes, in_channels]
        # gso: [batch, num_nodes, num_nodes]
        agg = torch.einsum("bij,btjf->btif", gso, x)  # aggregate
        out = self._combine(x, agg)  # combine, [batch, time, nodes, out_channels]
        return out


class ResidualGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, combination="gcn",
                 use_bias=True, mlp_layer_num=2, learn_eps=False,
                 mlp_dropout=0.1, activation="relu", use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.gnn_layer = GNNLayer(in_channels, out_channels, combination,
                                  use_bias, mlp_layer_num, learn_eps,
                                  mlp_dropout, activation)

        if use_residual and in_channels != out_channels:
            self.projector = nn.Linear(in_channels, out_channels)
        else:
            self.projector = None

    def forward(self, x, gso):
        if self.use_residual:
            residual = x if self.projector is None else self.projector(x)

        out = self.gnn_layer(x, gso)

        if self.use_residual:
            out = out + residual

        return out


class GCN(nn.Module):
    """
    GCN: Graph Convolutional Network

    $$ h_v^{(k)} = ReLU(W * MEAN { h_u^{(k-1)}, forall u in N(v) U {v} }) $$

    Args:
        input_dim: int, input dimension
        hidden_dim: int, hidden dimension
        output_dim: int, output dimension
        num_layers: int, number of layers
        use_bias: bool, whether to use bias
        use_residual: bool, whether to use residual connections
    Input:
        x: [batch, time, nodes, in_channels]
        gso: [batch, num_nodes, num_nodes], graph shift operator
            Required to be a symmetric or random walk normalized Laplacian matrix,
            with eigenvalues falling within the [-1, 1] interval
    Output:
        x: [batch, time, nodes, out_channels]
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 use_bias=True, use_residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layer_use_residual = use_residual and (in_dim == out_dim)
            # GCN uses ReLU activation function by default
            self.layers.append(ResidualGNNLayer(in_dim, out_dim, combination="gcn",
                                                use_bias=use_bias, activation="relu",
                                                use_residual=layer_use_residual))

            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(out_dim))

    def forward(self, x, gso):
        for i, layer in enumerate(self.layers):
            x = layer(x, gso)
            if i < len(self.norms):
                x = self.norms[i](x)
        return x


class GIN(nn.Module):
    """
    GIN: Graph Isomorphism Network

    $$ h_v^{(k)} = MLP^{(k)} ( (1 + epsilon^{(k)}) * h_v^{(k-1)} + SUM { h_u^{(k-1)}, forall u in N(v) } ) $$

    Args:
        input_dim: int, input dimension
        hidden_dim: int, hidden dimension
        output_dim: int, output dimension
        num_layers: int, number of layers
        mlp_dropout: float, dropout rate for MLP
        activation: str, activation function
        learn_eps: bool, whether to learn epsilon
        use_residual: bool, whether to use residual connections
    Input:
        x: [batch, time, nodes, in_channels]
        gso: [batch, num_nodes, num_nodes], graph shift operator
            Adjacency matrix within [0, 1] interval (with self-loops) is sufficient,
            no need for normalized Laplacian matrix
    Output:
        x: [batch, time, nodes, out_channels]
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 mlp_layer_num=2, mlp_dropout=0.1, activation="relu",
                 learn_eps=False, use_residual=True):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layer_use_residual = use_residual and (in_dim == out_dim)
            self.layers.append(ResidualGNNLayer(in_dim, out_dim, combination="gin",
                                                mlp_layer_num=mlp_layer_num, learn_eps=learn_eps,
                                                mlp_dropout=mlp_dropout, activation=activation,
                                                use_residual=layer_use_residual))

    def forward(self, x, gso):
        for layer in self.layers:
            x = layer(x, gso)
        return x
