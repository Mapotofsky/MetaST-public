import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.metast.modules.layers import MLP
from models.metast.modules.gnns import GCN, GIN
from models.metast.modules.regressors import RidgeRegressor
from models.metast.modules.fusers import *

from models.metast.pretrained_encoders.softclt_ts2vec.soft_ts2vec import TS2Vec


@gin.configurable()
def metast(device, hidden_dims, dropout, node_encoder_config, anchor_encoder_config, predictor_config):
    """MetaST model factory function."""
    return MetaST(
        device=device, hidden_dims=hidden_dims, dropout=dropout,
        node_encoder_config=node_encoder_config,
        anchor_encoder_config=anchor_encoder_config,
        predictor_config=predictor_config,
    )


class NodeEncoder(nn.Module):
    """
    Node feature encoding module.
    Supports optional pretrained encoder (TS2Vec) or default MLP encoder.
    """

    def __init__(
        self, device, input_dims, hidden_dims, dropout=0.1,
        use_pretrained_encoder=False, pretrained_config=None,
        ne_layer_num=2, activation="relu"
    ):
        super().__init__()
        self.device = device
        self.hidden_dims = hidden_dims
        self.use_pretrained_encoder = use_pretrained_encoder

        if use_pretrained_encoder and pretrained_config is not None:
            pretrained_encoder_config = dict(
                device=device,
                input_dims=input_dims, output_dims=hidden_dims, hidden_dims=hidden_dims,
                depth=pretrained_config.get("pe_depth", 10),
                lambda_=pretrained_config.get("pe_lambda_", 0.5),
                tau_temp=pretrained_config.get("pe_tau_temp", 0.5),
                soft_instance=pretrained_config.get("pe_soft_instance", False),
                soft_temporal=pretrained_config.get("pe_soft_temporal", False),
                batch_size=64,
            )
            self.encoder = TS2Vec(**pretrained_encoder_config)
            if "pe_path" in pretrained_config:
                self.encoder.load(pretrained_config["pe_path"])
            else:
                raise ValueError("pe_path is not provided")
        else:
            self.encoder = MLP(ne_layer_num, input_dims, hidden_dims, hidden_dims,
                               0, None, dropout, activation=activation)

    def forward(self, all_nodes):
        """
        Encode all node features.

        Args:
            all_nodes: [B, L, N, C] spatiotemporal data for all nodes
        Returns:
            encoded_features: [B, N, hidden_dims] encoded node features
        """
        B, _, N, _ = all_nodes.shape

        if self.use_pretrained_encoder:
            all_nodes_flat = rearrange(all_nodes, "b l n c -> (b n) l c")
            encoded_features = self.encoder.encode_tensor(
                all_nodes_flat, encoding_window="full_series"
            )  # [B * N, hidden_dims]
            encoded_features = rearrange(
                encoded_features, "(b n) c -> b n c", b=B, n=N
            )  # [B, N, hidden_dims]
        else:
            all_nodes_flat = rearrange(all_nodes, "b l n c -> (b n) l c")
            encoded_features = self.encoder(all_nodes_flat)  # [B * N, L, hidden_dims]
            # Temporal pooling to get node-level features
            encoded_features = F.max_pool1d(encoded_features.transpose(1, 2),
                                            kernel_size=encoded_features.size(1)).transpose(1, 2)
            encoded_features = rearrange(encoded_features, "(b n) 1 c -> b n c", b=B, n=N)

        return encoded_features


class AnchorEncoder(nn.Module):
    """
    Anchor vector generation module.
    Computes node similarity, builds local subgraphs, and generates anchor vectors.
    """

    def __init__(
        self, device, feature_num, hidden_dims, node_num, dropout=0.1,
        sim_metric="dot", k_neighbors=5,
        gnn_type="gcn", gnn_layer_num=2, gnn_use_residual=True,
        use_bias=True, learn_eps=False,
        mlp_layer_num=2, aggregation_type="mlp_mlp", activation="relu"
    ):
        super().__init__()
        self.device = device
        self.feature_num = feature_num
        self.hidden_dims = hidden_dims
        self.node_num = node_num + 1  # Include target node
        self.sim_metric = sim_metric
        self.k_neighbors = k_neighbors
        self.gnn_type = gnn_type
        self.gnn_layer_num = gnn_layer_num
        self.gnn_use_residual = gnn_use_residual
        self.use_bias = use_bias
        self.learn_eps = learn_eps
        self.mlp_layer_num = mlp_layer_num
        self.aggregation_type = aggregation_type

        # Attention mechanism encoder (used when sim_metric="attention")
        if sim_metric == "attention":
            self.query_proj = nn.Linear(hidden_dims, hidden_dims)
            self.key_proj = nn.Linear(hidden_dims, hidden_dims)
            self.scale_factor = hidden_dims ** 0.5

        # Initialize aggregation functions based on aggregation type
        if aggregation_type == "mlp_mlp":
            self.global_mlp = MLP(mlp_layer_num, 2 * hidden_dims, hidden_dims, hidden_dims,
                                  0, None, dropout, activation=activation)
            self.node_mlp = MLP(mlp_layer_num, self.node_num, hidden_dims, 1,
                                0, None, dropout, activation=activation)

            def aggregate_mlp_mlp(global_agg, local_agg):
                # input: [B, L, N + 1, hidden_dims]
                combined = torch.cat([global_agg, local_agg], dim=-1)  # [B, L, N + 1, 2 * hidden_dims]
                combined_weights = self.global_mlp(combined)  # [B, L, N + 1, hidden_dims]
                node_weights = self.node_mlp(combined_weights.mT).squeeze(-1)  # [B, L, hidden_dims]
                return node_weights
            self._aggregate = aggregate_mlp_mlp

        elif aggregation_type == "only_global":
            self.global_mlp = MLP(mlp_layer_num, hidden_dims, hidden_dims, hidden_dims,
                                  0, None, dropout, activation=activation)
            self.gating_mlp = nn.Sequential(
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.Sigmoid()
            )

            def aggregate_only_global(global_agg, local_agg):
                # input: [B, L, N + 1, hidden_dims]
                global_target = self.global_mlp(global_agg[:, :, -1, :])  # [B, L, hidden_dims]
                local_target = torch.zeros_like(global_target, device=self.device)
                gate_input = torch.cat([global_target, local_target], dim=-1)  # [B, L, 2 * hidden_dims]
                gate = self.gating_mlp(gate_input)  # [B, L, hidden_dims]
                fused = gate * global_target + (1 - gate) * local_target  # [B, L, hidden_dims]
                return fused
            self._aggregate = aggregate_only_global

        elif aggregation_type == "only_local":
            self.local_mlp = MLP(mlp_layer_num, hidden_dims, hidden_dims, hidden_dims,
                                 0, None, dropout, activation=activation)
            self.gating_mlp = nn.Sequential(
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.Sigmoid()
            )

            def aggregate_only_local(global_agg, local_agg):
                # input: [B, L, N + 1, hidden_dims]
                local_target = self.local_mlp(local_agg[:, :, -1, :])
                global_target = torch.zeros_like(local_target, device=self.device)
                gate_input = torch.cat([global_target, local_target], dim=-1)  # [B, L, 2 * hidden_dims]
                gate = self.gating_mlp(gate_input)  # [B, L, hidden_dims]
                fused = gate * global_target + (1 - gate) * local_target  # [B, L, hidden_dims]
                return fused
            self._aggregate = aggregate_only_local

        elif aggregation_type == "pool_gating":
            self.global_mlp = MLP(mlp_layer_num, hidden_dims, hidden_dims, hidden_dims,
                                  0, None, dropout, activation=activation)
            self.local_mlp = MLP(mlp_layer_num, hidden_dims, hidden_dims, hidden_dims,
                                 0, None, dropout, activation=activation)
            self.gating_mlp = nn.Sequential(
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.Sigmoid()
            )

            def aggregate_pool_gating(global_agg, local_agg):
                # input: [B, L, N + 1, hidden_dims]
                global_target = self.global_mlp(global_agg[:, :, -1, :])  # [B, L, hidden_dims]
                local_target = self.local_mlp(local_agg[:, :, -1, :])
                gate_input = torch.cat([global_target, local_target], dim=-1)  # [B, L, 2 * hidden_dims]
                gate = self.gating_mlp(gate_input)  # [B, L, hidden_dims]
                fused = gate * global_target + (1 - gate) * local_target  # [B, L, hidden_dims]
                return fused
            self._aggregate = aggregate_pool_gating

        elif aggregation_type == "attention_mlp":
            self.cross_attention = nn.MultiheadAttention(hidden_dims, num_heads=8, dropout=dropout, batch_first=True)
            self.node_mlp = MLP(mlp_layer_num, self.node_num, hidden_dims, 1,
                                0, None, dropout, activation=activation)

            def aggregate_attention_mlp(global_agg, local_agg):
                B, L, _, _ = global_agg.shape
                global_flat = rearrange(global_agg, "b l n c -> (b l) n c")
                local_flat = rearrange(local_agg, "b l n c -> (b l) n c")
                attended, _ = self.cross_attention(global_flat, local_flat, local_flat)  # global as query
                attended = rearrange(attended, "(b l) n c -> b l c n", b=B, l=L)  # [B, L, hidden_dims, N + 1]
                target_feats = self.node_mlp(attended).squeeze(-1)  # [B, L, hidden_dims]
                return target_feats
            self._aggregate = aggregate_attention_mlp

        else:
            raise ValueError(f"Unsupported aggregation_type: {aggregation_type}")

        # 根据GNN类型初始化GNN创建函数
        if gnn_type == "gcn":
            # Note: GCN requires symmetric normalization, may cause numerical instability
            # when used with dot/cosine metrics and pool_gating aggregation
            self._gnn = GCN(feature_num, hidden_dims, hidden_dims, gnn_layer_num,
                            use_bias=use_bias, use_residual=gnn_use_residual)

            def prepare_gso(sim_matrix, local_graph):
                return self._normalize_adj(sim_matrix), self._normalize_adj(local_graph)
            self._prepare_gso = prepare_gso

        elif gnn_type == "gin":
            self._gnn = GIN(feature_num, hidden_dims, hidden_dims, gnn_layer_num,
                            mlp_layer_num=mlp_layer_num, mlp_dropout=dropout,
                            activation=activation, learn_eps=learn_eps, use_residual=gnn_use_residual)

            def prepare_gso(sim_matrix, local_graph):
                return sim_matrix, local_graph
            self._prepare_gso = prepare_gso

        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}")

    def compute_similarity(self, node_features, target_idx=None):
        """
        Compute pairwise node similarity matrix.

        Args:
            node_features: [B, N + 1, hidden_dims] node features
            target_idx: target node index for training (masks similarity with last node)
        Returns:
            sim_matrix: [B, N + 1, N + 1] similarity matrix
        """
        if self.sim_metric == "dot":
            sim_matrix = torch.bmm(node_features, node_features.transpose(1, 2))
        elif self.sim_metric == "cosine":
            norm_feats = F.normalize(node_features, p=2, dim=2)
            sim_matrix = torch.bmm(norm_feats, norm_feats.transpose(1, 2))
        elif self.sim_metric == "attention":
            Q = self.query_proj(node_features)
            K = self.key_proj(node_features)
            sim_matrix = torch.bmm(Q, K.transpose(1, 2)) / self.scale_factor
            sim_matrix = F.softmax(sim_matrix, dim=-1)
        else:
            raise ValueError(f"Invalid similarity metric: {self.sim_metric}")

        # Training phase: mask target node similarity
        if target_idx is not None:
            mask = torch.ones_like(sim_matrix)
            mask[:, target_idx, -1] = 0
            mask[:, -1, target_idx] = 0
            sim_matrix = sim_matrix * mask

        return sim_matrix

    def build_local_graph(self, sim_matrix):
        """
        Build local subgraph by keeping top-k similar nodes.

        Args:
            sim_matrix: [B, N + 1, N + 1] global similarity matrix
        Returns:
            local_graph: [B, N + 1, N + 1] local subgraph adjacency matrix
        """
        B = sim_matrix.shape[0]
        k = min(self.k_neighbors, self.node_num - 1)

        target_similarities = sim_matrix[:, -1, :-1]
        _, topk_indices = torch.topk(target_similarities, k, dim=1)

        lg_mask = torch.zeros_like(sim_matrix, requires_grad=False, device=self.device)

        # Keep edges between target node and top-k neighbors
        lg_mask[:, -1, :-1].scatter_(1, topk_indices, 1)
        lg_mask[:, :-1, -1].scatter_(1, topk_indices, 1)

        # Build induced subgraph among top-k neighbors
        neighbor_mask = torch.zeros(B, self.node_num - 1, dtype=torch.bool, device=self.device)
        neighbor_mask.scatter_(1, topk_indices, True)
        neighbor_mask_expanded = (neighbor_mask.unsqueeze(2) & neighbor_mask.unsqueeze(1)).float()
        lg_mask[:, :-1, :-1] = lg_mask[:, :-1, :-1] + neighbor_mask_expanded

        # Add self-loops
        lg_mask[:, -1, -1] = 1
        batch_indices = torch.arange(B, device=self.device).unsqueeze(1)
        lg_mask[batch_indices, topk_indices, topk_indices] = 1

        # Apply mask to original similarity matrix
        local_graph = sim_matrix * lg_mask
        return local_graph

    def forward(self, node_features, all_nodes, target_idx=None):
        """
        Generate anchor vectors.

        Formula: anchor_vectors = g(GNN(G_global, x), GNN(G_local, x))
        where g: [2, L, N, C] -> [L, C]

        Args:
            node_features: [B, N, hidden_dims] encoded node features
            all_nodes: [B, L, N, C] original spatiotemporal data
            target_idx: target node index for training
        Returns:
            anchor_vectors: [B, L, hidden_dims] anchor vectors
        """
        # Compute similarity matrix and local graph
        sim_matrix = self.compute_similarity(node_features, target_idx)  # [B, N, N]
        local_graph = self.build_local_graph(sim_matrix)  # [B, N, N]

        # Prepare graph shift operators
        global_gso, local_gso = self._prepare_gso(sim_matrix, local_graph)

        # Apply GNN
        global_agg_feats = self._gnn(all_nodes, global_gso)
        local_agg_feats = self._gnn(all_nodes, local_gso)

        # Aggregate using predefined function g: 2 * [B, L, N, C] -> [L, C]
        anchor_vectors = self._aggregate(global_agg_feats, local_agg_feats)

        return anchor_vectors

    def _normalize_adj(self, adj_matrix):
        """
        Normalize adjacency matrix for GCN using symmetric normalization: D^(-1/2) * A * D^(-1/2)
        """
        identity = torch.eye(adj_matrix.size(-1), device=self.device)
        adj_with_self_loop = adj_matrix + identity.unsqueeze(0)

        degree = torch.sum(adj_with_self_loop, dim=-1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
        normalized_adj = torch.bmm(torch.bmm(degree_inv_sqrt, adj_with_self_loop), degree_inv_sqrt)

        return normalized_adj


class Predictor(nn.Module):
    """
    Meta-Forecaster module.
    Uses anchor_vector as conditional prompt through conditional INR + RidgeRegressor framework.

    Core idea:
    1. Process time coordinates to get time representations
    2. Fuse anchor_vector as conditional prompt into time representations
    3. Solve closed-form solution using conditional representations and historical data
    """

    def __init__(self, device, hidden_dims, x_len, y_len,
                 inr_layer_num=2, n_fourier_feats=128, scales=[1.0],
                 dropout=0.1, condition_type="concat", activation="relu",
                 mlp_layer_num=2, gru_layers=1, bidirectional=False, num_heads=8):
        super().__init__()
        self.device = device
        self.hidden_dims = hidden_dims
        self.x_len = x_len
        self.y_len = y_len
        self.condition_type = condition_type

        # Time representation network
        self.time_inr = MLP(inr_layer_num, 1, hidden_dims, hidden_dims,
                            n_fourier_feats, scales, dropout, activation=activation).to(self.device)

        # Initialize conditional fuser based on condition type
        if condition_type == "concat":
            self.conditional_fuser = ConcatConditionalFuser(hidden_dims, dropout, x_len, y_len,
                                                            mlp_layer_num, activation=activation)
        elif condition_type == "gru":
            self.conditional_fuser = GRUConditionalFuser(hidden_dims, dropout, x_len, y_len,
                                                         gru_layers, bidirectional)
        elif condition_type == "attention":
            self.conditional_fuser = AttentionConditionalFuser(hidden_dims, dropout, x_len, y_len,
                                                               num_heads)
        elif condition_type == "film":
            self.conditional_fuser = FiLMConditionalFuser(hidden_dims, dropout, x_len, y_len)
        else:
            raise ValueError(f"Unsupported condition_type: {condition_type}")

        self.ridge_regressor = RidgeRegressor(lambda_init=0.0)

    def get_time_coords(self, x_len, y_len):
        """Generate normalized time coordinates."""
        coords = torch.linspace(0, 1, x_len + y_len, device=self.device)
        return coords.unsqueeze(0).unsqueeze(-1)  # [1, total_len, 1]

    def forward(self, anchor_vectors, target_data, x_time, y_time):
        """
        Args:
            anchor_vectors: [B, L, hidden_dims] anchor vectors
            target_data: [B, L, C] target node historical data
            x_time: [L] historical time window timestamps
            y_time: [L] prediction time window timestamps
        Returns:
            forecast: [B, y_len] univariate prediction results
        """
        B = target_data.shape[0]

        # Generate time coordinates and get time representations
        time_coords = self.get_time_coords(self.x_len, self.y_len)
        time_coords = time_coords.repeat(B, 1, 1)
        time_reprs = self.time_inr(time_coords)

        # Fuse time representations with anchor conditions
        conditioned_reprs = self.conditional_fuser(time_reprs, anchor_vectors)

        # Separate historical and future conditional representations
        x_reprs = conditioned_reprs[:, :self.x_len, :]
        y_reprs = conditioned_reprs[:, self.x_len:, :]

        # Solve closed-form solution using historical conditional representations
        target_single = target_data[:, :, 0:1]  # For univariate prediction
        w, b = self.ridge_regressor(x_reprs, target_single)

        # Make predictions based on future conditional representations
        forecast = torch.einsum('btd,bdc->btc', y_reprs, w) + b
        forecast = forecast.squeeze(-1)

        return forecast


class MetaST(nn.Module):
    """
    MetaST main model class.
    Integrates node encoding, anchor generation, and meta-forecaster modules.
    """

    def __init__(
        self, device, hidden_dims, dropout,
        node_encoder_config, anchor_encoder_config, predictor_config
    ):
        super().__init__()
        self.device = device
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        node_encoder_config['device'] = device
        node_encoder_config['hidden_dims'] = hidden_dims
        node_encoder_config['dropout'] = dropout

        anchor_encoder_config['device'] = device
        anchor_encoder_config['hidden_dims'] = hidden_dims
        anchor_encoder_config['dropout'] = dropout

        predictor_config['device'] = device
        predictor_config['hidden_dims'] = hidden_dims
        predictor_config['dropout'] = dropout

        # Initialize three core modules
        self.node_encoder = NodeEncoder(**node_encoder_config)
        self.anchor_encoder = AnchorEncoder(**anchor_encoder_config)
        self.predictor = Predictor(**predictor_config)

    def forward(self, support, target, target_idx=None, x_time=None, y_time=None):
        """
        Args:
            support: [B, L, N, C] support set node data (seen nodes)
            target: [B, L, C] target node historical data
            target_idx: target node index for training (masks corresponding node in support)
            x_time: [L] historical window timestamps
            y_time: [L] prediction window timestamps
        Returns:
            forecast: [B, y_len] univariate prediction results
        """
        L = support.shape[1]

        # Generate default timestamps if not provided
        if x_time is None:
            x_time = torch.arange(L, device=self.device).float()
        if y_time is None:
            y_time = torch.arange(L, L + self.predictor.y_len, device=self.device).float()

        # Node feature encoding
        target_expanded = rearrange(target, "b l c -> b l 1 c")
        all_nodes = torch.cat([support, target_expanded], dim=2)
        node_features = self.node_encoder(all_nodes)

        # Anchor vector generation
        anchor_vectors = self.anchor_encoder(
            node_features, all_nodes, target_idx
        )  # [B, L, hidden_dims]

        # Conditional meta-forecasting
        forecast = self.predictor(
            anchor_vectors, target, x_time, y_time
        )  # [B, y_len]

        return forecast

    def inference(self, support, target, x_time=None, y_time=None):
        """Inference mode forward pass (without target_idx)."""
        return self.forward(support, target, None, x_time, y_time)

    def get_anchor_vectors(self, support, target, target_idx=None):
        """Get anchor vectors for analysis or visualization."""
        target_expanded = rearrange(target, "b l c -> b l 1 c")
        all_nodes = torch.cat([support, target_expanded], dim=2)

        node_features = self.node_encoder(all_nodes)

        anchor_vectors = self.anchor_encoder(node_features, all_nodes, target_idx)

        return anchor_vectors

    def get_similarity_matrix(self, support, target, target_idx=None):
        """Get node similarity matrix and local graph for analysis or visualization."""
        target_expanded = rearrange(target, "b l c -> b l 1 c")
        all_nodes = torch.cat([support, target_expanded], dim=2)
        node_features = self.node_encoder(all_nodes)
        sim_matrix = self.anchor_encoder.compute_similarity(node_features, target_idx)
        local_graph = self.anchor_encoder.build_local_graph(sim_matrix)

        return sim_matrix, local_graph
