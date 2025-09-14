import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


# === Utility: построить полный граф (полносвязный) ===
def build_fully_connected_edge_index(num_nodes, batch_size, device):
    """Создаёт edge_index для батча из B графов, каждый из которых полный"""
    row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
    edge_index_single = torch.stack([row.flatten(), col.flatten()], dim=0)  # [2, N^2]
    edge_indices = []
    for b in range(batch_size):
        edge_indices.append(edge_index_single + b * num_nodes)
    edge_index = torch.cat(edge_indices, dim=1).to(device)
    return edge_index


# === Графовый энкодер ===
class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim, edge_dim, hidden_dim=64, out_dim=128, heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        # первый слой: node_in_dim -> hidden_dim
        self.layers.append(GATv2Conv(node_in_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, concat=True))
        for _ in range(num_layers - 1):
            self.layers.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, concat=True))

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, node_features, edge_index, edge_attr, batch):
        x = node_features
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
        g = global_mean_pool(x, batch)   # [B, hidden_dim]
        return self.mlp_out(g)           # [B, out_dim]


# === Actor ===
class CustomActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-5, max_log_std=2,
                 gnn_hidden=64, gnn_out=128, gnn_layers=2, gnn_heads=4):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.device = device
        self.img_dim = observation_space["img"].shape[0]
        self.num_nodes = observation_space["graph"]["node_features"].shape[0]
        self.node_dim = observation_space["graph"]["node_features"].shape[1]
        self.edge_dim = observation_space["graph"]["edge_features"].shape[1]

        self.graph_encoder = GraphEncoder(
            node_in_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            heads=gnn_heads,
            num_layers=gnn_layers
        ).to(device)

        mlp_in = self.img_dim + gnn_out
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        ).to(device)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs, role=""):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])

        img = states["img"].to(self.device)  # [B, img_dim]
        node_feats = states["graph"]["node_features"].reshape(B * self.num_nodes, -1).to(self.device)
        edge_feats = states["graph"]["edge_features"].reshape(B * self.num_nodes, -1).to(self.device)

        batch = torch.repeat_interleave(torch.arange(B, device=self.device), repeats=self.num_nodes)
        edge_index = build_fully_connected_edge_index(self.num_nodes, B, device=self.device)

        graph_emb = self.graph_encoder(node_feats, edge_index, edge_feats, batch)  # [B, gnn_out]

        x = torch.cat([img, graph_emb], dim=-1)   # [B, img_dim + gnn_out]
        mu = self.net(x)
        return mu, self.log_std_parameter, {}


# === Critic ===
class CustomCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, gnn_hidden=64, gnn_out=128, gnn_layers=2, gnn_heads=4):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.device = device
        self.img_dim = observation_space["img"].shape[0]
        self.num_nodes = observation_space["graph"]["node_features"].shape[0]
        self.node_dim = observation_space["graph"]["node_features"].shape[1]
        self.edge_dim = observation_space["graph"]["edge_features"].shape[1]

        self.graph_encoder = GraphEncoder(
            node_in_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            heads=gnn_heads,
            num_layers=gnn_layers
        ).to(device)

        mlp_in = self.img_dim + gnn_out + self.num_actions
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        ).to(device)

    def compute(self, inputs, role=""):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])

        img = states["img"].to(self.device)
        node_feats = states["graph"]["node_features"].reshape(B * self.num_nodes, -1).to(self.device)
        edge_feats = states["graph"]["edge_features"].reshape(B * self.num_nodes, -1).to(self.device)
        actions = inputs["taken_actions"].to(self.device)

        batch = torch.repeat_interleave(torch.arange(B, device=self.device), repeats=self.num_nodes)
        edge_index = build_fully_connected_edge_index(self.num_nodes, B, device=self.device)

        graph_emb = self.graph_encoder(node_feats, edge_index, edge_feats, batch)  # [B, gnn_out]

        x = torch.cat([img, graph_emb, actions], dim=-1)
        q = self.net(x)
        return q, {}
