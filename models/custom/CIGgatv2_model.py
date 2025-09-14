import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import gymnasium as gym
from skrl.utils.spaces.torch import (
    compute_space_size,
    flatten_tensorized_space,
    sample_space,
    unflatten_tensorized_space,
)

from torch_geometric.nn import GATv2Conv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=64, out_dim=128, num_layers=2, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        # first layer: node_in_dim -> hidden_dim (per head handled inside)
        self.convs.append(GATv2Conv(node_in_dim, hidden_dim // heads, heads=heads, concat=True))
        for _ in range(max(0, num_layers - 1)):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, node_features, edge_index, batch):
        """
        node_features: [num_nodes_total, node_in_dim]  (i.e. B*N, F)
        edge_index: [2, num_edges_total] (with offsets already applied)
        batch: [num_nodes_total] long tensor mapping node -> graph index (0..B-1)
        """
        x = node_features
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        # readout: per-graph embedding
        g = global_mean_pool(x, batch)  # [B, hidden_dim]
        out = self.mlp_out(g)  # [B, out_dim]
        return out

# === Graph encoder as above ===
# from torch_geometric.nn import GATv2Conv, global_mean_pool
# define build_fully_connected_edge_index() from previous cell

class CustomActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2,
                 gnn_hidden=64, gnn_out=128, gnn_layers=2, gnn_heads=4):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.device = device
        self.observation_space = observation_space

        # sizes from space
        self.img_dim = observation_space["img"].shape[0]                # e.g. 36
        self.num_objects = observation_space["graph"]["node_features"].shape[0]  # N
        self.node_dim = observation_space["graph"]["node_features"].shape[1]    # node_dim
        self.edge_dim = observation_space["graph"]["edge_features"].shape[1]    # edge_dim

        # we'll concat node and edge features into node input
        self.node_input_dim = self.node_dim + self.edge_dim

        # Graph encoder
        self.graph_encoder = GraphEncoder(node_in_dim=self.node_input_dim,
                                          hidden_dim=gnn_hidden,
                                          out_dim=gnn_out,
                                          num_layers=gnn_layers,
                                          heads=gnn_heads).to(device)

        # final MLP: img + graph_emb
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
        # inputs["states"] is flat [B, total_dim]
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])

        img = states["img"]  # (B, img_dim)
        node_feats = states["graph"]["node_features"]  # (B, N, node_dim)
        edge_feats = states["graph"]["edge_features"]  # (B, N, edge_dim)

        # combine node + edge features per node (pragmatic choice)
        node_input = torch.cat([node_feats, edge_feats], dim=-1)  # (B, N, node_input_dim)
        # flatten nodes for PyG
        node_input_flat = node_input.reshape(B * self.num_objects, -1).to(self.device)  # (B*N, node_input_dim)
        # build batch vector
        batch = torch.repeat_interleave(torch.arange(B, device=self.device), repeats=self.num_objects)  # (B*N,)
        # build edge_index for full graphs repeated for B
        edge_index = build_fully_connected_edge_index(self.num_objects, B, device=self.device)  # [2, Etotal]

        # run GNN
        graph_emb = self.graph_encoder(node_input_flat, edge_index, batch)  # (B, gnn_out)

        # final concat
        x = torch.cat([img.to(self.device), graph_emb], dim=-1)  # (B, img_dim + gnn_out)
        mu = self.net(x)  # (B, action_dim)
        return mu, self.log_std_parameter, {}

class CustomCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 gnn_hidden=64, gnn_out=128, gnn_layers=2, gnn_heads=4):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.device = device
        self.observation_space = observation_space

        self.img_dim = observation_space["img"].shape[0]
        self.num_objects = observation_space["graph"]["node_features"].shape[0]
        self.node_dim = observation_space["graph"]["node_features"].shape[1]
        self.edge_dim = observation_space["graph"]["edge_features"].shape[1]

        self.node_input_dim = self.node_dim + self.edge_dim

        self.graph_encoder = GraphEncoder(node_in_dim=self.node_input_dim,
                                          hidden_dim=gnn_hidden,
                                          out_dim=gnn_out,
                                          num_layers=gnn_layers,
                                          heads=gnn_heads).to(device)

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

        img = states["img"]  # (B, img_dim)
        node_feats = states["graph"]["node_features"]  # (B, N, node_dim)
        edge_feats = states["graph"]["edge_features"]  # (B, N, edge_dim)
        actions = inputs["taken_actions"].to(self.device)  # (B, action_dim)

        node_input = torch.cat([node_feats, edge_feats], dim=-1)
        node_input_flat = node_input.reshape(B * self.num_objects, -1).to(self.device)
        batch = torch.repeat_interleave(torch.arange(B, device=self.device), repeats=self.num_objects)
        edge_index = build_fully_connected_edge_index(self.num_objects, B, device=self.device)

        graph_emb = self.graph_encoder(node_input_flat, edge_index, batch)  # (B, gnn_out)

        x = torch.cat([img.to(self.device), graph_emb, actions], dim=-1)
        q = self.net(x)
        return q, {}
