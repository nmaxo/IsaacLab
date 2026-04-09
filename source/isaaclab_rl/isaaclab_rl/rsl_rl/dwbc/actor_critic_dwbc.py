"""ActorCriticDWBC: dual-head actor + dual critic on proprioceptive observations only."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "selu": nn.SELU(),
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if act_name not in activations:
        raise ValueError(f"Unknown activation: {act_name}. Available: {list(activations.keys())}")
    return activations[act_name]


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: nn.Module,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation)
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class ActorCriticDWBC(nn.Module):
    """Dual-head actor and dual-head critic; both use the same proprio vector."""

    is_recurrent = False

    def __init__(
        self,
        num_prop: int,
        num_leg_actions: int,
        num_arm_actions: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        leg_control_head_hidden_dims: list[int] | None = None,
        arm_control_head_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: list[float] | float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"ActorCriticDWBC.__init__ ignoring unexpected kwargs: {list(kwargs.keys())}")

        actor_hidden_dims = actor_hidden_dims or [128]
        critic_hidden_dims = critic_hidden_dims or [128]
        leg_control_head_hidden_dims = leg_control_head_hidden_dims or [128, 128]
        arm_control_head_hidden_dims = arm_control_head_hidden_dims or [128, 128]

        self.num_leg_actions = num_leg_actions
        self.num_arm_actions = num_arm_actions
        self.num_actions = num_leg_actions + num_arm_actions
        self.num_prop = num_prop

        act_fn = get_activation(activation)

        if len(actor_hidden_dims) > 0:
            self.actor_backbone = build_mlp(num_prop, actor_hidden_dims[:-1], actor_hidden_dims[-1], act_fn, act_fn)
            backbone_out = actor_hidden_dims[-1]
        else:
            self.actor_backbone = nn.Identity()
            backbone_out = num_prop

        self.leg_action_head = build_mlp(backbone_out, leg_control_head_hidden_dims, num_leg_actions, act_fn, nn.Tanh())
        self.arm_action_head = build_mlp(backbone_out, arm_control_head_hidden_dims, num_arm_actions, act_fn, nn.Tanh())

        if len(critic_hidden_dims) > 0:
            self.critic_backbone = build_mlp(num_prop, critic_hidden_dims[:-1], critic_hidden_dims[-1], act_fn, act_fn)
            critic_out = critic_hidden_dims[-1]
        else:
            self.critic_backbone = nn.Identity()
            critic_out = num_prop

        self.leg_value_head = build_mlp(critic_out, leg_control_head_hidden_dims, 1, act_fn)
        self.arm_value_head = build_mlp(critic_out, arm_control_head_hidden_dims, 1, act_fn)

        if isinstance(init_noise_std, (list, tuple)):
            self.std = nn.Parameter(torch.tensor(init_noise_std, dtype=torch.float32))
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        print(
            f"ActorCriticDWBC: prop={num_prop}, leg_act={num_leg_actions}, arm_act={num_arm_actions}"
        )

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        ent = self.distribution.entropy()
        leg_ent = ent[:, : self.num_leg_actions].sum(dim=-1, keepdim=True)
        arm_ent = ent[:, self.num_leg_actions :].sum(dim=-1, keepdim=True)
        return torch.cat([leg_ent, arm_ent], dim=-1)

    def _actor_forward(self, obs_prop: torch.Tensor) -> torch.Tensor:
        backbone_out = self.actor_backbone(obs_prop)
        leg_out = self.leg_action_head(backbone_out)
        arm_out = self.arm_action_head(backbone_out)
        return torch.cat([leg_out, arm_out], dim=-1)

    def update_distribution(self, obs_prop: torch.Tensor):
        mean = self._actor_forward(obs_prop)
        self.distribution = Normal(mean, mean * 0.0 + self.std.clamp(min=1e-6))

    def act(self, obs_prop: torch.Tensor) -> torch.Tensor:
        self.update_distribution(obs_prop)
        return self.distribution.sample()

    def act_inference(self, obs_prop: torch.Tensor) -> torch.Tensor:
        return self._actor_forward(obs_prop)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(actions)
        leg_lp = log_prob[:, : self.num_leg_actions].sum(dim=-1, keepdim=True)
        arm_lp = log_prob[:, self.num_leg_actions :].sum(dim=-1, keepdim=True)
        return torch.cat([leg_lp, arm_lp], dim=-1)

    def evaluate(self, obs_prop: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 2): [leg_value, arm_value]."""
        backbone_out = self.critic_backbone(obs_prop)
        leg_val = self.leg_value_head(backbone_out)
        arm_val = self.arm_value_head(backbone_out)
        return torch.cat([leg_val, arm_val], dim=-1)
