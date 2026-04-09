"""ActorMultiCritic: one actor, N critics (one per reward group).

Each critic estimates the value function for its own reward group.
The actor is shared across all groups.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorMultiCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        num_critics: int = 2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorMultiCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_critics = num_critics
        self.obs_groups = obs_groups

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2
            num_critic_obs += obs[obs_group].shape[-1]

        # Shared actor
        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # Multiple critics (one per reward group)
        self.critics = nn.ModuleList()
        self.critic_obs_normalizers = nn.ModuleList()
        for i in range(num_critics):
            critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
            self.critics.append(critic)
            if critic_obs_normalization:
                self.critic_obs_normalizers.append(EmpiricalNormalization(num_critic_obs))
            else:
                self.critic_obs_normalizers.append(torch.nn.Identity())
            print(f"Critic {i} MLP: {critic}")

        self.critic_obs_normalization = critic_obs_normalization

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

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
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs_flat = self.get_actor_obs(obs)
        obs_flat = self.actor_obs_normalizer(obs_flat)
        self.update_distribution(obs_flat)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs_flat = self.get_actor_obs(obs)
        obs_flat = self.actor_obs_normalizer(obs_flat)
        return self.actor(obs_flat)

    def evaluate(self, obs, **kwargs):
        """Evaluate all critics. Returns tensor of shape (batch, num_critics)."""
        obs_flat = self.get_critic_obs(obs)
        values = []
        for i in range(self.num_critics):
            normalized = self.critic_obs_normalizers[i](obs_flat)
            values.append(self.critics[i](normalized))
        return torch.cat(values, dim=-1)  # (batch, num_critics)

    def evaluate_single(self, obs, critic_idx: int, **kwargs):
        """Evaluate a single critic. Returns tensor of shape (batch, 1)."""
        obs_flat = self.get_critic_obs(obs)
        normalized = self.critic_obs_normalizers[critic_idx](obs_flat)
        return self.critics[critic_idx](normalized)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            for normalizer in self.critic_obs_normalizers:
                normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
