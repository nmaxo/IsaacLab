"""RolloutStorage for DWBC PPO with 2-component (leg/arm) rewards, values, and advantages."""

from __future__ import annotations

import torch


class DwbcRolloutStorage:
    """Stores transitions for dual-reward PPO. Rewards/values/advantages are shape (..., 2)."""

    class Transition:
        def __init__(self):
            self.obs_prop: torch.Tensor | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None  # (num_envs, 2)
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None  # (num_envs, 2)
            self.actions_log_prob: torch.Tensor | None = None  # (num_envs, 2)
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        num_prop: int,
        actions_shape: list[int],
        device: str = "cpu",
    ):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_prop = num_prop

        T = num_transitions_per_env
        N = num_envs

        self.obs_prop = torch.zeros(T, N, num_prop, device=device)

        self.actions = torch.zeros(T, N, *actions_shape, device=device)
        self.rewards = torch.zeros(T, N, 2, device=device)
        self.dones = torch.zeros(T, N, 1, device=device).byte()

        self.values = torch.zeros(T, N, 2, device=device)
        self.actions_log_prob = torch.zeros(T, N, 2, device=device)
        self.mu = torch.zeros(T, N, *actions_shape, device=device)
        self.sigma = torch.zeros(T, N, *actions_shape, device=device)

        self.returns = torch.zeros(T, N, 2, device=device)
        self.advantages = torch.zeros(T, N, 2, device=device)

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow")

        self.obs_prop[self.step].copy_(transition.obs_prop)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob)
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        """GAE over 2-component rewards/values. last_values: (num_envs, 2)."""
        advantage = torch.zeros(self.num_envs, 2, device=self.device)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        obs_prop = self.obs_prop.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    obs_prop[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
