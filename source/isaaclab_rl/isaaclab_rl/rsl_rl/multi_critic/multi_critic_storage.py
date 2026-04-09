"""RolloutStorage that stores per-critic values and per-group rewards for multi-critic PPO."""

from __future__ import annotations

import torch
from tensordict import TensorDict


class MultiCriticRolloutStorage:
    """Rollout storage for multi-critic PPO.

    Stores separate values, rewards, returns and advantages for each critic/reward group.
    """

    class Transition:
        def __init__(self):
            self.observations: torch.Tensor | TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.group_rewards: dict[str, torch.Tensor] | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        num_critics: int,
        reward_group_names: list[str],
        obs,
        actions_shape,
        device="cpu",
        advantage_weights: dict[str, float] | None = None,
    ):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_critics = num_critics
        self.reward_group_names = reward_group_names
        self.actions_shape = actions_shape
        self.advantage_weights = advantage_weights

        # Observations
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )

        # Actions
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # Dones
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # Total reward (for logging compatibility)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # Per-group rewards: (num_transitions, num_envs, 1) each
        self.group_rewards = {
            name: torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            for name in reward_group_names
        }

        # Per-critic values: (num_transitions, num_envs, num_critics)
        self.values = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)

        # PPO quantities
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # Per-critic returns and advantages
        self.returns = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=self.device)

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow!")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # Per-critic values (num_envs, num_critics)
        self.values[self.step].copy_(transition.values)

        # Per-group rewards
        if transition.group_rewards is not None:
            for name, rew in transition.group_rewards.items():
                if name in self.group_rewards:
                    self.group_rewards[name][self.step].copy_(rew.view(-1, 1))

        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        """Compute per-critic returns and advantages using GAE.

        Args:
            last_values: (num_envs, num_critics) from the last observation.
            gamma: discount factor.
            lam: GAE lambda.
            normalize_advantage: whether to normalize advantages.
        """
        for c_idx, group_name in enumerate(self.reward_group_names):
            group_rew = self.group_rewards[group_name]  # (T, num_envs, 1)
            advantage = torch.zeros(self.num_envs, 1, device=self.device)

            for step in reversed(range(self.num_transitions_per_env)):
                if step == self.num_transitions_per_env - 1:
                    next_values = last_values[:, c_idx:c_idx+1]
                else:
                    next_values = self.values[step + 1, :, c_idx:c_idx+1]

                next_is_not_terminal = 1.0 - self.dones[step].float()
                current_values = self.values[step, :, c_idx:c_idx+1]

                delta = group_rew[step] + next_is_not_terminal * gamma * next_values - current_values
                advantage = delta + next_is_not_terminal * gamma * lam * advantage
                self.returns[step, :, c_idx:c_idx+1] = advantage + current_values

            # Per-critic advantage
            self.advantages[:, :, c_idx:c_idx+1] = self.returns[:, :, c_idx:c_idx+1] - self.values[:, :, c_idx:c_idx+1]

        # Normalize each critic's advantage independently, then sum
        if normalize_advantage:
            for c_idx in range(self.num_critics):
                adv = self.advantages[:, :, c_idx:c_idx+1]
                self.advantages[:, :, c_idx:c_idx+1] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Combined advantage: weighted sum of normalized per-critic advantages
        if self.advantage_weights:
            w = torch.tensor(
                [self.advantage_weights.get(name, 1.0) for name in self.reward_group_names],
                device=self.device,
                dtype=self.advantages.dtype,
            ).view(1, 1, -1)
            self.combined_advantages = (self.advantages * w).sum(dim=-1, keepdim=True)
        else:
            self.combined_advantages = self.advantages.sum(dim=-1, keepdim=True)  # (T, num_envs, 1)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.reshape(-1, self.num_critics)              # (batch, num_critics)
        returns = self.returns.reshape(-1, self.num_critics)            # (batch, num_critics)
        combined_advantages = self.combined_advantages.flatten(0, 1)    # (batch, 1)

        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    observations[batch_idx],           # obs_batch
                    actions[batch_idx],                 # actions_batch
                    values[batch_idx],                  # target_values_batch (num_critics)
                    combined_advantages[batch_idx],     # combined advantages
                    returns[batch_idx],                 # returns_batch (num_critics)
                    old_actions_log_prob[batch_idx],    # old log prob
                    old_mu[batch_idx],                  # old mu
                    old_sigma[batch_idx],               # old sigma
                )
