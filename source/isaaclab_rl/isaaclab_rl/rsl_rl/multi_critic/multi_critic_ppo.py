"""Multi-Critic PPO algorithm.

Splits rewards into groups, each with its own critic and value function.
Per-critic advantages are normalized independently and summed for the policy update.
Based on: "Multi-critic Learning for Whole-body End-effector Twist Tracking" (Vijayan et al.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_multi_critic import ActorMultiCritic
from .multi_critic_storage import MultiCriticRolloutStorage


class MultiCriticPPO:
    """Multi-Critic PPO: one actor, N critics (one per reward group)."""

    policy: ActorMultiCritic

    def __init__(
        self,
        policy: ActorMultiCritic,
        reward_group_names: list[str],
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
    ):
        self.device = device
        self.policy = policy
        self.policy.to(self.device)

        self.reward_group_names = reward_group_names
        self.num_critics = len(reward_group_names)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.storage: MultiCriticRolloutStorage = None  # type: ignore
        self.transition = MultiCriticRolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # no RND / symmetry support in multi-critic for now
        self.rnd = None

    def init_storage(self, num_envs, num_transitions_per_env, obs, actions_shape, advantage_weights=None):
        self.storage = MultiCriticRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            num_critics=self.num_critics,
            reward_group_names=self.reward_group_names,
            obs=obs,
            actions_shape=actions_shape,
            device=self.device,
            advantage_weights=advantage_weights,
        )

    def act(self, obs):
        self.transition.actions = self.policy.act(obs).detach()
        # evaluate all critics -> (num_envs, num_critics)
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        self.policy.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Extract per-group rewards from extras
        group_rewards = extras.get("reward_groups", None)
        self.transition.group_rewards = group_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            values = self.transition.values
            assert values is not None
            rewards = self.transition.rewards
            assert rewards is not None
            # Bootstrap total rewards
            rewards += self.gamma * torch.squeeze(
                values.mean(dim=-1, keepdim=True) * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
            self.transition.rewards = rewards
            # Bootstrap per-group rewards
            if group_rewards is not None:
                for c_idx, name in enumerate(self.reward_group_names):
                    if name in group_rewards:
                        group_rewards[name] = group_rewards[name] + self.gamma * (
                            values[:, c_idx] * extras["time_outs"].to(self.device)
                        )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        last_values = self.policy.evaluate(obs).detach()  # (num_envs, num_critics)
        self.storage.compute_returns(
            last_values, self.gamma, self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        per_critic_value_losses = [0.0] * self.num_critics

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,     # (batch, num_critics)
            advantages_batch,        # (batch, 1) combined
            returns_batch,           # (batch, num_critics)
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:

            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Actor forward pass
            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)

            # All critics forward pass -> (batch, num_critics)
            value_batch = self.policy.evaluate(obs_batch)

            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Adaptive learning rate via KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss (uses combined advantage)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Per-critic value losses
            total_value_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
            for c_idx in range(self.num_critics):
                v = value_batch[:, c_idx:c_idx+1]
                r = returns_batch[:, c_idx:c_idx+1]
                tv = target_values_batch[:, c_idx:c_idx+1]

                if self.use_clipped_value_loss:
                    v_clipped = tv + (v - tv).clamp(-self.clip_param, self.clip_param)
                    vl = (v - r).pow(2)
                    vl_clipped = (v_clipped - r).pow(2)
                    critic_loss = torch.max(vl, vl_clipped).mean()
                else:
                    critic_loss = (r - v).pow(2).mean()

                total_value_loss = total_value_loss + critic_loss
                per_critic_value_losses[c_idx] += critic_loss.item()

            loss = surrogate_loss + self.value_loss_coef * total_value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += total_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        per_critic_value_losses = [v / num_updates for v in per_critic_value_losses]

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        for c_idx, name in enumerate(self.reward_group_names):
            loss_dict[f"value_{name}"] = per_critic_value_losses[c_idx]

        return loss_dict
