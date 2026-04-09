"""PPO for DWBC-style dual rewards: separate value heads + advantage mixing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic_dwbc import ActorCriticDWBC
from .rollout_storage_dwbc import DwbcRolloutStorage


class PPO_DWBC:
    actor_critic: ActorCriticDWBC

    def __init__(
        self,
        actor_critic: ActorCriticDWBC,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = None,
        device: str = "cpu",
        mixing_schedule: list[float] | None = None,
        min_policy_std: list[float] | None = None,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        self.storage: DwbcRolloutStorage | None = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = DwbcRolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        if min_policy_std is not None:
            self.min_policy_std = torch.tensor(min_policy_std, device=self.device)
        else:
            self.min_policy_std = None

        self.mixing_schedule = mixing_schedule or [1.0, 0, 3000]
        self.counter = 0

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        num_prop: int,
        action_shape: list[int],
    ):
        self.storage = DwbcRolloutStorage(
            num_envs, num_transitions_per_env, num_prop, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs_prop: torch.Tensor) -> torch.Tensor:
        self.transition.actions = self.actor_critic.act(obs_prop).detach()
        self.transition.values = self.actor_critic.evaluate(obs_prop).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.obs_prop = obs_prop
        return self.transition.actions

    def process_env_step(self, rewards: torch.Tensor, arm_rewards: torch.Tensor, dones: torch.Tensor, infos: dict):
        self.transition.rewards = torch.stack([rewards.clone(), arm_rewards.clone()], dim=-1)
        self.transition.dones = dones

        if "time_outs" in infos:
            self.transition.rewards += (
                self.gamma * self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device)
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_obs_prop: torch.Tensor):
        last_values = self.actor_critic.evaluate(last_obs_prop).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def get_value_mixing_ratio(self) -> float:
        sched = self.mixing_schedule
        return min(max((self.counter - sched[1]) / max(sched[2], 1), 0), 1) * sched[0]

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0

        value_mixing_ratio = self.get_value_mixing_ratio()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_prop_b,
            actions_b,
            target_values_b,
            advantages_b,
            returns_b,
            old_actions_log_prob_b,
            old_mu_b,
            old_sigma_b,
        ) in generator:

            self.actor_critic.act(obs_prop_b)
            actions_log_prob_b = self.actor_critic.get_actions_log_prob(actions_b)
            value_b = self.actor_critic.evaluate(obs_prop_b)
            mu_b = self.actor_critic.action_mean
            sigma_b = self.actor_critic.action_std
            entropy_b = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_b / old_sigma_b + 1e-5)
                        + (old_sigma_b.pow(2) + (old_mu_b - mu_b).pow(2)) / (2.0 * sigma_b.pow(2))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            mixing_adv = torch.zeros_like(advantages_b)
            mixing_adv[..., 0] = advantages_b[..., 0] + value_mixing_ratio * advantages_b[..., 1]
            mixing_adv[..., 1] = advantages_b[..., 1] + value_mixing_ratio * advantages_b[..., 0]

            ratio = torch.exp(actions_log_prob_b - old_actions_log_prob_b)
            surrogate = -mixing_adv * ratio
            surrogate_clipped = -mixing_adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_b + (value_b - target_values_b).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_b - returns_b).pow(2)
                value_losses_clipped = (value_clipped - returns_b).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_b - value_b).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_b.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        self.storage.clear()
        self.counter += 1
        self._enforce_min_std()

        return {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "mixing_ratio": value_mixing_ratio,
            "learning_rate": self.learning_rate,
        }

    def _enforce_min_std(self):
        if self.min_policy_std is not None:
            current_std = self.actor_critic.std.detach()
            new_std = torch.max(current_std, self.min_policy_std).detach()
            self.actor_critic.std.data = new_std
