"""VecEnv wrapper that extracts per-group rewards for multi-critic training.

The wrapper reads reward_groups config from the environment and computes
per-group reward sums at each step, passing them through extras["reward_groups"].
"""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from isaaclab.envs import ManagerBasedRLEnv


class MultiCriticVecEnvWrapper(VecEnv):
    """Wraps IsaacLab env and extracts per-reward-group rewards for multi-critic PPO.

    The environment config must define `reward_groups`: a dict mapping group name
    to a list of reward term names belonging to that group.
    Example:
        reward_groups = {
            "locomotion": ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "flat_orientation_l2", ...],
            "manipulation": ["position_tracking", "orientation_tracking", ...],
        }
    """

    def __init__(self, env: ManagerBasedRLEnv, reward_groups: dict[str, list[str]], clip_actions: float | None = None):
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError(f"Expected ManagerBasedRLEnv, got {type(env)}")

        self.env = env
        self.clip_actions = clip_actions
        self.reward_groups = reward_groups

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        self._modify_action_space()

        # Build mapping from reward term name -> index in the reward manager
        self._build_reward_term_mapping()

        self.env.reset()

    def _build_reward_term_mapping(self):
        """Map reward term names to their indices in the reward manager."""
        rm = self.unwrapped.reward_manager
        self._term_name_to_idx = {name: idx for idx, name in enumerate(rm._term_names)}

        # Validate that all reward group terms exist
        for group_name, term_names in self.reward_groups.items():
            for term_name in term_names:
                if term_name not in self._term_name_to_idx:
                    available = list(self._term_name_to_idx.keys())
                    raise ValueError(
                        f"Reward term '{term_name}' in group '{group_name}' not found. "
                        f"Available terms: {available}"
                    )

    def _compute_group_rewards(self) -> dict[str, torch.Tensor]:
        """Compute per-group rewards from reward manager's step rewards.

        Returns dict {group_name: (num_envs,) tensor}.
        """
        rm = self.unwrapped.reward_manager
        dt = self.unwrapped.step_dt
        group_rewards = {}

        for group_name, term_names in self.reward_groups.items():
            group_sum = torch.zeros(self.num_envs, device=self.device)
            for term_name in term_names:
                idx = self._term_name_to_idx[term_name]
                # _step_reward stores value / dt (raw weighted reward without dt),
                # but the actual reward used in training is value * dt.
                # We want the same scale as the total reward, so multiply back by dt.
                group_sum += rm._step_reward[:, idx] * dt
            group_rewards[group_name] = group_sum

        return group_rewards

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed=-1):
        return self.unwrapped.seed(seed)

    def reset(self):
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self):
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # Extract per-group rewards (computed from reward_manager internal state)
        extras["reward_groups"] = self._compute_group_rewards()

        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):
        return self.env.close()

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
