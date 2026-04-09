"""VecEnv wrapper for DWBC: dual (leg/arm) rewards and policy observations only."""

from __future__ import annotations

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv
from isaaclab.envs import ManagerBasedRLEnv


class DwbcVecEnvWrapper(VecEnv):
    """Wraps IsaacLab env: two reward groups + policy observation group."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        reward_groups: dict[str, list[str]],
        clip_actions: float | None = None,
        reward_scale: float | None = None,
        action_delay_steps: int = 0,
    ):
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError(f"Expected ManagerBasedRLEnv, got {type(env)}")

        self.env = env
        self.clip_actions = clip_actions
        self.reward_groups = reward_groups
        self.reward_scale = reward_scale
        self.action_delay_steps = action_delay_steps

        group_names = list(reward_groups.keys())
        if len(group_names) != 2:
            raise ValueError(f"DWBC expects exactly 2 reward groups, got {len(group_names)}: {group_names}")
        self.leg_group_name = group_names[0]
        self.arm_group_name = group_names[1]

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        self._modify_action_space()
        self._build_reward_term_mapping()

        if self.action_delay_steps > 0:
            self._action_buf = torch.zeros(
                self.action_delay_steps + 1, self.num_envs, self.num_actions, device=self.device
            )

        self.env.reset()

    def _build_reward_term_mapping(self):
        rm = self.unwrapped.reward_manager
        self._term_name_to_idx = {name: idx for idx, name in enumerate(rm._term_names)}
        for group_name, term_names in self.reward_groups.items():
            for term_name in term_names:
                if term_name not in self._term_name_to_idx:
                    available = list(self._term_name_to_idx.keys())
                    raise ValueError(
                        f"Reward term '{term_name}' in group '{group_name}' not found. Available: {available}"
                    )

    def _compute_group_reward(self, group_name: str) -> torch.Tensor:
        rm = self.unwrapped.reward_manager
        scale = self.reward_scale if self.reward_scale is not None else self.unwrapped.step_dt
        group_sum = torch.zeros(self.num_envs, device=self.device)
        for term_name in self.reward_groups[group_name]:
            idx = self._term_name_to_idx[term_name]
            group_sum += rm._step_reward[:, idx] * scale
        return group_sum

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

    def _policy_obs(self, obs_dict: dict) -> torch.Tensor:
        policy_obs = obs_dict.get("policy")
        if policy_obs is None:
            raise KeyError(f"Observation dict must contain 'policy' key. Got: {list(obs_dict.keys())}")
        return policy_obs

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, extras = self.env.reset()
        return self._policy_obs(obs_dict), extras

    def get_observations(self) -> torch.Tensor:
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._policy_obs(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        if self.action_delay_steps > 0:
            self._action_buf = torch.roll(self._action_buf, shifts=-1, dims=0)
            self._action_buf[-1] = actions
            actions = self._action_buf[0]

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        leg_rewards = self._compute_group_reward(self.leg_group_name)
        arm_rewards = self._compute_group_reward(self.arm_group_name)

        obs_prop = self._policy_obs(obs_dict)
        return obs_prop, leg_rewards, arm_rewards, dones, extras

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
