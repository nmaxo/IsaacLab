# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pose2d command that uses EE target position as the base navigation goal.

Each step the goal is (ee_pose.x, ee_pose.y) in world, heading = atan2 to goal.
Output format matches UniformPose2dCommand: command shape (num_envs, 4)
[pos_command_b (3), heading_command_b (1)] for compatibility with navigation rewards.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class EeGoalPose2dCommand(CommandTerm):
    """Pose2d command derived from EE target: base goal = (ee_x, ee_y), heading to goal.

    Does not resample; target is updated every step from ee_pose command term.
    Command shape (num_envs, 4): pos_command_b (3), heading_command_b (1).
    """

    cfg: "EeGoalPose2dCommandCfg"

    def __init__(self, cfg: EeGoalPose2dCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._ee_command_name = cfg.ee_command_name
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def _ee_pose_term(self):
        return self._env.command_manager._terms[self._ee_command_name]

    @property
    def command(self) -> torch.Tensor:
        """Desired 2D-pose in base frame. Shape (num_envs, 4)."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        # Goal is driven by ee_pose; nothing to resample
        pass

    def _update_command(self):
        ee_term = self._ee_pose_term
        ee_target_w = ee_term.pose_command_w[:, :3]
        robot_pos_w = self.robot.data.root_pos_w[:, :3]
        # Base goal: same xy as EE target, height = robot z
        self.pos_command_w[:, 0] = ee_target_w[:, 0]
        self.pos_command_w[:, 1] = ee_target_w[:, 1]
        self.pos_command_w[:, 2] = robot_pos_w[:, 2]
        # Heading = toward goal
        dx = ee_target_w[:, 0] - robot_pos_w[:, 0]
        dy = ee_target_w[:, 1] - robot_pos_w[:, 1]
        self.heading_command_w[:] = torch.atan2(dy, dx)
        # To base frame (same as UniformPose2dCommand)
        target_vec = self.pos_command_w - robot_pos_w
        self.pos_command_b[:] = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w), target_vec
        )
        self.heading_command_b[:] = wrap_to_pi(
            self.heading_command_w - self.robot.data.heading_w
        )

    def _update_metrics(self):
        self.metrics["error_pos"] = torch.norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        )


@configclass
class EeGoalPose2dCommandCfg(CommandTermCfg):
    """Config for pose2d command that tracks EE target position."""

    class_type: type = EeGoalPose2dCommand

    asset_name: str = "robot"
    ee_command_name: str = "ee_pose"
