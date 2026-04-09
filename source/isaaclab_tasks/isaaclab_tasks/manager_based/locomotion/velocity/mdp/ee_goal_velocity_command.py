"""Velocity command that automatically points toward the EE target pose.

Instead of sampling a random heading, this command computes heading as
atan2(ee_target_y - robot_y, ee_target_x - robot_x) every step,
so the arrow always points toward the EE goal.

Linear velocity magnitude is proportional to XY distance to the EE target
(clamped to the configured range), so the robot speeds up when far and
slows down when close.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class EeGoalVelocityCommand(UniformVelocityCommand):
    """Velocity command that steers the robot toward the EE target.

    Each step:
    1. Reads the EE target world position from the 'ee_pose' command term.
    2. Computes heading = atan2(dy, dx) from robot to target.
    3. Sets linear velocity proportional to XY distance (clamped).
    4. Uses heading_control_stiffness to convert heading error -> angular velocity.
    """

    cfg: EeGoalVelocityCommandCfg

    def __init__(self, cfg: EeGoalVelocityCommandCfg, env: ManagerBasedEnv):
        cfg.heading_command = True
        super().__init__(cfg, env)
        self._ee_command_name = cfg.ee_command_name

    @property
    def _ee_pose_term(self):
        """Lazily resolve the ee_pose CommandTerm (available after all terms are created)."""
        return self._env.command_manager._terms[self._ee_command_name]

    def _resample_command(self, env_ids: Sequence[int]):
        """Override: don't sample random heading/velocity — computed in _update_command."""
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 0] = 0.0
        self.vel_command_b[env_ids, 2] = 0.0
        self.is_heading_env[env_ids] = True
        self.is_standing_env[env_ids] = False

    def _update_command(self):
        """Compute heading and speed from EE target position every step."""
        ee_term = self._ee_pose_term
        ee_target_w = ee_term.pose_command_w[:, :3]
        robot_pos_w = self.robot.data.root_pos_w[:, :3]

        dx = ee_target_w[:, 0] - robot_pos_w[:, 0]
        dy = ee_target_w[:, 1] - robot_pos_w[:, 1]
        dist_xy = torch.sqrt(dx * dx + dy * dy + 1e-6)

        heading_to_goal = torch.atan2(dy, dx)
        self.heading_target[:] = heading_to_goal

        speed = self.cfg.approach_speed * torch.clamp(dist_xy / self.cfg.slowdown_distance, max=1.0)
        self.vel_command_b[:, 0] = torch.clamp(
            speed,
            min=self.cfg.ranges.lin_vel_x[0],
            max=self.cfg.ranges.lin_vel_x[1],
        )

        heading_error = math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        self.vel_command_b[:, 2] = torch.clamp(
            self.cfg.heading_control_stiffness * heading_error,
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )
        self.vel_command_b[:, 1] = 0.0

    def _debug_vis_callback(self, event):
        """Зелёная стрелка = направление на цель в мире. Синяя = текущая скорость базы."""
        if not self.robot.is_initialized:
            return
        ee_term = self._ee_pose_term
        ee_target_w = ee_term.pose_command_w[:, :3]
        robot_pos_w = self.robot.data.root_pos_w[:, :3]
        dx = ee_target_w[:, 0] - robot_pos_w[:, 0]
        dy = ee_target_w[:, 1] - robot_pos_w[:, 1]
        dist_xy = torch.sqrt(dx * dx + dy * dy + 1e-6)
        heading_to_goal_w = torch.atan2(dy, dx)
        zeros = torch.zeros_like(heading_to_goal_w)
        goal_arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_to_goal_w)
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        goal_arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)
        goal_arrow_scale[:, 0] *= torch.clamp(dist_xy * 2.0, max=1.5)
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        self.goal_vel_visualizer.visualize(base_pos_w, goal_arrow_quat, goal_arrow_scale)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)


@configclass
class EeGoalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for velocity command that tracks EE target heading."""

    class_type: type = EeGoalVelocityCommand

    ee_command_name: str = MISSING
    """Name of the ee_pose command term in CommandManager (e.g. 'ee_pose')."""

    approach_speed: float = 0.4
    """Linear speed (m/s) when far from the EE target. Clamped to lin_vel_x range."""

    slowdown_distance: float = 0.5
    """Distance (m) at which the robot starts slowing down."""
