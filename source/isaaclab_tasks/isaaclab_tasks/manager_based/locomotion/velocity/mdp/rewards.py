# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# =============================================================================
# Gaussian (Φ) rewards — paper "Multi-critic Learning for Whole-body ..."
# Φ(v, σ²) = exp(-v^T v / σ²). Parameter std = sqrt(σ²).
# =============================================================================


def flat_orientation_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward flat base orientation: Φ(θ_roll,pitch, σ²) = exp(-||proj_gravity_xy||²/σ²)."""
    sq_error = mdp.flat_orientation_l2(env, asset_cfg)
    return torch.exp(-sq_error / (std * std))


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward torso height near target: Φ(h - ĥ, σ²) = exp(-(h-ĥ)²/σ²)."""
    sq_error = mdp.base_height_l2(env, target_height=target_height, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    return torch.exp(-sq_error / (std * std))


def lin_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward small vertical base velocity: Φ(v_z, σ²) = exp(-v_z²/σ²)."""
    sq_error = mdp.lin_vel_z_l2(env, asset_cfg)
    return torch.exp(-sq_error / (std * std))


def ang_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward small roll/pitch angular velocity: Φ(ω_xy, σ²) = exp(-||ω_xy||²/σ²)."""
    sq_error = mdp.ang_vel_xy_l2(env, asset_cfg)
    return torch.exp(-sq_error / (std * std))


def action_rate_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward smooth actions: Φ(a_t - a_{t-1}, σ²) = exp(-||Δa||²/σ²)."""
    sq_error = mdp.action_rate_l2(env)
    return torch.exp(-sq_error / (std * std))


def joint_torques_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward low joint torques: Φ(τ, σ²) = exp(-||τ||²/σ²)."""
    sq_error = mdp.joint_torques_l2(env, asset_cfg)
    return torch.exp(-sq_error / (std * std))


def joint_vel_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward low joint velocities: Φ(q̇, σ²) = exp(-||q̇||²/σ²)."""
    sq_error = mdp.joint_vel_l2(env, asset_cfg)
    return torch.exp(-sq_error / (std * std))


def action_rate_exp_slice(
    env: ManagerBasedRLEnv, std: float, start_idx: int, end_idx: int
) -> torch.Tensor:
    """Reward smooth actions on a slice: Φ(Δa[start:end], σ²) = exp(-||Δa||²/σ²)."""
    delta = env.action_manager.action[:, start_idx:end_idx] - env.action_manager.prev_action[:, start_idx:end_idx]
    sq_error = torch.sum(delta * delta, dim=1)
    return torch.exp(-sq_error / (std * std))


def arm_base_proximity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_body_names: str = "arm_link.*",
    base_body_name: str = "base",
    margin: float = 0.15,
) -> torch.Tensor:
    """Penalize when any arm link is closer than margin to the base (discourages self-intersection).

    Returns the number of envs where min(arm-to-base distance) < margin. Use with negative weight.
    With enabled_self_collisions=False the arm can pass through the base; this term teaches the policy to avoid it.
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_cfg.name]
    base_ids, _ = asset.find_bodies(base_body_name)
    arm_ids, _ = asset.find_bodies(arm_body_names)
    if not base_ids or not arm_ids:
        return torch.zeros(env.num_envs, device=env.device)
    base_pos = asset.data.body_pos_w[:, base_ids[0], :]
    arm_pos = asset.data.body_pos_w[:, arm_ids, :]
    dist = torch.norm(arm_pos - base_pos.unsqueeze(1), dim=-1)
    min_dist = dist.min(dim=1)[0]
    return (min_dist < margin).float()
