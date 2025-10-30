# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



#####CUSTOM REWARDS#####

def stability_reward(
    env: "ManagerBasedRLEnv",
    command_name: str,
    position_threshold: float = 0.3,
    orientation_threshold: float = 0.3,
    lin_velocity_threshold: float = 0.4,
    ang_velocity_threshold: float = 0.4
) -> torch.Tensor:
    """
    Награда за полную стабильность у цели.
    Робот должен быть: на месте, правильно ориентирован и неподвижен.
    """
    command = env.command_manager.get_command(command_name)

    # Используем готовые функции для позиции и ориентации
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    position_error = distance

    # Скорости робота
    lin_vel = torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=-1)
    ang_vel = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2])

    # Проверяем условия
    at_position = position_error < position_threshold
    lin_stopped = lin_vel < lin_velocity_threshold
    ang_stopped = ang_vel < ang_velocity_threshold

    # Полная стабильность
    fully_stable = at_position & lin_stopped & ang_stopped

    # Градуированная награда
    reward = at_position.float() * 0.25
    reward += lin_stopped.float() * 0.25
    reward += ang_stopped.float() * 0.25
    reward += fully_stable.float() * 1.0

    return reward


def distance_based_velocity_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    activation_distance: float = 1.0
) -> torch.Tensor:
    """
    Штраф за скорость, который увеличивается по мере приближения к цели.
    """
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)

    # Используем функции для скоростей
    lin_vel = torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=-1)
    ang_vel = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2])

    # Штраф пропорционален скорости и близости к цели
    penalty_factor = torch.clamp(1.0 - distance / activation_distance, 0.0, 1.0)
    penalty = penalty_factor * (lin_vel + ang_vel * 0.5)

    return penalty


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore

    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)