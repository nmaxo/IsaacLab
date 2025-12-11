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
    if isinstance(asset_cfg.body_ids, slice):
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids][:, 0]
    else:
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]

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
    # Если body_ids - это срез, получаем все элементы и берём первый
    if isinstance(asset_cfg.body_ids, slice):
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids][:, 0]
    else:
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
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
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        index = body_ids.start or 0
    else:
        index = body_ids[0]

    curr_quat_w = asset.data.body_quat_w[:, index]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


##### НОВЫЕ ФУНКЦИИ НАГРАД #####

def goal_reached_bonus(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    position_threshold: float = 0.05,
    orientation_threshold: float = 0.,
) -> torch.Tensor:
    """
    Большая награда за успешное достижение цели.
    Выдается только когда робот находится в пределах заданных порогов по позиции и ориентации.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Проверка позиции
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    if isinstance(asset_cfg.body_ids, slice):
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids][:, 0]
    else:
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    position_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    
    # Проверка ориентации
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        index = body_ids.start or 0
    else:
        index = body_ids[0]

    curr_quat_w = asset.data.body_quat_w[:, index]
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    
    # Условие достижения цели
    goal_reached = (position_error < position_threshold) & (orientation_error < orientation_threshold)
    
    return goal_reached.float()


def episode_failure_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    failure_distance: float = 0.3,
) -> torch.Tensor:
    """
    Штраф за неудачное завершение эпизода.
    Применяется, когда эпизод заканчивается, но цель не достигнута.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Вычисляем текущую ошибку позиции
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    if isinstance(asset_cfg.body_ids, slice):
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids][:, 0]
    else:
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    
    # Проверяем, близки ли мы к концу эпизода
    # Используем episode_length_buf для определения
    max_episode_length = env.max_episode_length
    current_step = env.episode_length_buf
    
    # Применяем штраф только в конце эпизода, если цель не достигнута
    near_end = current_step >= (max_episode_length - 1)
    failed = distance > failure_distance
    
    penalty = (near_end & failed).float()
    
    return penalty


def time_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Прогрессивный штраф за каждый шаг эпизода.
    Мотивирует агента выполнять задачу быстрее.
    """
    # Постоянный штраф за каждый шаг
    return torch.ones(env.num_envs, device=env.device)





# ============================================================
# ДВЕ ГЛАВНЫЕ ФУНКЦИИ С ДИНАМИЧЕСКИМИ ВЕСАМИ
# ============================================================

def navigation_rewards_combined(
    env, 
    command_name: str = "ee_pose",
    transition_distance: float = 0.5,
    k: float = 5.0
) -> torch.Tensor:
    """
    Навигационная награда с весом w_nav(d), где
    w_nav(d) = (1 - tanh(k*(D - d)/D)) / 2
    """

    base_asset_cfg = SceneEntityCfg("robot", body_names='gripper_link')
    
    # --- Основные компоненты (НЕ меняем) ---
    pos_tracking = position_command_error(
        env, asset_cfg=base_asset_cfg, command_name=command_name
    )
    
    pos_tracking_fine = position_command_error_tanh(
        env, std=0.5, asset_cfg=base_asset_cfg, command_name=command_name
    )
    
    # stability = stability_reward(
    #     env, command_name=command_name,
    #     position_threshold=0.2,
    #     orientation_threshold=0.4,
    #     lin_velocity_threshold=0.2,
    #     ang_velocity_threshold=0.2
    # )
    
    # vel_penalty = distance_based_velocity_penalty(env, command_name)

    base_reward = (
        -0.15 * pos_tracking +
        0.25 * pos_tracking_fine 
    )

    # --- distance ---
    d = position_command_error(
        env, asset_cfg=base_asset_cfg, command_name=command_name
    )

    # --- Новый вес: w_nav(d) ---
    x = k * (transition_distance - d) / transition_distance
    w_nav = (1.0 - torch.tanh(x)) / 2.0

    return base_reward * w_nav



def manipulation_rewards_combined(
    env,
    command_name: str = "ee_pose",
    transition_distance: float = 0.5,
    k: float = 5.0
) -> torch.Tensor:
    """
    Манипуляционная награда с весом w_manip(d), где
    w_manip(d) = (1 + tanh(k*(D - d)/D)) / 2
    """

    ee_asset_cfg = SceneEntityCfg("robot", body_names='gripper_link')
    
    # --- Основные компоненты (НЕ меняем) ---
    ee_pos_tracking = position_command_error(
        env, asset_cfg=ee_asset_cfg, command_name=command_name
    )
    
    ee_pos_tracking_fine = position_command_error_tanh(
        env, std=0.25, asset_cfg=ee_asset_cfg, command_name=command_name
    )
    
    ee_orientation = orientation_command_error(
        env, asset_cfg=ee_asset_cfg, command_name=command_name
    )

    base_reward = (
        -0.45 * ee_pos_tracking +
        0.35 * ee_pos_tracking_fine +
        (-0.3) * ee_orientation
    )

    # --- distance ---
    d = ee_pos_tracking

    # --- Новый вес: w_manip(d) ---
    x = k * (transition_distance - d) / transition_distance
    w_manip = (1.0 + torch.tanh(x)) / 2.0

    return base_reward * w_manip



#######################################################################################################################################################################
def goal_reached_bool(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    position_threshold: float = 0.05,
    orientation_threshold: float = 0.,
) -> torch.Tensor:
    """
    termination for goal_reachrd completion,
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Проверка позиции
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    if isinstance(asset_cfg.body_ids, slice):
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids][:, 0]
    else:
        curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    position_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    
    # Проверка ориентации
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        index = body_ids.start or 0
    else:
        index = body_ids[0]

    curr_quat_w = asset.data.body_quat_w[:, index]
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    
    # Условие достижения цели
    goal_reached = (position_error < position_threshold) & (orientation_error < orientation_threshold)
    
    return goal_reached



