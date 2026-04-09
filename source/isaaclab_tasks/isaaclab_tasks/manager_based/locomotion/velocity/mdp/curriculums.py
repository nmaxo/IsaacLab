# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_target_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "ee_pose",
    success_threshold: float = 0.15,
    difficulty_step: float = 0.05,
    success_pos_threshold: float = 0.1,
    success_ori_threshold: float = 0.2,
) -> dict[str, float]:
    """Per-env curriculum that adjusts EE target difficulty based on tracking error.

    At each environment reset the curriculum checks the final position error.
    If the error is below ``success_threshold`` the difficulty is increased,
    if it is above ``3 * success_threshold`` it is decreased.

    Also tracks success rate: an episode is "successful" if both position error
    < ``success_pos_threshold`` and orientation error < ``success_ori_threshold``
    at the end of the episode.

    The difficulty value lives on the command term's ``difficulty`` buffer and
    is used by :class:`UniformPoseFixedCommand` to interpolate between easy and hard ranges.

    Returns:
        Dict with mean difficulty and success rate for logging.
    """
    ee_cmd = env.command_manager._terms[command_name]
    pos_error = ee_cmd.metrics["position_error"][env_ids]
    ori_error = ee_cmd.metrics["orientation_error"][env_ids]

    # --- Success rate tracking (end-of-episode: pos < 0.1 m, ori < 0.2 rad) ---
    if not hasattr(ee_cmd, "_success_buf"):
        from collections import deque
        ee_cmd._success_buf = deque(maxlen=1000)

    success = (pos_error < success_pos_threshold) & (ori_error < success_ori_threshold)
    ee_cmd._success_buf.extend(success.float().cpu().tolist())
    success_rate = sum(ee_cmd._success_buf) / max(len(ee_cmd._success_buf), 1)

    # --- Curriculum update: same threshold (0.15) for "move up" so difficulty grows when close enough ---
    move_up = pos_error < success_threshold
    move_down = (pos_error > success_threshold * 3.0) & (~move_up)

    ee_cmd.difficulty[env_ids] = torch.clamp(
        ee_cmd.difficulty[env_ids]
        + difficulty_step * move_up.float()
        - difficulty_step * move_down.float(),
        0.0,
        1.0,
    )
    return {
        "mean_difficulty": ee_cmd.difficulty.mean().item(),
        "success_rate": success_rate,
    }


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_distance_ratio: float = 0.5,
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    Args:
        move_up_distance_ratio: Fraction of terrain size (e.g. size[0]) beyond which the robot is
            considered to have "walked far enough" and terrain level is increased. Default 0.5 (half of 8m = 4m).
            For loco-manip with nearby goals (e.g. 0.5--2.5 m), use a smaller value (e.g. 0.25–0.3).

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] * move_up_distance_ratio
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
