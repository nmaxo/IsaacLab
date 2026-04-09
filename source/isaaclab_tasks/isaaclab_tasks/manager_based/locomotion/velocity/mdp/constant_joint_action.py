# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term that applies a constant joint position target (e.g. to fix an arm in place)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ConstantJointPositionAction(ActionTerm):
    """Applies a constant joint position target every step. Action dim is 0 (no policy input)."""

    cfg: ConstantJointPositionActionCfg
    _asset: Articulation
    _joint_ids: list[int]
    _target: torch.Tensor

    def __init__(self, cfg: ConstantJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._num_joints = len(self._joint_ids)
        pos = cfg.position
        if isinstance(pos, (int, float)):
            self._target = torch.full(
                (self.num_envs, self._num_joints), float(pos), device=self.device, dtype=torch.float32
            )
        else:
            self._target = torch.tensor(pos, device=self.device, dtype=torch.float32).unsqueeze(0).expand(
                self.num_envs, -1
            )
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)


@configclass
class ConstantJointPositionActionCfg(ActionTermCfg):
    """Configuration for constant joint position action (0-dim action, just applies fixed targets)."""

    class_type: type[ActionTerm] = ConstantJointPositionAction

    asset_name: str = "robot"
    """Name of the articulation asset."""
    joint_names: list[str] = ["arm_joint.*"]
    """Joint name expressions to apply the constant position to."""
    position: float | list[float] = 0.0
    """Target position(s). If float, same for all joints; if list, per joint."""
