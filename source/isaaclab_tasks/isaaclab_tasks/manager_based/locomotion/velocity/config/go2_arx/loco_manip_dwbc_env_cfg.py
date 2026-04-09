"""Go2-ARX DWBC environment config.

Same observations as Go2-PRX FBC (policy only) + DWBC-style domain randomization events.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .loco_manip_env_cfg import (
    Go2ArxFBCEventCfg,
    Go2ArxFBCObservationsCfg,
    Go2ArxLocoManipEnvCfg,
    Go2ArxLocoManipFlatEnvCfg,
)


@configclass
class Go2ArxDwbcObservationsCfg(Go2ArxFBCObservationsCfg):
    """Policy observations only (same as loco-manip FBC)."""


# ── DWBC-style motor strength (domain rand buffer; not exposed as observation) ──


def randomize_motor_strength(
    env,
    env_ids,
    leg_strength_range: tuple[float, float] = (0.7, 1.3),
    arm_strength_range: tuple[float, float] = (0.7, 1.3),
    num_leg_joints: int = 12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize per-joint multipliers; stored on env for optional logging."""
    from isaaclab.assets import Articulation
    import torch

    asset: Articulation = env.scene[asset_cfg.name]
    num_joints = asset.data.joint_pos.shape[1]

    if not hasattr(env, "_dwbc_motor_strength"):
        env._dwbc_motor_strength = torch.ones(env.num_envs, num_joints, device=env.device)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    n = env_ids.shape[0]
    leg_strength = torch.empty(n, num_leg_joints, device=env.device).uniform_(
        leg_strength_range[0], leg_strength_range[1]
    )
    arm_joints = num_joints - num_leg_joints
    arm_strength = torch.empty(n, arm_joints, device=env.device).uniform_(
        arm_strength_range[0], arm_strength_range[1]
    )
    env._dwbc_motor_strength[env_ids] = torch.cat([leg_strength, arm_strength], dim=-1)


@configclass
class Go2ArxDwbcEventCfg(Go2ArxFBCEventCfg):
    """DWBC-like domain randomization (friction, mass, COM, motor scale, pushes)."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 3.0),
            "dynamic_friction_range": (0.3, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1000,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 2.5),
            "operation": "add",
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.15, 0.15),
            },
        },
    )

    randomize_motor = EventTerm(
        func=randomize_motor_strength,
        mode="startup",
        params={
            "leg_strength_range": (0.7, 1.3),
            "arm_strength_range": (0.7, 1.3),
            "num_leg_joints": 12,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class Go2ArxLocoManipDwbcEnvCfg(Go2ArxLocoManipEnvCfg):
    """Go2-ARX DWBC (rough terrain): FBC policy obs + DWBC events."""

    observations: Go2ArxDwbcObservationsCfg = Go2ArxDwbcObservationsCfg()
    events: Go2ArxDwbcEventCfg = Go2ArxDwbcEventCfg()


@configclass
class Go2ArxLocoManipDwbcEnvCfg_PLAY(Go2ArxLocoManipDwbcEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2ArxLocoManipDwbcFlatEnvCfg(Go2ArxLocoManipFlatEnvCfg):
    """Go2-ARX DWBC (flat terrain)."""

    observations: Go2ArxDwbcObservationsCfg = Go2ArxDwbcObservationsCfg()
    events: Go2ArxDwbcEventCfg = Go2ArxDwbcEventCfg()


@configclass
class Go2ArxLocoManipDwbcFlatEnvCfg_PLAY(Go2ArxLocoManipDwbcFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
