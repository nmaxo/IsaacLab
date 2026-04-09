"""Go2-ARX Full Body Control environment.

Main goal: robot walks to a distant target and reaches it with arm_link6 (EE).
Single goal is the EE target pose sampled 0.5–2.5 m from the env origin.
Robot must WALK (legs) and REACH (arm) simultaneously.

Multi-critic PPO with 3 critics (no advantage weighting — pure sum):
  - locomotion:      velocity tracking + base stability (L2 penalties)
  - manipulation:    EE position (tanh) + orientation tracking
  - contact_schedule: foot air time + slide penalty

Reward strategy (hybrid, avoids "standing still" local minimum):
  - TRACKING rewards (exp/tanh): positive weight — reward matching commands
  - REGULARIZATION (L2): negative weight — penalize bad behavior, ~0 when standing
  This ensures robot gets reward ONLY by actively tracking, not by standing still.

The base_velocity command is derived FROM the EE target: heading towards
the target's XY projection. This way there is only ONE user-facing goal
(the EE pose), but the locomotion critic receives a velocity signal
that "guides" the legs towards it.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import (
    orientation_command_error,
    position_command_error_tanh_curriculum,
)
from isaaclab_tasks.manager_based.full_body_control.mdp.rewards import position_command_error
from isaaclab_assets.robots.go2_arx import GO2_ARX_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


# ============================================================
# Scene (rough terrain, self-collisions disabled — handled via reward penalties)
# ============================================================

@configclass
class Go2ArxFBCSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Same as working rough_env: self-collisions OFF in physics engine,
    # arm/body overlap is discouraged via joint_pos_limits + arm_undesired_contacts rewards.
    robot: ArticulationCfg = GO2_ARX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


# ============================================================
# Events (domain randomization)
# ============================================================

@configclass
class Go2ArxFBCEventCfg:
    # startup: randomize friction
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # startup: randomize base mass
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset: external forces (zero by default, placeholder)
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset: base pose
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    # reset: legs — exact default pose so robot starts standing.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            ),
        },
    )

    # reset: arm to default pose so robot doesn't start with arm in fallen/weird pose (causes instant fall).
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_joint.*"]),
        },
    )

    # interval: random push
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


# ============================================================
# Actions: legs + arm
# ============================================================

@configclass
class Go2ArxFBCActionsCfg:
    joint_pos_legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,
        use_default_offset=True,
    )
    joint_pos_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_joint.*"],
        scale=0.2,
        use_default_offset=True,
    )


# ============================================================
# Observations
# ============================================================

@configclass
class Go2ArxFBCObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
        )

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================
# Commands
# ============================================================

@configclass
class Go2ArxFBCCommandsCfg:
    base_velocity = mdp.EeGoalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.8,
        debug_vis=True,
        ee_command_name="ee_pose",
        approach_speed=0.5,
        slowdown_distance=0.3,
        ranges=mdp.EeGoalVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.5),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    ee_pose = mdp.UniformPoseFixedCommandCfg(
        asset_name="robot",
        body_name="arm_link6",
        resampling_time_range=(8.0, 15.0),
        debug_vis=True,
        enable_curriculum=True,
        ranges=mdp.UniformPoseFixedCommandCfg.Ranges(
            pos_x=(0.5, 2.5),
            pos_y=(-1.5, 1.5),
            pos_z=(0.15, 0.85),
            roll=(-0.3, 0.3),
            pitch=(-0.3, 0.3),
            yaw=(-0.3, 0.3),
        ),
        easy_ranges=mdp.UniformPoseFixedCommandCfg.Ranges(
            pos_x=(0.3, 0.8),
            pos_y=(-0.4, 0.4),
            pos_z=(0.15, 0.55),
            roll=(-0.05, 0.05),
            pitch=(-0.05, 0.05),
            yaw=(-0.05, 0.05),
        ),
    )


# ============================================================
# Rewards — paper-style: all Φ(x,σ)=exp(-||x||²/σ²) with positive weights.
# Based on: "Multi-critic Learning for Whole-body End-effector Twist Tracking"
# (Vijayan et al., Table 2). Adapted for pose tracking (vs twist tracking).
# ============================================================

@configclass
class Go2ArxFBCRewardsCfg:

    # ─── LOCOMOTION GROUP ──────────────────────────────────────────────────

    # Velocity tracking — main reward for "walk toward goal". Must dominate over standing still.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.1},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.05},
    )

    # Base stability: enough to prefer not falling, but not so high that standing beats walking.
    base_height_exp = RewTerm(
        func=mdp.base_height_exp,
        weight=1.0,
        params={
            "target_height": 0.34,
            "std": 0.1,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    flat_orientation_exp = RewTerm(
        func=mdp.flat_orientation_exp,
        weight=0.5,
        params={"std": 0.1},
    )
    lin_vel_z_exp = RewTerm(
        func=mdp.lin_vel_z_exp,
        weight=0.5,
        params={"std": 0.2},
    )
    ang_vel_xy_exp = RewTerm(
        func=mdp.ang_vel_xy_exp,
        weight=1.5,
        params={"std": 0.2},
    )

    # Alive: moderate bonus; termination penalty to discourage falling (especially on rough).
    is_alive = RewTerm(func=mdp.is_alive, weight=0.3)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-50.0)

    # Contacts
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # Leg pose: small penalty for deviating from default (like rough_env), encourages natural standing.
    leg_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            ),
        },
    )

    # Smoothness (Φ-kernel)
    robot_action_rate_exp = RewTerm(
        func=mdp.action_rate_exp_slice,
        weight=0.001,
        params={"std": 0.1, "start_idx": 0, "end_idx": 12},
    )
    robot_joint_torque_exp = RewTerm(
        func=mdp.joint_torques_exp,
        weight=0.00001,
        params={
            "std": 40.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
        },
    )
    robot_joint_vel_exp = RewTerm(
        func=mdp.joint_vel_exp,
        weight=0.0001,
        params={
            "std": 4.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
        },
    )

    # Contact schedule: reward stepping when command is non-zero (so walking beats standing).
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    # ─── MANIPULATION GROUP ─────────────────────────────────────────────────
    # EE position: curriculum on std (easy=wide → hard=tight). Main task reward.
    ee_position_tracking = RewTerm(
        func=position_command_error_tanh_curriculum,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm_link6"),
            "std_easy": 0.2,
            "std_hard": 0.05,
            "command_name": "ee_pose",
        },
    )

    # L2 penalty for distance to target — strong gradient "move EE toward goal" when far.
    ee_position_error_l2 = RewTerm(
        func=position_command_error,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm_link6"),
            "command_name": "ee_pose",
        },
    )

    # EE orientation: penalty scaled so it doesn't dominate and zero out manipulation value.
    ee_orientation_tracking = RewTerm(
        func=orientation_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm_link6"),
            "command_name": "ee_pose",
        },
    )

    # Arm contacts (penalizes arm touching anything — base, legs, ground)
    arm_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["arm_link.*"]),
            "threshold": 1.0,
        },
    )

    # Penalize arm too close to base (self-intersection); physics has enabled_self_collisions=False
    arm_base_proximity_penalty = RewTerm(
        func=mdp.arm_base_proximity_penalty,
        weight=-1.0,
        params={"margin": 0.15},
    )

    # Joint limits — keeps arm away from extreme positions that cause self-collision
    arm_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_joint.*"])},
    )

    # Arm smoothness (Φ-kernel)
    arm_action_rate_exp = RewTerm(
        func=mdp.action_rate_exp_slice,
        weight=0.1,
        params={"std": 0.5, "start_idx": 12, "end_idx": 18},
    )
    arm_joint_torque_exp = RewTerm(
        func=mdp.joint_torques_exp,
        weight=0.00001,
        params={
            "std": 40.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_joint.*"]),
        },
    )
    arm_joint_vel_exp = RewTerm(
        func=mdp.joint_vel_exp,
        weight=0.0001,
        params={
            "std": 4.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_joint.*"]),
        },
    )


# ============================================================
# Terminations
# ============================================================

@configclass
class Go2ArxFBCTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},
    )


# ============================================================
# Curriculum
# ============================================================

@configclass
class Go2ArxFBCCurriculumCfg:
    # Loco-manip goals are 0.5–2.5 m away, so 4 m threshold is rarely reached. Use 0.25 (2 m) so terrain level can increase.
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"move_up_distance_ratio": 0.25},
    )
    ee_target = CurrTerm(
        func=mdp.ee_target_curriculum,
        params={
            "command_name": "ee_pose",
            "success_threshold": 0.1,
            "difficulty_step": 0.05,
            "success_pos_threshold": 0.02,
            "success_ori_threshold": 999.0,  # ignore orientation for success metric
        },
    )


# ============================================================
# Reward groups -> 2 critics: locomotion (incl. contact) + manipulation
# ============================================================

REWARD_GROUPS = {
    "locomotion": [
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "base_height_exp",
        "flat_orientation_exp",
        "lin_vel_z_exp",
        "ang_vel_xy_exp",
        "is_alive",
        "is_terminated",
        "undesired_contacts",
        "leg_joint_deviation",
        "robot_action_rate_exp",
        "robot_joint_torque_exp",
        "robot_joint_vel_exp",
        "feet_air_time",
        "feet_slide",
    ],
    "manipulation": [
        "ee_position_tracking",
        "ee_position_error_l2",
        "ee_orientation_tracking",
        "arm_undesired_contacts",
        "arm_base_proximity_penalty",
        "arm_joint_pos_limits",
        "arm_action_rate_exp",
        "arm_joint_torque_exp",
        "arm_joint_vel_exp",
    ],
}


# ============================================================
# Main env config
# ============================================================

@configclass
class Go2ArxLocoManipEnvCfg(ManagerBasedRLEnvCfg):
    """Go2-ARX Full Body Control: walk to target and reach it with EE (rough terrain)."""

    scene: Go2ArxFBCSceneCfg = Go2ArxFBCSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: Go2ArxFBCActionsCfg = Go2ArxFBCActionsCfg()
    observations: Go2ArxFBCObservationsCfg = Go2ArxFBCObservationsCfg()
    events: Go2ArxFBCEventCfg = Go2ArxFBCEventCfg()
    commands: Go2ArxFBCCommandsCfg = Go2ArxFBCCommandsCfg()
    rewards: Go2ArxFBCRewardsCfg = Go2ArxFBCRewardsCfg()
    terminations: Go2ArxFBCTerminationsCfg = Go2ArxFBCTerminationsCfg()
    curriculum: Go2ArxFBCCurriculumCfg = Go2ArxFBCCurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 0.0025       # 400 Hz
        self.decimation = 8        # 50 Hz control
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Scale down terrain heights for small Go2 robot
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Terrain curriculum
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class Go2ArxLocoManipEnvCfg_PLAY(Go2ArxLocoManipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Reduce terrain for play
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # No random pushing during play
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ============================================================
# Flat terrain variant — easier to learn locomotion first
# ============================================================

@configclass
class Go2ArxLocoManipFlatEnvCfg(Go2ArxLocoManipEnvCfg):
    """Go2-ARX loco-manip on flat terrain (no rough terrain curriculum).

    Use this for initial training. Robot learns to walk toward the EE goal
    without the distraction of falling on rough terrain.
    """

    def __post_init__(self):
        super().__post_init__()

        # Flat surface — no terrain generator
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No height scanner (returns zeros on flat, just wastes obs space)
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # base_height_exp: without sensor_cfg height is measured in world frame (fine on flat)
        self.rewards.base_height_exp.params.pop("sensor_cfg", None)

        # No terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class Go2ArxLocoManipFlatEnvCfg_PLAY(Go2ArxLocoManipFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
