"""Husky UR5 DWBC environment config.

Same as FBC but rewards split into locomotion/manipulation for DWBC dual-critic;
policy observations only (no privileged group).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.full_body_control.mdp as mdp
from isaaclab_tasks.manager_based.full_body_control.mdp.custom_mdp import DiffDriveVelocityActionCfg
from isaaclab_assets.robots import ur5_husky

UR5M_CFG = ur5_husky.UR5M_CFG


# ============================================================
# Scene (identical to FBC_env_cfg)
# ============================================================

@configclass
class HuskyDwbcSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = UR5M_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


# ============================================================
# Events (identical to FBC_env_cfg)
# ============================================================

@configclass
class HuskyDwbcEventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )


# ============================================================
# Actions (identical to FBC_env_cfg)
# ============================================================

@configclass
class HuskyDwbcActionsCfg:
    vel_actions: ActionTerm = DiffDriveVelocityActionCfg(
        asset_name="robot",
        joint_names=["front_left_wheel_joint", "front_right_wheel_joint",
                     "rear_left_wheel_joint", "rear_right_wheel_joint"],
        wheel_radius=0.17775,
        wheel_base=0.5708,
        max_linear_speed=3.0,
        max_angular_speed=2.5,
        scale=1.0,
        linear_velocity_sign=-1.0,
    )
    arm_actions: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        scale=0.4,
        use_default_offset=False,
    )


# ============================================================
# Observations (same policy obs as FBC)
# ============================================================

@configclass
class HuskyDwbcObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()


# ============================================================
# Commands (identical to FBC_env_cfg — single ee_pose command)
# ============================================================

@configclass
class HuskyDwbcCommandsCfg:
    ee_pose = mdp.UniformPoseFixedCommandCfg(
        asset_name="robot",
        body_name="gripper_link",
        resampling_time_range=(10.0, 18.0),
        debug_vis=True,
        ranges=mdp.UniformPoseFixedCommandCfg.Ranges(
            pos_x=(-1.5, 1.5),
            pos_y=(-2.2, 2.2),
            pos_z=(0.6, 0.95),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ============================================================
# Rewards — same as FBC_env_cfg, split into two groups
# ============================================================

@configclass
class HuskyDwbcRewardsCfg:
    # ─── LOCOMOTION GROUP (base movement, regularization) ─────────────

    approach_penalty = RewTerm(
        func=mdp.distance_based_velocity_penalty,
        weight=-0.3,
        params={"command_name": "ee_pose"},
    )
    stability_bonus = RewTerm(
        func=mdp.stability_reward,
        weight=3.0,
        params={
            "command_name": "ee_pose",
            "position_threshold": 0.15,
            "orientation_threshold": 0.4,
            "lin_velocity_threshold": 0.15,
            "ang_velocity_threshold": 0.3,
        },
    )
    time_penalty = RewTerm(
        func=mdp.time_penalty,
        weight=-0.01,
        params={},
    )
    action_smoothness = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.008,
    )
    joint_velocity_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc_penalty = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.000008,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-6,
        params={},
    )

    # ─── MANIPULATION GROUP (EE tracking, goal reaching) ──────────────

    position_tracking_global = RewTerm(
        func=mdp.position_command_error,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
        },
    )
    position_tracking_1 = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "std": 0.12,
            "command_name": "ee_pose",
        },
    )
    position_tracking_2 = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "std": 0.3,
            "command_name": "ee_pose",
        },
    )
    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
        },
    )
    goal_reached_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
            "position_threshold": 0.08,
            "orientation_threshold": 0.3,
        },
    )


# ============================================================
# Terminations (identical to FBC_env_cfg)
# ============================================================

@configclass
class HuskyDwbcTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_reached = DoneTerm(
        func=mdp.goal_reached_bool,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
            "position_threshold": 0.05,
            "orientation_threshold": 0.2,
        },
    )


# ============================================================
# Reward groups for DWBC dual-critic
# ============================================================

REWARD_GROUPS = {
    "locomotion": [
        "approach_penalty",
        "stability_bonus",
        "time_penalty",
        "action_smoothness",
        "joint_velocity_penalty",
        "joint_acc_penalty",
        "torque_penalty",
    ],
    "manipulation": [
        "position_tracking_global",
        "position_tracking_1",
        "position_tracking_2",
        "orientation_tracking",
        "goal_reached_bonus",
    ],
}


# ============================================================
# Main env config
# ============================================================

@configclass
class HuskyDwbcEnvCfg(ManagerBasedRLEnvCfg):
    """Husky UR5 DWBC: same env as FBC with dual-critic reward groups."""

    scene: HuskyDwbcSceneCfg = HuskyDwbcSceneCfg(num_envs=4096, env_spacing=10.5)
    actions: HuskyDwbcActionsCfg = HuskyDwbcActionsCfg()
    observations: HuskyDwbcObservationsCfg = HuskyDwbcObservationsCfg()
    events: HuskyDwbcEventCfg = HuskyDwbcEventCfg()
    commands: HuskyDwbcCommandsCfg = HuskyDwbcCommandsCfg()
    rewards: HuskyDwbcRewardsCfg = HuskyDwbcRewardsCfg()
    terminations: HuskyDwbcTerminationsCfg = HuskyDwbcTerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 18.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0


@configclass
class HuskyDwbcEnvCfg_PLAY(HuskyDwbcEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
