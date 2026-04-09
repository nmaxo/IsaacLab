"""Husky UR5 IK Trajectory Tracking environment.

Pipeline adapted from Go2-ARX legged_gym-style:
- 8 actions: 2 (DiffDrive base) + 6 (UR5 arm via IK delta)
- EE goal in spherical coordinates with trajectory interpolation
- DLS IK correction on arm at every substep
- Rewards: EE tracking + regularization with only_positive_rewards clipping
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.full_body_control.mdp as mdp
from isaaclab_tasks.manager_based.full_body_control.mdp.custom_mdp import DiffDriveVelocityActionCfg
from isaaclab_tasks.manager_based.full_body_control.mdp import husky_ik_mdp as hk_mdp
from isaaclab_assets.robots import ur5_husky

UR5M_CFG = ur5_husky.UR5M_CFG


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class HuskyIkSceneCfg(InteractiveSceneCfg):
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


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class HuskyIkEventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
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
    reset_ee_goal = EventTerm(
        func=hk_mdp.reset_ee_goal,
        mode="reset",
        params={"command_name": "ee_goal"},
    )


# ---------------------------------------------------------------------------
# Actions: DiffDrive (2) + Arm IK delta (6) = 8 total
# ---------------------------------------------------------------------------

@configclass
class HuskyIkActionsCfg:
    vel_actions = DiffDriveVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ],
        wheel_radius=0.17775,
        wheel_base=0.5708,
        max_linear_speed=3.0,
        max_angular_speed=2.5,
        scale=1.0,
        linear_velocity_sign=-1.0,
    )
    arm_ik_delta = hk_mdp.HuskyArmIkDeltaJointPositionActionCfg(
        asset_name="robot",
        ee_body_name="gripper_link",
        arm_joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        action_scale=0.3,
        dls_lambda=0.1,
        command_name="ee_goal",
    )


# ---------------------------------------------------------------------------
# Commands: EE goal (spherical trajectory)
# ---------------------------------------------------------------------------

@configclass
class HuskyIkCommandsCfg:
    ee_goal = hk_mdp.HuskyEeGoalCommandCfg(
        asset_name="robot",
        ee_body_name="gripper_link",
        resampling_time_range=(1.0, 2.0),
        debug_vis=True,
        local_axis_z_offset=0.3,
        traj_time_range_s=(0.8, 1.5),
        hold_time_range_s=(0.3, 0.6),
        sphere_ranges=hk_mdp.HuskyEeGoalCommandCfg.SphereRanges(
            pos_l=(0.4, 0.9),
            pos_p=(-math.pi / 4, math.pi / 3),
            pos_y=(-math.pi / 4, math.pi / 4),
        ),
        collision_upper_limits=(0.3, 0.25, 0.1),
        collision_lower_limits=(-0.3, -0.25, -0.5),
        underground_limit=-0.1,
        num_collision_check_samples=10,
        visualize_env_index=-1,
        track_points=10,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@configclass
class HuskyIkObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        local_gripper_pos = ObsTerm(func=hk_mdp.local_gripper_pos)
        curr_ee_goal_cart = ObsTerm(func=hk_mdp.curr_ee_goal_cart)
        gripper_to_goal = ObsTerm(func=hk_mdp.gripper_to_goal)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

@configclass
class HuskyIkRewardsCfg:
    total = RewTerm(func=hk_mdp.husky_ik_total_reward, weight=1.0)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class HuskyIkTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ---------------------------------------------------------------------------
# Full env config
# ---------------------------------------------------------------------------

@configclass
class HuskyIkEnvCfg(ManagerBasedRLEnvCfg):
    """Husky UR5 IK trajectory tracking environment."""

    scene: HuskyIkSceneCfg = HuskyIkSceneCfg(num_envs=4096, env_spacing=10.5)
    actions: HuskyIkActionsCfg = HuskyIkActionsCfg()
    observations: HuskyIkObservationsCfg = HuskyIkObservationsCfg()
    events: HuskyIkEventCfg = HuskyIkEventCfg()
    commands: HuskyIkCommandsCfg = HuskyIkCommandsCfg()
    rewards: HuskyIkRewardsCfg = HuskyIkRewardsCfg()
    terminations: HuskyIkTerminationsCfg = HuskyIkTerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 60.0

        self.viewer.eye = (5.0, 5.0, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        self.commands.ee_goal.debug_vis = True
        self.commands.ee_goal.visualize_env_index = -1


@configclass
class HuskyIkEnvCfg_PLAY(HuskyIkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
