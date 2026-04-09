"""Husky UR5 World-Frame IK Trajectory Tracking — base-arm cooperation.

Goals are in WORLD coordinates at distances requiring the base to drive.
The arm does precise positioning via IK once the base brings it within reach.
This forces learned cooperation between diff-drive base and UR5 arm.
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


@configclass
class HuskyWorldIkSceneCfg(InteractiveSceneCfg):
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


@configclass
class HuskyWorldIkEventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )
    reset_ee_goal = EventTerm(
        func=hk_mdp.reset_world_ee_goal,
        mode="reset",
        params={"command_name": "ee_goal"},
    )


@configclass
class HuskyWorldIkActionsCfg:
    vel_actions = DiffDriveVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "front_left_wheel_joint", "front_right_wheel_joint",
            "rear_left_wheel_joint", "rear_right_wheel_joint",
        ],
        wheel_radius=0.17775,
        wheel_base=0.5708,
        max_linear_speed=2.0,
        max_angular_speed=5.0,
        scale=1.0,
        linear_velocity_sign=-1.0,
    )
    arm_ik_delta = hk_mdp.HuskyArmIkDeltaJointPositionActionCfg(
        asset_name="robot",
        ee_body_name="gripper_link",
        arm_joint_names=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ],
        action_scale=0.3,
        dls_lambda=0.1,
        command_name="ee_goal",
    )


@configclass
class HuskyWorldIkCommandsCfg:
    ee_goal = hk_mdp.HuskyWorldEeGoalCommandCfg(
        asset_name="robot",
        ee_body_name="gripper_link",
        resampling_time_range=(4.0, 10.0),
        debug_vis=True,
        local_axis_z_offset=0.3,
        arm_reach=0.85,
        goal_distance_range=(0.8, 1.8),
        goal_z_range=(0.3, 0.8),
        traj_time_range_s=(2.0, 5.0),
        hold_time_range_s=(1.0, 2.0),
        curve_xy_amplitude_range=(0.08, 0.35),
        curve_z_amplitude_range=(0.00, 0.22),
        curve_harmonic_range=(1, 3),
        visualize_env_index=-1,
        enable_curriculum=False,
        # common_step_counter += 1 each vectorized env.step (~ num_iterations * num_steps_per_env).
        # curriculum_p = smoothstep(counter / ramp_steps). For ~300 iters × 256 steps ≈ 76.8k → p≈0.17 use ramp≈285k.
        curriculum_ramp_steps=285_000,
        curriculum_schedule_step_interval=1,
        easy_goal_distance_range=(0.55, 1.05),
        easy_goal_z_range=(0.42, 0.58),
        easy_traj_time_range_s=(3.5, 6.5),
        easy_hold_time_range_s=(1.4, 2.4),
    )


@configclass
class HuskyWorldIkObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Goal info for the base (where to drive)
        goal_in_base = ObsTerm(func=hk_mdp.goal_in_base_frame)
        goal_distance = ObsTerm(func=hk_mdp.goal_distance_xy)

        # Arm-level info
        local_gripper_pos = ObsTerm(func=hk_mdp.local_gripper_pos)
        curr_ee_goal_local = ObsTerm(func=hk_mdp.curr_ee_goal_cart)
        gripper_to_goal = ObsTerm(func=hk_mdp.gripper_to_world_goal)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HuskyWorldIkRewardsCfg:
    total = RewTerm(func=hk_mdp.husky_world_ik_total_reward, weight=1.0)


@configclass
class HuskyWorldIkTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_flipped = DoneTerm(
        func=hk_mdp.base_flipped,
        params={"threshold": 0.5},
    )
    # Kill rare exploded envs before they poison PPO batches.
    bad_state = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.2, "asset_cfg": SceneEntityCfg("robot")},
    )
    # Unphysical hops / contact explosions on flat ground (normal base ~0.3–0.8 m Z).
    too_high = DoneTerm(
        func=hk_mdp.root_height_above_maximum,
        params={"maximum_height": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class HuskyWorldIkEnvCfg(ManagerBasedRLEnvCfg):
    """Husky UR5 world-frame IK tracking — base-arm cooperation."""

    scene: HuskyWorldIkSceneCfg = HuskyWorldIkSceneCfg(num_envs=4096, env_spacing=12.0)
    actions: HuskyWorldIkActionsCfg = HuskyWorldIkActionsCfg()
    observations: HuskyWorldIkObservationsCfg = HuskyWorldIkObservationsCfg()
    events: HuskyWorldIkEventCfg = HuskyWorldIkEventCfg()
    commands: HuskyWorldIkCommandsCfg = HuskyWorldIkCommandsCfg()
    rewards: HuskyWorldIkRewardsCfg = HuskyWorldIkRewardsCfg()
    terminations: HuskyWorldIkTerminationsCfg = HuskyWorldIkTerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 25.0
        self.sim.dt = 1.0 / 60.0

        self.viewer.eye = (8.0, 8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class HuskyWorldIkEnvCfg_PLAY(HuskyWorldIkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
