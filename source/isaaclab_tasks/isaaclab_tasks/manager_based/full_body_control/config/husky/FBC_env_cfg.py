# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


import isaaclab_tasks.manager_based.full_body_control.mdp as mdp
from isaaclab_tasks.manager_based.full_body_control.mdp.custom_mdp import DiffDriveVelocityAction, DiffDriveVelocityActionCfg
from isaaclab_assets.robots import ur5_husky # isort: skip



UR5M_CFG = ur5_husky.UR5M_CFG



@configclass
class FBCSceneCfg(InteractiveSceneCfg):

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: ArticulationCfg = UR5M_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )



@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0, 0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
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


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    vel_actions: ActionTerm = DiffDriveVelocityActionCfg(
    asset_name="robot",
    joint_names=["front_left_wheel_joint", "front_right_wheel_joint","rear_left_wheel_joint", "rear_right_wheel_joint"],
    wheel_radius=0.1651,  # Радиус колес Husky
    wheel_base=0.7,    # Расстояние между левыми и правыми колесами
    )
    arm_actions : ActionTerm = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], scale=0.5, use_default_offset=True
        )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.2,
        params={"std":1.5,"asset_cfg": SceneEntityCfg("robot", body_names='base_link'), "command_name": "ee_pose"},
    )


    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.4,
        params={"std":0.3,"asset_cfg": SceneEntityCfg("robot", body_names='base_link'), "command_name": "ee_pose"},
    )
    # orientation_tracking = RewTerm(
    #     func=mdp.heading_command_error_abs,
    #     weight=-0.2,
    #     params={"command_name": "pose_command"},
    # )
    stability_at_goal = RewTerm(
        func=mdp.stability_reward,
        weight=0.4,
        params={
            "command_name": "ee_pose",
            "position_threshold": 0.2,
            "orientation_threshold": 0.4,
            "lin_velocity_threshold": 0.2,
            "ang_velocity_threshold": 0.2
        }
    )

    vel_pen = RewTerm(func = mdp.distance_based_velocity_penalty, 
                      weight = -0.2, params={"command_name": "ee_pose"})
    
    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.15,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"), "std": 0.4 ,"command_name": "ee_pose"},
    )
    
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"), "command_name": "ee_pose"},
    )

    # # action penalty
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )


# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    
#     # ВСЁ КАК БЫЛО (ваша рабочая конфигурация)
#     position_tracking = RewTerm(
#         func=mdp.position_command_error_tanh,
#         weight=0.5,
#         params={"std": 2.0, "command_name": "pose_command"},
#     )
    
#     position_tracking_fine_grained = RewTerm(
#         func=mdp.position_command_error_tanh,
#         weight=0.5,
#         params={"std": 0.2, "command_name": "pose_command"},
#     )
    
#     orientation_tracking = RewTerm(
#         func=mdp.heading_command_error_abs,
#         weight=-0.2,
#         params={"command_name": "pose_command"},
#     )
    
#     stability_at_goal = RewTerm(
#         func=mdp.stability_reward,
#         weight=2.0,  # начните с малого веса
#         params={
#             "command_name": "pose_command",
#             "position_threshold": 0.5,  # чуть шире чем 0.2 из fine_grained
#             "orientation_threshold": 0.4,
#             "lin_velocity_threshold": 0.3,
#             "ang_velocity_threshold": 0.3
#         }
#     )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseFixedCommandCfg(
            asset_name="robot",
            body_name='gripper_link',
            resampling_time_range=(15.0, 15.0),
            debug_vis=True,
            ranges=mdp.UniformPoseFixedCommandCfg.Ranges(
                pos_x=(-3.0,3.0),
                pos_y=(-3.2, 3.2),
                pos_z=(0.7, 0.9),
                roll=(0.0, 0.0),
                pitch=(0.0,0.0),  # depends on end-effector axis
                yaw=(0, 0),
            ),
        )
    


    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True) 
    # goal_reached = DoneTerm(func=mdp.goal_reached_termination, params={"threshold": 0.1})



@configclass
class FBCEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: FBCSceneCfg = FBCSceneCfg(num_envs=4096, env_spacing=10.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

class FBCEnvCfg_PLAY(FBCEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
