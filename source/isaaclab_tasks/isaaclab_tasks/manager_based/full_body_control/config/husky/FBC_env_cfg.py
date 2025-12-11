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
from isaaclab.managers import CurriculumTermCfg as CurrTerm

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
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0, 0)},
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
            "position_range": (0.5, 1.5),
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
    wheel_base=0.7,# Расстояние между левыми и правыми колесами
    scale= 0.6
    )
    arm_actions : ActionTerm = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], scale=0.3, use_default_offset=False
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


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    # position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-2.25,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names='base_link'), "command_name": "ee_pose"},
    # )


    # position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=0.85,
    #     params={"std":0.3,"asset_cfg": SceneEntityCfg("robot", body_names='base_link'), "command_name": "ee_pose"},
    # )

    # stability_at_goal = RewTerm(
    #     func=mdp.stability_reward,
    #     weight=0.05,
    #     params={
    #         "command_name": "ee_pose",
    #         "position_threshold": 0.2,
    #         "orientation_threshold": 0.4,
    #         "lin_velocity_threshold": 0.2,
    #         "ang_velocity_threshold": 0.2
    #     }
    # )

    # vel_pen = RewTerm(func = mdp.distance_based_velocity_penalty, 
    #                   weight = -0.2, params={"command_name": "ee_pose"})
    
    # # task terms
    # end_effector_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.65,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"), "command_name": "ee_pose"},
    # )
    # end_effector_position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=2.85,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"), "std": 0.35 ,"command_name": "ee_pose"},
    # )
    
    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"), "command_name": "ee_pose"},
    # )


        # Большая награда за успешное достижение цели
    task_completion_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=5.0,  # Большой бонус за достижение цели
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
            "position_threshold": 0.05,  # Точный порог для достижения
            "orientation_threshold": 0.4,
        }
    )
    
    # # Штраф за неудачное завершение эпизода
    # failure_penalty = RewTerm(
    #     func=mdp.episode_failure_penalty,
    #     weight=-2.0,  # Штраф за провал
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
    #         "command_name": "ee_pose",
    #         "failure_distance": 0.25,  # Если дальше этого расстояния в конце
    #     }
    # )
    
    # Прогрессивный штраф за затягивание эпизода
    # time_penalty = RewTerm(
    #     func=mdp.time_penalty,
    #     weight=-0.05,  # Небольшой штраф за каждый шаг
    #     params={}
    # )
    nav_rew = RewTerm(
        func = mdp.navigation_rewards_combined,
        weight=1,
        params={}


    )
    manip_rew = RewTerm(
        func = mdp.manipulation_rewards_combined,
        weight=3,
        params={}

    )




    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    torque_pen = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5e-8,
        params={}

    )




@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseFixedCommandCfg(
            asset_name="robot",
            body_name='gripper_link',
            resampling_time_range=(18.0, 18.0),
            debug_vis=True,
            ranges=mdp.UniformPoseFixedCommandCfg.Ranges(
                pos_x=(-0.4,0.4),
                pos_y=(-0.4,0.4),
                pos_z=(0.6, 0.95),
                roll=(0.0, 0.0),
                pitch=(0.0,0.0),  # depends on end-effector axis
                yaw=(0, 0),
            ),
        )
    


    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True) 
    goal_reached = DoneTerm(func=mdp.goal_reached_bool,    params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
            "position_threshold": 0.05,  # Точный порог для достижения
            "orientation_threshold": 0.2,
        })

# @configclass
# class CurriculumCfg:
#     success_rate = CurrTerm(
#         func=mdp.compute_success_rate,
#         params={
#             "command_name": "ee_pose",       
#             "position_threshold": 0.05,      
#             "orientation_threshold": 0.2     
#         }
#     )


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
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 18
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
