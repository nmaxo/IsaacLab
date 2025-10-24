# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
# from isaaclab_assets import ALOHA_CFG  # isort: skip


##
# Environment configuration
# THis file is the overall configuration of the UR5 fined tuned for simulation
# You can have different configuration for different scenarios

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os
import math

stiffness_wheel_const = 0  # Жесткость для управления скоростью колес
damping_wheel_const =30   # Демпфирование для управления скоростью колес
UR5M_CFG = ArticulationCfg(
    prim_path="/ur5",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join("/home/maksim/IsaacLab/source/isaaclab_assets/data/husky_asset/FINAL_HUSKY.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=2.0,
            max_angular_velocity=2.0,
            max_depenetration_velocity=2.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,  # 4
            solver_velocity_iteration_count=2,  # 0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": math.radians(-7.0),             # 132.0°
            "shoulder_lift_joint": math.radians(-85.0),           # -8.9°
            "elbow_joint": math.radians(113.0),                   # -86.3°
            "wrist_1_joint": math.radians(-117.0),                # -104.0°
            "wrist_2_joint": math.radians(-90.0),                 # -1.0°
            "wrist_3_joint": math.radians(-8.0),                  # 33.0°
            "robotiq_85_left_knuckle_joint": math.radians(0.0),   # 26.0°
            "robotiq_85_right_knuckle_joint": math.radians(0.0),  # 26.0°
        }
    ),
    actuators={
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit_sim=0.5,  # 0.5,
            effort_limit_sim=300.0,  # 300
            stiffness=2000.0,        # 2000
            damping=100.0,           # 100
        ),
        "gripper_actuator": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"],
            effort_limit_sim=0.6,    # 0.6
            velocity_limit_sim=5.0,  # 5
            stiffness=1000,          # 6oo
            damping=100,             # 40
        ),
        "wheel_actuators": ImplicitActuatorCfg(
        joint_names_expr=[
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_right_wheel_joint",
            "rear_left_wheel_joint",
        ],
        effort_limit=100,
        velocity_limit=100.0,
        stiffness={
        "front_left_wheel_joint": stiffness_wheel_const,
        "front_right_wheel_joint": stiffness_wheel_const,
        "rear_right_wheel_joint": stiffness_wheel_const,
        "rear_left_wheel_joint": stiffness_wheel_const,
        },
        damping={
        "front_left_wheel_joint": damping_wheel_const,
        "front_right_wheel_joint": damping_wheel_const,
        "rear_right_wheel_joint": damping_wheel_const,
        "rear_left_wheel_joint": damping_wheel_const,
        },
        )

    }
)


@configclass
class UR5MMReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to UR5
        self.scene.robot = UR5M_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_link"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], scale=1.0, use_default_offset=True
        )

        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "gripper_link"
        self.commands.ee_pose.ranges.pitch = (0,0)


@configclass
class UR5MMReachEnvCfg_PLAY(UR5MMReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
