# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree Go2 robot with ARX manipulator arm.

The following configuration is available:

* :obj:`GO2_ARX_CFG`: Unitree Go2 with ARX 6-DOF arm, DC motor model for legs, implicit actuators for arm.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

GO2_ARX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/maksim/legged-robots-manipulation/loco_manipulation_gym/resources/robots/go2_arx/urdf/go2_arx/go2_arx.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            # legs (from loco_manipulation config)
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
            # arm (folded, neutral)
            "arm_joint1": 0.0,
            "arm_joint2": 0.0,
            "arm_joint3": 0.0,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint.*"],
            effort_limit=20.0,
            stiffness=20.0,
            damping=0.5,
        ),
    },
)
"""Configuration for Unitree Go2 with ARX manipulator arm.

Leg actuator parameters match the loco_manipulation_gym config.
Arm uses implicit actuators with position control.

For locomotion-only training: in the velocity env the arm is held at a fixed pose
(ConstantJointPositionAction), so it does not move randomly. Self-collisions are
disabled (enabled_self_collisions=False). To disable arm collision with the
ground/terrain, disable collision on the arm link prims in the USD (e.g. in Isaac Sim).
"""
