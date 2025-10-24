# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

# CLI arguments
parser = argparse.ArgumentParser(description="Launch UR5 robot in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import os
import math

# from isaacsim.robot_setup.assembler import RobotAssembler
# import omni

current_dir=os.getcwd()

UR5_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Husky",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(current_dir, "source/isaaclab_assets/data/husky_asset", "Mobile_Husky_UR5.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=2.0,
            max_angular_velocity=2.0,
            max_depenetration_velocity=2.0,
            enable_gyroscopic_forces=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,  # 4
            solver_velocity_iteration_count=6,  # 0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "ur5_shoulder_pan_joint": math.radians(7.0),             # 132.0°
            "ur5_shoulder_lift_joint": math.radians(45.0),           # -8.9°
            "ur5_elbow_joint": math.radians(122.0),                   # -86.3°
            "ur5_wrist_1_joint": math.radians(-117.0),                # -104.0°
            "ur5_wrist_2_joint": math.radians(-90.0),                 # -1.0°
            "ur5_wrist_3_joint": math.radians(-8.0),                  # 33.0°
            "finger_joint": math.radians(0.0),   # 26.0°
            "right_outer_knuckle_joint": math.radians(0.0),  # 26.0°
        }
    ),
    actuators={
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
            "ur5_shoulder_pan_joint",    
            "ur5_shoulder_lift_joint",   
            "ur5_elbow_joint",                
            "ur5_wrist_1_joint",                
            "ur5_wrist_2_joint",            
            "ur5_wrist_3_joint",       
            ],
            velocity_limit_sim=0.5,  # 0.5,
            effort_limit_sim=300.0,  # 300
            stiffness=2000.0,        # 2000
            damping=100.0,           # 100
        ),
        "gripper_actuator": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint","right_outer_knuckle_joint"],
            effort_limit_sim=0.6,    # 0.6
            velocity_limit_sim=5.0,  # 5
            stiffness=1000,          # 6oo
            damping=100,             # 40
        ),
    }
)




HUSKY_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Husky",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(current_dir, "source/isaaclab_assets/data/husky_asset","husky_with_sensors.usd")
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
                "front_left_wheel_joint",
                "front_right_wheel_joint",
            ],
            damping=None,
            stiffness=None,
        )
    },
)
##
# Scene config
##


class UR5SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)),
    )

    ur5 = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/UR5")


    # husky = HUSKY_CFG.replace(prim_path="{ENV_REGEX_NS}/Husky")
        # Prim path to the base robot
    # robot_base = "/World/envs/env_0/Husky"
    # # Prim path to the mount point of the base robot
    # robot_base_mount = "/World/envs/env_0/Husky/put_ur5"
    # # Prim path to the attach robot
    # robot_attach = "/World/envs/env_0/UR5"
    # # Prim path to the mount point of the attach robot
    # robot_attach_mount = "/World/envs/env_0/UR5/world/base_link"
    # # Assembly namespace
    # assembly_namespace = "Gripper"
    # variant_name = "my_assembled_robot"


    # stage = omni.usd.get_context().get_stage()
    # assembler = RobotAssembler()


    # # Begin the Assembly process - Creates a session layer and attach it to the current stage, where all the modifications necessary for the assembly will be made.
    # assembler.begin_assembly(stage, robot_base, robot_base_mount, robot_attach, robot_attach_mount, assembly_namespace, variant_name)

    # # Perform any Additional transformations on the Attach robot pose here directly through USD.

    # assembler.assemble()


##
# Main loop
##

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    while simulation_app.is_running():
        sim.step()
        scene.update(sim.get_physics_dt())


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((2.5, 0.0, 1.8), (0.0, 0.0, 0.5))

    scene_cfg = UR5SceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: UR5 loaded in simulation.")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
