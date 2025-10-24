# add_new_robot.py (Jetbot + UR5, без Dofbot)
# Copyright ...
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import math

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Jetbot + UR5 scene")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.types import ArticulationAction

# константы для колёс
stiffness_wheel_const = 500.0
damping_wheel_const = .0

package_root = os.path.dirname(os.path.abspath(__file__))

# Jetbot
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None),
    },
)
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
        pos=(0.0, 5.0, 0.0),
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


class NewRobotsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    UR5 = UR5M_CFG.replace(prim_path="{ENV_REGEX_NS}/UR5")
    # Assuming a stage context containing a Jetbot at /World/Jetbot
    jetbot = WheeledRobot(prim_path="/World/envs/env_0/Jetbot",
            name="BOT",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
            position = np.array([0,1,0]),
            create_robot = True
            )
    #wrap the articulation in the interface class
    # jetbot = WheeledRobot(prim_path="{ENV_REGEX_NS}/Jetbot",
    #                     name="BOT",
    #                     wheel_dof_names=["left_wheel_joint", "right_wheel_joint"]
    #                     )


def get_available_joints(scene: InteractiveScene, robot_name: str) -> list:
    """
    Возвращает список доступных джоинтов робота
    
    Args:
        scene: Интерактивная сцена
        robot_name: Имя робота в сцене (например, "Jetbot" или "UR5")
    
    Returns:
        list: Список имен джоинтов
    """
    if robot_name not in scene.keys():
        print(f"Робот '{robot_name}' не найден в сцене")
        return []
    
    robot = scene[robot_name]
    
    try:
        joint_names = robot.data.joint_names
        return joint_names
    except AttributeError:
        print(f"Не удалось получить информацию о джоинтах для {robot_name}")
        return []



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0




    while simulation_app.is_running():

        print("Wheel actions:", action.joint_velocities)
        print("Wheel DOFs:", jetbot.get_wheel_velocities())
        # ==== UR5 ====
        # --- Позиции ---

        # --- Скорости ---
        ur5_vel_target = torch.zeros((1, 16))

        # Колёса (10–13)
        ur5_vel_target[0, 0:4] = torch.tensor([5.0, 5.0, 5.0, 5.0])

        # Отправляем в сим
        scene["UR5"].set_joint_velocity_target(ur5_vel_target)

        # print(scene["UR5"].data.joint_names)
        # print(scene['Jetbot'].data.joint_names)
        # шаг симуляции
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        scene.update(sim_dt)




def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.5, 0.0, 3.2), (0.0, 0.0, 0.5))

    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()   