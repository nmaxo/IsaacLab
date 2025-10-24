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
from isaacsim.robot.wheeled_robots.controllers import WheelBasePoseController
# --- CLI ---
parser = argparse.ArgumentParser(description="Jetbot with WheelBasePoseController")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- App launcher ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === Сцена ===
class JetbotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )


def main():
    # 1️⃣ Создаём симуляцию
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.5, 0.0, 2.5), (0.0, 0.0, 0.5))

    # 2️⃣ Создаём сцену
    scene_cfg = JetbotSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 3️⃣ Загружаем Jetbot
    jetbot_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"
    jetbot = WheeledRobot(
        prim_path="/World/Jetbot",
        name="Jetbot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        wheel_radius=0.03,
        wheel_base=0.12,
        create_robot=True,
        usd_path=jetbot_usd,
    )

    # 4️⃣ Инициализируем контроллер позы
    controller = WheelBasePoseController(name="jetbot_pose_controller", robot=jetbot)

    print("[INFO] WheelBasePoseController initialized successfully.")

    # 5️⃣ Основной цикл симуляции
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    sim.reset()

    # Задаем начальное и целевое положение
    start_position = np.array([0.0, 0.0, 0.0])
    start_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion (x,y,z,w)
    goal_position = np.array([1.0, 1.0, 0.0])  # цель (1 м вперед, 1 м вбок)

    while simulation_app.is_running():
        # Получаем текущее положение робота
        root_state = jetbot.data.root_state_w[0].cpu().numpy()
        curr_position = root_state[0:3]
        curr_orientation = root_state[3:7]

        # Вызываем контроллер для движения к цели
        action: ArticulationAction = controller.forward(
            start_position=curr_position,
            start_orientation=curr_orientation,
            goal_position=goal_position,
            lateral_velocity=0.3,
            yaw_velocity=0.5,
            heading_tol=0.05,
            position_tol=0.04,
        )

        jetbot.apply_action(action)

        # Обновляем сцену
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        scene.update(sim_dt)

        # Когда достигли цели — можно поменять цель
        if np.linalg.norm(curr_position[:2] - goal_position[:2]) < 0.05:
            goal_position = np.array([
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
                0.0
            ])
            print(f"[INFO] New goal: {goal_position}")

if __name__ == "__main__":
    main()
    simulation_app.close()