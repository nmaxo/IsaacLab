# env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
 
import gymnasium as gym
import torch
import math
import numpy as np
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from .scene_manager import Scene_manager
from .control_manager import Control_module
from .path_manager import Path_manager
from .memory_manager import Memory_manager
from .asset_manager import Asset_paths
import omni.kit.commands
import omni.usd
import datetime
# from torch.utils.tensorboard import SummaryWriter
##
# Pre-defined configs
##
from isaaclab_assets.robots.aloha import ALOHA_CFG
from isaaclab.markers import CUBOID_MARKER_CFG
from transformers import CLIPProcessor, CLIPModel
Asset_paths_manager = Asset_paths()

class WheeledRobotEnvWindow(BaseEnvWindow):
    def __init__(self, env: 'WheeledRobotEnv', window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class WheeledRobotEnvCfg(DirectRLEnvCfg):
    episode_length_s = 512.0
    decimation = 4
    action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        shape=(2,)
    )
    # Observation space is now the ResNet18 embedding size (512)
    m = 1  # Например, 3 эмбеддинга и действия
    observation_space = gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(m * (512 + 2),),  # m * (embedding_size + action_size) + 2 (скорости)
        dtype="float32"
    )
    state_space = 0
    debug_vis = False

    ui_window_class_type = WheeledRobotEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="min",
        #     restitution_combine_mode="min",
        #     static_friction=0.05,
        #     dynamic_friction=0.01,
        #     restitution=0.0,
        # ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=15, replicate_physics=True)
    robot: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    wheel_radius = 0.068
    wheel_distance = 0.34
    lin_vel_reward_scale = 0.1
    ang_vel_reward_scale = 0.05
    distance_to_goal_reward_scale = 15.0
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/box2_Link/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.35, 0, 1.1), rot=(0.99619469809,0,0.08715574274,0), convention="world"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0, 0.9), rot=(1,0,0,0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0, focus_distance=2.0, horizontal_aperture=36, clipping_range=(0.2, 10.0)
        ),
        width=224,
        height=224,
    )
    kitchen = sim_utils.UsdFileCfg(
        usd_path=Asset_paths_manager.kitchen_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
            rigid_body_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.1,
        history_length=1,
        debug_vis=True,
        filter_prim_paths_expr=["/World/envs/env_.*"],
    )

    # table = sim_utils.UsdFileCfg(
    #     usd_path=Asset_paths_manager.table_usd_path,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=True,
    #         kinematic_enabled=True,
    #         rigid_body_enabled=True,
    #     ),
    #     collision_props=sim_utils.CollisionPropertiesCfg(
    #         collision_enabled=True,
    #     ),
    # )

    # Конфигурация миски (цели)
    bowl = sim_utils.UsdFileCfg(
        usd_path=Asset_paths_manager.bowl_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,  # Миска неподвижна
            rigid_body_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,  # Отключаем коллизии
        ),
    )

class WheeledRobotEnv(DirectRLEnv):
    cfg: WheeledRobotEnvCfg

    def __init__(self, cfg: WheeledRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._actions[:, 1] = 0.0
        self._left_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._right_wheel_vel = torch.zeros(self.num_envs, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["moves", "contact_penalty"]
        }
        self._left_wheel_id = self._robot.find_joints("left_wheel")[0]
        self._right_wheel_id = self._robot.find_joints("right_wheel")[0]

        self.set_debug_vis(self.cfg.debug_vis)
        self.Debug = True
        self.event_update_counter = 0
        self.episode_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.scene_manager = Scene_manager(self.num_envs, self.device, num_obstacles=3)
        self.use_controller = False
        if self.use_controller:
            self.path_manager = Path_manager(scene_manager=self.scene_manager, ratio=8.0, shift=[5, 4], device=self.device)
            self.control_module = Control_module(num_envs=self.num_envs, device=self.device)
        self.memory_on = False
        if self.memory_on:
            self.memory_manager = Memory_manager(
                num_envs=self.num_envs,
                embedding_size=512,  # Размер эмбеддинга ResNet18
                action_size=2,      # Размер действия (линейная и угловая скорость)
                history_length=25,  # n = 10, можно настроить
                device=self.device
            )
        self.count = 0
        self._debug_log_enabled = True
        self._debug_envs_to_log = list(range(min(5, self.num_envs)))
        self._inconsistencies = []
        self._debug_step_counter = 0
        self._debug_log_frequency = 10
        self.turn_on_controller = False
        self.turn_on_controller_step = 0
        self.my_episode_step = 0
        self.my_episode_lenght = 256
        self.turn_off_controller_step = 0
        self.use_obstacles = False
        self.imitation = False
        self.previous_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Initialize ResNet18 for image embeddings
        self.resnet18 = models.resnet18(pretrained=True).to(self.device)
        self.resnet18.eval()  # Set to evaluation mode
        # Remove the final fully connected layer to get embeddings
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        # Image preprocessing for ResNet18
        transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.success_rate = 0
        self.sr_stack_capacity = 0
        self.episode_completion_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.success_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.history_index = 0
        self.history_len = torch.zeros(self.num_envs, device=self.device)
        self._step_update_counter = 0
        self.mean_radius = 2.5
        self.max_angle_error = torch.pi / 6
        self.cur_angle_error = torch.pi / 6
        self.warm = True
        self._obstacle_update_counter = 0
        self.has_contact = torch.full((self.num_envs,), True, dtype=torch.bool, device=self.device)
        self.sim = SimulationContext.instance()
        self.obstacle_positions = None
        self.key = None
        self.success_ep_num = 0
        # self.run = wandb.init(project="aloha_direct")
        self.first_ep = True
        self.first_ep_step = 0
        self.second_ep = True
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
        name = "dev"
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self.episode_count = 0
        self.total_episode_length = 0.0
        # self.tensorboard_writer = SummaryWriter(log_dir=f"/home/xiso/IsaacLab/logs/tensorboard/navigation_rl_{name}_{timestamp}")
        self.tensorboard_step = 0
        self.cur_step = 0
        self.print_config_info()

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()  # Установить в режим оценки
        self.second_try = 0

        # Инициализация стеков для хранения успехов (1 - успех, 0 - неуспех)
        self.success_stacks = [[] for _ in range(self.num_envs)]  # Список списков для каждой среды
        self.max_stack_size = 15  # Максимальный размер стека
        self.sr_stack_full = False

    def print_config_info(self):
        print("__________[ CONGIFG INFO ]__________")
        print(f"|")
        print(f"| Start mean radius is: {self.mean_radius}")
        print(f"|")
        print(f"| Start amx angle is: {self.max_angle_error}")
        print(f"|")
        print(f"| Use controller: {self.use_controller}")
        print(f"|")
        print(f"| Full imitation: {self.imitation}")
        print(f"|")
        print(f"| Use memory: {self.memory_on}")
        print(f"|")
        print(f"_______[ CONGIFG INFO CLOSE ]_______")

    def _setup_scene(self):
        from isaaclab.sensors import ContactSensor
        import time
        from pxr import Usd
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=True)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.set_env()
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # light_cfg = sim_utils.DomeLightCfg(intensity=300.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

    def set_env(self):
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        self.obstacle_positions = None
        self.chair_prims = [[] for _ in range(self.cfg.scene.num_envs)]
        spawn_from_usd(
            prim_path="/World/envs/env_.*/Kitchen",
            cfg=self.cfg.kitchen,
            translation=(5.0, 4.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        # Спавн стола
        # spawn_from_usd(
        #     prim_path="/World/envs/env_.*/Table",
        #     cfg=self.cfg.table,
        #     translation=(-4.5, 0.0, 0.0),  # Стол в центре локальной системы
        #     orientation=(0.7071, 0.0, 0.0, 0.7071),
        # )
        goal_pos = (-4.5, 0, 0.65)  # z=0.8 для поверхности стола
        spawn_from_usd(
            prim_path="/World/envs/env_.*/Bowl",
            cfg=self.cfg.bowl,
            translation=goal_pos,
            orientation=(0.0, 0.0, 0.7071, 0.7071),
        )

        stage = omni.usd.get_context().get_stage()
        for env_id in range(self.cfg.scene.num_envs):
            for prim_path in [
                f"/World/envs/env_{env_id}/Kitchen",
                # f"/World/envs/env_{env_id}/Table",
                f"/World/envs/env_{env_id}/Bowl",
            ]:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    raise RuntimeError(f"Failed to create prim at {prim_path}")
                # print(f"Created prim {prim_path}, Type: {prim.GetTypeName()}")
        import random

        self.chair_objects = [[] for _ in range(6)]
        grid_x = [-2.0, -1.0]
        grid_y = [-1.0, 0.0, 1.0]
        initial_positions = [(-2.0, -1.0, 0.0), (-2.0, 0.0, 0.0), (-2.0, 1.0, 0.0),
                            (-1.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0)]  # Позиции для ключа [7, 7]
        # for env_id in range(self.cfg.scene.num_envs):
        for i in range(3):
            chair_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Chair_{i}",  # Уникальный путь для каждого стула в каждой среде
                spawn=sim_utils.UsdFileCfg(
                    usd_path=Asset_paths_manager.chair_usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0 + initial_positions[i][0], initial_positions[i][1], 0.0)),
            )
            chair_object = RigidObject(cfg=chair_cfg)
            self.chair_objects[i] = chair_object  # Добавляем в список для конкретной среды

    def _get_observations(self) -> dict:
        self.tensorboard_step += 1
        self.cur_step += 1
        self.episode_lengths += 1
        
        # Получение RGB изображений с камеры
        camera_data = self._tiled_camera.data.output["rgb"].clone()  # Shape: (num_envs, 224, 224, 3)
        
        # Преобразование изображений для CLIP
        # CLIP ожидает изображения в формате PIL или тензоры с правильной нормализацией
        images = camera_data.cpu().numpy().astype(np.uint8)  # Конвертация в numpy uint8
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        # Получение эмбеддингов изображений
        with torch.no_grad():
            image_embeddings = self.clip_model.get_image_features(**inputs)  # Shape: (num_envs, 512)
        
        # Получение скоростей робота
        root_lin_vel_w = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1).unsqueeze(-1)
        root_ang_vel_w = self._robot.data.root_ang_vel_w[:, 2].unsqueeze(-1)
        
        # Обновление памяти, если используется
        if self.memory_on:
            velocities = torch.cat([root_lin_vel_w, root_ang_vel_w], dim=-1)
            self.memory_manager.update(image_embeddings, velocities)
            memory_data = self.memory_manager.get_observations()  # Shape: (num_envs, m * (512 + 2))
            obs = torch.cat([memory_data], dim=-1)
        else:
            obs = torch.cat([image_embeddings, root_lin_vel_w, root_ang_vel_w], dim=-1)
        
        observations = {"policy": obs}
        return observations

    # The rest of the methods (_pre_physics_step, _apply_action, _get_rewards, etc.) remain unchanged
    # as they are not affected by the observation space change.

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_distance
        self.my_episode_step += 1
        self._step_update_counter += 1
        if self.turn_on_controller or self.imitation:
            self.turn_on_controller_step += 1
            # Получаем текущую ориентацию (yaw) из кватерниона
            quat = self._robot.data.root_quat_w
            siny_cosp = 2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
            cosy_cosp = 1 - 2 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3])
            yaw = torch.atan2(siny_cosp, cosy_cosp)
            env_ids = self._robot._ALL_INDICES
            linear_speed, angular_speed = self.control_module.pure_pursuit_controller(
                self.to_local(self._robot.data.root_pos_w[:, :2],env_ids),
                yaw
            )
            # angular_speed = -angular_speed 
            self._actions[:, 0] = (linear_speed / 0.5) - 1
            self._actions[:, 1] = angular_speed / 2.5
        else:
            linear_speed = 0.5*(self._actions[:, 0] + 1.0) # [num_envs], всегда > 0
            angular_speed = 2.5*self._actions[:, 1]  # [num_envs], оставляем как есть от RL
        # print("vel is: ", linear_speed, angular_speed)
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r
        # self._left_wheel_vel = torch.zeros(1, device=self.device)
        # self._right_wheel_vel = torch.zeros(1, device=self.device)
        return

        return self._actions

    def _apply_action(self):
        wheel_velocities = torch.stack([self._left_wheel_vel, self._right_wheel_vel], dim=1).unsqueeze(-1)
        # print("wheel_velocities: ", wheel_velocities)
        # print("true wheel velocities: ", self._robot.data.joint_vel[:, 2:4])
        self._robot.set_joint_velocity_target(wheel_velocities, joint_ids=[self._left_wheel_id, self._right_wheel_id])

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        lin_vel_reward = lin_vel * 0.005
        ang_vel = torch.abs(self._robot.data.root_ang_vel_w[:, 2])
        ang_vel_reward = ang_vel * 0.005
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        # print("root_pos_w ", root_pos_w)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        out_of_bounds_penalty_scale = -15.0
        out_of_bounds = distance_to_goal > 10.0
        out_of_bounds_penalty = out_of_bounds.float() * out_of_bounds_penalty_scale
        goal_reached, num_subs = self.goal_reached(distance_to_goal, get_num_subs=True)
        goal_reached_reward_scale = 20.0
        num_subs_scale = 0.03
        goal_reached_reward = goal_reached.float() * goal_reached_reward_scale
        moves = 5*(self.previous_distance_to_goal-distance_to_goal)
        self.previous_distance_to_goal = distance_to_goal

        quat = self._robot.data.root_quat_w
        siny_cosp = 2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
        cosy_cosp = 1 - 2 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3])
        psi = torch.atan2(siny_cosp, cosy_cosp)
        # Compute angle to goal (theta)
        delta_pos = self._desired_pos_w[:, :2] - root_pos_w
        theta = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])
        # Compute heading error (alpha = theta - psi)
        alpha = theta - psi
        # Normalize alpha to [-pi, pi]
        alpha = torch.atan2(torch.sin(alpha), torch.cos(alpha))

        # Lyapunov derivative components
        rho = distance_to_goal
        u = lin_vel
        omega = ang_vel
        # Compute V_dot = -rho * u * cos(alpha) + alpha * (-omega + u * sin(alpha) / rho)
        term1 = -rho * u * torch.cos(alpha)
        term2 = alpha * (-omega + torch.where(rho > 0.01, u * torch.sin(alpha) / rho, torch.zeros_like(rho)))
        V_dot = term1 + term2

        # Lyapunov reward: Encourage negative V_dot for stability (scale to balance with other rewards)
        lyapunov_reward_scale = 1 # Adjustable scale
        lyapunov_reward = torch.max(-V_dot * lyapunov_reward_scale, torch.tensor(0.0))  # Negative V_dot gives positive reward


        #reward = (-1 + goal_reached_reward + lin_vel_reward + ang_vel_reward + out_of_bounds_penalty + moves)
        has_contact = self.get_contact()
        # print("contact ", has_contact.long())
        contact_penalty = -15 * has_contact.long()
        num_subs_reward = num_subs * num_subs_scale
        IL_reward = 0
        if self.turn_on_controller:
            IL_reward = 0.3
        time_out = self.is_time_out(self.my_episode_lenght-1)
        time_out_penalty = -5 * time_out.float()
        
        reward = (-0.05 + goal_reached_reward + contact_penalty + time_out_penalty)
        if torch.any(has_contact) or torch.any(goal_reached) or torch.any(time_out):
            sr = self.update_success_rate()
            # print("sr: ", sr)
        #     print("reward ", reward)
        #     print("goal_reached ", goal_reached)
        #     print("has_contact ", has_contact)
        #     print("time_out ", time_out)
        check = {
            "moves":moves,
            "contact_penalty": contact_penalty,
        }
        for key, value in check.items():
            self._episode_sums[key] += value

        # if self.tensorboard_step % 100 == 0:
        #     self.tensorboard_writer.add_scalar("Metrics/reward", torch.sum(reward), self.tensorboard_step)
        return reward
    
    def quat_rotate(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Вращение вектора vec кватернионом quat.
        quat: [N, 4] (w, x, y, z)
        vec: [N, 3]
        Возвращает: [N, 3] - вектор vec, повернутый кватернионом quat
        """
        w, x, y, z = quat.unbind(dim=1)
        vx, vy, vz = vec.unbind(dim=1)

        # Кватернионное умножение q * v
        qw = -x*vx - y*vy - z*vz
        qx = w*vx + y*vz - z*vy
        qy = w*vy + z*vx - x*vz
        qz = w*vz + x*vy - y*vx

        # Обратный кватернион q*
        rw = w
        rx = -x
        ry = -y
        rz = -z

        # Результат (q * v) * q*
        rx_new = qw*rx + qx*rw + qy*rz - qz*ry
        ry_new = qw*ry - qx*rz + qy*rw + qz*rx
        rz_new = qw*rz + qx*ry - qy*rx + qz*rw

        return torch.stack([rx_new, ry_new, rz_new], dim=1)


    def goal_reached(self, distance_to_goal: torch.Tensor, angle_threshold: float = 10, get_num_subs=False) -> torch.Tensor:
        """
        Проверяет достижение цели с учётом расстояния и направления взгляда робота.
        distance_to_goal: [N] расстояния до цели
        angle_threshold: максимально допустимый угол в радианах между направлением взгляда и вектором на цель
        Возвращает: [N] булев тензор, True если цель достигнута
        """

        # Проверка по расстоянию (например, радиус достижения stored в self.radius)
        close_enough = distance_to_goal <= 1.3

        # Получаем ориентацию робота в виде кватерниона (w, x, y, z)
        root_quat_w = self._robot.data.root_quat_w  # shape [N, 4]

        # Локальный вектор взгляда робота (вперёд по оси X)
        local_forward = torch.tensor([1.0, 0.0, 0.0], device=root_quat_w.device, dtype=root_quat_w.dtype)
        local_forward = local_forward.unsqueeze(0).repeat(root_quat_w.shape[0], 1)  # [N, 3]

        # Вектор взгляда в мировых координатах
        forward_w = self.quat_rotate(root_quat_w, local_forward)  # [N, 3]

        # Вектор от робота к цели
        root_pos_w = self._robot.data.root_pos_w  # [N, 3]
        to_goal = self._desired_pos_w - root_pos_w  # [N, 3]

        # Нормализуем векторы
        forward_w_norm = torch.nn.functional.normalize(forward_w[:, :2] , dim=1)
        to_goal_norm = torch.nn.functional.normalize(to_goal[:, :2] , dim=1)

        # Косинус угла между векторами взгляда и направления на цель
        cos_angle = torch.sum(forward_w_norm * to_goal_norm, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # для безопасности
        # direction_to_goal = to_goal
        # yaw_g = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])

        # Вычисляем угол между векторами
        angle = torch.acos(cos_angle)
        angle_degrees = torch.abs(angle) * 180.0 / 3.141592653589793
        # Проверяем, что угол меньше порога
        facing_goal = angle_degrees < angle_threshold

        # Итоговое условие: близко к цели и смотрит в её сторону
        # print(distance_to_goal, angle_degrees)
        
        conditions = torch.stack([close_enough, facing_goal], dim=1)  # shape [N, M]
        num_conditions_met = conditions.sum(dim=1)  # shape [N], количество True в каждой строк

        # self.step_counter += torch.ones_like(self.step_counter) #torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # enouth_steps = self.step_counter > 4
        # returns = torch.logical_and(torch.logical_and(close_enough, facing_goal), enouth_steps)
        # self.step_counter = torch.where(returns, torch.zeros_like(self.step_counter), self.step_counter)
        returns = torch.logical_and(close_enough, facing_goal)
        # if torch.any(returns):
        #     print(close_enough, facing_goal)
        # print("returns", returns)
        if get_num_subs == False:
            return returns
        return returns, num_conditions_met

    def get_contact(self):
        force_matrix = self.scene["contact_sensor"].data.net_forces_w
        force_matrix[..., 2] = 0
        forces_magnitude = torch.norm(torch.norm(force_matrix, dim=2), dim=1)  # shape: [batch_size, num_contacts]
        # вычисляем модуль силы для каждого контакта
        if force_matrix is not None and force_matrix.numel() > 0:
            contact_forces = torch.norm(force_matrix, dim=-1)
            if contact_forces.dim() >= 3:
                has_contact = torch.any(contact_forces > 0.1, dim=(1, 2))
            else:
                has_contact = torch.any(contact_forces > 0.1, dim=1) if contact_forces.dim() == 2 else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # print("c ", has_contact)
            num_contacts_per_env = torch.sum(contact_forces > 0.1, dim=1)
            high_contact_envs = num_contacts_per_env >= 1
        else:
            print("force_matrix_w is None or empty")
        # if torch.any(high_contact_envs):
        #     print("high_contact_envs ", high_contact_envs)
        return high_contact_envs

    def update_SR_history(self):
        self.episode_completion_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.success_history = torch.zeros((self.num_envs*4, self.num_envs), dtype=torch.bool, device=self.device)
        self.history_index = 0
        self.history_len = torch.zeros(self.num_envs, device=self.device)

    def update_success_rate(self) -> torch.Tensor:
        if self.turn_on_controller:
            return torch.tensor(self.success_rate, device=self.device)
        
        # Получаем завершенные эпизоды
        died, time_out = self._get_dones(self.my_episode_lenght - 1, inner=True)
        completed = died | time_out
        
        if torch.any(completed):
            # Получаем релевантные среды среди завершенных
            relevant_env_ids = self.scene_manager.get_relevant_env()
            # Фильтруем завершенные среды, оставляя только релевантные
            # print("completed ", completed)
            # print("relevant_env_ids ", relevant_env_ids)
            # print("self._robot._ALL_INDICES ", self._robot._ALL_INDICES)
            # print("self._robot._ALL_INDICES[completed] ", self._robot._ALL_INDICES[completed])
            relevant_completed = relevant_env_ids[(relevant_env_ids.view(1, -1) == self._robot._ALL_INDICES[completed].view(-1, 1)).any(dim=0)] #torch.intersect1d(self._robot._ALL_INDICES[completed], relevant_env_ids)
            # print("                      ")
            # print("                      ")
            # print("died", died)
            # print("time_out", time_out)
            # print("completed ", completed)
            # print("_robot._ALL_INDICES[completed] ", self._robot._ALL_INDICES[completed])
            # print("relevant_completed ", relevant_completed)
            if len(relevant_completed) > 0:
                # Вычисляем успехи для релевантных завершенных сред
                distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=1)
                # print("distance_to_goal ", distance_to_goal)
                success = self.goal_reached(distance_to_goal)
                # print("sucess, ", success)
                # Обновляем стеки для релевантных завершенных сред
                for env_id in self._robot._ALL_INDICES[completed]:
                    env_id = env_id.item()
                    if success[env_id] == False:
                        self.success_stacks[env_id].append(0)
                    elif env_id in relevant_completed:
                        self.success_stacks[env_id].append(1)
                    
                    if len(self.success_stacks[env_id]) > self.max_stack_size:
                        self.success_stacks[env_id].pop(0)
            # print("self.success_stacks ", self.success_stacks)
        # Вычисляем процент успеха для всех сред с непустыми стеками
        # Подсчитываем общий процент успеха по всем релевантным средам
        total_successes = 0
        total_elements = 0
        # print(self.success_stacks)
        for env_id in range(self.num_envs):
            stack = self.success_stacks[env_id]
            if len(stack) == 0:
                continue
            total_successes += sum(stack)
            total_elements += len(stack)
        
        # Вычисляем процент успеха
        # print("total_successes ", total_successes, total_elements)
        self.sr_stack_capacity = total_elements
        if total_elements > 0:
            self.success_rate = (total_successes / total_elements) * 100.0
        else:
            self.success_rate = 0.0
        if total_elements > self.num_envs * self.max_stack_size * 0.8:
            self.sr_stack_full = True
        # print(success_rates, self.success_rate)
        return self.success_rate
    
    def update_sr_stack(self):
        self.success_stacks = [[] for _ in range(self.num_envs)]  # Список списков для каждой среды
        self.sr_stack_full = False

    def _get_dones(self, my_episode_lenght = 256, inner=False) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.is_time_out(my_episode_lenght)
        root_pos_w = self._robot.data.root_pos_w[:, :2]
        distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, :2] - root_pos_w, dim=1)
        
        has_contact = self.get_contact()
        self.has_contact = has_contact
        died = torch.logical_or(
            torch.logical_or(self.goal_reached(distance_to_goal, get_num_subs=False), has_contact),
            time_out,
        )
        if torch.any(died):
            goal_reached = self.goal_reached(distance_to_goal)
            self.episode_counter += died.long()
            self.success_counter += goal_reached.long()
            self.event_update_counter += torch.sum(died).item()
        
        if not inner:
            self.episode_length_buf[died] = 0
        # print("died ", time_out, self.episode_length_buf)
        return died, time_out
    
    def is_time_out(self, max_episode_length=256):
        time_out = self.episode_length_buf >= max_episode_length
        return time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if self.turn_on_controller_step > self.my_episode_lenght and self.turn_on_controller:
            self.turn_on_controller_step = 0
            self.turn_on_controller = False
        if (self.mean_radius != 0) and self.use_controller and not self.turn_on_controller and self.my_episode_step > self.my_episode_lenght:
            self.my_episode_step = 0
            self.turn_on_controller_step = 0
            self.turn_off_controller_step = 0
            prob = lambda x: torch.rand(1).item() <= x
            self.turn_on_controller = prob(0.01*max(10, min(30, 100 - self.success_rate))) #torch.clamp(1 - self.success_rate, min=10.0, max=60.0)
            print(f"turn controller: {self.turn_on_controller} with SR {self.success_rate}")
        if env_ids is None or len(env_ids) == self.num_envs:# or len(self.scene_manager.get_selected_indices()) < self.num_envs:
            env_ids = self._robot._ALL_INDICES
        if self.use_obstacles:
            self._update_chairs(env_ids)
        self.update_from_counters(env_ids)
        env_ids = env_ids.to(dtype=torch.long)

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2], dim=1
        ).mean()
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids) #maybe this shuld be the first
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.zeros_like(self.episode_length_buf) #, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._desired_pos_w[env_ids, :2] = self._terrain.env_origins[env_ids, :2]
        min_radius_x = torch.tensor([1.1])
        min_radius = torch.sqrt(min_radius_x)[0]
        robot_pos, quaternion, goal_pos = self.scene_manager.reset(env_ids, self._terrain.env_origins, self.mean_radius, min_radius, self.cur_angle_error)
        # print("i'm in path_manager")
        if self.turn_on_controller or self.imitation:
            if self.turn_on_controller_step == 0:
                env_ids_for_control = self._robot._ALL_INDICES
                robot_pos_for_control = self._robot.data.default_root_state[env_ids_for_control, :2].clone()
                robot_pos_for_control[env_ids, :2] = robot_pos
                goal_pos_for_control = self._desired_pos_w[env_ids_for_control, :2].clone()
                goal_pos_for_control[env_ids, :2] = goal_pos
            else:
                env_ids_for_control = env_ids
                robot_pos_for_control = robot_pos
                goal_pos_for_control = goal_pos
            paths = self.path_manager.get_paths(
                env_ids=env_ids_for_control,
                start_positions=self.to_local(robot_pos_for_control,env_ids_for_control),
                target_positions=self.to_local(goal_pos_for_control,env_ids_for_control),
                device=self.device
            )
            # print("out path_manager, paths: ", paths, self.turn_on_controller_step)
            self.control_module.update(self.to_local(robot_pos_for_control, env_ids_for_control), self.to_local(goal_pos_for_control, env_ids_for_control), paths, env_ids_for_control)
        if self.memory_on:
            self.memory_manager.reset()
        # print("in reset robot pose ", robot_pos, goal_pos)
        self._desired_pos_w[env_ids, :2] = goal_pos
        self._desired_pos_w[env_ids, 2] = 0.6
        if self.tensorboard_step > 500:
            self.tensorboard_step = 0
            print("Custom/success_rate", self.success_rate)
            print("Custom/counter", self._step_update_counter)
        # self.logger.record("Custom/sucess_rate", self.success_rate)
        # self.logger.record("Custom/radius", self.mean_radius)
        # self.logger.record("Custom/angle", self.cur_angle_error)
        # self.logger.record("Custom/counter", self.counter)
        # extras["sucess_rate"] = self.success_rate
        # extras["radius"] = self.mean_radius
        # extras["angle"] = self.cur_angle_error
        # extras["step update counter"] = self._step_update_counter
        # print("self.extras ", self.extras)
        # self.extras["Episode"].update(extras)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] = robot_pos
        default_root_state[:, 2] = 0.1
        default_root_state[:, 3:7] = quaternion
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # self.tensorboard_writer.add_scalar("Metrics/Success_Rate", self.success_rate, self.tensorboard_step)
        # self.tensorboard_writer.add_scalar("Metrics/Max_angle", self.max_angle, self.tensorboard_step)
        # self.tensorboard_writer.add_scalar("Metrics/Mean_radius", self.mean_radius, self.tensorboard_step)
        # self.tensorboard_writer.add_scalar("Metrics/Contact", torch.count_nonzero(self.has_contact).item(), self.tensorboard_step)
        # self.tensorboard_writer.add_scalar("Metrics/Imitation", self.turn_on_controller, self.tensorboard_step)
        # Логируем длину эпизодов для сброшенных сред
        self.total_episode_length += torch.sum(self.episode_lengths[env_ids]).item()
        self.episode_count += len(env_ids)
        mean_episode_length = self.total_episode_length / self.episode_count if self.episode_count > 0 else 0.0
        # self.tensorboard_writer.add_scalar("Metrics/Mean_episode_length", mean_episode_length, self.tensorboard_step)
        # Сбрасываем счетчик длины для сброшенных сред
        self.episode_lengths[env_ids] = 0
            
    def to_local(self, pos, env_ids, env_origins=None):
        if env_origins is None:
            env_origins = self._terrain.env_origins
        # print("to local:")
        # print("pos: ", pos)
        # print("pos: ", env_ids)
        # print("env_origins: ", env_origins)
        # print("pos: ", env_origins[env_ids, :2])
        return pos[:, :2] - env_origins[env_ids, :2]

    def update_from_counters(self, env_ids: torch.Tensor):
        k = 2 if self.cur_angle_error >= self.max_angle_error and self.mean_radius > self.second_try else 1
        base = k*self.my_episode_lenght
        # print("self.success_rate ", self.success_rate)
        if self.warm and self.cur_step >= 2048:
            self.warm = False
            self.mean_radius = 0.1
            self.cur_angle_error = 0
            self._step_update_counter = 0
        elif not self.warm and self._step_update_counter >= base and not self.turn_on_controller and self.sr_stack_full:
            if self.success_rate >= 90:
                self.success_ep_num += 1
                self.foult_ep_num = 0
                if self.success_ep_num > 50:
                    self.second_try = max(self.mean_radius, self.second_try)
                    self.success_ep_num = 0
                    old_mr = self.mean_radius
                    old_a = self.cur_angle_error
                    self.cur_angle_error += self.max_angle_error / 2
                    print("sr: ", round(self.success_rate, 2), self.sr_stack_capacity)
                    if self.cur_angle_error > self.max_angle_error:
                        self.cur_angle_error = 0
                        self.mean_radius += 0.3
                        print(f"udate [ UP ] r: from {round(old_mr, 2)} to {round(self.mean_radius, 2)}")
                    else:
                        print(f"udate [ UP ] r: {round(self.mean_radius, 2)} a: from {round(old_a, 2)} to {round(self.cur_angle_error, 2)}")
                    self._step_update_counter = 0
                    self.update_sr_stack()
            elif self.success_rate <= 50 or self._step_update_counter >= 4 * self.my_episode_lenght:
                self.foult_ep_num += 1
                if self.foult_ep_num > 1 * self.my_episode_lenght:
                    self.success_ep_num = 0
                    self.cur_angle_error = 0
                    self.foult_ep_num = 0
                    old_mr = self.mean_radius
                    self.mean_radius += -0.1
                    self.mean_radius = max(self.mean_radius, 0.1)
                    self._step_update_counter = 0
                    print("sr: ", round(self.success_rate, 2), self.sr_stack_capacity)
                    print(f"udate [ DOWN ] r: from {round(old_mr, 2)} to {round(self.mean_radius, 2)}, a: {round(self.cur_angle_error, 2)}")
                    self.update_sr_stack()

        self._obstacle_update_counter += 1
        return None

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    def close(self):
        # self.tensorboard_writer.close()
        super().close()

    def _update_chairs(self, env_ids: torch.Tensor = None, mess=False):
        """
        Args:
            env_ids: torch.Tensor, индексы сред для обновления. Если None, обновляются все среды.
        """
        if self.first_ep or self.mean_radius < 1.4:
            if self.first_ep:
                self.first_ep = False
            min_num_active = 0
            max_num_active = 0
        else:
            min_num_active=0
            max_num_active=3
            # selected_indices = {}
            # for env_id in env_ids:
            #     config_key = '012'
            #     positions_for_obstacles = [int(ch) for ch in config_key] if config_key else []
            #     selected_indices[env_id.item()] = positions_for_obstacles
            #     print(f"[DEBUG] Env {env_id}: Obstacle positions indices: {positions_for_obstacles}")
            
            # print(f"[DEBUG] Generating obstacle positions for env_ids={env_ids.tolist()}...")
            # self.scene_manager.generate_obstacle_positions(
            #     mess=False,
            #     env_ids=env_ids,
            #     terrain_origins=torch.zeros((len(env_ids), 3), device=self.device),
            #     min_num_active=len(positions_for_obstacles),  # Используем последнее значение или можно адаптировать
            #     max_num_active=len(positions_for_obstacles),
            #     selected_indices=selected_indices
            # )
        self.scene_manager.generate_obstacle_positions(mess=False, env_ids=env_ids,terrain_origins=self._terrain.env_origins,
                                                        min_num_active=min_num_active,max_num_active=max_num_active)
        # Обновляем позиции стульев для всех сред
        # print("self.scene_manager.num_obstacles ", self.scene_manager.num_obstacles)
        for i in range(self.scene_manager.num_obstacles):  # Для каждого стула
            root_poses = torch.zeros((len(env_ids), 7), device=self.device)  # [num_envs, 7] (x, y, z, qw, qx, qy, qz)
            for j, env_id in enumerate(env_ids):
                # Получаем информацию о препятствии из obstacle_manager
                node = self.scene_manager.graphs[env_id.item()].graph.nodes[i]
                pos = node['position']
                # Заполняем позицию и ориентацию
                root_poses[j, 0] = pos[0]  # x
                root_poses[j, 1] = pos[1]  # y
                root_poses[j, 2] = pos[2]  # z
                root_poses[j, 3] = 1.0  # qw
                root_poses[j, 4:7] = 0.0  # qx, qy, qz

            # Добавляем смещение среды
            root_poses[:, :2] += self._terrain.env_origins[env_ids, :2]
            self.chair_objects[i].write_root_pose_to_sim(root_poses, env_ids=env_ids)