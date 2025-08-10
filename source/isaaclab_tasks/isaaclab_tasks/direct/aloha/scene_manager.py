import torch
import math
from scipy.spatial import ConvexHull
import random
# from .graph_manager import ObstacleGraph

import importlib.util
def import_class_from_path(module_path, class_name):
    spec = importlib.util.spec_from_file_location("custom_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

import os
current_dir = os.getcwd()
module_path = os.path.join(current_dir, "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/graph_manager.py")
ObstacleGraph = import_class_from_path(module_path, "ObstacleGraph")

class Scene_manager:
    def __init__(self, num_envs=1, device='cuda:0', num_obstacles=6):
        self.num_envs = num_envs
        self.device = device
        
        # Тензоры для хранения позиций и ориентаций
        self.robot_pos = torch.zeros((num_envs, 2), device=device)  # [x, y]
        self.robot_yaw = torch.zeros(num_envs, device=device)      # угол yaw
        self.goal_local_pos = torch.zeros((num_envs, 2), device=device)  # [x, y] цели
        
        # Параметры управления
        self.max_linear_speed = 1.0   # максимальная линейная скорость (м/с)
        self.max_angular_speed = 0.5  # максимальная угловая скорость (рад/с)
        self.angle_threshold = math.pi / 30  # порог угла (10 градусов) для начала движения вперед
        
        # Размеры объектов
        self.robot_radius = 0.5  # радиус робота (м)
        self.goal_radius = 0.3   # радиус цели (м)
        
        # Границы комнаты в локальной системе координат
        self.room_bounds = {
            'x_min': -4.3,
            'x_max': 5,
            'y_min': -4,
            'y_max': 4
        }
        self.room_bounds = {
            'x_min': -4.3,
            'x_max': 2,
            'y_min': -3,
            'y_max': 3
        }
        # Параметры для генерации расстояний цель-робот
        self.max_radius_values = 8
        self.radius_values = torch.zeros(num_envs, device=device)  
        self.angle_values = torch.tensor([0], device=device)
        
        # Расчёт максимального радиуса (расстояние цель-робот)
        room_diagonal = math.sqrt(self.room_bounds['x_max']**2 + self.room_bounds['y_max']**2)  # 5 м
        self.max_radius = room_diagonal - (self.robot_radius + self.goal_radius + 0.2)
        
        # Границы для проверки позиций с учётом радиусов
        self.robot_bounds = {
            'x_min': self.room_bounds['x_min'] + self.robot_radius + 0.5,  # -3.5
            'x_max': self.room_bounds['x_max'] - self.robot_radius - 0.5,  # 3.5
            'y_min': self.room_bounds['y_min'] + self.robot_radius + 1,  # -2.5
            'y_max': self.room_bounds['y_max'] - self.robot_radius - 1   # 2.5
        }
        self.goal_bounds = {
            'x_min': self.room_bounds['x_min'] + self.goal_radius,  # -3.7
            'x_max': self.room_bounds['x_max'] - self.goal_radius,  # 3.7
            'y_min': self.room_bounds['y_min'] + self.goal_radius,  # -2.7
            'y_max': self.room_bounds['y_max'] - self.goal_radius   # 2.7
        }

        self.room_vertices = torch.tensor([
            [self.room_bounds['x_min'] + self.robot_radius + self.goal_radius + 0.1, self.room_bounds['y_min'] + self.robot_radius + self.goal_radius + 0.1],
            [self.room_bounds['x_min'] + self.robot_radius + self.goal_radius + 0.1, self.room_bounds['y_max'] - self.robot_radius - self.goal_radius - 0.1],
            [self.room_bounds['x_max'] - self.robot_radius - self.goal_radius - 0.1, self.room_bounds['y_min'] + self.robot_radius + self.goal_radius + 0.1],
            [self.room_bounds['x_max'] - self.robot_radius - self.goal_radius - 0.1, self.room_bounds['y_max'] - self.robot_radius - self.goal_radius - 0.1] 
        ], device=self.device)

        # Дискретизация углов
        self.n_angles = 36  # Шаг дискретизации: pi/n
        self.num_angle_steps = 2 * self.n_angles  # 72 угла (0 до 2pi)
        self.angle_step = math.pi / self.n_angles
        self.discrete_angles = torch.arange(0, 2 * math.pi, self.angle_step, device=device)

        self.num_envs = num_envs
        self.device = device
        self.num_obstacles = num_obstacles
        grid_x = [-1.5]  # Уровни по X для активных позиций
        grid_y = [-1.0, 0.0, 1.0]  # Уровни по Y для активных позиций
        self.possible_positions = [(x, y, 0.0) for x in grid_x for y in grid_y]  # 6 возможных позиций
        self.graphs = [ObstacleGraph(num_obstacles, device) for _ in range(num_envs)]
        self.obstacle_radii = [0.5] * num_obstacles
        for graph in self.graphs:
            graph.set_radii(self.obstacle_radii)
        self.selected_indices = {}
        self.active_indices = None
        self.mess = None
        self.base_radius = 1.25 #1.35

        self.config_env_episode = {
            "angle_error": torch.zeros(num_envs, device=device),
            "dist_error": torch.zeros(num_envs, device=device),
            "relevant": torch.zeros(num_envs,  device=device, dtype=torch.bool)
        }

    def get_selected_indices(self, env_id=None):
        # print("[ SCENE MANAGER DEBUG ] selected_indices: ", self.selected_indices)
        # print("[ SCENE MANAGER DEBUG ] active_indices: ", self.active_indices)
        
        if env_id is not None and len(self.selected_indices) > 0:
            # print("env_id: ", env_id.item())
            # for key in sorted(self.selected_indices.keys()):
            #     print(f"{key}: {self.selected_indices[key]}")

            num_ind = len(self.selected_indices[env_id.item()])
            # if self.active_indices is not None:
            #     num_ind = len(self.active_indices[env_id.item()])
            # print(f"[ SCENE MANAGER DEBUG ] selected_indices[{env_id.item()}][:{num_ind}]: ", self.selected_indices[env_id.item()][:num_ind])
            return self.selected_indices[env_id.item()][:num_ind].clone()
        # print(self.selected_indices)
        return self.selected_indices

    def get_mess(self):
        return self.mess

    def compute_max_radii(self, goal_pos):
        """Вычисляет максимальный радиус для размещения робота относительно каждой цели."""
        # Расчёт расстояний от каждой цели до всех вершин
        distances = torch.norm(goal_pos.unsqueeze(1) - self.room_vertices.unsqueeze(0), dim=2)
        # Максимальное расстояние до вершин для каждой цели
        max_distances = torch.max(distances, dim=1)[0]  # [num_envs]
        # Максимальный радиус = максимальное расстояние - радиусы робота и цели
        max_radii = max_distances - (self.robot_radius + self.goal_radius + 0.2) # [num_envs]
        # Убедимся, что радиус не отрицательный
        max_radii = torch.clamp(max_radii, min=0.5)
        max_radii = 8*torch.ones(self.num_envs,device=self.device)
        return max_radii

    def _compute_valid_angles(self, goal_pos, radii):
        """Вычисляет булев тензор допустимых углов для каждой среды."""
        num_envs = goal_pos.shape[0]
        # Тензор [num_envs, num_angle_steps], изначально все углы допустимы
        valid_angles = torch.ones((num_envs, self.num_angle_steps), dtype=torch.bool, device=self.device)
        # Проверяем каждый угол
        for k in range(self.num_angle_steps):
            theta = self.discrete_angles[k]
            # Вычисляем позиции робота для всех сред
            rx = goal_pos[:, 0] + radii * torch.cos(theta)
            ry = goal_pos[:, 1] + radii * torch.sin(theta)
            
            # Проверяем, выходит ли робот за границы
            invalid = (
                (rx < self.robot_bounds['x_min']) |
                (rx > self.robot_bounds['x_max']) |
                (ry < self.robot_bounds['y_min']) |
                (ry > self.robot_bounds['y_max'])
            )
            valid_angles[:, k] = ~invalid
        return valid_angles

    def initialize_radii(self, num_envs, mean_radius, std_radius, max_radii=8, min_radius=1.2, max_radius=None, device='cuda'):
        """
        Инициализация радиусов для num_envs сред с нормальным распределением.
        
        Args:
            num_envs (int): Количество сред.
            mean_radius (float): Центр распределения радиусов (среднее).
            std_radius (float): Стандартное отклонение (дисперсия).
            min_radius (float): Минимальный радиус (по умолчанию 1.8).
            max_radius (float, optional): Глобальный максимальный радиус. Если None, используется self.max_radius_values.
            device (str): Устройство для тензоров (cuda или cpu).
        
        Returns:
            torch.Tensor: Тензор радиусов размером [num_envs].
        """
        if max_radius is None:
            max_radius = self.max_radius_values

        # Генерация радиусов из нормального распределения
        radii = torch.normal(mean=mean_radius, std=std_radius, size=(num_envs,), device=device)
        # print("radii ", radii)
        # Ограничение радиусов минимальным и глобальным максимальным значениями
        radii = torch.clamp(radii, min=min_radius, max=max_radius)

        # Учет локальных максимальных радиусов для каждой среды
        return radii
   
    def generate_goal_positions_local(self, env_ids):
        """Генерирует случайные позиции для целей в пределах комнаты."""
        num_envs = len(env_ids)
        
        sigma_x = (self.goal_bounds['x_max'] - self.goal_bounds['x_min']) / 4  # ~1/4 ширины
        sigma_y = (self.goal_bounds['y_max'] - self.goal_bounds['y_min']) / 4  # ~1/4 высоты
        x = torch.randn(num_envs, device=self.device) * sigma_x
        y = torch.randn(num_envs, device=self.device) * sigma_y

        # Ограничение координат в пределах goal_bounds
        x = torch.clamp(x, self.goal_bounds['x_min'] + 0.5, self.goal_bounds['x_max'] - 0.5)
        y = torch.clamp(y, self.goal_bounds['y_min'] + 0.5, self.goal_bounds['y_max'] - 0.5)
        
        # positions = torch.zeros((num_envs,2),  device=self.device) #+ torch.stack([x, y], dim=1)
        positions = torch.tensor([[-4.5, 0.0]], device=self.device).repeat(num_envs, 1)
        # # print("goal pos ", positions)
        return positions
    
    def shift_pos(self, env_ids, obj_pos, terrain_origins):
        return obj_pos[env_ids] + terrain_origins[env_ids, :2]

    def _generate_obj_positions(self, env_ids, goal_pos, mean_radius, min_radius, std_radius=0.1):
        """Генерирует позиции робота на заданном расстоянии от цели."""
        num_envs = len(env_ids)
        
        # Генерация радиусов
        max_radii = self.compute_max_radii(goal_pos)
        # Инициализация радиусов
        delta = 0.2
        base_radius = self.base_radius
        mean_radius = mean_radius + base_radius
        radii = self.initialize_radii(
            num_envs=num_envs,
            mean_radius=mean_radius,
            std_radius=(mean_radius-base_radius)*0.3,
            max_radii=8,
            min_radius=min_radius,
            max_radius=self.max_radius_values,
            device=self.device
        )
        # Вычисление допустимых углов
        valid_angles = self._compute_valid_angles(goal_pos, radii)
        # Выбор случайного допустимого угла для каждой среды
        theta = torch.zeros(num_envs, device=self.device)
        for i in range(num_envs):
            valid_indices = torch.where(valid_angles[i])[0]
            if valid_indices.numel() > 0:
                # Случайно выбираем индекс из допустимых
                idx = valid_indices[torch.randint(0, valid_indices.numel(), (1,), device=self.device)]
                theta[i] = self.discrete_angles[idx]
            else:
                theta[i] = torch.rand(1, device=self.device) * 2 * math.pi
        # Вычисление координат робота
        x = radii * torch.cos(theta)
        y = radii * torch.sin(theta)
        robot_offsets = torch.stack([x, y], dim=1)
        positions = goal_pos + robot_offsets
        return positions

    def check_bouds(self, env_ids, new_obj_pos):
        robot_valid = (
                (new_obj_pos[:, 0] >= self.robot_bounds['x_min']) &
                (new_obj_pos[:, 0] <= self.robot_bounds['x_max']) &
                (new_obj_pos[:, 1] >= self.robot_bounds['y_min']) &
                (new_obj_pos[:, 1] <= self.robot_bounds['y_max'])
            )
        # print("check_bouds ", robot_valid)
        if torch.any(~robot_valid):
            result = False
        else:
            result = True
        return result, robot_valid

    def get_checked_for_room_bounds_pos(self, env_ids, mean_radius, min_radius):
        invalid_env_ids = torch.tensor(range(len(env_ids)), device=self.device)
        max_iterations = 100
        iteration = 0
        obj_pos = self._generate_obj_positions(invalid_env_ids, self.goal_local_pos[invalid_env_ids], mean_radius, min_radius)
        # Проверяем коллизии
        while len(invalid_env_ids) > 0:
            num_invalid = len(invalid_env_ids)
            new_obj_pos = self._generate_obj_positions(invalid_env_ids, self.goal_local_pos[invalid_env_ids], mean_radius, min_radius)
            _, robot_valid = self.check_bouds(env_ids, new_obj_pos)
            valid_new = robot_valid
            valid_new_ids = invalid_env_ids[valid_new]
            # print("valid_new_ids", valid_new_ids)
            # print(valid_new)
            # print(invalid_env_ids)
            # print(obj_pos)
            # print(new_obj_pos)
            obj_pos[valid_new_ids] = new_obj_pos[valid_new]
            # print(valid_new, invalid_env_ids)
            invalid_env_ids = invalid_env_ids[~valid_new]
            iteration += 1
            if iteration > max_iterations:
                for i in range(len(self.graphs)):
                    print(i, self.graphs[i].get_graph_info())
                print(invalid_env_ids)
                print(self.goal_local_pos[invalid_env_ids])
                print(obj_pos)
                print("error in getting correction position")

        if len(invalid_env_ids) > 0:
            print(f"Warning: Could not generate valid positions for {len(invalid_env_ids)} environments")
        
        return obj_pos

    def check_obstacles_collision(self, obj_pos, env_ids, obstacle_positions=None, valid_mask=None):
        safety_margin = 0.3
        if obstacle_positions is None or valid_mask is None:
            num_envs = len(env_ids)
            env_ids = torch.tensor(range(num_envs), device=self.device)
            # print("i ", obj_pos)
            # Собираем активные препятствия
            obstacles_per_env = [self.graphs[env_id.item()].get_active_nodes() for env_id in env_ids]
            max_obstacles = max(len(obs) for obs in obstacles_per_env) if obstacles_per_env else 0
            if max_obstacles == 0:
                return False, None

            # Инициализируем тензоры
            obstacle_positions = torch.full((num_envs, max_obstacles, 2), 6666.0, device=self.device)
            obstacle_radii = torch.full((num_envs, max_obstacles), 0.001, device=self.device)
            valid_mask = torch.zeros((num_envs, max_obstacles), dtype=torch.bool, device=self.device)

            # Заполняем тензоры позиций и радиусов
            num_obstacles = torch.tensor([len(obs) for obs in obstacles_per_env], device=self.device)
            env_indices = torch.arange(num_envs, device=self.device).repeat_interleave(num_obstacles)
            obstacle_indices = torch.cat([torch.arange(n, device=self.device) for n in num_obstacles])
            positions = torch.tensor([obs['position'][:2] for env_id, obs_list in enumerate(obstacles_per_env) 
                                    for obs in obs_list], device=self.device)
            radii = torch.tensor([obs['radius'] for env_id, obs_list in enumerate(obstacles_per_env) 
                                for obs in obs_list], device=self.device)
            
            if len(env_indices) > 0:
                obstacle_positions[env_indices, obstacle_indices] = positions
                obstacle_radii[env_indices, obstacle_indices] = radii
                valid_mask[env_indices, obstacle_indices] = True

        # Итерации для устранения коллизий
        distances = torch.norm(obj_pos[env_ids, None, :] - obstacle_positions, dim=2)
        # print("2 ", obj_pos)
        # print("distances", distances)
        required_dists = self.robot_radius + 0.5 + safety_margin# obstacle_radii
        # print("required_dists", required_dists)
        collisions = (distances < required_dists) & valid_mask
        if torch.any(collisions):
            result = False
        else:
            result = True
        return result, collisions

    def get_checked_for_obstacles_rp(self, env_ids, obj_pos):
        """
        Проверяет позиции робота на коллизии с препятствиями и корректирует их векторно.
        
        Args:
            env_ids (torch.Tensor): Индексы сред для проверки.
        
        Returns:
            None: Обновляет obj_pos для сред с коллизиями.
        """
        num_envs = len(env_ids)
        env_ids = torch.tensor(range(num_envs), device=self.device)
        safety_margin = 0.4
        max_iterations = 10  # Ограничение на итерации
        # print("i ", obj_pos)
        # Собираем активные препятствия
        obstacles_per_env = [self.graphs[env_id.item()].get_active_nodes() for env_id in env_ids]
        max_obstacles = max(len(obs) for obs in obstacles_per_env) if obstacles_per_env else 0
        if max_obstacles == 0:
            return obj_pos

        # Инициализируем тензоры
        obstacle_positions = torch.full((num_envs, max_obstacles, 2), 6666.0, device=self.device)
        obstacle_radii = torch.full((num_envs, max_obstacles), 0.001, device=self.device)
        valid_mask = torch.zeros((num_envs, max_obstacles), dtype=torch.bool, device=self.device)

        # Заполняем тензоры позиций и радиусов
        num_obstacles = torch.tensor([len(obs) for obs in obstacles_per_env], device=self.device)
        env_indices = torch.arange(num_envs, device=self.device).repeat_interleave(num_obstacles)
        obstacle_indices = torch.cat([torch.arange(n, device=self.device) for n in num_obstacles])
        positions = torch.tensor([obs['position'][:2] for env_id, obs_list in enumerate(obstacles_per_env) 
                                for obs in obs_list], device=self.device)
        radii = torch.tensor([obs['radius'] for env_id, obs_list in enumerate(obstacles_per_env) 
                            for obs in obs_list], device=self.device)
        
        if len(env_indices) > 0:
            obstacle_positions[env_indices, obstacle_indices] = positions
            obstacle_radii[env_indices, obstacle_indices] = radii
            valid_mask[env_indices, obstacle_indices] = True

        # Генерируем случайное направление для каждой среды один раз
        directions = torch.randn((num_envs, 2), device=self.device)
        directions = directions / torch.norm(directions, dim=1, keepdim=True)  # Нормализуем

        # Итерации для устранения коллизий
        it_count = 0
        while True:
            it_count += 1
            if it_count > 100:
                print("error while in get_checked for obsacles_rp")
            # Итерации для устранения коллизий
            distances = torch.norm(obj_pos[env_ids, None, :] - obstacle_positions, dim=2)
            # print("2 ", obj_pos)
            # print("distances", distances)
            required_dists = self.robot_radius + 0.5 # obstacle_radii
            _, collisions = self.check_obstacles_collision(obj_pos, env_ids, obstacle_positions, valid_mask)
            # print("collisions", collisions)
            # Если нет коллизий, выходим
            has_collisions = torch.any(collisions, dim=1)
            # print("has_collisions", has_collisions)
            if not torch.any(has_collisions):
                break
            # print("aloha dance with obstacles")
            # Находим ближайшее препятствие для сред с коллизиями
            distances_masked = distances.clone()
            distances_masked[~valid_mask] = float('inf')
            # print("distances_masked", distances_masked)
            closest_indices = torch.argmin(distances_masked, dim=1)
            # print("closest_indices", closest_indices)
            env_ids_collisions = torch.where(has_collisions)[0]
            # print("env_ids_collisions", env_ids_collisions)
            abs_env_ids = env_ids[env_ids_collisions]
            # print("abs_env_ids", abs_env_ids)
            # Векторно вычисляем смещение
            closest_positions = obstacle_positions[env_ids_collisions, closest_indices[env_ids_collisions]]
            # print("closest_positions", closest_positions)
            closest_radii = obstacle_radii[env_ids_collisions, closest_indices[env_ids_collisions]]
            # print("closest_radii", closest_radii)
            vecs = obj_pos[abs_env_ids] - closest_positions
            # print("3 ", obj_pos)
            # print("vecs", vecs)
            dists = torch.norm(vecs, dim=1)
            # print("dists", dists)
            # Используем заранее сгенерированное направление для сред с коллизиями
            used_directions = directions[env_ids_collisions]
            # print("used_directions", used_directions)
            near_zero = dists < 1e-6
            # print("near_zero", near_zero)
            if torch.any(near_zero):
                used_directions[near_zero] = torch.randn((near_zero.sum(), 2), device=self.device)
                used_directions[near_zero] = used_directions[near_zero] / torch.norm(used_directions[near_zero], dim=1, keepdim=True)

            # Смещаем робота на границу препятствия в направлении used_directions
            displacements = used_directions * (closest_radii + self.robot_radius + safety_margin)[:, None]
            # print("displacements", displacements)
            obj_pos[abs_env_ids] = closest_positions + displacements
            # print("4 ", obj_pos, abs_env_ids)
            # print(f"self.robot_pos[{abs_env_ids}]", self.robot_pos[abs_env_ids])
        return obj_pos

    def get_pos(self, env_ids, terrain_origins=None,mean_radius=2,min_radius=1.2):
        """Проверяет позиции роботов и целей, перегенерирует для недопустимых."""
        it_count = 0
        while True:
            it_count += 1
            if it_count > 100:
                print("error while in scene manager get_pos")
            obj_pos = self.get_checked_for_room_bounds_pos(env_ids, mean_radius, min_radius)
            # print("check 1: ", obj_pos, env_ids)
            obj_pos = self.get_checked_for_obstacles_rp(env_ids, obj_pos)
            # print("check 2: ", obj_pos, env_ids)
            if self.check_bouds(env_ids, obj_pos):
                break
        if terrain_origins is None:
            return obj_pos
        return obj_pos + terrain_origins[env_ids, :2]

    def reset(self, env_ids, terrain_origins,mean_radius=1.3,min_radius=1.2,max_angle=0):
        """Инициализация стартовых позиций и целей при сбросе."""
        num_envs = len(env_ids)       
        # Проверка и корректировка позиций
        self.goal_local_pos[env_ids] = self.generate_goal_positions_local(
            env_ids)
        self.robot_pos[env_ids] = self.get_pos(
            env_ids, terrain_origins, mean_radius, min_radius)

        # Генерация угловых ошибок
        random_sign = torch.sign(torch.rand(num_envs, device=self.device) - 0.5)
        angle_errors = torch.normal(mean=max_angle, std=max_angle/3, size=(num_envs,), device=self.device)
        angle_errors = torch.clamp(angle_errors, min=0, max=torch.pi / 4)
        # Начальная ориентация робота
        goal_global_pos = self.shift_pos(env_ids, self.goal_local_pos, terrain_origins)
        direction_to_goal = goal_global_pos - self.robot_pos[env_ids]
        yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])

        # print(env_ids)
        # print(self.robot_pos[env_ids])
        # print(goal_global_pos[env_ids])
        # print("self.config_env_episode ", self.config_env_episode)
        # print("env_ids ", env_ids)
        # print("self.robot_pos[env_ids] ", self.robot_pos[env_ids])
        # print("self.goal_global_pos[env_ids] ", goal_global_pos)
        # print("angle_errors ", angle_errors)
        self.config_env_episode["dist_error"][env_ids] = torch.norm(self.robot_pos[env_ids] - goal_global_pos, dim=1)
        self.config_env_episode["angle_error"][env_ids] = angle_errors
        # print(self.config_env_episode["dist_error"][env_ids], self.config_env_episode["angle_error"][env_ids])
        # Порог для близости (выберите значение по задаче)
        angle_threshold = max(math.pi/18, 0.1 * max_angle)  # например, 10% от max_angle
        dist_threshold = max(0.1, 0.1 * mean_radius)  # например, 10% от mean_radius
        angle_close = torch.abs(self.config_env_episode["angle_error"][env_ids] - max_angle) <= angle_threshold
        dist_close = torch.abs(self.config_env_episode["dist_error"][env_ids] - self.base_radius - mean_radius) <= dist_threshold
        self.config_env_episode["relevant"][env_ids] = angle_close & dist_close
        # Возвращаем данные для симуляции
        self.robot_yaw[env_ids] = yaw + angle_errors * random_sign
        quaternion = torch.zeros((num_envs, 4), device=self.device)
        quaternion[:, 0] = torch.cos(self.robot_yaw[env_ids] / 2.0)  # w
        quaternion[:, 3] = torch.sin(self.robot_yaw[env_ids] / 2.0)  # z
        
        return self.robot_pos[env_ids], quaternion, goal_global_pos

    def get_relevant_env(self):
        return torch.where(self.config_env_episode["relevant"])[0]


    def generate_obstacle_positions(self, mess=False, env_ids=None, terrain_origins=None, min_num_active=0, max_num_active=None, mean_obs_rad=4,selected_indices=None):
        """
        Выбирает активные стулья и распределяет им случайные позиции из сетки.
        Неактивные стулья остаются на дефолтных позициях.
        
        Args:
            env_ids (torch.Tensor, optional): Индексы сред для обновления. Если None, обновляются все среды.
            terrain_origins (torch.Tensor, optional): Смещения локальных координат для каждой среды [num_envs, 3].
        
        Returns:
            List[ObstacleGraph]: Список графов для каждой среды.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        if terrain_origins is None:
            terrain_origins = torch.zeros((self.num_envs, 3), device=self.device)
        
        num_envs = len(env_ids)
        max_attempts = 10
        max_obstacles = self.num_obstacles
        max_positions = len(self.possible_positions)
        
        # Инициализируем позиции для всех сред
        positions = torch.stack([self.graphs[env_id.item()].default_positions 
                                for env_id in env_ids], dim=0)  # [num_envs, num_obstacles, 3]
        
        valid_envs = None
        for attempt in range(max_attempts):
            # Генерируем количество активных узлов для каждой среды
            if max_num_active == None:
                max_num_active = min(max_obstacles, max_positions)
            else:
                max_num_active = min(max_obstacles, max_positions, max_num_active)
            if selected_indices is not None:
                num_active = torch.tensor([len(selected_indices[env_id.item()]) for env_id in env_ids], device=self.device)
            else:
                num_active = torch.randint(min_num_active, max_num_active + 1, 
                                    size=(num_envs,), device=self.device)  # [num_envs]
            
            # Маска для сред с num_active == 0
            zero_active_mask = num_active == 0
            
            # Обрабатываем среды с num_active == 0
            if torch.any(zero_active_mask):
                terrain_offset = terrain_origins[env_ids[zero_active_mask], :2]  # [num_zero_active, 2]
                positions[zero_active_mask, :, :2] += terrain_offset[:, None, :]  # [num_zero_active, num_obstacles, 2]
                for env_idx, env_id in enumerate(env_ids[zero_active_mask]):
                    self.graphs[env_id.item()].update_positions([], positions[env_idx])
            
            # Обрабатываем среды с num_active > 0
            active_mask = ~zero_active_mask
            if not torch.any(active_mask):
                continue
            
            active_env_ids = env_ids[active_mask]  # [num_active_envs]
            active_num_active = num_active[active_mask]  # [num_active_envs]
            # print("num_active ", active_num_active)
            # Генерируем случайные индексы для активных узлов
            indices = torch.randperm(max_obstacles, device=self.device)[:active_num_active.max()]  # [max(num_active)]
            # if isinstance(indices, torch.Tensor):
            #     indices = indices.clone().detach().to(dtype=torch.long, device=self.device)[:n_active]
            # else:
            #     indices = torch.tensor(indices, device=self.device, dtype=torch.long)[:n_active]
            active_indices = [indices[:n].tolist() for n in active_num_active]  # Список списков для каждой среды
            self.active_indices = active_indices
            self.mess = mess
            if mess:
                max_num_env = active_num_active.max()
                # Initialize selected_positions as a list of empty lists
                selected_positions = [[] for _ in range(len(active_env_ids))]
                
                # Sequentially add obstacles up to max(active_num_active)
                for step in range(active_num_active.max().item()):
                    # Identify environments that need another obstacle
                    need_obstacle = active_num_active > step  # [num_active_envs]
                    if not torch.any(need_obstacle):
                        break
                    
                    # Select environments that still need obstacles
                    current_env_ids = active_env_ids[need_obstacle]  # [num_current_envs]
                    
                    # Generate one position per environment using get_pos
                    new_positions = self.get_pos(
                        env_ids=current_env_ids,
                        mean_radius=mean_obs_rad,
                        min_radius=3
                    )  # [num_current_envs, 2]
                    new_positions_3d = torch.cat([new_positions, torch.zeros((len(current_env_ids), 1), device=self.device)], dim=1)  # [num_current_envs, 3]
                    
                    # Update selected_positions and graphs for these environments
                    for env_idx, env_id in enumerate(current_env_ids):
                        local_env_idx = (active_env_ids == env_id).nonzero(as_tuple=True)[0].item()
                        selected_positions[local_env_idx].append(new_positions_3d[env_idx:env_idx+1])
                        # Temporarily update graph to include this obstacle
                        self.graphs[env_id.item()].update_positions([active_indices[local_env_idx][step]], new_positions_3d[env_idx:env_idx+1])
                
                # Convert selected_positions to list of tensors
                selected_positions = [torch.cat(pos_list, dim=0) if pos_list else torch.tensor([], device=self.device).reshape(0, 3) 
                                    for pos_list in selected_positions]
            else:
                # Выбираем случайные позиции из possible_positions
                possible_positions = torch.tensor(self.possible_positions, 
                                                device=self.device, dtype=torch.float32)  # [num_positions, 3]
                # print("possible_positions: ", possible_positions)
                if selected_indices is None:
                    selected_indices = {}
                    for env_id in env_ids:
                        selected_indices[env_id.item()] = torch.randperm(max_positions, device=self.device)[:active_num_active.max()]  # [max(num_active)]
                # print("selected_indices: ", selected_indices)
                # selected_positions = [possible_positions[indices[:n]] for n in active_num_active]  # Список тензоров [num_active_i, 3]
                selected_positions = [possible_positions[selected_indices.get(env_id.item(), []).clone().detach()] for env_id in active_env_ids]
                # print("env_ids: ", env_ids)
                for env_id in env_ids:
                    self.selected_indices[env_id.item()] = selected_indices[env_id.item()]
                # print("selected_indices: ", self.selected_indices)
            # Обновляем позиции для активных узлов
            for env_idx, (env_id, indices, sel_pos) in enumerate(zip(active_env_ids, active_indices, selected_positions)):
                positions[env_idx, indices] = sel_pos[:len(indices)]  # Обновляем активные узлы
                self.graphs[env_id.item()].update_positions(indices, sel_pos[:len(indices)])
            
            # Проверяем минимальное расстояние между активными узлами
            valid_envs = torch.ones(num_envs, dtype=torch.bool, device=self.device)
            for env_idx, env_id in enumerate(env_ids):
                if zero_active_mask[env_idx]:
                    continue
                active_edges = [(u, v) for u, v, d in self.graphs[env_id.item()].graph.edges(data=True) 
                                if d['weight'] != float('inf')]
                if not active_edges:
                    valid_envs[env_idx] = True
                    continue
                min_dist = min(self.graphs[env_id.item()].graph.edges[u, v]['weight'] for u, v in active_edges)
                min_required = min(node['radius'] for node in self.graphs[env_id.item()].get_active_nodes()) * 2
                valid_envs[env_idx] = min_dist >= min_required
                # print("not all envsd vcalid: ", min_dist, min_required, active_edges)
            
            # Если все среды валидны, выходим
            if torch.all(valid_envs):
                break
            # else:
            #     print("not all envsd vcalid: ", valid_envs[env_idx], env_idx)
        
        # Для невалидных сред устанавливаем дефолтные позиции
        if valid_envs is not None:
            invalid_envs = ~valid_envs

            if torch.any(invalid_envs):
                invalid_env_ids = env_ids[invalid_envs]
                positions[invalid_envs] = torch.stack([self.graphs[env_id.item()].default_positions 
                                                    for env_id in invalid_env_ids], dim=0)
                # terrain_offset = terrain_origins[invalid_env_ids, :2]
                # positions[invalid_envs, :, :2] += terrain_offset[:, None, :]
                for env_idx, env_id in enumerate(invalid_env_ids):
                    self.graphs[env_id.item()].update_positions([], positions[env_idx])
            
        # # Обновляем active_nodes и выводим отладочную информацию
        # for env_id in env_ids:
        #     print(env_id, self.graphs[env_id.item()].get_graph_info())
        
        return self.graphs
    
    def print_graph_info(self, env_ids=None):
        for env_id in env_ids:
            print(env_id, self.graphs[env_id.item()].get_graph_info())

    def get_obstacles(self, env_id: int):
        """
        Возвращает информацию о препятствиях для указанной среды.

        Args:
            env_id (int): Индекс среды.

        Returns:
            tuple: (None, None, list) - Пустые значения для обратной совместимости и список ID активных препятствий.
        """
        active_nodes = self.graphs[env_id].get_active_nodes()
        obstacles_id = [i for i in range(self.num_obstacles) if self.graphs[env_id].active[i]]
        return None, None, obstacles_id


'''
### **Переменные класса `Scene_manager`**

#### **Инициализация и основные параметры**

13. **`max_radius_values`**:
    - **Тип**: `float`
    - **Назначение**: Глобальный максимальный радиус для расстояния между роботом и целью (по умолчанию 8).
    - **Пример**: Ограничивает максимальное расстояние при генерации позиций.

14. **`radius_values`**:
    - **Тип**: `torch.Tensor` `[num_envs]`
    - **Назначение**: Хранит радиусы (расстояния робот-цель) для каждой среды.
    - **Пример**: `radius_values[0] = 2.0` — расстояние между роботом и целью в среде 0 равно 2 метра.

15. **`angle_values`**:
    - **Тип**: `torch.Tensor` `[1]`
    - **Назначение**: Хранит угол (по умолчанию `[0]`), используется для инициализации.
    - **Пример**: Не активно используется в коде, возможно, зарезервировано для будущих функций.

16. **`max_radius`**:
    - **Тип**: `float`
    - **Назначение**: Максимально допустимое расстояние между роботом и целью, рассчитанное как диагональ комнаты минус радиусы робота, цели и отступ (0.2 м).
    - **Пример**: Для комнаты 10x8 м с радиусами робота 0.5 м и цели 0.3 м: `max_radius ≈ 6.2 м`.

17. **`robot_bounds`**:
    - **Тип**: `dict`
    - **Назначение**: Границы для позиций робота с учетом его радиуса и дополнительных отступов.
    - **Пример**: `{'x_min': -3.5, 'x_max': 3.5, 'y_min': -2.5, 'y_max': 2.5}` — допустимая область для центра робота.

18. **`goal_bounds`**:
    - **Тип**: `dict`
    - **Назначение**: Границы для позиций цели с учетом ее радиуса.
    - **Пример**: `{'x_min': -3.7, 'x_max': 3.7, 'y_min': -2.7, 'y_max': 2.7}` — допустимая область для центра цели.

19. **`room_vertices`**:
    - **Тип**: `torch.Tensor` `[4, 2]`
    - **Назначение**: Координаты углов комнаты с учетом радиусов робота и цели (для расчета допустимых радиусов).
    - **Пример**: Углы комнаты с отступами: `[[-3.9, -2.9], [-3.9, 2.9], [3.9, -2.9], [3.9, 2.9]]`.

20. **`n_angles`**:
    - **Тип**: `int`
    - **Назначение**: Количество шагов дискретизации углов (по умолчанию 36, шаг `π/36` ≈ 5 градусов).
    - **Пример**: Используется для дискретизации углов при генерации позиций робота.

21. **`num_angle_steps`**:
    - **Тип**: `int`
    - **Назначение**: Общее количество угловых шагов (по умолчанию `2 * n_angles = 72`, от 0 до 2π).
    - **Пример**: Определяет, сколько углов проверяется для размещения робота.

22. **`angle_step`**:
    - **Тип**: `float`
    - **Назначение**: Шаг угловой дискретизации (`π/n_angles`).
    - **Пример**: Для `n_angles=36`, `angle_step ≈ 0.087 радиан (5 градусов)`.

23. **`discrete_angles`**:
    - **Тип**: `torch.Tensor` `[num_angle_steps]`
    - **Назначение**: Тензор дискретных углов от 0 до 2π с шагом `angle_step`.
    - **Пример**: `discrete_angles = [0, 0.087, 0.174, ..., 6.195]` — массив углов для генерации позиций.

24. **`possible_positions`**:
    - **Тип**: `list[tuple]`
    - **Назначение**: Список возможных позиций для препятствий, заданных в виде сетки (x, y, z=0).
    - **Пример**: `[(-1.5, -1.0, 0.0), (-1.5, 0.0, 0.0), (-1.5, 1.0, 0.0)]` — три возможные позиции для препятствий.

25. **`graphs`**:
    - **Тип**: `list[ObstacleGraph]`
    - **Назначение**: Список объектов `ObstacleGraph` для каждой среды, управляющих препятствиями.
    - **Пример**: Каждый граф хранит позиции, радиусы и связи препятствий в одной среде.

26. **`obstacle_radii`**:
    - **Тип**: `list[float]`
    - **Назначение**: Список радиусов препятствий (по умолчанию `[0.5] * num_obstacles`).
    - **Пример**: Все препятствия имеют радиус 0.5 м.

27. **`selected_indices`**:
    - **Тип**: `dict`
    - **Назначение**: Словарь, хранящий индексы выбранных позиций для активных препятствий в каждой среде.
    - **Пример**: `selected_indices[0] = [0, 2]` — в среде 0 активны препятствия с индексами 0 и 2.

28. **`active_indices`**:
    - **Тип**: `list[list[int]]` или `None`
    - **Назначение**: Список индексов активных препятствий для каждой среды (используется временно в `generate_obstacle_positions`).
    - **Пример**: `[[0, 1], [2], []]` — активные препятствия для трех сред.

29. **`mess`**:
    - **Тип**: `bool` или `None`
    - **Назначение**: Флаг, указывающий, генерировать ли случайные позиции препятствий (`True`) или использовать фиксированную сетку (`False`).
    - **Пример**: Если `mess=True`, препятствия размещаются случайно с помощью `get_pos`.

30. **`base_radius`**:
    - **Тип**: `float`
    - **Назначение**: Базовый радиус для генерации позиций робота или препятствий (по умолчанию 1).
    - **Пример**: Добавляется к `mean_radius` для смещения позиций.

31. **`config_env_episode`**:
    - **Тип**: `dict`
    - **Назначение**: Словарь, хранящий информацию о текущем эпизоде для каждой среды:
      - `angle_error`: Тензор `[num_envs]`, угол ошибки ориентации робота относительно цели.
      - `dist_error`: Тензор `[num_envs]`, расстояние между роботом и целью.
      - `relevant`: Булев тензор `[num_envs]`, указывающий, является ли конфигурация среды "релевантной" (в пределах допустимых порогов).
    - **Пример**: `config_env_episode["dist_error"][0] = 2.0` — расстояние до цели в среде 0 равно 2 м.

---

### **Методы класса `Scene_manager`**

1. **`__init__(num_envs=1, device='cuda:0', num_obstacles=6)`**:
   - **Назначение**: Инициализирует объект `Scene_manager`, задавая начальные параметры, тензоры и границы.
   - **Параметры**:
     - `num_envs`: Количество сред.
     - `device`: Устройство для вычислений.
     - `num_obstacles`: Количество препятствий.
   - **Действия**:
     - Инициализирует тензоры для позиций и ориентаций.
     - Задает границы комнаты и объектов.
     - Создает список объектов `ObstacleGraph` для каждой среды.
     - Устанавливает радиусы препятствий.
   - **Пример**: `Scene_manager(num_envs=64, device='cuda:0')` создает менеджер для 64 сред на GPU.

2. **`get_selected_indices(env_id=None)`**:
   - **Назначение**: Возвращает индексы выбранных позиций препятствий для указанной среды или всех сред.
   - **Параметры**:
     - `env_id`: Индекс среды (если `None`, возвращает весь словарь `selected_indices`).
   - **Возвращает**: Тензор с индексами или словарь `selected_indices`.
   - **Пример**: `get_selected_indices(0)` возвращает `[0, 2]` для среды 0.

3. **`get_mess()`**:
   - **Назначение**: Возвращает значение флага `mess`.
   - **Возвращает**: `bool` или `None`.
   - **Пример**: Если `mess=True`, указывает, что препятствия размещаются случайно.

4. **`compute_max_radii(goal_pos)`**:
   - **Назначение**: Вычисляет максимальный радиус для размещения робота относительно цели с учетом границ комнаты.
   - **Параметры**:
     - `goal_pos`: Тензор `[num_envs, 2]` с позициями целей.
   - **Возвращает**: Тензор `[num_envs]` с максимальными радиусами.
   - **Действия**:
     - Рассчитывает расстояния от целей до углов комнаты.
     - Вычитает радиусы робота и цели, а также отступ (0.2 м).
     - Ограничивает радиус минимальным значением (0.5 м).
   - **Пример**: Для цели в `[-4.5, 0]` возвращает радиус, ограниченный границами комнаты.

5. **`_compute_valid_angles(goal_pos, radii)`**:
   - **Назначение**: Определяет допустимые углы для размещения робота на заданном радиусе от цели.
   - **Параметры**:
     - `goal_pos`: Тензор `[num_envs, 2]` с позициями целей.
     - `radii`: Тензор `[num_envs]` с радиусами.
   - **Возвращает**: Булев тензор `[num_envs, num_angle_steps]`, где `True` — допустимый угол.
   - **Действия**:
     - Проверяет, попадают ли позиции робота (для каждого угла из `discrete_angles`) в границы `robot_bounds`.
   - **Пример**: Для радиуса 2 м и цели `[0, 0]` проверяет 72 угла и возвращает маску допустимых.

6. **`initialize_radii(num_envs, mean_radius, std_radius, max_radii=8, min_radius=1.2, max_radius=None, device='cuda')`**:
   - **Назначение**: Генерирует радиусы для размещения робота с нормальным распределением.
   - **Параметры**:
     - `num_envs`: Количество сред.
     - `mean_radius`: Среднее значение радиуса.
     - `std_radius`: Стандартное отклонение радиуса.
     - `max_radii`: Максимальный радиус (по умолчанию 8).
     - `min_radius`: Минимальный радиус (по умолчанию 1.2).
     - `max_radius`: Глобальный максимум (если `None`, используется `max_radius_values`).
     - `device`: Устройство для тензоров.
   - **Возвращает**: Тензор `[num_envs]` с радиусами.
   - **Действия**:
     - Генерирует радиусы с нормальным распределением.
     - Ограничивает их между `min_radius` и минимальным из `max_radii`, `max_radius`.
   - **Пример**: `initialize_radii(64, 2.0, 0.5)` возвращает 64 радиуса около 2 м.

7. **`generate_goal_positions_local(env_ids)`**:
   - **Назначение**: Генерирует случайные позиции целей в пределах `goal_bounds`.
   - **Параметры**:
     - `env_ids`: Тензор с индексами сред.
   - **Возвращает**: Тензор `[num_envs, 2]` с позициями целей.
   - **Действия**:
     - Генерирует координаты x, y из нормального распределения с сигмой, равной 1/4 ширины/высоты области.
     - Ограничивает координаты в пределах `goal_bounds`.
     - В текущем коде возвращает фиксированную позицию `[-4.5, 0.0]` для всех сред.
   - **Пример**: Для `env_ids=[0, 1]` возвращает `[[-4.5, 0.0], [-4.5, 0.0]]`.

8. **`shift_pos(env_ids, obj_pos, terrain_origins)`**:
   - **Назначение**: Смещает позиции объектов (например, цели) на величину `terrain_origins`.
   - **Параметры**:
     - `env_ids`: Тензор с индексами сред.
     - `obj_pos`: Тензор с локальными позициями `[num_envs, 2]`.
     - `terrain_origins`: Тензор с глобальными смещениями `[num_envs, 3]`.
   - **Возвращает**: Тензор `[num_envs, 2]` с глобальными позициями.
   - **Пример**: Для `obj_pos=[[0, 0]]`, `terrain_origins=[[1, 2, 0]]` возвращает `[[1, 2]]`.

9. **`_generate_obj_positions(env_ids, goal_pos, mean_radius, min_radius, std_radius=0.1)`**:
   - **Назначение**: Генерирует позиции робота на заданном расстоянии от цели.
   - **Параметры**:
     - `env_ids`: Тензор с индексами сред.
     - `goal_pos`: Тензор `[num_envs, 2]` с позициями целей.
     - `mean_radius`: Средний радиус для размещения.
     - `min_radius`: Минимальный радиус.
     - `std_radius`: Стандартное отклонение радиуса.
   - **Возвращает**: Тензор `[num_envs, 2]` с позициями робота.
   - **Действия**:
     - Вычисляет максимальные радиусы с помощью `compute_max_radii`.
     - Генерирует радиусы через `initialize_radii`.
     - Определяет допустимые углы через `_compute_valid_angles`.
     - Выбирает случайный допустимый угол и вычисляет позицию робота.
   - **Пример**: Для цели `[0, 0]` и радиуса 2 м может вернуть `[2, 0]` (если угол 0).

10. **`check_bouds(env_ids, new_obj_pos)`**:
    - **Назначение**: Проверяет, находятся ли позиции робота в пределах `robot_bounds`.
    - **Параметры**:
      - `env_ids`: Тензор с индексами сред.
      - `new_obj_pos`: Тензор `[num_envs, 2]` с позициями робота.
    - **Возвращает**: Кортеж `(result, robot_valid)`:
      - `result`: `bool`, `True`, если все позиции валидны.
      - `robot_valid`: Булев тензор `[num_envs]`, указывающий валидность каждой позиции.
    - **Пример**: Для `new_obj_pos=[[0, 0], [10, 0]]` возвращает `(False, [True, False])`.

11. **`get_checked_for_room_bounds_pos(env_ids, mean_radius, min_radius)`**:
    - **Назначение**: Генерирует позиции робота, проверяя их на соответствие `robot_bounds`.
    - **Параметры**:
      - `env_ids`: Тензор с индексами сред.
      - `mean_radius`: Средний радиус.
      - `min_radius`: Минимальный радиус.
    - **Возвращает**: Тензор `[num_envs, 2]` с валидными позициями робота.
    - **Действия**:
      - Итеративно генерирует позиции через `_generate_obj_positions`.
      - Проверяет их через `check_bouds`, перегенерируя для невалидных сред.
      - Останавливается после `max_iterations` (100) или когда все позиции валидны.
    - **Пример**: Возвращает позиции робота, не выходящие за границы комнаты.

12. **`check_obstacles_collision(obj_pos, env_ids, obstacle_positions=None, valid_mask=None)`**:
    - **Назначение**: Проверяет коллизии между роботом и препятствиями.
    - **Параметры**:
      - `obj_pos`: Тензор `[num_envs, 2]` с позициями робота.
      - `env_ids`: Тензор с индексами сред.
      - `obstacle_positions`: Тензор `[num_envs, max_obstacles, 2]` с позициями препятствий (если `None`, собирается из графов).
      - `valid_mask`: Булев тензор `[num_envs, max_obstacles]`, указывающий активные препятствия (если `None`, собирается).
    - **Возвращает**: Кортеж `(result, collisions)`:
      - `result`: `bool`, `True`, если нет коллизий.
      - `collisions`: Булев тензор `[num_envs, max_obstacles]`, указывающий наличие коллизий.
    - **Действия**:
      - Собирает позиции и радиусы активных препятствий, если они не переданы.
      - Вычисляет расстояния между роботом и препятствиями.
      - Проверяет, меньше ли расстояние суммы радиусов робота и препятствия плюс отступ (`safety_margin=0.3`).
    - **Пример**: Для робота в `[0, 0]` и препятствия в `[0.5, 0]` с радиусами 0.5 м возвращает коллизию.

13. **`get_checked_for_obstacles_rp(env_ids, obj_pos)`**:
    - **Назначение**: Корректирует позиции робота для устранения коллизий с препятствиями.
    - **Параметры**:
      - `env_ids`: Тензор с индексами сред.
      - `obj_pos`: Тензор `[num_envs, 2]` с позициями робота.
    - **Возвращает**: Тензор `[num_envs, 2]` с откорректированными позициями.
    - **Действия**:
      - Собирает позиции и радиусы активных препятствий.
      - Генерирует случайные направления для смещения.
      - Итеративно смещает робота от ближайшего препятствия на безопасное расстояние.
      - Проверяет коллизии через `check_obstacles_collision`.
    - **Пример**: Если робот в `[0, 0]` сталкивается с препятствием в `[0.5, 0]`, смещает его, например, в `[1.3, 0]`.

14. **`get_pos(env_ids, terrain_origins=None, mean_radius=2, min_radius=1.2)`**:
    - **Назначение**: Генерирует валидные позиции робота с учетом границ комнаты и препятствий.
    - **Параметры**:
      - `env_ids`: Тензор с индексами сред.
      - `terrain_origins`: Тензор `[num_envs, 3]` с глобальными смещениями (если `None`, смещение не применяется).
      - `mean_radius`: Средний радиус.
      - `min_radius`: Минимальный радиус.
    - **Возвращает**: Тензор `[num_envs, 2]` с позициями робота.
    - **Действия**:
      - Вызывает `get_checked_for_room_bounds_pos` для проверки границ.
      - Вызывает `get_checked_for_obstacles_rp` для устранения коллизий.
      - Итеративно повторяет, пока позиции не станут валидными.
    - **Пример**: Возвращает позиции робота, не пересекающиеся с препятствиями и границами.

15. **`reset(env_ids, terrain_origins, mean_radius=1.3, min_radius=1.2, max_angle=0)`**:
    - **Назначение**: Сбрасывает состояние среды, генерируя новые позиции и ориентации робота и цели.
    - **Параметры**:
      - `env_ids`: Тензор с индексами сред.
      - `terrain_origins`: Тензор `[num_envs, 3]` с глобальными смещениями.
      - `mean_radius`: Средний радиус для размещения робота.
      - `min_radius`: Минимальный радиус.
      - `max_angle`: Максимальный угол ошибки ориентации.
    - **Возвращает**: Кортеж `(robot_pos, quaternion, goal_global_pos)`:
      - `robot_pos`: Тензор `[num_envs, 2]` с позициями робота.
      - `quaternion`: Тензор `[num_envs, 4]` с кватернионами ориентации робота.
      - `goal_global_pos`: Тензор `[num_envs, 2]` с глобальными позициями целей.
    - **Действия**:
      - Генерирует позиции целей через `generate_goal_positions_local`.
      - Генерирует позиции робота через `get_pos`.
      - Вычисляет ориентацию робота (yaw) с учетом направления к цели и случайной ошибки угла.
      - Рассчитывает ошибки расстояния и угла, определяет "релевантность" среды.
      - Преобразует yaw в кватернион для ориентации.
    - **Пример**: Для среды 0 может вернуть позицию робота `[2, 0]`, кватернион `[cos(0.5), 0, 0, sin(0.5)]`, цель `[-4.5, 0]`.

16. **`get_relevant_env()`**:
    - **Назначение**: Возвращает индексы сред, которые считаются "релевантными" (где ошибки угла и расстояния в пределах порогов).
    - **Возвращает**: Тензор с индексами релевантных сред.
    - **Пример**: Если `relevant=[True, False]`, возвращает `[0]`.

17. **`generate_obstacle_positions(mess=False, env_ids=None, terrain_origins=None, min_num_active=0, max_num_active=None, mean_obs_rad=4, selected_indices=None)`**:
    - **Назначение**: Генерирует позиции препятствий, либо из фиксированной сетки, либо случайно.
    - **Параметры**:
      - `mess`: Если `True`, препятствия размещаются случайно через `get_pos`.
      - `env_ids`: Тензор с индексами сред (если `None`, все среды).
      - `terrain_origins`: Тензор `[num_envs, 3]` с глобальными смещениями (если `None`, нули).
      - `min_num_active`: Минимальное количество активных препятствий.
      - `max_num_active`: Максимальное количество активных препятствий.
      - `mean_obs_rad`: Средний радиус для случайного размещения препятствий.
      - `selected_indices`: Словарь с индексами выбранных позиций (если `None`, генерируются случайно).
    - **Возвращает**: Список объектов `ObstacleGraph` для каждой среды.
    - **Действия**:
      - Определяет количество активных препятствий для каждой среды.
      - Для `mess=False` выбирает позиции из `possible_positions`.
      - Для `mess=True` генерирует случайные позиции через `get_pos`.
      - Проверяет минимальное расстояние между препятствиями.
      - Обновляет графы препятствий (`graphs`) новыми позициями.
    - **Пример**: Для `mess=False` выбирает позиции из сетки, например, `[-1.5, -1.0, 0]`.

18. **`print_graph_info(env_ids=None)`**:
    - **Назначение**: Выводит информацию о графе препятствий для указанных сред.
    - **Параметры**:
      - `env_ids`: Тензор с индексами сред.
    - **Действия**: Вызывает метод `get_graph_info` для каждого графа.
    - **Пример**: Выводит позиции и связи активных препятствий для среды 0.

19. **`get_obstacles(env_id)`**:
    - **Назначение**: Возвращает информацию о препятствиях для указанной среды.
    - **Параметры**:
      - `env_id`: Индекс среды (целочисленный).
    - **Возвращает**: Кортеж `(None, None, obstacles_id)`:
      - Первые два значения — заглушки для обратной совместимости.
      - `obstacles_id`: Список индексов активных препятствий.
    - **Пример**: Для среды 0 может вернуть `(None, None, [0, 2])`, если активны препятствия 0 и 2.

---

### **Общее назначение класса**
Класс `Scene_manager` управляет сценой в симуляции, где робот должен достигать цели, избегая препятствий. Он отвечает за:
- Генерацию случайных позиций робота и цели с учетом границ комнаты и коллизий.
- Управление препятствиями через объект `ObstacleGraph`.
- Проверку и корректировку позиций для избежания коллизий.
- Инициализацию и сброс состояния среды для новых эпизодов.
- Хранение и предоставление данных о позициях, ориентациях и релевантности сред.

Класс оптимизирован для параллельной обработки нескольких сред (`num_envs`) с использованием тензоров PyTorch, что делает его подходящим для обучения с подкреплением или симуляций в реальном времени.
'''