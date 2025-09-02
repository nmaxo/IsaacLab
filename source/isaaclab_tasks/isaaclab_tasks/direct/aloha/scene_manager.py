import torch
import math
from scipy.spatial import ConvexHull
import random
import json

# from .graph_manager import ObstacleGraph
# from .graph_manager import ObstacleGraph, ObjectType # Убедитесь, что ObjectType импортируется

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
ObjectType = import_class_from_path(module_path, "ObjectType")

class Scene_manager:
    def __init__(self, num_envs=1, device='cuda:0', num_obstacles=6, num_goals=3):
        self.num_envs = num_envs
        self.device = device
        config_path='config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.objects_config = self.config['objects']

        self.object_indices = {}
        self.object_type_map = {}
        start_idx = 0
        for obj in self.objects_config:
            name = obj['name']
            count = obj['count']
            end_idx = start_idx + count
            indices = list(range(start_idx, end_idx))
            self.object_indices[name] = indices
            
            # Сохраняем Enum значение типа для каждого индекса
            type_enum = getattr(ObjectType, obj['type'].upper(), ObjectType.UNKNOWN)
            for i in indices:
                self.object_type_map[i] = type_enum
            
            start_idx = end_idx

        # Тензоры для хранения позиций и ориентаций
        self.robot_pos = torch.zeros((num_envs, 2), device=device)  # [x, y]
        self.robot_yaw = torch.zeros(num_envs, device=device)      # угол yaw
        self.goal_local_pos = torch.zeros((num_envs, 2), device=device)  # [x, y] цели
        self.active_goal_node_idx = torch.zeros(num_envs, dtype=torch.long, device=device) # Индекс узла цели

        # Параметры управления
        self.max_linear_speed = 1.0   # максимальная линейная скорость (м/с)
        self.max_angular_speed = 0.5  # максимальная угловая скорость (рад/с)
        self.angle_threshold = math.pi / 30  # порог угла (10 градусов) для начала движения вперед
        
        # Размеры объектов
        self.robot_radius = 0.5  # радиус робота (м)
        self.goal_radius = 0.3   # радиус цели (м)
        
        # Границы комнаты в локальной системе координат
        self.room_bounds = {
            'x_min': -4.5,
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
            'x_min': self.room_bounds['x_min'] + 0.5,  # -3.5
            'x_max': self.room_bounds['x_max'] - self.robot_radius - 1,  # 3.5
            'y_min': self.room_bounds['y_min'] + self.robot_radius + 1.5,  # -2.5
            'y_max': self.room_bounds['y_max'] - self.robot_radius - 1.5   # 2.5
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
        grid_x = [-2.2]  # Уровни по X для активных позиций
        grid_y = [-1.1, 0.0, 1.1]  # Уровни по Y для активных позиций
        self.possible_positions = [(x, y, 0.0) for x in grid_x for y in grid_y]  # 6 возможных позиций
        self.graphs = [ObstacleGraph(self.objects_config, device) for _ in range(num_envs)]
        self.goal_graphs = [ObstacleGraph(num_goals, device) for _ in range(num_envs)]
        self.obstacle_radii = [0.5] * num_obstacles
        for graph in self.graphs:
            graph.set_radii(self.obstacle_radii)
        self.selected_indices = {}
        self.active_indices = None
        self.mess = None
        self.base_radius = 1.25

        self.config_env_episode = {
            "angle_error": torch.zeros(num_envs, device=device),
            "dist_error": torch.zeros(num_envs, device=device),
            "relevant": torch.zeros(num_envs,  device=device, dtype=torch.bool),
            "local_robots_start_position": torch.zeros(self.num_envs, 2, device=self.device)
        }

    def get_selected_indices(self, env_id=None):
        # print("[ SCENE MANAGER DEBUG ] selected_indices: ", self.selected_indices)
        # print("[ SCENE MANAGER DEBUG ] active_indices: ", self.active_indices)
        try:
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
        except Exception as e:
            print(f"[SCENE MANAGER ERROR] Exception: {e}")
            return None

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
        """Выбирает случайную активную цель из графа для каждой среды."""
        num_envs = len(env_ids)
        positions = torch.zeros((num_envs, 2), device=self.device)
        
        for i, env_id in enumerate(env_ids):
            graph = self.graphs[env_id.item()]
            
            # Находим все активные узлы, которые являются целями
            active_goal_indices = [
                idx for idx, data in graph.graph.nodes(data=True)
                if data['active'] and data['type'] == ObjectType.POSSIBLE_GOAL.value
            ]
            
            if not active_goal_indices:
                # Если активных целей нет, ставим дефолтную позицию (аварийный случай)
                print(f"[WARNING] No active goals found for env {env_id.item()}. Using default goal.")
                chosen_goal_pos = torch.tensor([-4.5, 0.0], device=self.device)
                self.active_goal_node_idx[env_id] = -1 # Признак отсутствия цели
            else:
                # Выбираем случайную цель из активных
                chosen_goal_node_idx = random.choice(active_goal_indices)
                self.active_goal_node_idx[env_id] = chosen_goal_node_idx
                chosen_goal_pos = torch.tensor(graph.graph.nodes[chosen_goal_node_idx]['position'][:2], device=self.device)
            
            positions[i] = chosen_goal_pos
            
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
            std_radius=(mean_radius-base_radius)*0.2,
            max_radii=5,
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
        max_iterations = 50
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
            if (iteration + 1) % max_iterations == 0:
                print("[ error ] in getting correction position, iteration: ", iteration)
                for i in invalid_env_ids:
                    print(f"graph for {i.item()} env: ", self.graphs[i.item()].get_graph_info())
                print("invalid_env_ids: ", invalid_env_ids)
                print("goal_local_pos: ", self.goal_local_pos[invalid_env_ids])
                print("obj_pos: ", obj_pos)
                print("[ error ] in getting correction position, end log")
                print("___________________")
                

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
            obstacles_per_env = []
            for env_id in env_ids:
                graph = self.graphs[env_id.item()]
                nodes_with_radius = [
                    {'position': data['position'], 'radius': data['radius']}
                    for _, data in graph.graph.nodes(data=True)
                    if data['active'] and data['radius'] > 0
                ]
                obstacles_per_env.append(nodes_with_radius)
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
        required_dists = self.robot_radius + 0.51# obstacle_radii
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
        
        safety_margin = 0.4
        max_iterations = 10  # Ограничение на итерации
        # print("i ", obj_pos)
        # Собираем активные препятствия
        obstacles_per_env = []
        for env_id in env_ids:
            graph = self.graphs[env_id.item()]
            nodes_with_radius = [
                {'position': data['position'], 'radius': data['radius']}
                for _, data in graph.graph.nodes(data=True)
                if data['active'] and data['radius'] > 0
            ]
            obstacles_per_env.append(nodes_with_radius)
        env_ids = torch.tensor(range(num_envs), device=self.device)
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
                # used_directions[near_zero] = torch.randn((near_zero.sum(), 2), device=self.device)
                used_directions[near_zero] = used_directions[near_zero] / torch.norm(used_directions[near_zero], dim=1, keepdim=True)

            # Смещаем робота на границу препятствия в направлении used_directions
            displacements = used_directions * (closest_radii + self.robot_radius + safety_margin)[:, None]
            # print("displacements", displacements)
            obj_pos[abs_env_ids] = closest_positions + displacements
            # print("4 ", obj_pos, abs_env_ids)
            # print(f"self.robot_pos[{abs_env_ids}]", self.robot_pos[abs_env_ids])
        return obj_pos

    def get_pos(self, env_ids, terrain_origins=None, mean_radius=2.0, min_radius=1.2):
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

    def print_start_info(self, env_ids):
        for env_id in env_ids:
            print(f"[ DEBUG ]: env: {env_id.item()}")
            print("start position:")
            print(self.config_env_episode["local_robots_start_position"][env_id.item()])
            print("obstacles positions:")
            print(self.graphs[env_id.item()].get_graph_info())
        _, collisions = self.check_obstacles_collision(self.config_env_episode["local_robots_start_position"][env_ids], env_ids)
        print("Collisions:")
        print(collisions)

    def reset(self, env_ids, terrain_origins, mean_radius=1.3,min_radius=1.2,max_angle=0.0):
        """Инициализация стартовых позиций и целей при сбросе."""
        num_envs = len(env_ids)       
        # Проверка и корректировка позиций
        self.goal_local_pos[env_ids] = self.generate_goal_positions_local(
            env_ids)
        
        self.robot_pos[env_ids] = self.get_pos(
            env_ids, terrain_origins, mean_radius, min_radius)
        self.config_env_episode["local_robots_start_position"][env_ids] = self.robot_pos[env_ids].clone()
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

    def get_start_dist_error(self):
        return self.config_env_episode["dist_error"]

    def get_relevant_env(self):
        return torch.where(self.config_env_episode["relevant"])[0]


    def place_scene_objects(self, mess=False, env_ids=None, min_num_obstacles=0, max_num_obstacles=None, min_num_goals=1, max_num_goals=None):
        """Размещает все объекты на сцене согласно их типам и правилам."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        # Шаг 1: Размещаем стационарные объекты (столы)
        for obj_cfg in self.objects_config:
            if obj_cfg['type'] == 'fixed_obstacle':
                indices = self.object_indices[obj_cfg['name']]
                positions = torch.tensor(obj_cfg['placement_grid'], device=self.device, dtype=torch.float32)
                for env_id in env_ids:
                    # Столы всегда активны и находятся на своих местах
                    self.graphs[env_id.item()].update_positions(indices, positions)

        # Шаг 2: Размещаем перемещаемые объекты (стулья)
        chair_cfg = next((obj for obj in self.objects_config if obj['type'] == 'obstacle'), None)
        if chair_cfg:
            if max_num_obstacles is None:
                max_num_obstacles = chair_cfg['count']
            
            num_active_chairs = torch.randint(min_num_obstacles, max_num_obstacles + 1, size=(len(env_ids),), device=self.device)
            
            chair_indices = self.object_indices[chair_cfg['name']]
            possible_chair_pos = torch.tensor(chair_cfg['placement_grid'], device=self.device, dtype=torch.float32)

            for i, env_id in enumerate(env_ids):
                graph = self.graphs[env_id.item()]
                num_active = num_active_chairs[i].item()
                
                # Выбираем случайные стулья для активации
                active_chair_indices = random.sample(chair_indices, num_active)
                # Выбираем случайные позиции для них
                pos_indices = torch.randperm(len(possible_chair_pos), device=self.device)[:num_active]
                
                graph.positions[active_chair_indices] = possible_chair_pos[pos_indices]
                graph.active[active_chair_indices] = True

        # Шаг 3: Размещаем цели (миски) на столах
        goal_cfg = next((obj for obj in self.objects_config if obj['type'] == 'possible_goal'), None)
        table_cfg = next((obj for obj in self.objects_config if obj['type'] == 'fixed_obstacle'), None)
        
        if goal_cfg and table_cfg:
            if max_num_goals is None:
                max_num_goals = goal_cfg['count']

            num_active_goals = torch.randint(min_num_goals, max_num_goals + 1, size=(len(env_ids),), device=self.device)
            goal_indices = self.object_indices[goal_cfg['name']]
            table_indices = self.object_indices[table_cfg['name']]

            for i, env_id in enumerate(env_ids):
                graph = self.graphs[env_id.item()]
                num_active = num_active_goals[i].item()
                
                # Выбираем случайные миски для активации
                active_goal_indices = random.sample(goal_indices, num_active)
                # Выбираем случайные столы для размещения мисок
                target_table_indices = random.choices(table_indices, k=num_active)

                for goal_idx, table_idx in zip(active_goal_indices, target_table_indices):
                    table_pos = graph.positions[table_idx]
                    table_size = graph.sizes[table_idx]
                    
                    # Генерируем случайную позицию на поверхности стола
                    # Отступ от края 0.1
                    rand_x = (torch.rand(1, device=self.device).item() - 0.5) * (table_size[0] - 0.2)
                    rand_y = (torch.rand(1, device=self.device).item() - 0.5) * (table_size[1] - 0.2)
                    
                    goal_pos_z = table_pos[2] + table_size[2] # Поверхность стола
                    graph.positions[goal_idx] = torch.tensor([table_pos[0] + rand_x, table_pos[1] + rand_y, goal_pos_z], device=self.device)
                    graph.active[goal_idx] = True

        # Шаг 4: Обновляем рёбра и состояние графов после всех размещений
        for env_id in env_ids:
            self.graphs[env_id.item()]._update_edges()

        return self.graphs
    
    def print_graph_info(self, env_ids=None):
        for env_id in env_ids:
            print(env_id, self.graphs[env_id.item()].get_graph_info())
    
    def get_scene_embedding(self, env_ids):
        """
        Формирует эмбеддинг сцены как конкатенацию координат препятствий (x, y)
        и фиксированной цели [-4.5, 0].
        """
        embeddings = []
        for env_id in env_ids:
            emb = self.graphs[env_id.item()].graph_to_tensor()  # [num_chairs, 4] или [3, 4]
            # Берём только первые 2 координаты (x, y)
            chair_positions = emb[:, :2]  # [3, 2]
            # Добавляем цель
            goal_pos = torch.tensor([-4.5, 0.0], device=emb.device, dtype=emb.dtype).unsqueeze(0)  # [1, 2]
            # Конкатенация
            scene_vec = torch.cat([chair_positions.flatten(), goal_pos.flatten()])  # [3*2 + 2] = [8]
            embeddings.append(scene_vec)

        return torch.stack(embeddings, dim=0)  # [batch_size, 8]

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
