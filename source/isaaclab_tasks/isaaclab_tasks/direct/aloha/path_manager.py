import json
import torch
import os
from pathlib import Path

class Path_manager:
    def __init__(self, scene_manager, ratio: float = 4.0, shift: list = [5, 4], device: str = 'cpu'):
        """
        Инициализирует менеджер путей для загрузки путей из all_paths.json и преобразования координат.

        Args:
            log_dir (str): Путь к директории с all_paths.json.
            ratio (float): Масштабный коэффициент для преобразования координат (по умолчанию 4.0).
            shift (list): Смещение координат [shift_x, shift_y] (по умолчанию [5, 4]).
            device (str): Устройство для хранения путей (по умолчанию 'cpu').
        """
        self.device = device
        self.ratio = ratio
        self.shift = torch.tensor(shift, device=device, dtype=torch.float32)
        self.all_paths = {}
        self.paths_file = os.path.join("data", "all_paths.json")
        self._load_paths()
        self.scene_manager = scene_manager

    def _load_paths(self):
        """
        Загружает пути из all_paths.json.
        Формат: {config_key: {target_node: {start_node: path}}}
        """
        if os.path.exists(self.paths_file):
            try:
                with open(self.paths_file, 'r') as f:
                    loaded_paths = json.load(f)
                for config_key, targets in loaded_paths.items():
                    self.all_paths[config_key] = {}
                    for target_str, nodes in targets.items():
                        target = tuple(map(int, target_str.split(',')))
                        self.all_paths[config_key][target] = {}
                        for node_str, path in nodes.items():
                            node = tuple(map(int, node_str.split(',')))
                            self.all_paths[config_key][target][node] = [tuple(p) for p in path]
                print(f"Loaded {len(self.all_paths)} configurations from {self.paths_file}")
            except Exception as e:
                print(f"Error loading paths file: {e}")
                self.all_paths = {}
        else:
            print(f"No paths file found at {self.paths_file}")
            self.all_paths = {}

    def real_to_grid(self, real_point: torch.Tensor) -> torch.Tensor:
        """
        Преобразует реальные координаты (x, y) в сеточные.

        Args:
            real_point (torch.Tensor): Реальные координаты [num_envs, 2].

        Returns:
            torch.Tensor: Сеточные координаты [num_envs, 2], целочисленные.
        """
        grid_x = torch.round((real_point[:, 0] + self.shift[0]) * self.ratio).to(torch.int32)
        grid_y = torch.round((real_point[:, 1] + self.shift[1]) * self.ratio).to(torch.int32)
        return torch.stack([grid_x, grid_y], dim=-1)

    def grid_to_real(self, grid_point: torch.Tensor) -> torch.Tensor:
        """
        Преобразует сеточные координаты (x, y) в реальные.

        Args:
            grid_point (torch.Tensor): Сеточные координаты [..., 2].

        Returns:
            torch.Tensor: Реальные координаты [..., 2].
        """
        # print(grid_point)
        # print(grid_point[:, 0])
        # print(grid_point[..., 0])
        real_x = grid_point[..., 0] / self.ratio - self.shift[0]
        real_y = grid_point[..., 1] / self.ratio - self.shift[1]
        return torch.stack([real_x, real_y], dim=-1)
    
    def get_paths(self, env_ids: torch.Tensor, start_positions: torch.Tensor, target_positions: torch.Tensor, device: str = 'cuda:0'):
        """
        Возвращает пути для заданных конфигураций, стартовых и целевых позиций в реальных координатах.

        Args:
            env_ids (torch.Tensor): Индексы сред [num_envs].
            start_positions (torch.Tensor): Стартовые позиции в реальных координатах [num_envs, 2].
            target_positions (torch.Tensor): Целевые позиции в реальных координатах [num_envs, 2].
            device (str): Устройство для возвращаемых тензоров.

        Returns:
            torch.Tensor: Пути в реальных координатах [num_envs, 50, 2].
        """
        configs = []
        for env_id in env_ids:
            active_pos = self.scene_manager.get_active_obstacle_positions(env_id)
            config = ','.join([f"{x:.1f}_{y:.1f}_{z:.1f}" for x,y,z in sorted(active_pos)])
            configs.append(config)
        # print("configs: ", configs)
        # Преобразуем реальные координаты в сеточные
        start_nodes = self.real_to_grid(start_positions)
        target_nodes = self.real_to_grid(target_positions)
        # print("target pos: ", target_positions, target_nodes)
        max_path_length = 15
        self.max_path_length = max_path_length
        paths = []
        for i, (config, start, target) in enumerate(zip(configs, start_nodes.tolist(), target_nodes.tolist())):
            start = tuple(start)
            target = tuple(target)
            path = self.all_paths.get(config, {}).get(target, {}).get(start, [])
            # print(f"[ PATH MANAGER DEBUG ] start: {start}, target: {target}, cinfig: {config}")
            # Если путь не найден, ищем ближайшие узлы
            if not path:
                # print(f"No exact path for env {env_ids[i]}, config {config}, start {start}, target {target}")
                # Ищем ближайший целевой узел
                target_dict = self.all_paths.get(config, {})
                if target_dict:
                    nearest_target = self.find_nearest_node(target, set(target_dict.keys()))
                    if nearest_target:
                        # Ищем ближайший стартовый узел для найденного целевого
                        start_dict = target_dict.get(nearest_target, {})
                        if start_dict:
                            nearest_start = self.find_nearest_node(start, set(start_dict.keys()))
                            if nearest_start:
                                path = target_dict[nearest_target].get(nearest_start, [])
                                # print(f"Using nearest path: start {nearest_start}, target {nearest_target}")
            # print(f"[ PATH MANAGER DEBUG ] path: ", path)
            paths.append(path)

        # Создаем тензор путей в сеточных координатах
        path_tensor = torch.full((len(env_ids), max_path_length, 2), -7777.0, device=device, dtype=torch.float32)
        for i, (env_id, path) in enumerate(zip(env_ids, paths)):
            if path:
                path_length = len(path)
                if path_length > max_path_length:
                    print(f"[DEBUG] Warning: Path for env {env_id} truncated from {path_length} to {max_path_length}")
                    path = path[-max_path_length:]  # Берем последние max_path_length точек
                path_tensor[i, -len(path):] = torch.tensor(path, device=device, dtype=torch.float32)
            else:
                print(f"No path found for env {env_id}, config {configs[i]}, start {start_nodes[i].tolist()}, target {target_nodes[i].tolist()}")
                # Заполняем последнюю точку стартовой позицией
                path_tensor[i, -1] = start_nodes[i].to(device=device, dtype=torch.float32)

        # Преобразуем пути в реальные координаты
        # print(f"[ PATH MANAGER DEBUG ] path before grid to real: ", path_tensor)
        path_tensor = self.grid_to_real(path_tensor.to(device=device))
        # print(f"[ PATH MANAGER DEBUG ] path after grid to real: ", path_tensor)
        # print(f"[ PATH MANAGER DEBUG ] PathManager.get_paths: env_ids={env_ids.tolist()}, paths shape={path_tensor.shape}")
        return path_tensor

    def find_nearest_node(self, target: tuple, nodes: set) -> tuple:
        """
        Находит ближайший узел из набора узлов к целевой точке (манхэттенское расстояние).

        Args:
            target (tuple): Целевая точка в сеточных координатах (x, y).
            nodes (set): Набор узлов в сеточных координатах.

        Returns:
            tuple: Ближайший узел или None, если набор узлов пуст.
        """
        if not nodes:
            return None
        min_distance = float('inf')
        nearest_node = None
        target_x, target_y = target
        for node in nodes:
            x, y = node
            distance = abs(target_x - x) + abs(target_y - y)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        return nearest_node