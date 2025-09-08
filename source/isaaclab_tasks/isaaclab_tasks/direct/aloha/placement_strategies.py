import torch
import random
from abc import ABC, abstractmethod
# ПРИМЕЧАНИЕ: ObstacleGraph больше не используется. Вместо него передаются тензоры
# из нового VectorizedSceneManager.

class PlacementStrategy(ABC):
    def __init__(self, device: str, **kwargs):
        self.device = device

    @abstractmethod
    def apply(self, env_ids: torch.Tensor, object_indices: torch.Tensor, scene_data: dict, mess: bool):
        # scene_data будет словарем с тензорами: 'positions', 'active' и т.д.
        pass

class GridPlacement(PlacementStrategy):
    def __init__(self, device: str, grid_coordinates: list[list[float]]):
        super().__init__(device)
        self.grid = torch.tensor(grid_coordinates, device=self.device, dtype=torch.float32)

    def apply(
        self,
        env_ids: torch.Tensor,
        obj_indices_to_place: torch.Tensor, # Форма: (num_envs, num_to_place)
        scene_data: dict,
        mess: bool
    ):
        num_to_place = obj_indices_to_place.shape[1]
        if num_to_place == 0:
            return

        num_grid_points = len(self.grid)
        
        # Генерируем случайные перестановки позиций на сетке для каждого окружения
        # torch.rand(...).argsort() - эффективный способ получить батч перестановок
        pos_indices = torch.rand(len(env_ids), num_grid_points, device=self.device).argsort(dim=1)[:, :num_to_place]
        
        # Выбираем позиции из сетки согласно сгенерированным индексам
        selected_positions = self.grid[pos_indices] # Форма: (num_envs, num_to_place, 3)

        # Обновляем глобальные тензоры позиций и состояний
        # Используем scatter_ для эффективного обновления по индексам
        env_idx_tensor = env_ids.view(-1, 1).expand_as(obj_indices_to_place)
        scene_data['positions'][env_idx_tensor, obj_indices_to_place] = selected_positions
        scene_data['active'][env_idx_tensor, obj_indices_to_place] = True
        scene_data['on_surface_idx'][env_idx_tensor, obj_indices_to_place] = -1 # Размещение на сетке = на полу
        scene_data['surface_level'][env_idx_tensor, obj_indices_to_place] = 0

class OnSurfacePlacement(PlacementStrategy):
    def __init__(self, device: str, surface_indices: list[int], margin: float):
        super().__init__(device)
        self.surface_indices = torch.tensor(surface_indices, device=self.device, dtype=torch.long)
        self.margin = margin

    def apply(
        self,
        env_ids: torch.Tensor,
        obj_indices_to_place: torch.Tensor, # Форма: (num_envs, num_to_place)
        scene_data: dict,
        mess: bool
    ):
        num_to_place = obj_indices_to_place.shape[1]
        if num_to_place == 0:
            return

        num_envs_to_process = len(env_ids)

        # 1. Находим все активные поверхности в нужных окружениях
        active_mask = scene_data['active'][env_ids][:, self.surface_indices] # (num_envs, num_surfaces)
        
        # 2. Для каждого окружения выбираем случайную активную поверхность
        # Используем torch.multinomial для векторизованного выбора
        # Добавляем малые значения для стабильности, если нет активных поверхностей
        probs = active_mask.float() + 1e-9
        chosen_surface_rel_idx = torch.multinomial(probs, num_to_place, replacement=True) # (num_envs, num_to_place)
        target_surface_idx = self.surface_indices[chosen_surface_rel_idx] # (num_envs, num_to_place)

        # 3. Собираем данные о выбранных поверхностях
        env_idx_tensor = env_ids.view(-1, 1).expand_as(target_surface_idx)
        surface_pos = scene_data['positions'][env_idx_tensor, target_surface_idx]
        surface_size = scene_data['sizes'][env_idx_tensor, target_surface_idx]
        
        # Собираем размеры размещаемых объектов
        obj_size = scene_data['sizes'][env_idx_tensor, obj_indices_to_place]

        # 4. Вычисляем новые позиции объектов
        rand_xy = torch.zeros(num_envs_to_process, num_to_place, 2, device=self.device)
        if mess:
            # Генерируем случайные смещения сразу для всех
            margin_tensor = torch.tensor([self.margin * 2, self.margin * 2], device=self.device)
            max_offsets = (surface_size[..., :2] - margin_tensor)
            rand_xy = (torch.rand_like(rand_xy) - 0.5) * max_offsets
        
        pos_z = surface_pos[..., 2] + surface_size[..., 2] + obj_size[..., 2] / 2.0
        new_pos_xy = surface_pos[..., :2] + rand_xy
        
        # 5. Обновляем глобальные тензоры
        scene_data['positions'][env_idx_tensor, obj_indices_to_place] = torch.cat([new_pos_xy, pos_z.unsqueeze(-1)], dim=-1)
        scene_data['active'][env_idx_tensor, obj_indices_to_place] = True
        scene_data['on_surface_idx'][env_idx_tensor, obj_indices_to_place] = target_surface_idx
        surface_levels = scene_data['surface_level'][env_idx_tensor, target_surface_idx]
        scene_data['surface_level'][env_idx_tensor, obj_indices_to_place] = surface_levels + 1