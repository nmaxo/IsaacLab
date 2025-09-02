import torch
import networkx as nx
from enum import Enum
import random

# ADDED: Enum для кодирования типов объектов
class ObjectType(Enum):
    OBSTACLE = 0        # Перемещаемые препятствия (стулья)
    FIXED_OBSTACLE = 1  # Стационарные препятствия (столы)
    POSSIBLE_GOAL = 2   # Потенциальные цели (миски)
    STUFF = 3           # Декоративные объекты (коробки)
    UNKNOWN = 4

class ObstacleGraph:
    # CHANGED: Инициализация теперь принимает полную конфигурацию
    def __init__(self, objects_config: list, device='cuda:0'):
        """
        Args:
            objects_config (list): Список словарей с конфигурацией для каждого объекта из JSON.
            device (str or torch.device): Устройство для тензоров (cuda или cpu).
        """
        self.device = device
        self.graph = nx.Graph()
        
        # CHANGED: Считаем общее количество узлов и храним конфиг
        self.objects_config = objects_config
        self.num_nodes = sum(obj['count'] for obj in self.objects_config)

        self._initialize_nodes()

    def _initialize_nodes(self):
        """Инициализирует узлы с дефолтными позициями и атрибутами из конфига."""
        self.default_positions = torch.zeros(self.num_nodes, 3, device=self.device)
        self.positions = torch.zeros(self.num_nodes, 3, device=self.device)
        self.radii = torch.zeros(self.num_nodes, device=self.device)
        self.sizes = torch.zeros(self.num_nodes, 3, device=self.device)
        self.types = torch.full((self.num_nodes,), ObjectType.UNKNOWN.value, dtype=torch.long, device=self.device)
        self.active = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        self.names = [''] * self.num_nodes

        # Отображение строки типа в Enum
        type_map = {
            "obstacle": ObjectType.OBSTACLE,
            "fixed_obstacle": ObjectType.FIXED_OBSTACLE,
            "possible_goal": ObjectType.POSSIBLE_GOAL,
            "stuff": ObjectType.STUFF,
        }

        # Заполняем атрибуты для каждого объекта из конфига
        node_idx = 0
        for obj_cfg in self.objects_config:
            count = obj_cfg['count']
            
            # Дефолтные позиции за сценой
            base_x = 10.0 + node_idx * 2.0
            y_pos = torch.linspace(-count / 2.0, count / 2.0, count, device=self.device)
            
            for i in range(count):
                self.names[node_idx] = obj_cfg['name']
                self.types[node_idx] = type_map.get(obj_cfg['type'], ObjectType.UNKNOWN).value
                self.sizes[node_idx] = torch.tensor(obj_cfg['size'], device=self.device)
                self.radii[node_idx] = obj_cfg.get('radius', 0.0) # Используем .get для 'stuff'
                self.default_positions[node_idx] = torch.tensor([base_x, y_pos[i], 0.0], device=self.device)
                node_idx += 1

        self.positions = self.default_positions.clone()
        
        # Инициализируем NetworkX граф
        for i in range(self.num_nodes):
            self.graph.add_node(i, 
                                name=self.names[i], 
                                radius=self.radii[i].item(),
                                size=tuple(self.sizes[i].tolist()),
                                type=self.types[i].item(),
                                position=tuple(self.positions[i].tolist()),
                                default_position=tuple(self.default_positions[i].tolist()),
                                active=self.active[i].item())
        self._update_edges()

    def _update_edges(self):
        """Обновляет рёбра на основе текущих позиций узлов."""
        self.graph.remove_edges_from(list(self.graph.edges))
        dists = torch.cdist(self.positions[:, :2], self.positions[:, :2])
        active_pairs = self.active[:, None] & self.active[None, :]
        dists = torch.where(active_pairs, dists, torch.tensor(float('inf'), device=self.device))
        
        for i in range(self.num_nodes):
            # Синхронизируем атрибуты узла в графе
            node_data = self.graph.nodes[i]
            node_data['position'] = tuple(self.positions[i].tolist())
            node_data['active'] = self.active[i].item()
            node_data['radius'] = self.radii[i].item()
            node_data['size'] = tuple(self.sizes[i].tolist())
            node_data['type'] = self.types[i].item()
            node_data['name'] = self.names[i]
            for j in range(i + 1, self.num_nodes):
                self.graph.add_edge(i, j, weight=dists[i, j].item())

    # REMOVED: set_radii (теперь все устанавливается при инициализации)

    def update_positions(self, active_indices, new_positions):
        """Обновляет позиции активных узлов и рёбра."""
        self.active[:] = False
        if active_indices and new_positions.numel() > 0:
            active_indices_tensor = torch.tensor(active_indices, device=self.device, dtype=torch.long)
            self.active[active_indices_tensor] = True
            self.positions[active_indices_tensor] = new_positions.clone().detach()
            
        # Сбрасываем неактивные узлы на дефолтные позиции
        self.positions[~self.active] = self.default_positions[~self.active]
        self._update_edges()

    def get_active_nodes(self):
        """Возвращает список словарей с данными активных узлов."""
        active_mask = self.active
        active_indices = torch.where(active_mask)[0]
        
        active_nodes_data = []
        for i in active_indices:
            active_nodes_data.append({
                'name': self.names[i],
                'radius': self.radii[i].item(),
                'size': tuple(self.sizes[i].tolist()),
                'type': self.types[i].item(),
                'position': tuple(self.positions[i].tolist()),
                'active': True,
                'default_position': tuple(self.default_positions[i].tolist())
            })
        return active_nodes_data
    
    def graph_to_tensor(self):
        """Преобразует состояние графа в тензор для RL агента."""
        # Нормируем позиции и размеры для стабильности сети
        norm_positions = self.positions / 10.0 
        norm_sizes = self.sizes / 2.0
        # Кодируем тип one-hot вектором
        type_one_hot = torch.nn.functional.one_hot(self.types, num_classes=len(ObjectType)).float()
        
        # Конкатенируем все признаки: [x, y, z, size_x, size_y, size_z, type_0, ..., type_N]
        features = torch.cat([norm_positions, norm_sizes, self.radii.unsqueeze(-1), type_one_hot], dim=-1)
        return features.flatten() # Возвращаем плоский вектор

    def get_graph_info(self):
        """Возвращает отформатированную информацию о графе для отладки."""
        info = f"ObstacleGraph with {self.num_nodes} nodes:\n"
        info += "Nodes:\n"
        for i in range(self.num_nodes):
            info += f"  Node {i} ({self.names[i]}):\n"
            info += f"    Type: {ObjectType(self.types[i].item()).name}\n"
            info += f"    Radius: {self.radii[i]:.2f}\n"
            info += f"    Size: ({self.sizes[i, 0]:.2f}, {self.sizes[i, 1]:.2f}, {self.sizes[i, 2]:.2f})\n"
            info += f"    Position: ({self.positions[i, 0]:.2f}, {self.positions[i, 1]:.2f}, {self.positions[i, 2]:.2f})\n"
            info += f"    Active: {self.active[i]}\n"
        
        active_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True) if d['weight'] != float('inf')]
        info += f"\nEdges ({len(active_edges)} active):\n"
        if not active_edges:
            info += "  No active edges.\n"
        else:
            for u, v, data in active_edges:
                info += f"  Edge ({u}, {v}): Weight = {data['weight']:.2f}\n"
        
        return info