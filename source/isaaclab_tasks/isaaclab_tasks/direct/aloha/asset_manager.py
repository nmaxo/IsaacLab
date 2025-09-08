import json
import random
from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
import os

class AssetManager:
    """Управляет загрузкой конфига и созданием ассетов в симуляции."""
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)['objects']
        
        self.sim_objects = {}

    def spawn_assets_in_scene(self):
        """Создает все объекты из конфига в симуляторе Isaac Lab."""
        current_dir = os.getcwd()
        for obj_cfg in self.config:
            name = obj_cfg['name']
            count = obj_cfg['count']
            usd_paths = obj_cfg['usd_paths']
            
            self.sim_objects[name] = []
            for i in range(count):
                random_usd_path = random.choice(usd_paths)
                usd_path = os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", random_usd_path)
                rot = (1.0, 0.0, 0.0, 0.0) 
                if name == "bowl":
                    # Для миски используем Z-up ориентацию (кватернион [1, 0, 0, 0])
                    rot = (0.0, 0.7071, 0.0, 0.7071)
                instance_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/{name}_{i}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=usd_path,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            rigid_body_enabled=True,
                            kinematic_enabled=True,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            articulation_enabled=False,  # Отключаем articulation root
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(4.0 + i, 6.0, 0.0),  # Начальная позиция
                        rot=rot    # Кватернион для Z-up ориентации
                    ),
                )
                obj_instance = RigidObject(cfg=instance_cfg)
                self.sim_objects[name].append(obj_instance)
        return self.sim_objects