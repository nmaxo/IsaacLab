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
        
        self.sim_objects = {}    # name -> [RigidObject, ...]
        self.prim_paths = {}     # name -> [prim_path_str, ...]
        self.all_prim_paths = [] # список prim_path для всех объектов в порядке индексов (глобальный mapping)

    def spawn_assets_in_scene(self):
        """Создает все объекты из конфига в симуляторе Isaac Lab."""
        current_dir = os.getcwd()
        for obj_cfg in self.config:
            name = obj_cfg['name']
            types = obj_cfg['type']
            if "info" in types:
                print("info with ", name)
                continue  
            count = obj_cfg['count']
            usd_paths = obj_cfg['usd_paths']
            
            self.sim_objects[name] = []
            self.prim_paths[name] = []
            for i in range(count):
                random_usd_path = random.choice(usd_paths)
                usd_path = os.path.join(
                    current_dir,
                    "source/isaaclab_assets/data/aloha_assets",
                    random_usd_path
                )
                rot = (1.0, 0.0, 0.0, 0.0) 
                if name == "bowl":
                    # Для миски используем Z-up ориентацию (кватернион [0, 0.7071, 0, 0.7071])
                    rot = (0.0, 0.7071, 0.0, 0.7071)

                prim_path = f"/World/envs/env_.*/{name}_{i}"

                instance_cfg = RigidObjectCfg(
                    prim_path=prim_path,
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
                        pos=(4.0 + i, 6.0, 0.0),  # Начальная позиция (кладбище)
                        rot=rot
                    ),
                )
                obj_instance = RigidObject(cfg=instance_cfg)

                # сохраняем сам объект
                self.sim_objects[name].append(obj_instance)
                # сохраняем prim_path
                self.prim_paths[name].append(prim_path)
                self.all_prim_paths.append(prim_path)

        return self.sim_objects
