import os
current_dir = os.getcwd()

class Asset_paths:
    def __init__(self):
        print("Текущая директория:", current_dir)
        self.kitchen_usd_path = new_path = os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", "scenes/scenes_sber_kitchen_for_BBQ/kitchen_new_simple.usd")
        self.table_usd_path =  new_path = os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", "scenes/scenes_sber_kitchen_for_BBQ/table/table_new.usd")
        self.bowl_usd_path =  new_path = os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", "objects/bowl.usd")
        self.aloha_usd_path =  new_path = os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", "aloha/ALOHA_with_sensor_02.usd")
        self.chair_usd_path =  new_path = os.path.join(current_dir, "source/isaaclab_assets/data/aloha_assets", "scenes/obstacles/chair2.usd")
        self.log_usd_path =  new_path = os.path.join(current_dir, "logs"),
