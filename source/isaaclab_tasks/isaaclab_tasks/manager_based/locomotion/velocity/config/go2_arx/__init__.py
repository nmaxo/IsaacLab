# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Arx-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2ArxFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArxFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Arx-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2ArxFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArxFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Arx-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2ArxRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArxRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Arx-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2ArxRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArxRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

##
# Loco-manipulation (multi-critic)
##

gym.register(
    id="Go2Arx-LocoManip-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:Go2ArxLocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_multi_critic_cfg:Go2ArxLocoManipMultiCriticPPORunnerCfg",
    },
)

gym.register(
    id="Go2Arx-LocoManip-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:Go2ArxLocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_multi_critic_cfg:Go2ArxLocoManipMultiCriticPPORunnerCfg",
    },
)

gym.register(
    id="Go2Arx-LocoManip-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:Go2ArxLocoManipFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_multi_critic_cfg:Go2ArxLocoManipMultiCriticPPORunnerCfg",
    },
)

gym.register(
    id="Go2Arx-LocoManip-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:Go2ArxLocoManipFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_multi_critic_cfg:Go2ArxLocoManipMultiCriticPPORunnerCfg",
    },
)

##
# LeggedGym-style (asymmetric PPO + privileged heights)
##

gym.register(
    id="Go2Arx-LeggedGym-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leggedgym_env_cfg:Go2ArxLeggedGymEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_leggedgym_ppo_cfg:Go2ArxLeggedGymPPORunnerCfg",
    },
)

gym.register(
    id="Go2Arx-LeggedGym-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leggedgym_env_cfg:Go2ArxLeggedGymEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_leggedgym_ppo_cfg:Go2ArxLeggedGymPPORunnerCfg",
    },
)

##
# DWBC loco-manipulation (dual-head actor/critic + DAgger)
##

gym.register(
    id="Go2Arx-LocoManip-DWBC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_dwbc_env_cfg:Go2ArxLocoManipDwbcEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwbc_cfg:Go2ArxDwbcRunnerCfg",
    },
)

gym.register(
    id="Go2Arx-LocoManip-DWBC-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_dwbc_env_cfg:Go2ArxLocoManipDwbcEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwbc_cfg:Go2ArxDwbcRunnerCfg",
    },
)

gym.register(
    id="Go2Arx-LocoManip-DWBC-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_dwbc_env_cfg:Go2ArxLocoManipDwbcFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwbc_cfg:Go2ArxDwbcRunnerCfg",
    },
)

gym.register(
    id="Go2Arx-LocoManip-DWBC-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_dwbc_env_cfg:Go2ArxLocoManipDwbcFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwbc_cfg:Go2ArxDwbcRunnerCfg",
    },
)

##
# DWBC-Orig (match DWBC widowGo1 rewards/goals more closely)
##

gym.register(
    id="Go2Arx-LocoManip-DWBC-Orig-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_dwbc_orig_env_cfg:Go2ArxLocoManipDwbcOrigEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwbc_cfg:Go2ArxDwbcOrigRunnerCfg",
    },
)

gym.register(
    id="Go2Arx-LocoManip-DWBC-Orig-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_dwbc_orig_env_cfg:Go2ArxLocoManipDwbcOrigEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwbc_cfg:Go2ArxDwbcOrigRunnerCfg",
    },
)
