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
    id="Isaac-FBC-Husky-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_env_cfg:FBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FBCEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-FBC-Husky-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_env_cfg:FBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FBCEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

##
# DWBC loco-manipulation (dual-head actor/critic + DAgger)
##

gym.register(
    id="Isaac-FBC-Husky-DWBC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_dwbc_env_cfg:HuskyDwbcEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwbc_cfg:HuskyDwbcRunnerCfg",
    },
)

gym.register(
    id="Isaac-FBC-Husky-DWBC-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_dwbc_env_cfg:HuskyDwbcEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwbc_cfg:HuskyDwbcRunnerCfg",
    },
)

##
# IK-based EE trajectory tracking (Go2-ARX style pipeline for Husky)
##

gym.register(
    id="Isaac-FBC-Husky-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_ik_env_cfg:HuskyIkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ik_ppo_cfg:HuskyIkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-FBC-Husky-IK-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_ik_env_cfg:HuskyIkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ik_ppo_cfg:HuskyIkPPORunnerCfg",
    },
)

##
# World-frame IK tracking (base-arm cooperation)
##

gym.register(
    id="Isaac-FBC-Husky-WorldIK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_world_ik_env_cfg:HuskyWorldIkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_world_ik_ppo_cfg:HuskyWorldIkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-FBC-Husky-WorldIK-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_world_ik_env_cfg:HuskyWorldIkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_world_ik_ppo_cfg:HuskyWorldIkPPORunnerCfg",
    },
)

##
# World-frame IK + cabinet (approach + open)
##

gym.register(
    id="Isaac-FBC-Husky-WorldIK-Cabinet-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_world_ik_cabinet_env_cfg:HuskyWorldIkCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_world_ik_cabinet_ppo_cfg:HuskyWorldIkCabinetPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-FBC-Husky-WorldIK-Cabinet-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.FBC_world_ik_cabinet_env_cfg:HuskyWorldIkCabinetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_world_ik_cabinet_ppo_cfg:HuskyWorldIkCabinetPPORunnerCfg"
        ),
    },
)
