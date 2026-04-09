# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FBCEnvPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 256  # Увеличено для лучшей статистики
    max_iterations = 500    # Больше итераций
    save_interval = 100
    experiment_name = "new_fbc"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,  # Больше исследования вначале
        actor_hidden_dims=[512, 256, 128],  # Больше капасити
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Увеличено для исследования
        num_learning_epochs=8,  # Больше эпох
        num_mini_batches=32,    # Уменьшено (256*4096/32 = разумный batch)
        learning_rate=3.0e-4,   # Немного ниже для стабильности
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.04,        # Более консервативно
        max_grad_norm=1.0,
    )  