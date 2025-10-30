# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FBCEnvPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # увеличено с 8
    max_iterations = 1500
    save_interval = 50
    experiment_name = "husky_nav_baseline"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # увеличено с 0.5
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256],  # увеличена емкость
        critic_hidden_dims=[256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,  # уменьшено с 1.0
        use_clipped_value_loss=True,
        clip_param=0.1,  # уменьшено с 0.2
        entropy_coef=0.01,  # увеличено с 0.005
        num_learning_epochs=4,  # уменьшено с 5
        num_mini_batches=8,  # уменьшено с 16
        learning_rate=3.0e-4,  # уменьшено с 1.0e-3
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,  # увеличено с 0.01
        max_grad_norm=0.5,  # уменьшено с 1.0
    )