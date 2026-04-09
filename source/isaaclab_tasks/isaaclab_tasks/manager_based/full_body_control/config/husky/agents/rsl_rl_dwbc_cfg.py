"""DWBC PPO config for Husky UR5 loco-manipulation.

Dual-head actor: 2 wheel actions (DiffDrive) + 6 arm actions (UR5 joints).
Privileged encoder for the actor; dual value heads; advantage mixing.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.dwbc import DwbcActorCriticCfg, DwbcAlgorithmCfg, DwbcRunnerCfg

from ..FBC_dwbc_env_cfg import REWARD_GROUPS


@configclass
class HuskyDwbcRunnerCfg(DwbcRunnerCfg):
    num_steps_per_env = 256
    max_iterations = 1000
    save_interval = 100
    experiment_name = "new_fbc_dwbc"

    reward_groups = REWARD_GROUPS

    policy = DwbcActorCriticCfg(
        num_leg_actions=2,
        num_arm_actions=6,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        leg_control_head_hidden_dims=[256, 128],
        arm_control_head_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=0.8,
    )

    algorithm = DwbcAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=32,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        mixing_schedule=[0.0, 0.0, 0.0],
        min_policy_std=[
            0.28, 0.28,
            0.28, 0.28, 0.28, 0.28, 0.28, 0.28,
        ],
    )
