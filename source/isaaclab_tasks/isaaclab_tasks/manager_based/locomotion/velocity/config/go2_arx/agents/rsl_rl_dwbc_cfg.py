"""DWBC PPO config for Go2-ARX loco-manipulation (dual-head, policy observations only)."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.dwbc import DwbcActorCriticCfg, DwbcAlgorithmCfg, DwbcRunnerCfg

from ..loco_manip_env_cfg import REWARD_GROUPS
from ..loco_manip_dwbc_orig_env_cfg import REWARD_GROUPS_DWBC_ORIG


@configclass
class Go2ArxDwbcRunnerCfg(DwbcRunnerCfg):
    num_steps_per_env = 40
    max_iterations = 40000
    save_interval = 500
    experiment_name = ""

    reward_groups = REWARD_GROUPS

    policy = DwbcActorCriticCfg(
        num_leg_actions=12,
        num_arm_actions=6,
        actor_hidden_dims=[128],
        critic_hidden_dims=[128],
        leg_control_head_hidden_dims=[128, 128],
        arm_control_head_hidden_dims=[128, 128],
        activation="elu",
        init_noise_std=[0.8, 1.0, 1.0] * 4 + [1.0] * 6,
    )

    algorithm = DwbcAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=1.0,
        mixing_schedule=[1.0, 0, 3000],
        min_policy_std=[
            0.15, 0.25, 0.25,
            0.15, 0.25, 0.25,
            0.15, 0.25, 0.25,
            0.15, 0.25, 0.25,
            0.2, 0.2, 0.2, 0.05, 0.05, 0.05,
        ],
    )


@configclass
class Go2ArxDwbcOrigRunnerCfg(Go2ArxDwbcRunnerCfg):
    """Runner cfg for DWBC-Orig env variant (DWBC-like rewards/goals)."""

    reward_groups = REWARD_GROUPS_DWBC_ORIG
    reward_scale = 0.01
    action_delay_steps = 1
