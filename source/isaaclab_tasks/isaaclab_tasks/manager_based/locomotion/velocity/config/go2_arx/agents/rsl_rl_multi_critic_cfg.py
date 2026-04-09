"""Multi-critic PPO config for Go2-ARX Full Body Control.

Two critics (locomotion includes contact rewards):
  - locomotion:    walk stably toward EE target + foot contact rewards
  - manipulation:  reach EE target pose
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.multi_critic import (
    RslRlMultiCriticPpoActorCriticCfg,
    RslRlMultiCriticPpoAlgorithmCfg,
    RslRlMultiCriticRunnerCfg,
)

from ..loco_manip_env_cfg import REWARD_GROUPS


@configclass
class Go2ArxLocoManipMultiCriticPPORunnerCfg(RslRlMultiCriticRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = ""

    reward_groups = REWARD_GROUPS

    policy = RslRlMultiCriticPpoActorCriticCfg(
        num_critics=2,
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="relu",
    )

    algorithm = RslRlMultiCriticPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
