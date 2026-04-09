"""RSL-RL PPO config for Husky world-IK cabinet task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HuskyWorldIkCabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 256
    max_iterations = 5000
    save_interval = 200
    experiment_name = "husky_world_ik_cabinet_coop"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=32,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )
