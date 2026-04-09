"""Configuration dataclasses for multi-critic PPO."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlMultiCriticPpoActorCriticCfg:
    """Configuration for the multi-critic PPO actor-critic networks."""

    class_name: str = "ActorMultiCritic"

    num_critics: int = 2
    """Number of critic networks (one per reward group)."""

    init_noise_std: float = MISSING
    noise_std_type: Literal["scalar", "log"] = "scalar"
    actor_obs_normalization: bool = MISSING
    critic_obs_normalization: bool = MISSING
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    activation: str = MISSING


@configclass
class RslRlMultiCriticPpoAlgorithmCfg:
    """Configuration for the multi-critic PPO algorithm."""

    class_name: str = "MultiCriticPPO"

    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    entropy_coef: float = MISSING
    desired_kl: float = MISSING
    max_grad_norm: float = MISSING
    value_loss_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    normalize_advantage_per_mini_batch: bool = False


@configclass
class RslRlMultiCriticRunnerCfg:
    """Configuration for the multi-critic on-policy runner."""

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    obs_groups: dict | None = None
    clip_actions: float | None = None

    policy: RslRlMultiCriticPpoActorCriticCfg = MISSING
    algorithm: RslRlMultiCriticPpoAlgorithmCfg = MISSING

    reward_groups: dict[str, list[str]] = MISSING
    """Mapping from reward group name to list of reward term names.
    Example: {"locomotion": ["track_lin_vel_xy_exp", ...], "manipulation": ["ee_pos_tracking", ...]}
    """

    advantage_weights: dict[str, float] | None = None
    """Per-group advantage weights. If None, all groups weighted equally (1.0).
    Example: {"locomotion": 0.5, "manipulation": 1.0}
    """
