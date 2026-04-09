"""Configuration dataclasses for DWBC (Deep Whole-Body Control) PPO.

Based on: "Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion"
(Fu, Cheng, Pathak, CoRL 2022)
"""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Literal

from isaaclab.utils import configclass


@configclass
class DwbcActorCriticCfg:
    """Configuration for the DWBC actor-critic networks."""

    class_name: str = "ActorCriticDWBC"

    num_leg_actions: int = MISSING
    """Number of leg action dimensions."""

    num_arm_actions: int = MISSING
    """Number of arm action dimensions."""

    actor_hidden_dims: list[int] = MISSING
    """Hidden dimensions for the shared actor backbone."""

    critic_hidden_dims: list[int] = MISSING
    """Hidden dimensions for the shared critic backbone."""

    leg_control_head_hidden_dims: list[int] = MISSING
    """Hidden dimensions for the leg action head."""

    arm_control_head_hidden_dims: list[int] = MISSING
    """Hidden dimensions for the arm action head."""

    activation: str = "elu"

    init_noise_std: list[float] | float = 1.0
    """Initial action noise std. Can be per-action or scalar."""


@configclass
class DwbcAlgorithmCfg:
    """Configuration for the DWBC PPO algorithm."""

    class_name: str = "PPO_DWBC"

    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = "fixed"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float | None = None
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2

    mixing_schedule: list[float] = field(default_factory=lambda: [1.0, 0, 3000])
    """[max_ratio, start_iter, ramp_iters]. Advantage mixing ratio ramps linearly."""

    min_policy_std: list[float] | None = None
    """Minimum per-action std. Enforced after each PPO update."""


@configclass
class DwbcRunnerCfg:
    """Configuration for the DWBC on-policy runner."""

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    obs_groups: dict | None = None
    clip_actions: float | None = None
    reward_scale: float | None = None
    """Reward scaling factor. If None, uses env step_dt (Isaac Lab default).
    Set to 0.01 (=1/100) to match original DWBC reward scaling."""
    action_delay_steps: int = 0
    """Number of steps to delay action application. Original DWBC uses 1."""

    policy: DwbcActorCriticCfg = MISSING
    algorithm: DwbcAlgorithmCfg = MISSING

    reward_groups: dict[str, list[str]] = MISSING
    """Mapping from reward group name to list of reward term names.
    Must contain exactly 2 groups: one for legs, one for arm.
    Example: {"locomotion": ["track_lin_vel_xy_exp", ...], "manipulation": ["ee_pos_tracking", ...]}
    """
