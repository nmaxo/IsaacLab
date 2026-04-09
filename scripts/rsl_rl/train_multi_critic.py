"""Train a loco-manipulation policy using multi-critic PPO.

Usage:
    python scripts/rsl_rl/train_multi_critic.py --task Go2Arx-LocoManip-v0 --num_envs 4096

Based on: "Multi-critic Learning for Whole-body End-effector Twist Tracking"
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# Parse CLI args
parser = argparse.ArgumentParser(description="Train multi-critic PPO for loco-manipulation.")
parser.add_argument("--task", type=str, default="Go2Arx-LocoManip-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- After sim app is launched -----

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_rl.rsl_rl.multi_critic import (
    MultiCriticOnPolicyRunner,
    MultiCriticVecEnvWrapper,
)

import isaaclab_tasks  # noqa: F401 — registers gym tasks


def main():
    # Load env config from registry (resolves env_cfg_entry_point string to class instance)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Get reward groups from the env config module
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_arx.loco_manip_env_cfg import REWARD_GROUPS
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_arx.agents.rsl_rl_multi_critic_cfg import (
        Go2ArxLocoManipMultiCriticPPORunnerCfg,
    )

    runner_cfg = Go2ArxLocoManipMultiCriticPPORunnerCfg()
    if args_cli.max_iterations is not None:
        runner_cfg.max_iterations = args_cli.max_iterations

    runner_cfg.seed = args_cli.seed

    # Wrap environment with multi-critic wrapper
    env_wrapped = MultiCriticVecEnvWrapper(
        env,
        reward_groups=REWARD_GROUPS,
        clip_actions=runner_cfg.clip_actions,
    )

    # Setup log directory
    log_root_path = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Convert config to dict for the runner
    train_cfg = {
        "seed": runner_cfg.seed,
        "device": runner_cfg.device,
        "num_steps_per_env": runner_cfg.num_steps_per_env,
        "max_iterations": runner_cfg.max_iterations,
        "save_interval": runner_cfg.save_interval,
        "experiment_name": runner_cfg.experiment_name,
        "logger": runner_cfg.logger,
        "obs_groups": runner_cfg.obs_groups,
        "advantage_weights": getattr(runner_cfg, "advantage_weights", None),
        "policy": {
            "class_name": runner_cfg.policy.class_name,
            "num_critics": runner_cfg.policy.num_critics,
            "init_noise_std": runner_cfg.policy.init_noise_std,
            "noise_std_type": runner_cfg.policy.noise_std_type,
            "actor_obs_normalization": runner_cfg.policy.actor_obs_normalization,
            "critic_obs_normalization": runner_cfg.policy.critic_obs_normalization,
            "actor_hidden_dims": runner_cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": runner_cfg.policy.critic_hidden_dims,
            "activation": runner_cfg.policy.activation,
        },
        "algorithm": {
            "class_name": runner_cfg.algorithm.class_name,
            "num_learning_epochs": runner_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": runner_cfg.algorithm.num_mini_batches,
            "learning_rate": runner_cfg.algorithm.learning_rate,
            "schedule": runner_cfg.algorithm.schedule,
            "gamma": runner_cfg.algorithm.gamma,
            "lam": runner_cfg.algorithm.lam,
            "entropy_coef": runner_cfg.algorithm.entropy_coef,
            "desired_kl": runner_cfg.algorithm.desired_kl,
            "max_grad_norm": runner_cfg.algorithm.max_grad_norm,
            "value_loss_coef": runner_cfg.algorithm.value_loss_coef,
            "use_clipped_value_loss": runner_cfg.algorithm.use_clipped_value_loss,
            "clip_param": runner_cfg.algorithm.clip_param,
            "normalize_advantage_per_mini_batch": runner_cfg.algorithm.normalize_advantage_per_mini_batch,
        },
    }

    # Create runner
    runner = MultiCriticOnPolicyRunner(
        env=env_wrapped,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=runner_cfg.device,
    )

    # Resume if needed
    if runner_cfg.resume:
        print("[INFO] Resume training is not yet implemented for multi-critic. Starting fresh.")

    # Train
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    # Cleanup
    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
