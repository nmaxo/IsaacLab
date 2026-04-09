"""Play (visualize) a trained multi-critic PPO checkpoint.

Usage:
    python scripts/rsl_rl/play_multi_critic.py \
        --task Go2Arx-LocoManip-Play-v0 \
        --checkpoint logs/rsl_rl/go2_arx_fbc_multi_critic/<run>/model_500.pt \
        --num_envs 50

    # Or auto-find latest checkpoint:
    python scripts/rsl_rl/play_multi_critic.py \
        --task Go2Arx-LocoManip-Play-v0 --num_envs 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play multi-critic PPO checkpoint.")
parser.add_argument("--task", type=str, default="Go2Arx-LocoManip-Play-v0")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint. If None, auto-find latest.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- After sim app is launched -----

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.multi_critic import (
    MultiCriticOnPolicyRunner,
    MultiCriticVecEnvWrapper,
)

import isaaclab_tasks  # noqa: F401


def _build_train_cfg(runner_cfg):
    """Convert the dataclass runner config to a dict (same as in train_multi_critic.py)."""
    return {
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


def main():
    # Load env config (Play variant: fewer envs, no noise)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Load runner config
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_arx.loco_manip_env_cfg import REWARD_GROUPS
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_arx.agents.rsl_rl_multi_critic_cfg import (
        Go2ArxLocoManipMultiCriticPPORunnerCfg,
    )

    runner_cfg = Go2ArxLocoManipMultiCriticPPORunnerCfg()

    # Wrap env with multi-critic wrapper (needed for runner to know reward_groups / num_actions)
    env_wrapped = MultiCriticVecEnvWrapper(
        env,
        reward_groups=REWARD_GROUPS,
        clip_actions=runner_cfg.clip_actions,
    )

    train_cfg = _build_train_cfg(runner_cfg)

    # Create runner (log_dir=None — no logging during play)
    runner = MultiCriticOnPolicyRunner(
        env=env_wrapped,
        train_cfg=train_cfg,
        log_dir=None,
        device=runner_cfg.device,
    )

    # Find checkpoint
    if args_cli.checkpoint is not None:
        resume_path = args_cli.checkpoint
    else:
        log_root_path = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, ".*", "model_.*.pt")

    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path, load_optimizer=False)

    # Get inference policy
    policy = runner.get_inference_policy(device=env_wrapped.device)

    dt = env.unwrapped.step_dt
    obs = env_wrapped.get_observations().to(runner_cfg.device)

    print("[INFO] Playing... Press Ctrl+C to stop.")
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env_wrapped.step(actions)
            obs = obs.to(runner_cfg.device)

        # Real-time pacing
        if args_cli.real_time:
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
