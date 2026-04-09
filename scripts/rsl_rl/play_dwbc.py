"""Play (visualize) a trained DWBC checkpoint.

Usage:
    ./isaaclab.sh -p scripts/rsl_rl/play_dwbc.py --task Go2Arx-LocoManip-DWBC-Orig-Play-v0 --num_envs 50
    ./isaaclab.sh -p scripts/rsl_rl/play_dwbc.py --task Go2Arx-LocoManip-DWBC-Play-v0 --checkpoint logs/.../model_500.pt
    ./isaaclab.sh -p scripts/rsl_rl/play_dwbc.py --task Isaac-FBC-Husky-DWBC-Play-v0 --checkpoint logs/.../model_500.pt
"""

from __future__ import annotations

import argparse
import importlib
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play DWBC PPO checkpoint.")
parser.add_argument("--task", type=str, default="Go2Arx-LocoManip-DWBC-Orig-Play-v0")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint. If None, auto-find latest.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- After sim app is launched -----

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, get_checkpoint_path  # noqa: E402
from isaaclab_rl.rsl_rl.dwbc import DwbcOnPolicyRunner, DwbcVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402


def _load_runner_cfg(task_name: str):
    """Dynamically load the DwbcRunnerCfg from the gym registry entry point."""
    spec = gym.spec(task_name)
    cfg_entry = spec.kwargs.get("rsl_rl_cfg_entry_point")
    if cfg_entry is None:
        raise ValueError(f"Task '{task_name}' has no 'rsl_rl_cfg_entry_point' in gym kwargs.")
    module_path, class_name = cfg_entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def _build_train_cfg(runner_cfg):
    return {
        "seed": runner_cfg.seed,
        "device": runner_cfg.device,
        "num_steps_per_env": runner_cfg.num_steps_per_env,
        "max_iterations": runner_cfg.max_iterations,
        "save_interval": runner_cfg.save_interval,
        "experiment_name": runner_cfg.experiment_name,
        "logger": runner_cfg.logger,
        "obs_groups": runner_cfg.obs_groups,
        "reward_groups": runner_cfg.reward_groups,
        "policy": runner_cfg.policy.to_dict(),
        "algorithm": runner_cfg.algorithm.to_dict(),
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    runner_cfg = _load_runner_cfg(args_cli.task)

    env_wrapped = DwbcVecEnvWrapper(
        env,
        reward_groups=runner_cfg.reward_groups,
        clip_actions=runner_cfg.clip_actions,
        reward_scale=getattr(runner_cfg, "reward_scale", None),
        action_delay_steps=getattr(runner_cfg, "action_delay_steps", 0),
    )

    train_cfg = _build_train_cfg(runner_cfg)

    runner = DwbcOnPolicyRunner(
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

    policy = runner.get_inference_policy(device=env_wrapped.device)

    dt = env.unwrapped.step_dt
    obs_prop = env_wrapped.get_observations()
    obs_prop = obs_prop.to(runner_cfg.device)

    print("[INFO] Playing... Press Ctrl+C to stop.")
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs_prop)
            obs_prop, *_ = env_wrapped.step(actions)
            obs_prop = obs_prop.to(runner_cfg.device)

        if args_cli.real_time:
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

