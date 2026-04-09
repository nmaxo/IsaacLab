"""Train a loco-manipulation policy using DWBC PPO (dual-head actor/critic, policy obs only).

Usage:
    ./isaaclab.sh -p scripts/rsl_rl/train_dwbc.py --task Go2Arx-LocoManip-DWBC-v0 --num_envs 4096 --headless
    ./isaaclab.sh -p scripts/rsl_rl/train_dwbc.py --task Isaac-FBC-Husky-DWBC-v0 --num_envs 4096 --headless

Resume from checkpoint (continue training from same weights + optimizer):
    ./isaaclab.sh -p scripts/rsl_rl/train_dwbc.py --task Go2Arx-LocoManip-DWBC-v0 --resume
    ./isaaclab.sh -p scripts/rsl_rl/train_dwbc.py --task Go2Arx-LocoManip-DWBC-v0 --resume --checkpoint model_500.pt
"""

from __future__ import annotations

import argparse
import importlib
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# Parse CLI args (must happen before AppLauncher)
parser = argparse.ArgumentParser(description="Train DWBC PPO for loco-manipulation.")
parser.add_argument("--task", type=str, default="Go2Arx-LocoManip-DWBC-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint (same log run, latest or --checkpoint).")
parser.add_argument("--load_run", type=str, default=None, help="Run folder name to resume from (regex). Default: latest run.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load (regex, e.g. model_500.pt or model_.*.pt). Default: latest.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- After sim app is launched -----

import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, get_checkpoint_path  # noqa: E402

from isaaclab_rl.rsl_rl.dwbc import DwbcOnPolicyRunner, DwbcVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402 — registers gym tasks


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


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    runner_cfg = _load_runner_cfg(args_cli.task)
    if args_cli.max_iterations is not None:
        runner_cfg.max_iterations = args_cli.max_iterations
    runner_cfg.seed = args_cli.seed

    env_wrapped = DwbcVecEnvWrapper(
        env,
        reward_groups=runner_cfg.reward_groups,
        clip_actions=runner_cfg.clip_actions,
        reward_scale=getattr(runner_cfg, "reward_scale", None),
        action_delay_steps=getattr(runner_cfg, "action_delay_steps", 0),
    )

    log_root_path = os.path.join("logs", "rsl_rl", runner_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = None
    if args_cli.resume:
        ckpt_arg = args_cli.checkpoint
        # Full path to a .pt file: use as-is (e.g. logs/rsl_rl/2026-03-18_14-46-04/model_3000.pt)
        if ckpt_arg and (os.path.sep in ckpt_arg or ckpt_arg.startswith("/")):
            path_abs = os.path.abspath(ckpt_arg)
            if os.path.isfile(path_abs):
                resume_path = path_abs
                log_dir = os.path.dirname(resume_path)
                print(f"[INFO] Resuming from given path: {resume_path}, log_dir: {log_dir}")
        if resume_path is None:
            load_run = args_cli.load_run if args_cli.load_run is not None else ".*"
            load_ckpt = ckpt_arg if ckpt_arg and not (os.path.sep in ckpt_arg or ckpt_arg.startswith("/")) else "model_.*.pt"
            try:
                resume_path = get_checkpoint_path(log_root_path, load_run, load_ckpt)
            except ValueError:
                # e.g. runs under logs/rsl_rl/ instead of logs/rsl_rl/<experiment_name>/
                log_parent = os.path.dirname(log_root_path)
                resume_path = get_checkpoint_path(log_parent, load_run, load_ckpt)
            log_dir = os.path.dirname(resume_path)
            print(f"[INFO] Resuming: loading {resume_path}, logging to existing run: {log_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, timestamp)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] Logging to: {log_dir}")

    train_cfg = {
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

    runner = DwbcOnPolicyRunner(
        env=env_wrapped,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=runner_cfg.device,
    )

    if resume_path is not None:
        print(f"[INFO] Loading checkpoint (model + optimizer): {resume_path}")
        runner.load(resume_path, load_optimizer=True)

    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

