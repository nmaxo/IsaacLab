"""Evaluate World-IK policy: command segments per env, log + plots.

For each trajectory type: ``num_envs`` parallel envs, each collects ``segments_per_env``
sequential segments (a segment ends when ``goal_pos_w`` is resampled, or env episode, or cap).

Default: 50 envs × 5 segments = 250 rollouts per trajectory type.

Usage
-----
python scripts/reinforcement_learning/rsl_rl/eval_world_ik.py \
    --task Isaac-FBC-Husky-WorldIK-Play-v0 \
    --num_envs 50 \
    --segments_per_env 5
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate World-IK trajectory tracking.")
parser.add_argument(
    "--num_envs", type=int, default=50, help="Parallel environments."
)
parser.add_argument(
    "--segments_per_env",
    type=int,
    default=5,
    help="Number of command segments to record per env (per trajectory type).",
)
parser.add_argument("--task", type=str, default="Isaac-FBC-Husky-WorldIK-Play-v0", help="Task id.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point."
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--max_segment_steps", type=int, default=5000, help="Safety cap steps per segment (per env)."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric."
)
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use pre-trained checkpoint."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.num_envs:
    args_cli.num_envs = 50

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Heavy imports after AppLauncher
# ---------------------------------------------------------------------------
import gymnasium as gym
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ---------------------------------------------------------------------------
# Trajectory type definitions
# ---------------------------------------------------------------------------
TRAJECTORY_TYPES: dict[str, dict] = {
    "straight_flat": {
        "curve_xy_amplitude_range": (0.0, 0.0),
        "curve_z_amplitude_range": (0.0, 0.0),
        "curve_harmonic_range": (1, 1),
        "goal_z_range": (0.5, 0.5),
        "label": "Прямая (фикс. Z)",
    },
    "straight_z": {
        "curve_xy_amplitude_range": (0.0, 0.0),
        "curve_z_amplitude_range": (0.0, 0.0),
        "curve_harmonic_range": (1, 1),
        "goal_z_range": (0.3, 0.8),
        "label": "Прямая (перем. Z)",
    },
    "sinusoidal_xy": {
        "curve_xy_amplitude_range": (0.15, 0.30),
        "curve_z_amplitude_range": (0.0, 0.0),
        "curve_harmonic_range": (1, 1),
        "goal_z_range": (0.5, 0.5),
        "label": "Синусоида XY",
    },
    "sinusoidal_full": {
        "curve_xy_amplitude_range": (0.15, 0.30),
        "curve_z_amplitude_range": (0.08, 0.20),
        "curve_harmonic_range": (1, 1),
        "goal_z_range": (0.3, 0.8),
        "label": "Синусоида XY+Z",
    },
    "high_harmonic": {
        "curve_xy_amplitude_range": (0.10, 0.25),
        "curve_z_amplitude_range": (0.05, 0.15),
        "curve_harmonic_range": (2, 3),
        "goal_z_range": (0.3, 0.8),
        "label": "Высокая гармоника",
    },
}


def apply_trajectory_type(env, traj_cfg: dict):
    """Override command term sampling ranges for a specific trajectory type."""
    ee_term = env.unwrapped.command_manager.get_term("ee_goal")
    ee_term._curve_xy_range = traj_cfg["curve_xy_amplitude_range"]
    ee_term._curve_z_range = traj_cfg["curve_z_amplitude_range"]
    ee_term._curve_harmonic_range = traj_cfg["curve_harmonic_range"]
    z_lo, z_hi = traj_cfg["goal_z_range"]
    ee_term._goal_z_min = z_lo
    ee_term._goal_z_max = z_hi
    ee_term._z_hard = traj_cfg["goal_z_range"]
    ee_term._z_easy = traj_cfg["goal_z_range"]


def _append_step_to_buffers(
    robot,
    ee_term,
    ee_body_id: int,
    rewards: torch.Tensor,
    env_ids: list[int],
    buffers: list[dict[str, list]],
    sim_time_s: list[float],
    step_dt: float,
):
    body_pos = robot.data.body_pos_w[env_ids, ee_body_id].detach().cpu().numpy()
    desired = ee_term.curr_goal_pos_w[env_ids].detach().cpu().numpy()
    rew = rewards[env_ids].detach().cpu().numpy()
    for k, i in enumerate(env_ids):
        gp = body_pos[k].copy()
        d = desired[k].copy()
        buffers[i]["gripper"].append(gp)
        buffers[i]["desired"].append(d)
        buffers[i]["error"].append(float(np.linalg.norm(d - gp)))
        buffers[i]["reward"].append(float(rew[k]))
        sim_time_s[i] += step_dt


def _append_t0_single_env(robot, ee_term, ee_body_id: int, i: int, buffers, sim_time_s: list[float]):
    """Post-reset / post-segment-start snapshot for env i (reward=0, no sim time advance)."""
    gp = robot.data.body_pos_w[i, ee_body_id].detach().cpu().numpy().copy()
    d = ee_term.curr_goal_pos_w[i].detach().cpu().numpy().copy()
    buffers[i]["gripper"].append(gp)
    buffers[i]["desired"].append(d)
    buffers[i]["error"].append(float(np.linalg.norm(d - gp)))
    buffers[i]["reward"].append(0.0)


def run_segments_per_env(
    env,
    policy,
    ee_term,
    ee_body_id: int,
    *,
    segments_per_env: int,
    step_dt: float,
    goal_match_atol: float = 1.0e-4,
    max_segment_steps: int = 5000,
) -> tuple[list[dict[str, np.ndarray]], list[dict]]:
    """Collect ``segments_per_env`` segments per environment (parallel rollout).

    Segment boundary: same as before (``goal_pos_w`` change, or ``env_episode``, or cap).
    """
    device = env.unwrapped.device
    n = env.num_envs
    robot = env.unwrapped.scene["robot"]

    env.reset()
    obs = env.get_observations()

    seg_goal_ref = ee_term.goal_pos_w.detach().clone()
    collecting = torch.ones(n, device=device, dtype=torch.bool)
    seg_done_count = torch.zeros(n, device=device, dtype=torch.long)

    buffers: list[dict[str, list]] = [
        {"gripper": [], "desired": [], "error": [], "reward": []} for _ in range(n)
    ]
    sim_time_s = [0.0 for _ in range(n)]
    manifest_rows: list[dict] = []
    completed_segments: list[dict[str, np.ndarray | dict]] = []

    for i in range(n):
        _append_t0_single_env(robot, ee_term, ee_body_id, i, buffers, sim_time_s)

    step_in_segment = torch.zeros(n, device=device, dtype=torch.long)

    def _finalize_segment(i_: int, how: str):
        sidx = int(seg_done_count[i_].item())
        err_list = buffers[i_]["error"]
        meta = {
            "env_id": int(i_),
            "segment_index": sidx,
            "num_steps": len(err_list),
            "duration_sim_s": float(sim_time_s[i_]),
            "mean_error_m": float(np.mean(err_list)) if err_list else 0.0,
            "ended": how,
        }
        manifest_rows.append(meta)
        completed_segments.append(
            {
                "gripper": np.asarray(buffers[i_]["gripper"], dtype=np.float64),
                "desired": np.asarray(buffers[i_]["desired"], dtype=np.float64),
                "error": np.asarray(err_list, dtype=np.float64),
                "reward": np.asarray(buffers[i_]["reward"], dtype=np.float64),
                "meta": dict(meta),
            }
        )
        seg_done_count[i_] += 1
        buffers[i_] = {"gripper": [], "desired": [], "error": [], "reward": []}
        sim_time_s[i_] = 0.0
        step_in_segment[i_] = 0
        collecting[i_] = False

        if int(seg_done_count[i_].item()) < segments_per_env:
            seg_goal_ref[i_] = ee_term.goal_pos_w[i_].detach().clone()
            collecting[i_] = True
            _append_t0_single_env(robot, ee_term, ee_body_id, i_, buffers, sim_time_s)

    while simulation_app.is_running():
        if bool(torch.all(seg_done_count >= segments_per_env)):
            break

        cap_ids = [
            i
            for i in range(n)
            if collecting[i]
            and int(seg_done_count[i].item()) < segments_per_env
            and int(step_in_segment[i].item()) >= max_segment_steps
        ]
        if cap_ids:
            print(f"[WARN] max_segment_steps={max_segment_steps} reached for envs {cap_ids}.")
            for i in cap_ids:
                _finalize_segment(i, "max_steps_cap")

        if bool(torch.all(seg_done_count >= segments_per_env)):
            break

        active_mask = collecting & (seg_done_count < segments_per_env)
        if not bool(active_mask.any()):
            break

        with torch.no_grad():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)

        resampled = torch.max(torch.abs(ee_term.goal_pos_w - seg_goal_ref), dim=-1)[0] > goal_match_atol
        done_b = dones.bool() if dones.dtype != torch.bool else dones

        ids_list = active_mask.nonzero(as_tuple=False).flatten().tolist()
        for i in ids_list:
            step_in_segment[i] += 1

        record_ids: list[int] = []
        finished: list[tuple[int, str]] = []
        for i in ids_list:
            if resampled[i]:
                finished.append((i, "goal_resampled"))
            elif done_b[i]:
                record_ids.append(i)
                finished.append((i, "env_episode"))
            else:
                record_ids.append(i)

        if record_ids:
            _append_step_to_buffers(
                robot, ee_term, ee_body_id, rewards, record_ids, buffers, sim_time_s, step_dt
            )

        for i, how in finished:
            _finalize_segment(i, how)

    completed_segments.sort(key=lambda s: (s["meta"]["env_id"], s["meta"]["segment_index"]))

    total_expected = n * segments_per_env
    n_rs = sum(1 for r in manifest_rows if r["ended"] == "goal_resampled")
    print(
        f"  Segments recorded: {len(completed_segments)}/{total_expected} "
        f"(goal_resampled={n_rs}, segments_per_env={segments_per_env}, num_envs={n})"
    )

    return completed_segments, manifest_rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_3d_trajectory(episodes, traj_name, label, save_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    n_ep = len(episodes)
    for i, ep in enumerate(episodes):
        a = 0.9 if n_ep <= 5 else max(0.08, min(0.5, 6.0 / n_ep))
        ax.plot(ep["gripper"][:, 0], ep["gripper"][:, 1], ep["gripper"][:, 2],
                alpha=a, color="C0", label="Факт" if i == 0 else None)
        ax.plot(ep["desired"][:, 0], ep["desired"][:, 1], ep["desired"][:, 2],
                "--", alpha=a * 0.7, color="C1", label="Цель" if i == 0 else None)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Траектория: {label}")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{traj_name}_3d.png"), dpi=150)
    plt.close(fig)


def plot_xy_projection(episodes, traj_name, label, save_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    n_ep = len(episodes)
    for i, ep in enumerate(episodes):
        a = 0.9 if n_ep <= 5 else max(0.08, min(0.5, 6.0 / n_ep))
        ax.plot(ep["gripper"][:, 0], ep["gripper"][:, 1],
                alpha=a, color="C0", label="Факт" if i == 0 else None)
        ax.plot(ep["desired"][:, 0], ep["desired"][:, 1],
                "--", alpha=a * 0.7, color="C1", label="Цель" if i == 0 else None)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"XY проекция: {label}")
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{traj_name}_xy.png"), dpi=150)
    plt.close(fig)


def plot_error_over_time(episodes, traj_name, label, save_dir, dt: float):
    fig, ax = plt.subplots(figsize=(12, 4))
    n_ep = len(episodes)
    for i, ep in enumerate(episodes):
        t = np.arange(len(ep["error"])) * dt
        a = 0.85 if n_ep <= 8 else max(0.06, min(0.45, 8.0 / n_ep))
        ax.plot(t, ep["error"], alpha=a, color="C0", label="сегменты" if i == 0 else None)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("||ошибка|| (м)")
    ax.set_title(f"Ошибка трекинга: {label}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{traj_name}_error_time.png"), dpi=150)
    plt.close(fig)


def plot_error_components(episodes, traj_name, label, save_dir, dt: float):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    comp_names = ["X", "Y", "Z"]
    n_ep = len(episodes)
    for i, ep in enumerate(episodes):
        err_vec = ep["desired"] - ep["gripper"]
        t = np.arange(len(err_vec)) * dt
        a = 0.75 if n_ep <= 6 else max(0.06, min(0.4, 8.0 / n_ep))
        for c in range(3):
            axes[c].plot(t, err_vec[:, c], alpha=a, label="сегменты" if i == 0 else None)
    for c in range(3):
        axes[c].set_ylabel(f"Ошибка {comp_names[c]} (м)")
        axes[c].grid(True, alpha=0.3)
    h0, l0 = axes[0].get_legend_handles_labels()
    if h0:
        axes[0].legend(h0, l0, fontsize=7, loc="upper right")
    axes[2].set_xlabel("Время (с)")
    axes[0].set_title(f"Компоненты ошибки: {label}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{traj_name}_error_components.png"), dpi=150)
    plt.close(fig)


def plot_summary_bar(summary: dict[str, dict], save_dir: str):
    names = list(summary.keys())
    labels = [summary[n]["label"] for n in names]
    means = [summary[n]["mean_error"] for n in names]
    medians = [summary[n]["median_error"] for n in names]
    maxes = [summary[n]["max_error"] for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, means, w, label="Средняя", color="#4C72B0")
    ax.bar(x, medians, w, label="Медианная", color="#55A868")
    ax.bar(x + w, maxes, w, label="Макс", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Ошибка (м)")
    ax.set_title("Сводка ошибок по типам траекторий")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "summary_bar.png"), dpi=150)
    plt.close(fig)


def plot_all_types_error(all_data: dict, save_dir: str, dt: float):
    """Single plot with mean error curve per trajectory type."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for traj_name, episodes in all_data.items():
        max_len = max(len(ep["error"]) for ep in episodes)
        padded = np.full((len(episodes), max_len), np.nan)
        for i, ep in enumerate(episodes):
            padded[i, :len(ep["error"])] = ep["error"]
        mean_curve = np.nanmean(padded, axis=0)
        t = np.arange(max_len) * dt
        label = TRAJECTORY_TYPES[traj_name]["label"]
        ax.plot(t, mean_curve, label=label, linewidth=1.5)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Средняя ||ошибка|| (м)")
    ax.set_title("Средняя ошибка трекинга по типам")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "all_types_error.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- checkpoint ----
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # ---- output directory ----
    eval_dir = os.path.abspath(os.path.join(
        "logs", "eval", agent_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    ))
    os.makedirs(eval_dir, exist_ok=True)
    print(f"[INFO] Eval results will be saved to: {eval_dir}")

    # ---- create env ----
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---- load policy ----
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- resolve EE body id ----
    ee_term = env.unwrapped.command_manager.get_term("ee_goal")
    ee_body_id = ee_term._ee_body_id
    dt = env.unwrapped.step_dt

    # ---- run evaluations ----
    all_data: dict[str, list] = {}
    summary: dict[str, dict] = {}
    full_manifest: dict = {
        "step_dt": float(dt),
        "num_envs": int(env.num_envs),
        "segments_per_env": int(args_cli.segments_per_env),
        "checkpoint": str(resume_path),
        "task": args_cli.task,
        "segment_end": "goal_pos_w resampled (new EE waypoint) or env episode / cap",
        "trajectory_types": {},
    }

    for traj_name, traj_cfg in TRAJECTORY_TYPES.items():
        print(f"\n{'='*60}")
        print(f"Trajectory type: {traj_name} — {traj_cfg['label']}")
        print(f"{'='*60}")

        apply_trajectory_type(env, traj_cfg)

        segments, manifest_rows = run_segments_per_env(
            env,
            policy,
            ee_term,
            ee_body_id,
            segments_per_env=int(args_cli.segments_per_env),
            step_dt=float(dt),
            max_segment_steps=int(args_cli.max_segment_steps),
        )
        episodes = [
            {
                "gripper": s["gripper"],
                "desired": s["desired"],
                "error": s["error"],
                "reward": s["reward"],
                "meta": s["meta"],
            }
            for s in segments
        ]
        all_data[traj_name] = episodes

        rows_out = []
        for row in manifest_rows:
            r = dict(row)
            r["trajectory_type"] = traj_name
            r["trajectory_label"] = traj_cfg["label"]
            rows_out.append(r)
        full_manifest["trajectory_types"][traj_name] = rows_out

        # Пулы по всем шагам всех сегментов
        all_errors = np.concatenate([ep["error"] for ep in episodes])
        mean_err = float(np.mean(all_errors))
        median_err = float(np.median(all_errors))
        max_err = float(np.max(all_errors))
        std_err = float(np.std(all_errors))
        pct_5cm = float(np.mean(all_errors < 0.05)) * 100
        pct_10cm = float(np.mean(all_errors < 0.10)) * 100
        seg_means = [float(np.mean(ep["error"])) for ep in episodes]
        mean_of_segment_means = float(np.mean(seg_means)) if seg_means else 0.0
        n_rs = sum(1 for r in manifest_rows if r["ended"] == "goal_resampled")
        n_ep = sum(1 for r in manifest_rows if r["ended"] == "env_episode")
        n_cap = sum(1 for r in manifest_rows if r["ended"] == "max_steps_cap")

        summary[traj_name] = {
            "label": traj_cfg["label"],
            "mean_error": mean_err,
            "median_error": median_err,
            "max_error": max_err,
            "std_error": std_err,
            "success_5cm_pct": pct_5cm,
            "success_10cm_pct": pct_10cm,
            "num_steps": len(all_errors),
            "mean_of_segment_means": mean_of_segment_means,
            "num_segments_recorded": len(episodes),
            "segments_goal_resampled": n_rs,
            "segments_env_episode": n_ep,
            "segments_max_cap": n_cap,
        }

        plot_3d_trajectory(episodes, traj_name, traj_cfg["label"], eval_dir)
        plot_xy_projection(episodes, traj_name, traj_cfg["label"], eval_dir)
        plot_error_over_time(episodes, traj_name, traj_cfg["label"], eval_dir, dt)
        plot_error_components(episodes, traj_name, traj_cfg["label"], eval_dir, dt)

    manifest_path = os.path.join(eval_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, ensure_ascii=False, indent=2)

    # ---- cross-type plots ----
    plot_summary_bar(summary, eval_dir)
    plot_all_types_error(all_data, eval_dir, dt)

    # ---- save raw data ----
    npz_data = {}
    for traj_name, episodes in all_data.items():
        for ep in episodes:
            env_id = int(ep["meta"]["env_id"])
            seg_ix = int(ep["meta"]["segment_index"])
            suf = f"{traj_name}_env{env_id:02d}_seg{seg_ix:02d}"
            npz_data[f"{suf}_gripper"] = ep["gripper"]
            npz_data[f"{suf}_desired"] = ep["desired"]
            npz_data[f"{suf}_error"] = ep["error"]
            npz_data[f"{suf}_reward"] = ep["reward"]
    np.savez_compressed(os.path.join(eval_dir, "rollout_data.npz"), **npz_data)

    # ---- print summary table ----
    print(f"\n{'='*100}")
    print(
        f"{'Тип траектории':<20} {'Ср(шаг)':<8} {'Ср(сегм)':<8} {'Мед.':<8} {'Макс':<8} {'СКО':<8} "
        f"{'<5см':<7} {'<10см':<7} {'шагов':<8} {'Nсег':<6} {'↦цель':<7} {'↦эпиз':<7} {'cap':<5}"
    )
    print(f"{'-'*100}")
    for traj_name, s in summary.items():
        print(
            f"{s['label']:<20} {s['mean_error']:<8.4f} {s['mean_of_segment_means']:<8.4f} "
            f"{s['median_error']:<8.4f} {s['max_error']:<8.4f} {s['std_error']:<8.4f} "
            f"{s['success_5cm_pct']:<7.1f} {s['success_10cm_pct']:<7.1f} {s['num_steps']:<8} "
            f"{s['num_segments_recorded']:<6} {s['segments_goal_resampled']:<7} "
            f"{s['segments_env_episode']:<7} {s['segments_max_cap']:<5}"
        )
    print(f"{'='*100}")
    print(f"\n[INFO] All results saved to: {eval_dir}")
    print(f"[INFO] Segment metadata: {manifest_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
