#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Debug tool for Husky cabinet end-effector/handle orientation matching.

Usage:
    ./isaaclab.sh -p scripts/tools/debug_husky_cabinet_orientation.py --task Isaac-FBC-Husky-WorldIK-Cabinet-Play-v0
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Debug gripper vs handle orientation in Husky cabinet task.")
parser.add_argument("--task", type=str, default="Isaac-FBC-Husky-WorldIK-Cabinet-Play-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--max_steps", type=int, default=1500, help="Maximum simulation steps.")
parser.add_argument("--print_every", type=int, default=30, help="Print metrics every N simulation steps.")
parser.add_argument(
    "--freeze_after_reset",
    action="store_true",
    default=False,
    help="Print orientation snapshot right after reset and exit (no stepping).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--output_path",
    type=str,
    default="",
    help="Optional path to save suggested quaternion and debug scalars.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _fmt(v: torch.Tensor) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v.detach().cpu().tolist()) + "]"


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    if "ee_frame" not in unwrapped.scene.keys() or "cabinet_frame" not in unwrapped.scene.keys():
        raise RuntimeError("Task must provide both `ee_frame` and `cabinet_frame` frame transformers.")

    def print_snapshot(step: int):
        ee_quat = unwrapped.scene["ee_frame"].data.target_quat_w[:, 0, :]
        handle_quat = unwrapped.scene["cabinet_frame"].data.target_quat_w[:, 0, :]

        ee_rot = math_utils.matrix_from_quat(ee_quat)
        handle_rot = math_utils.matrix_from_quat(handle_quat)

        # Axis-angle error (world): how much ee should rotate to match handle.
        delta_axis_angle = math_utils.quat_box_minus(handle_quat, ee_quat)
        delta_deg = torch.norm(delta_axis_angle, dim=1) * (180.0 / torch.pi)

        # Recommended ee_target_quat_in_handle to copy into command cfg:
        # goal_ee_quat = handle_quat * ee_target_quat_in_handle
        # => ee_target_quat_in_handle = inv(handle_quat) * goal_ee_quat
        # If we want goal_ee_quat == current ee_quat at debug moment:
        suggested = math_utils.quat_mul(math_utils.quat_inv(handle_quat), ee_quat)
        suggested = math_utils.quat_unique(suggested)

        env_id = 0
        ee_x = ee_rot[env_id, :, 0]
        ee_y = ee_rot[env_id, :, 1]
        ee_z = ee_rot[env_id, :, 2]
        h_x = handle_rot[env_id, :, 0]
        h_y = handle_rot[env_id, :, 1]
        h_z = handle_rot[env_id, :, 2]

        print(f"\n[step {step}] env={env_id}")
        print(f"  ee_quat_w            = {_fmt(ee_quat[env_id])}")
        print(f"  handle_quat_w        = {_fmt(handle_quat[env_id])}")
        print(f"  ee axis x/y/z        = {_fmt(ee_x)}  {_fmt(ee_y)}  {_fmt(ee_z)}")
        print(f"  handle axis x/y/z    = {_fmt(h_x)}  {_fmt(h_y)}  {_fmt(h_z)}")
        print(f"  delta_angle_deg      = {delta_deg[env_id].item():.3f}")
        print(f"  suggested ee_in_handle quat (wxyz) = {_fmt(suggested[env_id])}")

        if "log" in unwrapped.extras:
            log = unwrapped.extras["log"]
            for key in [
                "metric/align_ee",
                "metric/align_grasp",
                "metric/ee_to_handle",
                "metric/quat_err_deg",
                "metric/init/align_ee",
                "metric/init/quat_err_deg",
            ]:
                if key in log:
                    print(f"  {key:24s}= {log[key]:.6f}")

        if args_cli.output_path:
            out_dir = os.path.dirname(args_cli.output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args_cli.output_path, "w", encoding="utf-8") as f:
                f.write(
                    " ".join(
                        [
                            f"suggested_w={suggested[env_id, 0].item():.9f}",
                            f"suggested_x={suggested[env_id, 1].item():.9f}",
                            f"suggested_y={suggested[env_id, 2].item():.9f}",
                            f"suggested_z={suggested[env_id, 3].item():.9f}",
                            f"delta_angle_deg={delta_deg[env_id].item():.6f}",
                        ]
                    )
                )

        return suggested[env_id]

    if args_cli.freeze_after_reset:
        # One sensor update tick so frame-transformer data is populated after reset.
        env.step(torch.zeros(env.action_space.shape, device=unwrapped.device))
        print_snapshot(step=0)
        env.close()
        return

    step = 0
    while simulation_app.is_running() and step < args_cli.max_steps:
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            env.step(actions)
            step += 1

            if step % max(args_cli.print_every, 1) != 0:
                continue
            print_snapshot(step=step)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

