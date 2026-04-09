"""MDP utilities for Go2-ARX LeggedGym-style environment."""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.envs.mdp.observations import height_scan as height_scan_mdp
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# EE goal command (trajectory in spherical coords, collision check)
# -----------------------------------------------------------------------------


def _cart2sphere(cart: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sphere = torch.zeros_like(cart)
    r = torch.sqrt(torch.sum(cart * cart, dim=-1) + eps)
    sphere[:, 0] = r
    sphere[:, 1] = torch.atan2(cart[:, 2], cart[:, 0])
    sphere[:, 2] = torch.asin(torch.clamp(cart[:, 1] / r, -1.0 + 1e-6, 1.0 - 1e-6))
    return sphere


def _sphere2cart(sphere: torch.Tensor) -> torch.Tensor:
    """Convert (r, azimuth, elevation) to cartesian.

    Matches legged_gym:
      sphere[...,1] = atan2(cart_z, cart_x)   (azimuth)
      sphere[...,2] = asin(cart_y / r)        (elevation)

    Therefore:
      x = r * cos(elev) * cos(azim)
      y = r * sin(elev)
      z = r * cos(elev) * sin(azim)
    """

    cart = torch.zeros_like(sphere)
    r = sphere[:, 0]
    azim = sphere[:, 1]
    elev = sphere[:, 2]
    cart[:, 0] = r * torch.cos(elev) * torch.cos(azim)
    cart[:, 1] = r * torch.sin(elev)
    cart[:, 2] = r * torch.cos(elev) * torch.sin(azim)
    return cart


class Go2ArxEeGoalCommand(CommandTerm):
    """Generates a moving EE goal (curr_ee_goal_cart) in the base-yaw frame.

    Exposes buffers used by observations and rewards:
    - local_axis_z_offset (scalar)
    - local_gripper_pos (N,3) computed outside (observation term)
    - curr_ee_goal_cart (N,3)
    - goal_world_pos (N,3) for debug/metrics (optional)
    """

    cfg: "Go2ArxEeGoalCommandCfg"

    def __init__(self, cfg: "Go2ArxEeGoalCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]

        body_ids, _ = self._robot.find_bodies(cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one EE body match for '{cfg.ee_body_name}', got: {body_ids}")
        self._ee_body_id = body_ids[0]

        # buffers
        self.local_axis_z_offset = float(cfg.local_axis_z_offset)
        self.base_align_z_axis = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = torch.ones(self.num_envs, device=self.device)
        self.traj_total_timesteps = torch.ones(self.num_envs, device=self.device)

        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)

        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_world_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.collision_upper_limits = torch.tensor(cfg.collision_upper_limits, device=self.device, dtype=torch.float)
        self.collision_lower_limits = torch.tensor(cfg.collision_lower_limits, device=self.device, dtype=torch.float)
        self.underground_limit = float(cfg.underground_limit)

        self.num_collision_check_samples = int(cfg.num_collision_check_samples)
        self.collision_check_t = torch.linspace(0.0, 1.0, self.num_collision_check_samples, device=self.device)[
            None, None, :
        ]

        # pre-cache ranges
        self.goal_ee_l_ranges = torch.tensor(cfg.sphere_ranges.pos_l, device=self.device)
        self.goal_ee_p_ranges = torch.tensor(cfg.sphere_ranges.pos_p, device=self.device)
        self.goal_ee_y_ranges = torch.tensor(cfg.sphere_ranges.pos_y, device=self.device)

        self.traj_time_range_s = cfg.traj_time_range_s
        self.hold_time_range_s = cfg.hold_time_range_s

        # optional debug visualization (points along the track)
        self._debug_vis = bool(cfg.debug_vis)
        # -1 means: visualize for all envs [0..num_envs-1]
        self._vis_env_index = int(cfg.visualize_env_index)
        self._track_points = int(cfg.track_points)
        self._track_t = torch.linspace(0.0, 1.0, self._track_points, device=self.device)

        self._ee_track_markers: VisualizationMarkers | None = None
        if self._debug_vis:
            # 2 prototypes: cyan track points + red current point
            markers_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/Go2ArxEETrack",
                markers={
                    "track": sim_utils.SphereCfg(
                        radius=float(cfg.track_marker_radius),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.track_marker_color),
                    ),
                    "current": sim_utils.SphereCfg(
                        radius=float(cfg.current_marker_radius),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.current_marker_color),
                    ),
                },
            )
            self._ee_track_markers = VisualizationMarkers(markers_cfg)
            self._debug_update_every = max(int(cfg.debug_vis_update_every_steps), 1)
            self._debug_update_counter = 0

        # initialize immediately
        self._resample_command(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        # command is the current goal in base-yaw frame (cart)
        return self.curr_ee_goal_cart

    def _resample_ee_goal_sphere_once(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        self.ee_goal_sphere[env_ids, 0] = torch.rand(n, device=self.device) * (
            self.goal_ee_l_ranges[1] - self.goal_ee_l_ranges[0]
        ) + self.goal_ee_l_ranges[0]
        self.ee_goal_sphere[env_ids, 1] = torch.rand(n, device=self.device) * (
            self.goal_ee_p_ranges[1] - self.goal_ee_p_ranges[0]
        ) + self.goal_ee_p_ranges[0]
        self.ee_goal_sphere[env_ids, 2] = torch.rand(n, device=self.device) * (
            self.goal_ee_y_ranges[1] - self.goal_ee_y_ranges[0]
        ) + self.goal_ee_y_ranges[0]

    def _collision_check(self, env_ids: torch.Tensor) -> torch.Tensor:
        # interpolate in sphere space, convert to cart, check bounds and underground
        ee_target_all_sphere = torch.lerp(
            self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ..., None], self.collision_check_t
        ).squeeze(-1)  # (n,3,S)
        ee_target_all_sphere = torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)
        ee_target_cart = _sphere2cart(ee_target_all_sphere).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(
            torch.logical_and(
                torch.all(ee_target_cart < self.collision_upper_limits, dim=-1),
                torch.all(ee_target_cart > self.collision_lower_limits, dim=-1),
            ),
            dim=0,
        )
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def _resample_ee_goal(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        init_env_ids = env_ids.clone()
        self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
        for _ in range(10):
            self._resample_ee_goal_sphere_once(env_ids)
            bad = self._collision_check(env_ids)
            env_ids = env_ids[bad]
            if env_ids.numel() == 0:
                break
        self.goal_timer[init_env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        # initialize goals
        self._resample_ee_goal_sphere_once(env_ids)
        self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
        # sample trajectory timing in steps
        dt = self._env.step_dt
        traj_s = torch.rand(env_ids.numel(), device=self.device) * (
            self.traj_time_range_s[1] - self.traj_time_range_s[0]
        ) + self.traj_time_range_s[0]
        hold_s = torch.rand(env_ids.numel(), device=self.device) * (
            self.hold_time_range_s[1] - self.hold_time_range_s[0]
        ) + self.hold_time_range_s[0]
        self.traj_timesteps[env_ids] = traj_s / dt
        self.traj_total_timesteps[env_ids] = (traj_s + hold_s) / dt
        self.goal_timer[env_ids] = 0.0

    def _update_command(self):
        # update base align z axis
        self.base_align_z_axis[:, :2] = self._robot.data.root_pos_w[:, :2]
        self.base_align_z_axis[:, 2] = self.local_axis_z_offset

        t = torch.clamp(self.goal_timer / torch.clamp(self.traj_timesteps, min=1.0), 0.0, 1.0)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        self.curr_ee_goal_cart[:] = _sphere2cart(self.curr_ee_goal_sphere)
        self.goal_timer += 1.0
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        self._resample_ee_goal(resample_id)

        # compute world position of goal for metrics/debug: base_align + yaw_rot(goal_cart)
        yaw_quat = math_utils.yaw_quat(self._robot.data.root_quat_w)
        self.goal_world_pos[:] = self.base_align_z_axis + math_utils.quat_apply(yaw_quat, self.curr_ee_goal_cart)

        if self._ee_track_markers is not None:
            self._update_track_visuals()

    def _update_metrics(self):
        # Metrics are logged by Isaac Lab as `Metrics/<command_name>/<key>`.
        # We compute EE-to-goal distance in the base-yaw frame (same frame as commands/obs).
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id]
        root_pos_w = self._robot.data.root_pos_w
        yaw_quat = math_utils.yaw_quat(self._robot.data.root_quat_w)

        base_align = torch.zeros_like(root_pos_w)
        base_align[:, :2] = root_pos_w[:, :2]
        base_align[:, 2] = self.local_axis_z_offset

        local_gripper_pos = math_utils.quat_apply_inverse(yaw_quat, ee_pos_w - base_align)
        err = self.curr_ee_goal_cart - local_gripper_pos
        dist_l2 = torch.sum(err * err, dim=1)
        dist = torch.sqrt(dist_l2 + 1e-8)

        # Common metrics: mean distance and success-rate under thresholds.
        self.metrics["dist"] = dist
        self.metrics["dist_l2"] = dist_l2
        self.metrics["success_5cm"] = (dist < 0.05).to(dtype=torch.float)
        self.metrics["success_10cm"] = (dist < 0.10).to(dtype=torch.float)

    def _update_track_visuals(self):
        """Visualize EE trajectory track (points) and current position."""
        if self._ee_track_markers is None:
            return

        self._debug_update_counter += 1
        if self._debug_update_counter % self._debug_update_every != 0:
            return

        if self._vis_env_index == -1:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            if self._vis_env_index < 0 or self._vis_env_index >= self.num_envs:
                return
            env_ids = torch.tensor([self._vis_env_index], device=self.device, dtype=torch.long)

        num_e = env_ids.numel()
        T = self._track_points

        # shape: (E, T, 3)
        start = self.ee_start_sphere[env_ids]  # (E,3)
        goal = self.ee_goal_sphere[env_ids]  # (E,3)
        t = self._track_t.view(1, T, 1)  # (1,T,1)
        sphere_points = start.unsqueeze(1) + (goal - start).unsqueeze(1) * t
        cart_points = _sphere2cart(sphere_points.reshape(-1, 3)).reshape(num_e, T, 3)

        yaw_quat_env = math_utils.yaw_quat(self._robot.data.root_quat_w[env_ids])  # (E,4)
        yaw_quat_rep = yaw_quat_env.unsqueeze(1).repeat(1, T, 1).reshape(-1, 4)
        cart_world = math_utils.quat_apply(yaw_quat_rep, cart_points.reshape(-1, 3)).reshape(num_e, T, 3)

        # For visualization, place points relative to the current robot base frame origin
        # (root position in world). This ensures markers are visible consistently
        # across terrains/heightfields.
        root_pos_w = self._robot.data.root_pos_w[env_ids]  # (E,3)
        world_points = (cart_world + root_pos_w.unsqueeze(1).repeat(1, T, 1)).reshape(-1, 3)  # (E*T,3)

        # current position on the track (red dot for each env)
        traj_t = torch.clamp(
            self.goal_timer[env_ids] / torch.clamp(self.traj_timesteps[env_ids], min=1.0),
            0.0,
            1.0,
        )  # (E,)
        current_idx = torch.round(traj_t * (T - 1)).to(dtype=torch.long)  # (E,)
        flat_current_idx = env_ids.view(-1, 1).repeat(1, T).reshape(-1)  # not used
        flat_red = (torch.arange(num_e, device=self.device, dtype=torch.long) * T) + current_idx  # (E,)

        marker_indices = torch.zeros(num_e * T, device=self.device, dtype=torch.long)  # prototype 0 = track
        marker_indices[flat_red] = 1  # prototype 1 = current

        scales = torch.ones(num_e * T, 3, device=self.device)
        scales[flat_red, :] = 1.8

        self._ee_track_markers.visualize(
            translations=world_points,
            marker_indices=marker_indices,
            scales=scales,
        )


@configclass
class Go2ArxEeGoalCommandCfg(CommandTermCfg):
    class_type: type = Go2ArxEeGoalCommand

    asset_name: str = MISSING
    ee_body_name: str = MISSING

    local_axis_z_offset: float = 0.3

    traj_time_range_s: tuple[float, float] = (0.6, 1.2)
    hold_time_range_s: tuple[float, float] = (0.2, 0.4)

    @configclass
    class SphereRanges:
        pos_l: tuple[float, float] = MISSING
        pos_p: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING

    sphere_ranges: SphereRanges = MISSING

    collision_upper_limits: tuple[float, float, float] = (0.3, 0.15, -0.115)
    collision_lower_limits: tuple[float, float, float] = (-0.2, -0.15, -0.515)
    underground_limit: float = -0.57
    num_collision_check_samples: int = 10

    # debug visualization
    debug_vis: bool = False
    # -1 means all envs
    visualize_env_index: int = 0
    track_points: int = 10
    # Make markers large enough to be visible across many envs.
    track_marker_radius: float = 0.03
    current_marker_radius: float = 0.045
    track_marker_color: tuple[float, float, float] = (0.0, 1.0, 1.0)
    current_marker_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    debug_vis_update_every_steps: int = 1


def reset_ee_goal(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, command_name: str = "ee_goal"):
    """Reset helper for EE goal term."""
    term: Go2ArxEeGoalCommand = env.command_manager.get_term(command_name)
    term._resample_command(env_ids)  # noqa: SLF001 (internal call is fine for env reset)


# -----------------------------------------------------------------------------
# Arm action: delta joint positions + DLS IK correction
# -----------------------------------------------------------------------------


class Go2ArxArmIkDeltaJointPositionAction(ActionTerm):
    cfg: "Go2ArxArmIkDeltaJointPositionActionCfg"

    def __init__(self, cfg: "Go2ArxArmIkDeltaJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]
        self._command_name = cfg.command_name

        # resolve arm joints
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.arm_joint_names)
        if len(self._joint_ids) == 0:
            raise ValueError(f"No joints matched: {cfg.arm_joint_names}")

        # resolve ee body
        body_ids, body_names = self._asset.find_bodies(cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body match for {cfg.ee_body_name}, got {body_names}")
        self._body_idx = body_ids[0]

        # jacobian indexing (same logic as DifferentialInverseKinematicsAction)
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._action_scale = float(cfg.action_scale)
        self._lambda_val = float(cfg.dls_lambda)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._action_scale

    def apply_actions(self):
        # current joint positions
        q = self._asset.data.joint_pos[:, self._joint_ids]

        # --- compute local_gripper_pos in yaw-only frame (matches legged_gym) ---
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w
        yaw_quat = math_utils.yaw_quat(self._asset.data.root_quat_w)

        ee_term: Go2ArxEeGoalCommand = self._env.command_manager.get_term(self._command_name)
        base_align = torch.zeros_like(root_pos_w)
        base_align[:, :2] = root_pos_w[:, :2]
        base_align[:, 2] = ee_term.local_axis_z_offset

        local_gripper_pos = math_utils.quat_apply_inverse(yaw_quat, ee_pos_w - base_align)
        goal_local = self._env.command_manager.get_command(self._command_name)
        pos_err = goal_local[:, :3] - local_gripper_pos

        # --- Jacobian in world frame (matches legged_gym) ---
        # legged_gym uses whole_body_jacobian directly (world frame) with pos_err in yaw frame.
        # This is a frame mismatch in the original code, but we replicate it exactly.
        jac = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

        # --- DLS solve (position only, orientation error = 0) ---
        dpose = torch.cat([pos_err, torch.zeros(self.num_envs, 3, device=self.device)], dim=1).unsqueeze(-1)
        jT = jac.transpose(1, 2)
        lam = (self._lambda_val**2) * torch.eye(6, device=self.device)
        u = (jT @ torch.inverse(jac @ jT + lam) @ dpose).squeeze(-1)  # (N, num_joints)

        q_des = q + self._processed_actions + u
        self._asset.set_joint_position_target(q_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class Go2ArxArmIkDeltaJointPositionActionCfg(ActionTermCfg):
    class_type: type = Go2ArxArmIkDeltaJointPositionAction

    asset_name: str = MISSING
    ee_body_name: str = MISSING
    arm_joint_names: list[str] = MISSING
    command_name: str = "ee_goal"

    action_scale: float = 1.0
    dls_lambda: float = 0.05


# -----------------------------------------------------------------------------
# Observations helpers (match legged_gym ordering)
# -----------------------------------------------------------------------------


def dof_err_rel_default(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def scaled_velocity_commands(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """Return velocity command scaled as in legged_gym: [vx*2, vy*2, wz*0.25]."""
    cmd = env.command_manager.get_command(command_name)  # (N, 3) — lin_x, lin_y, ang_z
    scale = torch.tensor([2.0, 2.0, 0.25], device=env.device)
    return cmd[:, :3] * scale


def local_gripper_pos(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ee_term: Go2ArxEeGoalCommand = env.command_manager.get_term("ee_goal")
    ee_pos_w = asset.data.body_pos_w[:, asset.find_bodies("arm_link6")[0][0]]
    root_pos_w = asset.data.root_pos_w
    yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    base_align = torch.zeros_like(root_pos_w)
    base_align[:, :2] = root_pos_w[:, :2]
    base_align[:, 2] = ee_term.local_axis_z_offset
    return math_utils.quat_apply_inverse(yaw_quat, ee_pos_w - base_align)


def curr_ee_goal_cart(env: "ManagerBasedRLEnv", command_name: str = "ee_goal") -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, :3]


def gripper_to_goal(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return local_gripper_pos(env) - curr_ee_goal_cart(env)


def policy_obs_passthrough(env: "ManagerBasedRLEnv") -> torch.Tensor:
    raise RuntimeError(
        "policy_obs_passthrough is not supported during ObservationManager initialization. "
        "Repeat policy terms explicitly in the privileged observation group instead."
    )


# -----------------------------------------------------------------------------
# Terminations helpers
# -----------------------------------------------------------------------------


def illegal_contact_hip(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return torch.any(forces > threshold, dim=1)


# -----------------------------------------------------------------------------
# Rewards (match legged_gym math; weights scaled by step_dt in env cfg)
# -----------------------------------------------------------------------------


def termination_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # penalize terminal (not timeout) -> 1 for terminated else 0
    return env.termination_manager.terminated.to(dtype=torch.float)


def tracking_lin_vel_exp(env: "ManagerBasedRLEnv", command_name: str, sigma: float) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    asset: Articulation = env.scene["robot"]
    err = torch.sum(torch.square(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-err / max(sigma, 1e-6))


def tracking_ang_vel_exp(env: "ManagerBasedRLEnv", command_name: str, sigma: float) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    asset: Articulation = env.scene["robot"]
    err = torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-err / max(sigma, 1e-6))


def ang_vel_xy_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def orientation_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_torques_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def joint_acc_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel
    qd_prev = env.action_manager.prev_action.new_zeros(qd.shape)  # fallback
    if hasattr(env, "_prev_joint_vel"):
        qd_prev = env._prev_joint_vel
    acc = (qd_prev - qd) / max(env.step_dt, 1e-6)
    env._prev_joint_vel = qd.clone()
    return torch.sum(torch.square(acc), dim=1)


def base_height_l2(env: "ManagerBasedRLEnv", target_height: float = 0.25) -> torch.Tensor:
    """Base height penalty matching legged_gym definition.

    legged_gym: base_height = mean(root_z - measured_heights)
    where measured_heights = terrain hit z values (actual ground height under each point).
    """
    from isaaclab.sensors import RayCaster

    sensor: RayCaster = env.scene.sensors["height_scanner"]
    sensor_z = sensor.data.pos_w[:, 2]  # (N,) — sensor origin z (= root_z + 20)
    hit_z = sensor.data.ray_hits_w[..., 2]  # (N, 187) — z of ray hit points

    # measured_heights in legged_gym = actual terrain z under each point
    # hit_z IS the terrain z directly from ray caster
    root_z = env.scene["robot"].data.root_pos_w[:, 2]  # (N,)
    base_height = torch.mean(root_z.unsqueeze(1) - hit_z, dim=1)  # (N,)
    # clamp for protection against ray misses (hit_z = very large negative)
    base_height = torch.clamp(base_height, -2.0, 5.0)

    # debug (first 500 calls)
    if not hasattr(env, "_bh_debug_cnt"):
        env._bh_debug_cnt = 0
    env._bh_debug_cnt += 1
    if env._bh_debug_cnt <= 5:
        i = 0
        print(
            f"[BaseHeightDebug] sensor_z={sensor_z[i]:.3f} root_z={root_z[i]:.3f} "
            f"hit_z_mean={hit_z[i].mean():.3f} hit_z_min={hit_z[i].min():.3f} hit_z_max={hit_z[i].max():.3f} "
            f"base_height={base_height[i]:.4f} target={target_height} penalty={(base_height[i]-target_height)**2:.6f}"
        )

    return torch.square(base_height - target_height)


def feet_air_time(env: "ManagerBasedRLEnv", threshold: float = 0.5) -> torch.Tensor:
    # reuse mdp contact sensor air-time through ContactSensor buffers
    sensor: ContactSensor = env.scene.sensors["contact_forces"]
    # feet names match ".*_foot"
    asset: Articulation = env.scene["robot"]
    foot_ids, _ = asset.find_bodies(".*_foot")
    first_contact = sensor.compute_first_contact(env.step_dt)[:, foot_ids]
    last_air_time = sensor.data.last_air_time[:, foot_ids]
    rew = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    cmd = env.command_manager.get_command("base_velocity")
    rew *= torch.norm(cmd[:, :2], dim=1) > 0.1
    return rew


def penalized_contacts(env: "ManagerBasedRLEnv") -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors["contact_forces"]
    asset: Articulation = env.scene["robot"]
    ids, _ = asset.find_bodies([".*_thigh", ".*_calf"])
    forces = sensor.data.net_forces_w_history[:, :, ids, :].norm(dim=-1).max(dim=1)[0]
    return torch.sum((forces > 0.1).float(), dim=1)


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    a = env.action_manager.action
    prev = env.action_manager.prev_action
    return torch.sum(torch.square(prev - a), dim=1)


def joint_pos_limits(env: "ManagerBasedRLEnv", margin: float = 0.0) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    q = asset.data.joint_pos
    lo = asset.data.soft_joint_pos_limits[..., 0] + margin
    hi = asset.data.soft_joint_pos_limits[..., 1] - margin
    out = -(q - lo).clamp(max=0.0) + (q - hi).clamp(min=0.0)
    return torch.sum(out, dim=1)


def object_distance_exp(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # exp(-||local_gripper - goal||^2 / 0.1)
    d = torch.sum(torch.square(gripper_to_goal(env)), dim=1)
    return torch.exp(-d / 0.1)


def object_distance_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.sum(torch.square(gripper_to_goal(env)), dim=1)


def leggedgym_total_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Total reward with legged_gym semantics (only-positive + termination-after-clip).

    Implements Go2ArxRoughCfg.rewards.scales exactly, but *without* the final dt scaling.
    The env config multiplies this term by step_dt.
    """

    # 1) sum non-termination terms
    r_tracking_lin = 2.0 * tracking_lin_vel_exp(env, command_name="base_velocity", sigma=0.25)
    r_tracking_ang = 0.5 * tracking_ang_vel_exp(env, command_name="base_velocity", sigma=0.25)
    r_ang_vel_xy = -0.1 * ang_vel_xy_l2(env)
    r_orientation = -0.5 * orientation_l2(env)
    r_torques = -0.0001 * joint_torques_l2(env)
    r_joint_acc = -2.5e-9 * joint_acc_l2(env)
    r_base_height = -0.2 * base_height_l2(env, target_height=0.25)
    r_feet_air = 1.0 * feet_air_time(env, threshold=0.5)
    r_contacts = -1.0 * penalized_contacts(env)
    r_action_rate = -0.001 * action_rate_l2(env)
    r_joint_lim = -10.0 * joint_pos_limits(env, margin=0.0)
    r_obj_dist_exp = 2.0 * object_distance_exp(env)
    r_obj_dist_l2 = -1.0 * object_distance_l2(env)

    r = (
        r_tracking_lin + r_tracking_ang + r_ang_vel_xy + r_orientation
        + r_torques + r_joint_acc + r_base_height + r_feet_air
        + r_contacts + r_action_rate + r_joint_lim
        + r_obj_dist_exp + r_obj_dist_l2
    )

    # periodic debug logging (every 100 steps, env 0 only)
    if not hasattr(env, "_reward_debug_counter"):
        env._reward_debug_counter = 0
    env._reward_debug_counter += 1
    if env._reward_debug_counter % 100 == 0:
        i = 0
        print(
            f"[RewardDebug step={env._reward_debug_counter}] "
            f"track_lin={r_tracking_lin[i]:.4f} track_ang={r_tracking_ang[i]:.4f} "
            f"ang_xy={r_ang_vel_xy[i]:.4f} orient={r_orientation[i]:.4f} "
            f"torq={r_torques[i]:.4f} acc={r_joint_acc[i]:.4f} "
            f"base_h={r_base_height[i]:.4f} feet_air={r_feet_air[i]:.4f} "
            f"contacts={r_contacts[i]:.4f} act_rate={r_action_rate[i]:.4f} "
            f"j_lim={r_joint_lim[i]:.4f} obj_exp={r_obj_dist_exp[i]:.4f} "
            f"obj_l2={r_obj_dist_l2[i]:.4f} sum_raw={r[i]:.4f}"
        )

    # 2) only_positive_rewards
    r = torch.clamp(r, min=0.0)

    # 3) termination reward after clipping (no terminal reward for time-outs)
    # legged_gym: reset_buf * ~time_out_buf, scaled by termination=-1
    terminated = env.termination_manager.terminated
    time_out = env.termination_manager.time_outs
    r += -1.0 * (terminated & ~time_out).to(dtype=r.dtype)

    return r

