"""MDP utilities for Husky UR5 IK-based EE trajectory tracking.

Adapted from Go2-ARX legged_gym-style pipeline:
- Spherical EE goal command with trajectory interpolation and collision checking
- Arm action: delta joint positions + DLS IK correction toward current EE goal
- Observation helpers (local gripper pos, goal, error)
- Reward terms for EE tracking + base regularization
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Spherical <-> Cartesian conversions
# ---------------------------------------------------------------------------

def _cart2sphere(cart: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sphere = torch.zeros_like(cart)
    r = torch.sqrt(torch.sum(cart * cart, dim=-1) + eps)
    sphere[:, 0] = r
    sphere[:, 1] = torch.atan2(cart[:, 2], cart[:, 0])
    sphere[:, 2] = torch.asin(torch.clamp(cart[:, 1] / r, -1.0 + 1e-6, 1.0 - 1e-6))
    return sphere


def _sphere2cart(sphere: torch.Tensor) -> torch.Tensor:
    """Convert (r, azimuth, elevation) to cartesian.

    Convention (same as legged_gym):
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


# ---------------------------------------------------------------------------
# EE Goal Command (spherical trajectory with collision check)
# ---------------------------------------------------------------------------


class HuskyEeGoalCommand(CommandTerm):
    """Generates a moving EE goal (curr_ee_goal_cart) in the base-yaw frame.

    Buffers used by observations and rewards:
    - local_axis_z_offset (scalar): vertical offset for the yaw-frame origin
    - curr_ee_goal_cart (N,3): current interpolated goal in base-yaw frame
    - goal_world_pos (N,3): goal in world frame (for debug/metrics)
    """

    cfg: "HuskyEeGoalCommandCfg"

    def __init__(self, cfg: "HuskyEeGoalCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]

        body_ids, _ = self._robot.find_bodies(cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one EE body match for '{cfg.ee_body_name}', got: {body_ids}")
        self._ee_body_id = body_ids[0]

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

        self.goal_ee_l_ranges = torch.tensor(cfg.sphere_ranges.pos_l, device=self.device)
        self.goal_ee_p_ranges = torch.tensor(cfg.sphere_ranges.pos_p, device=self.device)
        self.goal_ee_y_ranges = torch.tensor(cfg.sphere_ranges.pos_y, device=self.device)

        self.traj_time_range_s = cfg.traj_time_range_s
        self.hold_time_range_s = cfg.hold_time_range_s

        # debug visualization
        self._debug_vis = bool(cfg.debug_vis)
        self._vis_env_index = int(cfg.visualize_env_index)
        self._track_points = int(cfg.track_points)
        self._track_t = torch.linspace(0.0, 1.0, self._track_points, device=self.device)

        self._ee_track_markers: VisualizationMarkers | None = None
        if self._debug_vis:
            markers_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/HuskyEETrack",
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

        self._resample_command(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        return self.curr_ee_goal_cart

    def _resample_ee_goal_sphere_once(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        self.ee_goal_sphere[env_ids, 0] = (
            torch.rand(n, device=self.device) * (self.goal_ee_l_ranges[1] - self.goal_ee_l_ranges[0])
            + self.goal_ee_l_ranges[0]
        )
        self.ee_goal_sphere[env_ids, 1] = (
            torch.rand(n, device=self.device) * (self.goal_ee_p_ranges[1] - self.goal_ee_p_ranges[0])
            + self.goal_ee_p_ranges[0]
        )
        self.ee_goal_sphere[env_ids, 2] = (
            torch.rand(n, device=self.device) * (self.goal_ee_y_ranges[1] - self.goal_ee_y_ranges[0])
            + self.goal_ee_y_ranges[0]
        )

    def _collision_check(self, env_ids: torch.Tensor) -> torch.Tensor:
        ee_target_all_sphere = torch.lerp(
            self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ..., None], self.collision_check_t
        ).squeeze(-1)
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
        self._resample_ee_goal_sphere_once(env_ids)
        self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
        dt = self._env.step_dt
        traj_s = (
            torch.rand(env_ids.numel(), device=self.device) * (self.traj_time_range_s[1] - self.traj_time_range_s[0])
            + self.traj_time_range_s[0]
        )
        hold_s = (
            torch.rand(env_ids.numel(), device=self.device) * (self.hold_time_range_s[1] - self.hold_time_range_s[0])
            + self.hold_time_range_s[0]
        )
        self.traj_timesteps[env_ids] = traj_s / dt
        self.traj_total_timesteps[env_ids] = (traj_s + hold_s) / dt
        self.goal_timer[env_ids] = 0.0

    def _update_command(self):
        self.base_align_z_axis[:, :2] = self._robot.data.root_pos_w[:, :2]
        self.base_align_z_axis[:, 2] = self.local_axis_z_offset

        t = torch.clamp(self.goal_timer / torch.clamp(self.traj_timesteps, min=1.0), 0.0, 1.0)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        self.curr_ee_goal_cart[:] = _sphere2cart(self.curr_ee_goal_sphere)
        self.goal_timer += 1.0
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        self._resample_ee_goal(resample_id)

        yaw_quat = math_utils.yaw_quat(self._robot.data.root_quat_w)
        self.goal_world_pos[:] = self.base_align_z_axis + math_utils.quat_apply(yaw_quat, self.curr_ee_goal_cart)

        if self._ee_track_markers is not None:
            self._update_track_visuals()

    def _update_metrics(self):
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

        self.metrics["dist"] = dist
        self.metrics["dist_l2"] = dist_l2
        self.metrics["success_5cm"] = (dist < 0.05).to(dtype=torch.float)
        self.metrics["success_10cm"] = (dist < 0.10).to(dtype=torch.float)

    def _update_track_visuals(self):
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

        start = self.ee_start_sphere[env_ids]
        goal = self.ee_goal_sphere[env_ids]
        t = self._track_t.view(1, T, 1)
        sphere_points = start.unsqueeze(1) + (goal - start).unsqueeze(1) * t
        cart_points = _sphere2cart(sphere_points.reshape(-1, 3)).reshape(num_e, T, 3)

        yaw_quat_env = math_utils.yaw_quat(self._robot.data.root_quat_w[env_ids])
        yaw_quat_rep = yaw_quat_env.unsqueeze(1).repeat(1, T, 1).reshape(-1, 4)
        cart_world = math_utils.quat_apply(yaw_quat_rep, cart_points.reshape(-1, 3)).reshape(num_e, T, 3)

        root_pos_w = self._robot.data.root_pos_w[env_ids]
        world_points = (cart_world + root_pos_w.unsqueeze(1).repeat(1, T, 1)).reshape(-1, 3)

        traj_t = torch.clamp(
            self.goal_timer[env_ids] / torch.clamp(self.traj_timesteps[env_ids], min=1.0), 0.0, 1.0
        )
        current_idx = torch.round(traj_t * (T - 1)).to(dtype=torch.long)
        flat_red = (torch.arange(num_e, device=self.device, dtype=torch.long) * T) + current_idx

        marker_indices = torch.zeros(num_e * T, device=self.device, dtype=torch.long)
        marker_indices[flat_red] = 1

        scales = torch.ones(num_e * T, 3, device=self.device)
        scales[flat_red, :] = 1.8

        self._ee_track_markers.visualize(
            translations=world_points,
            marker_indices=marker_indices,
            scales=scales,
        )


@configclass
class HuskyEeGoalCommandCfg(CommandTermCfg):
    class_type: type = HuskyEeGoalCommand

    asset_name: str = MISSING
    ee_body_name: str = MISSING

    local_axis_z_offset: float = 0.3

    traj_time_range_s: tuple[float, float] = (0.8, 1.5)
    hold_time_range_s: tuple[float, float] = (0.3, 0.6)

    @configclass
    class SphereRanges:
        pos_l: tuple[float, float] = MISSING
        pos_p: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING

    sphere_ranges: SphereRanges = MISSING

    collision_upper_limits: tuple[float, float, float] = (0.4, 0.3, 0.1)
    collision_lower_limits: tuple[float, float, float] = (-0.3, -0.3, -0.5)
    underground_limit: float = -0.1
    num_collision_check_samples: int = 10

    debug_vis: bool = False
    visualize_env_index: int = -1
    track_points: int = 10
    track_marker_radius: float = 0.03
    current_marker_radius: float = 0.045
    track_marker_color: tuple[float, float, float] = (0.0, 1.0, 1.0)
    current_marker_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    debug_vis_update_every_steps: int = 1


def reset_ee_goal(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, command_name: str = "ee_goal"):
    cmd_term = env.command_manager.get_term(command_name)
    cmd_term._resample_command(env_ids)


# ---------------------------------------------------------------------------
# World-frame EE Goal Command (requires base locomotion for reaching)
# ---------------------------------------------------------------------------


class HuskyWorldEeGoalCommand(CommandTerm):
    """Generates EE goals in WORLD coordinates, forcing base-arm cooperation.

    The goal is sampled at a configurable distance from the robot's current position,
    at a height reachable by the UR5 arm. The robot must drive its base to bring the
    arm within reach, then use the arm for precise positioning.

    Optional curriculum: smoothly interpolates (smoothstep) sampling ranges for XY distance,
    height span, and motion segment duration from "easy" to "hard" using
    ``common_step_counter / curriculum_ramp_steps``. If
    ``curriculum_schedule_step_interval > 0``, the counter is snapped to multiples of that
    interval so the schedule only changes every N simulator steps (piecewise-constant ramps).

    command property returns the goal as a 6D vector in base frame:
      [goal_xy_in_base(2), goal_z_world(1), gripper_to_goal_base(3)]
    But for IK and observations we expose individual helpers.
    """

    cfg: "HuskyWorldEeGoalCommandCfg"

    def __init__(self, cfg: "HuskyWorldEeGoalCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]

        body_ids, _ = self._robot.find_bodies(cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one EE body for '{cfg.ee_body_name}', got: {body_ids}")
        self._ee_body_id = body_ids[0]

        self.local_axis_z_offset = float(cfg.local_axis_z_offset)
        self.arm_reach = float(cfg.arm_reach)

        # Goal in world frame (the "true" target)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Waypoint sequence: start → goal, interpolated over time
        self.start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = torch.ones(self.num_envs, device=self.device)
        self.traj_total_timesteps = torch.ones(self.num_envs, device=self.device)
        # Curved trajectory shaping buffers (all zero => straight line).
        self.curve_amp_xy = torch.zeros(self.num_envs, device=self.device)
        self.curve_amp_z = torch.zeros(self.num_envs, device=self.device)
        self.curve_harmonic = torch.ones(self.num_envs, device=self.device)

        # Curriculum: easy → hard ranges (hard = goal_distance_range / goal_z / traj / hold in cfg)
        self._curriculum_enabled = bool(cfg.enable_curriculum)
        self._ramp_steps = max(1, int(cfg.curriculum_ramp_steps))
        self._curriculum_step_snap = max(0, int(cfg.curriculum_schedule_step_interval))
        self._dist_easy = cfg.easy_goal_distance_range
        self._dist_hard = cfg.goal_distance_range
        self._z_easy = cfg.easy_goal_z_range
        self._z_hard = cfg.goal_z_range
        self._traj_easy = cfg.easy_traj_time_range_s
        self._traj_hard = cfg.traj_time_range_s
        self._hold_easy = cfg.easy_hold_time_range_s
        self._hold_hard = cfg.hold_time_range_s
        self._curve_xy_range = cfg.curve_xy_amplitude_range
        self._curve_z_range = cfg.curve_z_amplitude_range
        self._curve_harmonic_range = cfg.curve_harmonic_range
        self._curriculum_p = 1.0
        self._goal_dist_min = float(self._dist_hard[0])
        self._goal_dist_max = float(self._dist_hard[1])
        self._goal_z_min = float(self._z_hard[0])
        self._goal_z_max = float(self._z_hard[1])
        self._traj_t_min = float(self._traj_hard[0])
        self._traj_t_max = float(self._traj_hard[1])
        self._hold_t_min = float(self._hold_hard[0])
        self._hold_t_max = float(self._hold_hard[1])

        # Debug visualization
        self._debug_vis = bool(cfg.debug_vis)
        self._vis_env_index = int(cfg.visualize_env_index)

        self._ee_markers: VisualizationMarkers | None = None
        if self._debug_vis:
            markers_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/HuskyWorldEE",
                markers={
                    "goal": sim_utils.SphereCfg(
                        radius=0.06,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "current": sim_utils.SphereCfg(
                        radius=0.04,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self._ee_markers = VisualizationMarkers(markers_cfg)

        self._refresh_curriculum_ranges()
        self._resample_command(torch.arange(self.num_envs, device=self.device))

    def _smoothstep01(self, t: float) -> float:
        t = min(1.0, max(0.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _refresh_curriculum_ranges(self):
        """Update scalar sampling bounds from curriculum progress (every call)."""
        if not self._curriculum_enabled:
            self._curriculum_p = 1.0
            self._goal_dist_min, self._goal_dist_max = float(self._dist_hard[0]), float(self._dist_hard[1])
            self._goal_z_min, self._goal_z_max = float(self._z_hard[0]), float(self._z_hard[1])
            self._traj_t_min, self._traj_t_max = float(self._traj_hard[0]), float(self._traj_hard[1])
            self._hold_t_min, self._hold_t_max = float(self._hold_hard[0]), float(self._hold_hard[1])
            return

        counter = int(self._env.common_step_counter)
        if self._curriculum_step_snap > 0:
            counter = (counter // self._curriculum_step_snap) * self._curriculum_step_snap
        prog = float(counter) / float(self._ramp_steps)
        p = self._smoothstep01(prog)
        self._curriculum_p = p

        self._goal_dist_min = self._dist_easy[0] + p * (self._dist_hard[0] - self._dist_easy[0])
        self._goal_dist_max = self._dist_easy[1] + p * (self._dist_hard[1] - self._dist_easy[1])
        self._goal_z_min = self._z_easy[0] + p * (self._z_hard[0] - self._z_easy[0])
        self._goal_z_max = self._z_easy[1] + p * (self._z_hard[1] - self._z_easy[1])
        self._traj_t_min = self._traj_easy[0] + p * (self._traj_hard[0] - self._traj_easy[0])
        self._traj_t_max = self._traj_easy[1] + p * (self._traj_hard[1] - self._traj_easy[1])
        self._hold_t_min = self._hold_easy[0] + p * (self._hold_hard[0] - self._hold_easy[0])
        self._hold_t_max = self._hold_easy[1] + p * (self._hold_hard[1] - self._hold_easy[1])

    @property
    def command(self) -> torch.Tensor:
        """Returns curr_goal in base-yaw frame (3D) for IK and obs."""
        root_pos_w = self._robot.data.root_pos_w
        yaw_quat = math_utils.yaw_quat(self._robot.data.root_quat_w)
        base_align = torch.zeros_like(root_pos_w)
        base_align[:, :2] = root_pos_w[:, :2]
        base_align[:, 2] = self.local_axis_z_offset
        goal_local = math_utils.quat_apply_inverse(yaw_quat, self.curr_goal_pos_w - base_align)
        return goal_local

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        root_pos = self._robot.data.root_pos_w[env_ids]

        dmin, dmax = self._goal_dist_min, self._goal_dist_max
        zmin, zmax = self._goal_z_min, self._goal_z_max
        tmin, tmax = self._traj_t_min, self._traj_t_max
        hmin, hmax = self._hold_t_min, self._hold_t_max

        # Random distance and angle from current robot position
        dist = torch.rand(n, device=self.device) * (dmax - dmin) + dmin
        angle = torch.rand(n, device=self.device) * 2.0 * math.pi - math.pi

        self.goal_pos_w[env_ids, 0] = root_pos[:, 0] + dist * torch.cos(angle)
        self.goal_pos_w[env_ids, 1] = root_pos[:, 1] + dist * torch.sin(angle)
        self.goal_pos_w[env_ids, 2] = torch.rand(n, device=self.device) * (zmax - zmin) + zmin

        # Start = current EE world position (or current goal for chained waypoints)
        ee_pos_w = self._robot.data.body_pos_w[env_ids, self._ee_body_id]
        self.start_pos_w[env_ids] = ee_pos_w.clone()

        dt = self._env.step_dt
        traj_s = torch.rand(n, device=self.device) * (tmax - tmin) + tmin
        hold_s = torch.rand(n, device=self.device) * (hmax - hmin) + hmin
        self.traj_timesteps[env_ids] = traj_s / dt
        self.traj_total_timesteps[env_ids] = (traj_s + hold_s) / dt
        self.goal_timer[env_ids] = 0.0
        self.curve_amp_xy[env_ids] = (
            torch.rand(n, device=self.device) * (self._curve_xy_range[1] - self._curve_xy_range[0])
            + self._curve_xy_range[0]
        )
        self.curve_amp_z[env_ids] = (
            torch.rand(n, device=self.device) * (self._curve_z_range[1] - self._curve_z_range[0])
            + self._curve_z_range[0]
        )
        harmonic_lo = int(self._curve_harmonic_range[0])
        harmonic_hi = int(self._curve_harmonic_range[1])
        self.curve_harmonic[env_ids] = torch.randint(
            low=max(1, harmonic_lo),
            high=max(2, harmonic_hi + 1),
            size=(n,),
            device=self.device,
        ).to(dtype=torch.float)

    def _update_command(self):
        self._refresh_curriculum_ranges()
        t = torch.clamp(self.goal_timer / torch.clamp(self.traj_timesteps, min=1.0), 0.0, 1.0)
        # Base line: start -> goal. Then add a bounded transverse shape so trajectories are not
        # always straight segments in world frame.
        line = torch.lerp(self.start_pos_w, self.goal_pos_w, t[:, None])
        delta = self.goal_pos_w - self.start_pos_w
        delta_xy = delta[:, :2]
        delta_xy_norm = torch.norm(delta_xy, dim=1, keepdim=True).clamp(min=1e-6)
        tangential_xy = delta_xy / delta_xy_norm
        normal_xy = torch.stack([-tangential_xy[:, 1], tangential_xy[:, 0]], dim=1)

        shape = torch.sin(math.pi * t)
        harmonic = torch.sin(self.curve_harmonic * math.pi * t) * shape
        curve_xy = normal_xy * (self.curve_amp_xy * harmonic)[:, None]
        curve_z = (self.curve_amp_z * harmonic)

        self.curr_goal_pos_w[:, :2] = line[:, :2] + curve_xy
        self.curr_goal_pos_w[:, 2] = line[:, 2] + curve_z
        self.goal_timer += 1.0
        resample_ids = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        if resample_ids.numel() > 0:
            self._resample_command(resample_ids)

        if self._ee_markers is not None:
            self._update_visuals()

    def _update_metrics(self):
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id]
        err = self.curr_goal_pos_w - ee_pos_w
        dist = torch.norm(err, dim=1)
        self.metrics["dist"] = dist
        self.metrics["dist_l2"] = torch.sum(err * err, dim=1)
        self.metrics["success_5cm"] = (dist < 0.05).to(dtype=torch.float)
        self.metrics["success_10cm"] = (dist < 0.10).to(dtype=torch.float)

        base_to_goal_xy = torch.norm(
            self.curr_goal_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=1
        )
        self.metrics["base_to_goal_xy"] = base_to_goal_xy
        self.metrics["need_drive"] = (base_to_goal_xy > self.arm_reach * 0.8).to(dtype=torch.float)
        p = float(self._curriculum_p)
        self.metrics["curriculum_p"] = torch.full((self.num_envs,), p, device=self.device)
        self.metrics["goal_z_span"] = torch.full(
            (self.num_envs,), float(self._goal_z_max - self._goal_z_min), device=self.device
        )

    def _update_visuals(self):
        if self._ee_markers is None:
            return
        if self._vis_env_index == -1:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.tensor([self._vis_env_index], device=self.device, dtype=torch.long)

        n = env_ids.numel()
        goal_pts = self.goal_pos_w[env_ids]
        curr_pts = self.curr_goal_pos_w[env_ids]
        all_pts = torch.cat([goal_pts, curr_pts], dim=0)
        indices = torch.cat([
            torch.zeros(n, device=self.device, dtype=torch.long),
            torch.ones(n, device=self.device, dtype=torch.long),
        ])
        self._ee_markers.visualize(translations=all_pts, marker_indices=indices)


@configclass
class HuskyWorldEeGoalCommandCfg(CommandTermCfg):
    class_type: type = HuskyWorldEeGoalCommand

    asset_name: str = MISSING
    ee_body_name: str = MISSING

    local_axis_z_offset: float = 0.3
    arm_reach: float = 0.85

    goal_distance_range: tuple[float, float] = (1.0, 4.0)
    goal_z_range: tuple[float, float] = (0.3, 0.9)

    traj_time_range_s: tuple[float, float] = (3.0, 8.0)
    hold_time_range_s: tuple[float, float] = (1.0, 3.0)

    enable_curriculum: bool = False
    curriculum_ramp_steps: int = 2_000_000
    curriculum_schedule_step_interval: int = 0
    easy_goal_distance_range: tuple[float, float] = (0.55, 1.05)
    easy_goal_z_range: tuple[float, float] = (0.42, 0.58)
    easy_traj_time_range_s: tuple[float, float] = (3.5, 6.5)
    easy_hold_time_range_s: tuple[float, float] = (1.4, 2.4)
    # Curvature/noise in trajectory shape between start and goal in world frame.
    # Zero ranges disable shaping and recover straight interpolation.
    curve_xy_amplitude_range: tuple[float, float] = (0.05, 0.30)
    curve_z_amplitude_range: tuple[float, float] = (0.00, 0.18)
    curve_harmonic_range: tuple[int, int] = (1, 3)

    debug_vis: bool = True
    visualize_env_index: int = -1


def reset_world_ee_goal(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, command_name: str = "ee_goal"):
    cmd_term: HuskyWorldEeGoalCommand = env.command_manager.get_term(command_name)
    cmd_term._resample_command(env_ids)


# ---------------------------------------------------------------------------
# Arm action: delta joint positions + DLS IK correction
# ---------------------------------------------------------------------------


class HuskyArmIkDeltaJointPositionAction(ActionTerm):
    """UR5 arm action: policy outputs delta joint positions, IK corrects toward EE goal.

    q_des = q_current + Δq_RL * action_scale + u_IK
    where u_IK comes from DLS solve on pose error between gripper and goal.
    """

    cfg: "HuskyArmIkDeltaJointPositionActionCfg"

    def __init__(self, cfg: "HuskyArmIkDeltaJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]
        self._command_name = cfg.command_name

        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.arm_joint_names)
        if len(self._joint_ids) == 0:
            raise ValueError(f"No joints matched: {cfg.arm_joint_names}")

        body_ids, body_names = self._asset.find_bodies(cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body match for {cfg.ee_body_name}, got {body_names}")
        self._body_idx = body_ids[0]

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
        self._ik_pos_weight = float(cfg.ik_pos_weight)
        self._ik_rot_weight = float(cfg.ik_rot_weight)

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
        q = self._asset.data.joint_pos[:, self._joint_ids]

        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w
        yaw_quat = math_utils.yaw_quat(self._asset.data.root_quat_w)

        ee_term: HuskyEeGoalCommand = self._env.command_manager.get_term(self._command_name)
        base_align = torch.zeros_like(root_pos_w)
        base_align[:, :2] = root_pos_w[:, :2]
        base_align[:, 2] = ee_term.local_axis_z_offset

        local_gripper_pos = math_utils.quat_apply_inverse(yaw_quat, ee_pos_w - base_align)
        goal_local = self._env.command_manager.get_command(self._command_name)
        pos_err = goal_local[:, :3] - local_gripper_pos
        pos_err = self._ik_pos_weight * pos_err

        rot_err = torch.zeros(self.num_envs, 3, device=self.device)
        if hasattr(ee_term, "curr_goal_quat_w"):
            goal_quat_w = ee_term.curr_goal_quat_w
            rot_err = math_utils.quat_box_minus(goal_quat_w, ee_quat_w)
            rot_err = self._ik_rot_weight * rot_err

        # Jacobian in world frame (replicating legged_gym frame mismatch)
        jac = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

        dpose = torch.cat([pos_err, rot_err], dim=1).unsqueeze(-1)
        jT = jac.transpose(1, 2)
        lam = (self._lambda_val ** 2) * torch.eye(6, device=self.device)
        u = (jT @ torch.inverse(jac @ jT + lam) @ dpose).squeeze(-1)

        u = torch.clamp(u, -0.5, 0.5)
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

        q_des = q + self._processed_actions + u
        self._asset.set_joint_position_target(q_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class HuskyArmIkDeltaJointPositionActionCfg(ActionTermCfg):
    class_type: type = HuskyArmIkDeltaJointPositionAction

    asset_name: str = MISSING
    ee_body_name: str = MISSING
    arm_joint_names: list[str] = MISSING
    command_name: str = "ee_goal"

    action_scale: float = 0.4
    dls_lambda: float = 0.05
    ik_pos_weight: float = 1.0
    ik_rot_weight: float = 0.35


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def local_gripper_pos(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_name: str = "gripper_link",
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """Gripper position in base-yaw frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_term: HuskyEeGoalCommand = env.command_manager.get_term(command_name)
    body_id = asset.find_bodies(ee_body_name)[0][0]
    ee_pos_w = asset.data.body_pos_w[:, body_id]
    root_pos_w = asset.data.root_pos_w
    yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    base_align = torch.zeros_like(root_pos_w)
    base_align[:, :2] = root_pos_w[:, :2]
    base_align[:, 2] = ee_term.local_axis_z_offset
    result = math_utils.quat_apply_inverse(yaw_quat, ee_pos_w - base_align)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def curr_ee_goal_cart(env: "ManagerBasedRLEnv", command_name: str = "ee_goal") -> torch.Tensor:
    """Current EE goal in base-yaw frame."""
    return env.command_manager.get_command(command_name)[:, :3]


def gripper_to_goal(
    env: "ManagerBasedRLEnv",
    ee_body_name: str = "gripper_link",
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """Vector from gripper to goal in base-yaw frame."""
    return local_gripper_pos(env, ee_body_name=ee_body_name, command_name=command_name) - curr_ee_goal_cart(
        env, command_name=command_name
    )


# --- World-frame observation helpers ---


def goal_in_base_frame(env: "ManagerBasedRLEnv", command_name: str = "ee_goal") -> torch.Tensor:
    """World goal position expressed in base body frame (3D).

    This tells the base where it needs to drive: x=forward, y=left, z=up.
    """
    cmd_term = env.command_manager.get_term(command_name)
    if not hasattr(cmd_term, "curr_goal_pos_w"):
        return curr_ee_goal_cart(env, command_name=command_name)
    goal_w = cmd_term.curr_goal_pos_w
    root_pos_w = env.scene["robot"].data.root_pos_w
    yaw_quat = math_utils.yaw_quat(env.scene["robot"].data.root_quat_w)
    diff = goal_w - root_pos_w
    return torch.nan_to_num(math_utils.quat_apply_inverse(yaw_quat, diff), nan=0.0)


def goal_distance_xy(env: "ManagerBasedRLEnv", command_name: str = "ee_goal") -> torch.Tensor:
    """Horizontal distance from base to goal (scalar per env, returned as (N,1))."""
    cmd_term = env.command_manager.get_term(command_name)
    if not hasattr(cmd_term, "curr_goal_pos_w"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    goal_w = cmd_term.curr_goal_pos_w
    root_pos_w = env.scene["robot"].data.root_pos_w
    d = torch.norm(goal_w[:, :2] - root_pos_w[:, :2], dim=1, keepdim=True)
    return d


def gripper_to_world_goal(
    env: "ManagerBasedRLEnv",
    ee_body_name: str = "gripper_link",
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """Vector from gripper to world goal in base-yaw frame (3D)."""
    cmd_term = env.command_manager.get_term(command_name)
    if not hasattr(cmd_term, "curr_goal_pos_w"):
        return gripper_to_goal(env, ee_body_name=ee_body_name, command_name=command_name)
    asset: Articulation = env.scene["robot"]
    body_id = asset.find_bodies(ee_body_name)[0][0]
    ee_pos_w = asset.data.body_pos_w[:, body_id]
    goal_w = cmd_term.curr_goal_pos_w
    yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    err_w = goal_w - ee_pos_w
    return torch.nan_to_num(math_utils.quat_apply_inverse(yaw_quat, err_w), nan=0.0)


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def object_distance_exp(
    env: "ManagerBasedRLEnv",
    ee_body_name: str = "gripper_link",
    command_name: str = "ee_goal",
    sigma: float = 0.1,
) -> torch.Tensor:
    """exp(-||gripper - goal||^2 / sigma)"""
    d = torch.sum(torch.square(gripper_to_goal(env, ee_body_name=ee_body_name, command_name=command_name)), dim=1)
    return torch.exp(-d / max(sigma, 1e-6))


def object_distance_l2(
    env: "ManagerBasedRLEnv",
    ee_body_name: str = "gripper_link",
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """||gripper - goal||^2"""
    return torch.sum(torch.square(gripper_to_goal(env, ee_body_name=ee_body_name, command_name=command_name)), dim=1)


def world_goal_distance_exp(
    env: "ManagerBasedRLEnv",
    ee_body_name: str = "gripper_link",
    command_name: str = "ee_goal",
    sigma: float = 0.2,
) -> torch.Tensor:
    """exp(-||gripper - world_goal||^2 / sigma) — works with HuskyWorldEeGoalCommand."""
    err = gripper_to_world_goal(env, ee_body_name=ee_body_name, command_name=command_name)
    d = torch.sum(torch.square(err), dim=1)
    return torch.exp(-d / max(sigma, 1e-6))


def world_goal_distance_l2(
    env: "ManagerBasedRLEnv",
    ee_body_name: str = "gripper_link",
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """||gripper - world_goal||^2"""
    err = gripper_to_world_goal(env, ee_body_name=ee_body_name, command_name=command_name)
    return torch.sum(torch.square(err), dim=1)


def base_progress_to_goal(
    env: "ManagerBasedRLEnv",
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """Reward for base moving toward the goal (dot product of velocity and direction)."""
    cmd_term = env.command_manager.get_term(command_name)
    if not hasattr(cmd_term, "curr_goal_pos_w"):
        return torch.zeros(env.num_envs, device=env.device)
    goal_w = cmd_term.curr_goal_pos_w
    root_pos_w = env.scene["robot"].data.root_pos_w
    direction = goal_w[:, :2] - root_pos_w[:, :2]
    dist = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-3)
    unit_dir = direction / dist
    base_vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]
    progress = torch.sum(unit_dir * base_vel_w, dim=1)
    return torch.clamp(progress, min=0.0)


def base_goal_distance_delta_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """Reward positive reduction of base-to-goal XY distance between steps."""
    cmd_term = env.command_manager.get_term(command_name)
    if not hasattr(cmd_term, "curr_goal_pos_w"):
        return torch.zeros(env.num_envs, device=env.device)
    goal_w = cmd_term.curr_goal_pos_w
    root_pos_w = env.scene["robot"].data.root_pos_w
    curr_dist = torch.norm(goal_w[:, :2] - root_pos_w[:, :2], dim=1)
    if not hasattr(env, "_prev_base_goal_dist_xy"):
        env._prev_base_goal_dist_xy = curr_dist.clone()
    delta = env._prev_base_goal_dist_xy - curr_dist
    env._prev_base_goal_dist_xy = curr_dist.clone()
    return torch.clamp(delta, min=0.0)


def base_reverse_velocity_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """Squared backward component of base XY velocity w.r.t. goal direction (deters zig-zag / backing up)."""
    cmd_term = env.command_manager.get_term(command_name)
    if not hasattr(cmd_term, "curr_goal_pos_w"):
        return torch.zeros(env.num_envs, device=env.device)
    goal_w = cmd_term.curr_goal_pos_w
    root_pos_w = env.scene["robot"].data.root_pos_w
    direction = goal_w[:, :2] - root_pos_w[:, :2]
    dist = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-3)
    unit_dir = direction / dist
    base_vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]
    along = torch.sum(unit_dir * base_vel_w, dim=1)
    return torch.square(torch.relu(-along))


def base_yaw_rate_w_sq(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Squared world-frame yaw rate (vertical axis) — penalizes tight weaving."""
    asset: Articulation = env.scene["robot"]
    return torch.square(asset.data.root_ang_vel_w[:, 2])


def base_lin_vel_xy_delta_sq(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Squared step-to-step change in world-frame base linear velocity (XY) — discourages surging / braking oscillation."""
    asset: Articulation = env.scene["robot"]
    v = asset.data.root_lin_vel_w[:, :2]
    if not hasattr(env, "_prev_world_base_lin_vel_xy"):
        env._prev_world_base_lin_vel_xy = v.clone()
    d = v - env._prev_world_base_lin_vel_xy
    env._prev_world_base_lin_vel_xy = v.clone()
    return torch.sum(torch.square(d), dim=1)


def gripper_chassis_proximity_penalty_sq(
    env: "ManagerBasedRLEnv",
    ee_body_name: str = "gripper_link",
    min_horizontal_clearance_m: float = 0.30,
    min_height_above_root_m: float = 0.06,
) -> torch.Tensor:
    """Penalize EE too close to the base (XY) or too low vs root — soft proxy for arm-through-chassis poses.

    With ``enabled_self_collisions=False``, PhysX does not resolve arm–body overlap; geometry keeps the policy
    from learning folds that look like self-collision.
    """
    asset: Articulation = env.scene["robot"]
    ee_id = asset.find_bodies(ee_body_name)[0][0]
    ee_w = asset.data.body_pos_w[:, ee_id]
    root_w = asset.data.root_pos_w
    delta = ee_w - root_w
    xy = torch.norm(delta[:, :2], dim=1)
    h = delta[:, 2]
    pen_xy = torch.square(torch.relu(min_horizontal_clearance_m - xy))
    pen_h = torch.square(torch.relu(min_height_above_root_m - h))
    return pen_xy + pen_h


def diff_drive_linear_command_delta_sq(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Squared step-to-step change in linear drive command (raw action index 0).

    Deters frantic forward/back oscillation; smooth throttle changes are cheap, sharp reversals cost a little.
    Turning via angular command (index 1) is not penalized here so the policy can prefer поворот instead of rocking.
    """
    a = env.action_manager.action
    prev = env.action_manager.prev_action
    if a.shape[-1] < 1:
        return torch.zeros(env.num_envs, device=env.device)
    d = a[:, 0] - prev[:, 0]
    return torch.square(d)


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    a = env.action_manager.action
    prev = env.action_manager.prev_action
    return torch.sum(torch.square(prev - a), dim=1)


def joint_torques_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def joint_acc_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel
    qd_prev = env.action_manager.prev_action.new_zeros(qd.shape)
    if hasattr(env, "_prev_joint_vel"):
        qd_prev = env._prev_joint_vel
    acc = (qd_prev - qd) / max(env.step_dt, 1e-6)
    env._prev_joint_vel = qd.clone()
    return torch.sum(torch.square(acc), dim=1)


def orientation_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize non-upright base orientation (projected gravity XY components)."""
    asset: Articulation = env.scene["robot"]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_flipped(env: "ManagerBasedRLEnv", threshold: float = 0.7) -> torch.Tensor:
    """Termination: True when the base is flipped (gravity z component < -threshold).

    projected_gravity_b[:,2] ≈ -1 when upright, ≈ +1 when flipped.
    """
    asset: Articulation = env.scene["robot"]
    return asset.data.projected_gravity_b[:, 2] > threshold


def root_height_above_maximum(
    env: "ManagerBasedRLEnv",
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root Z is above a ceiling (flat terrain; catches pops / unstable contacts).

    Complements :func:`isaaclab.envs.mdp.terminations.root_height_below_minimum` (fall-through only).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height


def ang_vel_xy_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize roll/pitch angular velocity."""
    asset: Articulation = env.scene["robot"]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def joint_pos_limits(env: "ManagerBasedRLEnv", margin: float = 0.0) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    q = asset.data.joint_pos
    lo = asset.data.soft_joint_pos_limits[..., 0] + margin
    hi = asset.data.soft_joint_pos_limits[..., 1] - margin
    out = -(q - lo).clamp(max=0.0) + (q - hi).clamp(min=0.0)
    return torch.sum(out, dim=1)


def husky_ik_total_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Total reward for Husky UR5 IK tracking.

    Positive rewards: EE tracking (exp + tanh-like).
    Penalties: joint limits, action rate, torques, joint acc, orientation, ang_vel_xy.
    Uses only_positive_rewards clipping + termination penalty.
    """

    r_obj_dist_exp = 3.0 * object_distance_exp(env, sigma=0.1)
    r_obj_dist_l2 = -1.5 * object_distance_l2(env)
    r_orientation = -0.2 * orientation_l2(env)
    r_ang_vel_xy = -0.05 * ang_vel_xy_l2(env)
    r_torques = -1e-8 * joint_torques_l2(env)
    r_joint_acc = -1e-9 * joint_acc_l2(env)
    r_action_rate = -0.005 * action_rate_l2(env)
    r_joint_lim = -5.0 * joint_pos_limits(env, margin=0.0)

    r = (
        r_obj_dist_exp + r_obj_dist_l2
        + r_orientation + r_ang_vel_xy
        + r_torques + r_joint_acc
        + r_action_rate + r_joint_lim
    )
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    if not hasattr(env, "_reward_debug_counter"):
        env._reward_debug_counter = 0
    env._reward_debug_counter += 1
    if env._reward_debug_counter % 100 == 0:
        i = 0
        print(
            f"[HuskyIK RewardDebug step={env._reward_debug_counter}] "
            f"obj_exp={r_obj_dist_exp[i]:.4f} obj_l2={r_obj_dist_l2[i]:.4f} "
            f"orient={r_orientation[i]:.4f} ang_xy={r_ang_vel_xy[i]:.4f} "
            f"torq={r_torques[i]:.4f} acc={r_joint_acc[i]:.4f} "
            f"act_rate={r_action_rate[i]:.4f} j_lim={r_joint_lim[i]:.4f} "
            f"sum_raw={r[i]:.4f}"
        )

    r = torch.clamp(r, min=0.0)

    terminated = env.termination_manager.terminated
    time_out = env.termination_manager.time_outs
    r += -1.0 * (terminated & ~time_out).to(dtype=r.dtype)

    return r


def husky_world_ik_total_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Total reward for Husky UR5 with world-frame goals (base-arm cooperation).

    Key design: base earns reward for driving toward goal, arm earns reward for
    precise EE positioning. Both are needed for high total reward.
    """

    # EE tracking in world frame (L2 clamped to avoid dominating when far away)
    r_ee_exp = 3.0 * world_goal_distance_exp(env, sigma=0.2)
    raw_l2 = world_goal_distance_l2(env)
    r_ee_l2 = -0.25 * torch.clamp(raw_l2, max=2.0)

    # Base progress toward goal — strong signal to drive
    r_progress = 3.0 * base_progress_to_goal(env)
    r_delta_dist = 2.0 * base_goal_distance_delta_reward(env)
    r_reverse = -0.75 * base_reverse_velocity_penalty(env)
    r_yaw_w = -0.04 * base_yaw_rate_w_sq(env)
    # Stronger: penalize rapid forward/back command changes (not angular — prefer поворот вместо качания)
    r_lin_d = -0.04 * diff_drive_linear_command_delta_sq(env)
    r_vel_d = -0.015 * base_lin_vel_xy_delta_sq(env)
    r_self_geom = -1.0 * gripper_chassis_proximity_penalty_sq(env)

    # Stability: strong penalty for tipping over
    r_orientation = -2.0 * orientation_l2(env)
    r_ang_vel_xy = -0.5 * ang_vel_xy_l2(env)

    # Regularization (gentle)
    r_torques = -1e-8 * joint_torques_l2(env)
    r_joint_acc = -1e-9 * joint_acc_l2(env)
    r_action_rate = -0.003 * action_rate_l2(env)
    r_joint_lim = -3.0 * joint_pos_limits(env, margin=0.0)

    r = (
        r_ee_exp + r_ee_l2
        + r_progress + r_delta_dist + r_reverse + r_yaw_w + r_lin_d + r_vel_d + r_self_geom
        + r_orientation + r_ang_vel_xy
        + r_torques + r_joint_acc
        + r_action_rate + r_joint_lim
    )
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    if not hasattr(env, "_reward_debug_counter_w"):
        env._reward_debug_counter_w = 0
    env._reward_debug_counter_w += 1
    if env._reward_debug_counter_w % 100 == 0:
        i = 0
        print(
            f"[HuskyWorldIK RewardDebug step={env._reward_debug_counter_w}] "
            f"ee_exp={r_ee_exp[i]:.4f} ee_l2={r_ee_l2[i]:.4f} "
            f"progress={r_progress[i]:.4f} delta_d={r_delta_dist[i]:.4f} "
            f"rev={r_reverse[i]:.4f} yaw_w={r_yaw_w[i]:.4f} lin_d={r_lin_d[i]:.4f} vel_d={r_vel_d[i]:.4f} "
            f"self_g={r_self_geom[i]:.4f} "
            f"orient={r_orientation[i]:.4f} ang_xy={r_ang_vel_xy[i]:.4f} "
            f"torq={r_torques[i]:.4f} acc={r_joint_acc[i]:.4f} "
            f"act_rate={r_action_rate[i]:.4f} j_lim={r_joint_lim[i]:.4f} "
            f"sum_raw={r[i]:.4f}"
        )

    r = torch.clamp(r, min=0.0)

    # Termination penalty (large) — flipping, etc.
    terminated = env.termination_manager.terminated
    time_out = env.termination_manager.time_outs
    r += -5.0 * (terminated & ~time_out).to(dtype=r.dtype)

    return r
