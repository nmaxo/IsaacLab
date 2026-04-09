"""Go2-ARX DWBC-Orig environment config.

Goal: mimic the original Deep-Whole-Body-Control (DWBC) pipeline (widowGo1) as closely as possible:
- Commands: independent base velocity commands + EE goal in sphere coordinates (l, pitch, yaw)
- Rewards: exp(-L1/sigma) where DWBC uses L1 inside exp; energy/contacts shaping like DWBC

This is an alternative env ID to avoid changing the existing Go2Arx DWBC setup.
"""

from __future__ import annotations

import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import (
    orientation_command_error,
    orientation_command_error_exp,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from .loco_manip_dwbc_env_cfg import (
    Go2ArxDwbcEventCfg,
    Go2ArxLocoManipDwbcEnvCfg,
    Go2ArxLocoManipDwbcEnvCfg_PLAY,
)


# -----------------------------------------------------------------------------
# DWBC-style policy observation functions
# -----------------------------------------------------------------------------


def obs_roll_pitch(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Body roll and pitch angles (2D). Original DWBC uses these instead of projected gravity."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.stack([roll, pitch], dim=-1)


def obs_foot_contacts_binary(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary foot contacts (4D for quadruped). 1 if force > threshold, else 0."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces_z.abs() > threshold).float()


def obs_ee_goal_sphere(
    env: "ManagerBasedRLEnv", command_name: str = "ee_pose"
) -> torch.Tensor:
    """EE goal in sphere coordinates (r, pitch, yaw) from DwbcSphereEePoseCommand. Shape: (N, 3)."""
    ee_term = env.command_manager.get_term(command_name)
    if hasattr(ee_term, "goal_sphere"):
        return ee_term.goal_sphere
    return env.command_manager.get_command(command_name)[:, :3]


# -----------------------------------------------------------------------------
# Commands (DWBC-style)
# -----------------------------------------------------------------------------


def _cart2sphere(xyz: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert xyz (N,3) to (r, pitch, yaw) with pitch in [-pi/2, pi/2], yaw in [-pi, pi]."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = torch.sqrt(x * x + y * y + z * z + eps)
    yaw = torch.atan2(y, x)
    pitch = torch.asin(torch.clamp(z / r, -1.0 + 1e-6, 1.0 - 1e-6))
    return torch.stack([r, pitch, yaw], dim=-1)


def _sphere2cart(rpy: torch.Tensor) -> torch.Tensor:
    """Convert (r, pitch, yaw) to xyz in the same convention as _cart2sphere."""
    r, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cp = torch.cos(pitch)
    x = r * cp * torch.cos(yaw)
    y = r * cp * torch.sin(yaw)
    z = r * torch.sin(pitch)
    return torch.stack([x, y, z], dim=-1)


class DwbcSphereEePoseCommand(CommandTerm):
    """EE goal fixed in world, sampled in sphere coordinates relative to env origin.

    Goal is sampled in world frame (env_origin + sphere offset), so it does not move.
    Command tensor returned is pose in base frame (goal expressed relative to robot) for the policy.
    goal_sphere is updated each step from current base-frame goal for reward.
    """

    cfg: "DwbcSphereEePoseCommandCfg"

    def __init__(self, cfg: "DwbcSphereEePoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]
        body_ids, _ = self._robot.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body match for {cfg.body_name}, got {body_ids}")
        self._ee_body_id = body_ids[0]

        # pose command buffers (in base frame)
        self._command = torch.zeros(self.num_envs, 7, device=self.device)
        self._command[:, 3] = 1.0  # identity quat

        # DWBC-style sphere goal buffers
        self.goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        # World-frame goal at resample time (for fixed-in-world visualization)
        self._command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._command_w[:, 3] = 1.0

        # metrics expected by existing curricula/logging utilities
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # per-env difficulty (0=init ranges, 1=final ranges)
        self.difficulty = torch.zeros(self.num_envs, device=self.device)

        # schedule: ramp difficulty by number of resamples (simple proxy)
        self._resample_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # initialize immediately
        self._resample_command(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = env_ids.numel()
        if n == 0:
            return

        # update per-env difficulty (linear ramp over N resamples)
        self._resample_count[env_ids] += 1
        self.difficulty[env_ids] = torch.clamp(self._resample_count[env_ids].float() / max(self.cfg.ramp_resamples, 1), 0.0, 1.0)

        d = self.difficulty[env_ids]

        def lerp_range(easy, hard):
            low = easy[0] + (hard[0] - easy[0]) * d
            high = easy[1] + (hard[1] - easy[1]) * d
            return low, high

        # sample (r, pitch, yaw)
        r_low, r_high = lerp_range(self.cfg.init_ranges.pos_l, self.cfg.ranges.pos_l)
        p_low, p_high = lerp_range(self.cfg.init_ranges.pos_p, self.cfg.ranges.pos_p)
        y_low, y_high = lerp_range(self.cfg.init_ranges.pos_y, self.cfg.ranges.pos_y)

        r = torch.rand(n, device=self.device) * (r_high - r_low) + r_low
        p = torch.rand(n, device=self.device) * (p_high - p_low) + p_low
        y = torch.rand(n, device=self.device) * (y_high - y_low) + y_low
        goal_sphere = torch.stack([r, p, y], dim=-1)

        # Goal fixed in world: env_origin + spherical offset (same convention as main loco_manip)
        goal_cart_offset = _sphere2cart(goal_sphere)
        goal_cart_offset[:, 2] += self.cfg.z_invariant_offset
        env_origins = self._env.scene.env_origins[env_ids]
        self._command_w[env_ids, :3] = env_origins + goal_cart_offset
        self._command_w[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(n, 1)
        # Set base-frame command and goal_sphere for resampled envs (so valid before first _update_command)
        root_pos = self._robot.data.root_pos_w[env_ids]
        root_quat = self._robot.data.root_quat_w[env_ids]
        self._command[env_ids, :3] = math_utils.quat_apply_inverse(root_quat, self._command_w[env_ids, :3] - root_pos)
        self._command[env_ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(n, 1)
        self.goal_sphere[env_ids] = _cart2sphere(self._command[env_ids, :3])

    def _update_command(self):
        """Compute goal in base frame from fixed world goal (for policy and reward)."""
        pos_w = self._command_w[:, :3]
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w
        self._command[:, :3] = math_utils.quat_apply_inverse(root_quat, pos_w - root_pos)
        self._command[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.goal_sphere[:] = _cart2sphere(self._command[:, :3])

    def _update_metrics(self):
        # Minimal metrics to satisfy CommandTerm interface.
        # (DWBC-style curricula for this command are implemented via ramp_resamples.)
        if "position_error" not in self.metrics:
            self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        if "orientation_error" not in self.metrics:
            self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Toggle visualization of EE goal pose in world frame."""
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Draw EE goal fixed in world (pose at resample time)."""
        if not self._robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self._command_w[:, :3], self._command_w[:, 3:7])


@configclass
class DwbcSphereEePoseCommandCfg(CommandTermCfg):
    class_type: type = DwbcSphereEePoseCommand

    asset_name: str = MISSING
    body_name: str = MISSING

    goal_pose_visualizer_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/ee_goal_pose")
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # DWBC: init vs final ranges
    @configclass
    class Ranges:
        pos_l: tuple[float, float] = MISSING
        pos_p: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    init_ranges: Ranges = MISSING

    # approximate schedule by resample count
    ramp_resamples: int = 100

    # z-invariant offset (DWBC used -0.165); keep 0 for Go2-ARX by default
    z_invariant_offset: float = 0.0


class DwbcUniformVelocityCommand(mdp.UniformVelocityCommand):
    """Uniform velocity command with DWBC-style init->final range interpolation."""

    cfg: "DwbcUniformVelocityCommandCfg"

    def __init__(self, cfg: "DwbcUniformVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.difficulty = torch.zeros(self.num_envs, device=self.device)
        self._resample_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = env_ids.numel()
        if n == 0:
            return

        self._resample_count[env_ids] += 1
        self.difficulty[env_ids] = torch.clamp(self._resample_count[env_ids].float() / max(self.cfg.ramp_resamples, 1), 0.0, 1.0)
        d = self.difficulty[env_ids]

        def lerp_tuple(easy, hard):
            low = easy[0] + (hard[0] - easy[0]) * d
            high = easy[1] + (hard[1] - easy[1]) * d
            return low, high

        x_low, x_high = lerp_tuple(self.cfg.init_ranges.lin_vel_x, self.cfg.ranges.lin_vel_x)
        y_low, y_high = lerp_tuple(self.cfg.init_ranges.lin_vel_y, self.cfg.ranges.lin_vel_y)
        w_low, w_high = lerp_tuple(self.cfg.init_ranges.ang_vel_z, self.cfg.ranges.ang_vel_z)

        r = torch.rand(n, device=self.device)
        self.vel_command_b[env_ids, 0] = r * (x_high - x_low) + x_low
        r = torch.rand(n, device=self.device)
        self.vel_command_b[env_ids, 1] = r * (y_high - y_low) + y_low
        r = torch.rand(n, device=self.device)
        self.vel_command_b[env_ids, 2] = r * (w_high - w_low) + w_low

        # DWBC had clips; apply via clamp
        if self.cfg.lin_vel_x_clip is not None:
            self.vel_command_b[env_ids, 0] = self.vel_command_b[env_ids, 0].clamp(-self.cfg.lin_vel_x_clip, self.cfg.lin_vel_x_clip)
        if self.cfg.ang_vel_yaw_clip is not None:
            self.vel_command_b[env_ids, 2] = self.vel_command_b[env_ids, 2].clamp(-self.cfg.ang_vel_yaw_clip, self.cfg.ang_vel_yaw_clip)


@configclass
class DwbcUniformVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type: type = DwbcUniformVelocityCommand

    @configclass
    class Ranges(mdp.UniformVelocityCommandCfg.Ranges):
        pass

    ranges: Ranges = MISSING

    @configclass
    class InitRanges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = MISSING

    init_ranges: InitRanges = MISSING
    ramp_resamples: int = 200

    lin_vel_x_clip: float | None = None
    ang_vel_yaw_clip: float | None = None


# -----------------------------------------------------------------------------
# Rewards (DWBC-style forms)
# -----------------------------------------------------------------------------


def _base_yaw_quat(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternion from root orientation."""
    yaw = math_utils.euler_xyz_from_quat(root_quat_w)[2]
    return math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)


def reward_survive(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def reward_tracking_ang_vel_yaw_exp_l1(env: "ManagerBasedRLEnv", sigma: float, command_name: str = "base_velocity") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    asset: Articulation = env.scene["robot"]
    err = torch.abs(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-err / max(sigma, 1e-6))


def reward_tracking_lin_vel_xy_exp_sq(env: "ManagerBasedRLEnv", sigma: float, command_name: str = "base_velocity") -> torch.Tensor:
    # DWBC used exp(-||e||^2/sigma); not sigma^2
    cmd = env.command_manager.get_command(command_name)
    asset: Articulation = env.scene["robot"]
    sq = torch.sum(torch.square(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-sq / max(sigma, 1e-6))


def reward_tracking_lin_vel_x_l1(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """DWBC: reward = -error + |cmd_x| so that small error gives high reward (scale 0.5)."""
    cmd = env.command_manager.get_command(command_name)
    asset: Articulation = env.scene["robot"]
    error = torch.abs(cmd[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return -error + torch.abs(cmd[:, 0])


def reward_tracking_lin_vel_x_exp_l1(
    env: "ManagerBasedRLEnv", sigma: float, command_name: str = "base_velocity"
) -> torch.Tensor:
    err = reward_tracking_lin_vel_x_l1(env, command_name=command_name)
    return torch.exp(-err / max(sigma, 1e-6))


def reward_tracking_ang_vel_yaw_l1(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    asset: Articulation = env.scene["robot"]
    return torch.abs(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])


def reward_tracking_lin_vel_y_l2(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    asset: Articulation = env.scene["robot"]
    return torch.square(cmd[:, 1] - asset.data.root_lin_vel_b[:, 1])


def reward_feet_stumble(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # Placeholder (DWBC term exists but IsaacLab does not have a direct equivalent here)
    return torch.zeros(env.num_envs, device=env.device)


def reward_hip_action_l2(env: "ManagerBasedRLEnv", indices: list[int]) -> torch.Tensor:
    a = env.action_manager.action[:, indices]
    return torch.sum(a * a, dim=1)


def reward_leg_action_l2(env: "ManagerBasedRLEnv", end_idx: int = 12) -> torch.Tensor:
    a = env.action_manager.action[:, :end_idx]
    return torch.sum(a * a, dim=1)


def reward_foot_contacts_z(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net forces history: (N, H, B, 3). Use latest sample.
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]
    return torch.sum(forces * forces, dim=1)


def reward_energy_square(env: "ManagerBasedRLEnv", joint_slice: slice = slice(0, 12)) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    tau = asset.data.applied_torque[:, joint_slice]
    qd = asset.data.joint_vel[:, joint_slice]
    return torch.sum(torch.square(tau * qd), dim=1)


def reward_arm_energy_abs_sum(env: "ManagerBasedRLEnv", joint_slice: slice = slice(12, 18)) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    tau = asset.data.applied_torque[:, joint_slice]
    qd = asset.data.joint_vel[:, joint_slice]
    return torch.sum(torch.abs(tau * qd), dim=1)


def reward_tracking_ee_sphere_exp_l1(
    env: "ManagerBasedRLEnv",
    sigma: float,
    sphere_error_scale: tuple[float, float, float],
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    ee_term = env.command_manager.get_term(command_name)
    if not hasattr(ee_term, "goal_sphere"):
        raise AttributeError("DWBC sphere EE command term must expose goal_sphere buffer.")
    goal_sphere = ee_term.goal_sphere  # (N,3)

    base_pos_w = asset.data.root_pos_w
    yaw_quat = _base_yaw_quat(asset.data.root_quat_w)
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    ee_local = math_utils.quat_apply_inverse(yaw_quat, ee_pos_w - base_pos_w)

    ee_sphere = _cart2sphere(ee_local)
    scale = torch.tensor(sphere_error_scale, device=env.device).unsqueeze(0)
    err = torch.sum(torch.abs(ee_sphere - goal_sphere) * scale, dim=1)
    return torch.exp(-err / max(sigma, 1e-6))


# -----------------------------------------------------------------------------
# Env config
# -----------------------------------------------------------------------------


@configclass
class Go2ArxDwbcOrigCommandsCfg:
    """DWBC-style commands: independent base velocity + sphere EE goal."""

    base_velocity = DwbcUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 3.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=DwbcUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.9),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=None,
        ),
        init_ranges=DwbcUniformVelocityCommandCfg.InitRanges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        ramp_resamples=200,
        lin_vel_x_clip=0.3,
        ang_vel_yaw_clip=0.6,
        debug_vis=True,
    )

    ee_pose = DwbcSphereEePoseCommandCfg(
        asset_name="robot",
        body_name="arm_link6",
        resampling_time_range=(1.5, 3.0),
        ranges=DwbcSphereEePoseCommandCfg.Ranges(
            pos_l=(0.15, 0.45),
            pos_p=(-math.pi / 4, math.pi / 6),
            pos_y=(-math.pi / 2, math.pi / 2),
        ),
        init_ranges=DwbcSphereEePoseCommandCfg.Ranges(
            pos_l=(0.3, 0.35),
            pos_p=(0.0, math.pi / 6),
            pos_y=(-math.pi / 6, math.pi / 6),
        ),
        ramp_resamples=200,
        z_invariant_offset=0.0,
        debug_vis=True,
    )


@configclass
class Go2ArxDwbcOrigRewardsCfg:
    """DWBC-style reward terms with DWBC-like weights."""

    # ---------------------------------------------------------------------
    # Locomotion terms (DWBC widowGo1 naming; many weights are 0 in the original cfg)
    # ---------------------------------------------------------------------
    survive = RewTerm(func=reward_survive, weight=0.2)

    # tracking
    tracking_lin_vel = RewTerm(func=reward_tracking_lin_vel_xy_exp_sq, weight=0.0, params={"sigma": 1.0})
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": 1.0})
    tracking_lin_vel_x_l1 = RewTerm(func=reward_tracking_lin_vel_x_l1, weight=0.5, params={"command_name": "base_velocity"})
    tracking_lin_vel_x_exp = RewTerm(func=reward_tracking_lin_vel_x_exp_l1, weight=0.0, params={"sigma": 1.0, "command_name": "base_velocity"})
    tracking_ang_vel_yaw_l1 = RewTerm(func=reward_tracking_ang_vel_yaw_l1, weight=0.0, params={"command_name": "base_velocity"})
    tracking_ang_vel_yaw_exp = RewTerm(func=reward_tracking_ang_vel_yaw_exp_l1, weight=0.15, params={"sigma": 1.0, "command_name": "base_velocity"})
    tracking_lin_vel_y_l2 = RewTerm(func=reward_tracking_lin_vel_y_l2, weight=0.0, params={"command_name": "base_velocity"})
    tracking_lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot")})

    # regularization
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    torques = RewTerm(func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot")})
    base_height = RewTerm(func=mdp.base_height_l2, weight=0.0, params={"target_height": 0.25, "asset_cfg": SceneEntityCfg("robot")})
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "command_name": "base_velocity", "threshold": 0.5})
    collision = RewTerm(func=mdp.undesired_contacts, weight=0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0})
    feet_stumble = RewTerm(func=reward_feet_stumble, weight=-0.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0)
    stand_still = RewTerm(func=mdp.stand_still_joint_deviation_l1, weight=0.0, params={"command_name": "base_velocity", "command_threshold": 0.06, "asset_cfg": SceneEntityCfg("robot")})

    # action/energy/contact (DWBC non-zero by default)
    hip_action_l2 = RewTerm(func=reward_hip_action_l2, weight=-0.01, params={"indices": [0, 3, 6, 9]})
    leg_action_l2 = RewTerm(func=reward_leg_action_l2, weight=-0.0, params={"end_idx": 12})
    foot_contacts_z = RewTerm(func=reward_foot_contacts_z, weight=-1e-4, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
    energy_square = RewTerm(func=reward_energy_square, weight=-6e-5, params={"joint_slice": slice(0, 12)})
    leg_energy = RewTerm(func=reward_arm_energy_abs_sum, weight=-0.0, params={"joint_slice": slice(0, 12)})
    leg_energy_abs_sum = RewTerm(func=reward_arm_energy_abs_sum, weight=-0.0, params={"joint_slice": slice(0, 12)})
    leg_energy_sum_abs = RewTerm(func=reward_arm_energy_abs_sum, weight=-0.0, params={"joint_slice": slice(0, 12)})

    # ---------------------------------------------------------------------
    # Manipulation terms (DWBC widowGo1 naming)
    # ---------------------------------------------------------------------
    # DWBC sphere_error_scale = 1/(range) per axis: l range 0.5 -> 2.0; p range 3π/5 -> 5/(3π); y range 6π/5 -> 5/(6π)
    tracking_ee_sphere = RewTerm(
        func=reward_tracking_ee_sphere_exp_l1,
        weight=0.55,
        params={
            "sigma": 1.0,
            "sphere_error_scale": (1.0 / (0.7 - 0.2), 1.0 / (math.pi * 3.0 / 5.0), 1.0 / (math.pi * 6.0 / 5.0)),
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="arm_link6"),
        },
    )
    tracking_ee_cart = RewTerm(
        func=reward_tracking_ee_sphere_exp_l1,
        weight=0.0,
        params={"sigma": 1.0, "sphere_error_scale": (1.0, 1.0, 1.0), "command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names="arm_link6")},
    )
    tracking_ee_orn = RewTerm(func=orientation_command_error_exp, weight=0.0, params={"std": 1.0, "command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names="arm_link6")})
    tracking_ee_orn_ry = RewTerm(func=orientation_command_error_exp, weight=0.0, params={"std": 1.0, "command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names="arm_link6")})
    arm_orientation = RewTerm(func=orientation_command_error, weight=-0.0, params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names="arm_link6")})
    arm_energy_abs_sum = RewTerm(func=reward_arm_energy_abs_sum, weight=-0.004, params={"joint_slice": slice(12, 18)})


REWARD_GROUPS_DWBC_ORIG = {
    "locomotion": [
        "survive",
        "tracking_lin_vel_x_l1",
        "tracking_ang_vel_yaw_exp",
        "hip_action_l2",
        "foot_contacts_z",
        "energy_square",
    ],
    "manipulation": [
        "tracking_ee_sphere",
        "arm_energy_abs_sum",
    ],
}


# -----------------------------------------------------------------------------
# DWBC-Orig Observations (policy-only; same terms as before minus privileged)
# -----------------------------------------------------------------------------


@configclass
class Go2ArxDwbcOrigObservationsCfg:
    """DWBC-style policy observations (Go2-ARX, ~69 dim).

    roll/pitch (2), base_ang_vel (3), joint_pos_rel (18), joint_vel (18),
    last_action (18), foot_contacts (4), velocity_commands (3), ee_goal_sphere (3)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        roll_pitch = ObsTerm(func=obs_roll_pitch, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        foot_contacts = ObsTerm(
            func=obs_foot_contacts_binary,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        ee_goal_sphere = ObsTerm(
            func=obs_ee_goal_sphere,
            params={"command_name": "ee_pose"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2ArxDwbcOrigActionsCfg:
    """DWBC-style action scales: larger than default to match original exploration range."""

    joint_pos_legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.4,
        use_default_offset=True,
    )
    joint_pos_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class Go2ArxLocoManipDwbcOrigEnvCfg(Go2ArxLocoManipDwbcEnvCfg):
    """Go2-ARX DWBC-Orig: DWBC-like rewards + command sampling + policy-only observations."""

    observations: Go2ArxDwbcOrigObservationsCfg = Go2ArxDwbcOrigObservationsCfg()
    commands: Go2ArxDwbcOrigCommandsCfg = Go2ArxDwbcOrigCommandsCfg()
    rewards: Go2ArxDwbcOrigRewardsCfg = Go2ArxDwbcOrigRewardsCfg()
    actions: Go2ArxDwbcOrigActionsCfg = Go2ArxDwbcOrigActionsCfg()
    events: Go2ArxDwbcEventCfg = Go2ArxDwbcEventCfg()


@configclass
class Go2ArxLocoManipDwbcOrigEnvCfg_PLAY(Go2ArxLocoManipDwbcOrigEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.randomize_motor = None
        self.events.randomize_com = None

