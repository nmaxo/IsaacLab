"""MDP terms for Husky UR5 cabinet task (approach + open) with IK-assisted arm control."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils import configclass

from . import husky_ik_mdp as hk_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class HuskyCabinetEeGoalCommand(CommandTerm):
    """Two-stage world-frame target:
    1) approach drawer handle
    2) pull along drawer opening direction.
    """

    cfg: "HuskyCabinetEeGoalCommandCfg"

    def __init__(self, cfg: "HuskyCabinetEeGoalCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._cabinet: Articulation = env.scene[cfg.cabinet_asset_name]

        ee_ids, _ = self._robot.find_bodies(cfg.ee_body_name)
        if len(ee_ids) != 1:
            raise ValueError(f"Expected one EE body for '{cfg.ee_body_name}', got {ee_ids}")
        self._ee_body_id = ee_ids[0]

        handle_ids, _ = self._cabinet.find_bodies(cfg.handle_body_name)
        if len(handle_ids) != 1:
            raise ValueError(f"Expected one handle body for '{cfg.handle_body_name}', got {handle_ids}")
        self._handle_body_id = handle_ids[0]

        self.local_axis_z_offset = float(cfg.local_axis_z_offset)
        self.switch_distance = float(cfg.switch_distance)

        self._approach_offset = torch.tensor(cfg.approach_offset_xyz, device=self.device, dtype=torch.float)
        pull_dir = torch.tensor(cfg.open_pull_direction_xyz, device=self.device, dtype=torch.float)
        pull_dir = pull_dir / pull_dir.norm().clamp(min=1e-6)
        self._pull_delta = pull_dir * float(cfg.open_pull_distance)

        self.handle_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.handle_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.handle_quat_w[:, 0] = 1.0
        self.curr_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.curr_goal_quat_w[:, 0] = 1.0
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._ee_target_quat_in_handle = torch.tensor(
            cfg.ee_target_quat_in_handle,
            device=self.device,
            dtype=torch.float,
        )

        self._debug_vis = bool(cfg.debug_vis)
        self._vis_env_index = int(cfg.visualize_env_index)
        self._markers: VisualizationMarkers | None = None
        if self._debug_vis:
            markers_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/HuskyCabinetCommand",
                markers={
                    "handle": sim_utils.SphereCfg(
                        radius=0.045,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),
                    ),
                    "goal": sim_utils.SphereCfg(
                        radius=0.055,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(markers_cfg)

        self._resample_command(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    @property
    def command(self) -> torch.Tensor:
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
        self.phase[env_ids] = 0
        self._read_handle_frame()
        approach_offset_b = self._approach_offset.unsqueeze(0).repeat(self.num_envs, 1)
        approach_goal = self.handle_pos_w + math_utils.quat_apply(self.handle_quat_w, approach_offset_b)
        self.curr_goal_pos_w[env_ids] = approach_goal[env_ids]
        goal_quat = math_utils.quat_mul(
            self.handle_quat_w,
            self._ee_target_quat_in_handle.unsqueeze(0).repeat(self.num_envs, 1),
        )
        self.curr_goal_quat_w[env_ids] = goal_quat[env_ids]

    def _update_command(self):
        self._read_handle_frame()
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id]
        ee_to_handle = torch.norm(self.handle_pos_w - ee_pos_w, dim=1)

        just_reached = (self.phase == 0) & (ee_to_handle < self.switch_distance)
        self.phase[just_reached] = 1

        # Cabinet-style: approach offset is defined in the handle frame, not world axes.
        approach_offset_b = self._approach_offset.unsqueeze(0).repeat(self.num_envs, 1)
        approach_goal = self.handle_pos_w + math_utils.quat_apply(self.handle_quat_w, approach_offset_b)
        open_goal = self.handle_pos_w + self._pull_delta
        phase_mask = self.phase.to(dtype=torch.bool).unsqueeze(1)
        self.curr_goal_pos_w[:] = torch.where(phase_mask, open_goal, approach_goal)
        self.curr_goal_quat_w[:] = math_utils.quat_mul(
            self.handle_quat_w,
            self._ee_target_quat_in_handle.unsqueeze(0).repeat(self.num_envs, 1),
        )
        if self._markers is not None:
            self._update_visuals()

    def _update_metrics(self):
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id]
        ee_to_handle = torch.norm(self.handle_pos_w - ee_pos_w, dim=1)
        self.metrics["ee_to_handle"] = ee_to_handle
        self.metrics["phase"] = self.phase.to(dtype=torch.float)

    def _read_handle_frame(self):
        """Read handle frame exactly like cabinet task (via cabinet_frame sensor) when available."""
        if "cabinet_frame" in self._env.scene.keys():
            tf_data = self._env.scene["cabinet_frame"].data
            # index 0 = configured drawer_handle_top target frame
            self.handle_pos_w[:] = tf_data.target_pos_w[..., 0, :]
            self.handle_quat_w[:] = tf_data.target_quat_w[..., 0, :]
        else:
            # Fallback: raw body frame (kept for robustness)
            self.handle_pos_w[:] = self._cabinet.data.body_pos_w[:, self._handle_body_id]
            self.handle_quat_w[:] = self._cabinet.data.body_quat_w[:, self._handle_body_id]

    def _update_visuals(self):
        if self._markers is None:
            return
        if self._vis_env_index == -1:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            if self._vis_env_index < 0 or self._vis_env_index >= self.num_envs:
                return
            env_ids = torch.tensor([self._vis_env_index], device=self.device, dtype=torch.long)
        handle = self.handle_pos_w[env_ids]
        goal = self.curr_goal_pos_w[env_ids]
        pts = torch.cat([handle, goal], dim=0)
        idx = torch.cat(
            [
                torch.zeros(handle.shape[0], device=self.device, dtype=torch.long),
                torch.ones(goal.shape[0], device=self.device, dtype=torch.long),
            ]
        )
        self._markers.visualize(translations=pts, marker_indices=idx)


@configclass
class HuskyCabinetEeGoalCommandCfg(CommandTermCfg):
    class_type: type = HuskyCabinetEeGoalCommand

    # Required by CommandTermCfg validation; this command updates every step internally.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    asset_name: str = MISSING
    ee_body_name: str = MISSING
    cabinet_asset_name: str = "cabinet"
    handle_body_name: str = "drawer_handle_top"

    local_axis_z_offset: float = 0.3
    approach_offset_xyz: tuple[float, float, float] = (-0.10, 0.0, 0.05)
    open_pull_direction_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0)
    open_pull_distance: float = 0.24
    switch_distance: float = 0.10
    # Desired end-effector orientation in handle frame (w, x, y, z).
    ee_target_quat_in_handle: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    debug_vis: bool = True
    visualize_env_index: int = -1


def reset_cabinet_ee_goal(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, command_name: str = "ee_goal"):
    cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term(command_name)
    cmd_term._resample_command(env_ids)


def cabinet_handle_in_base_frame(env: "ManagerBasedRLEnv", command_name: str = "ee_goal") -> torch.Tensor:
    cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term(command_name)
    root_pos_w = env.scene["robot"].data.root_pos_w
    yaw_quat = math_utils.yaw_quat(env.scene["robot"].data.root_quat_w)
    diff = cmd_term.handle_pos_w - root_pos_w
    return torch.nan_to_num(math_utils.quat_apply_inverse(yaw_quat, diff), nan=0.0)


def cabinet_handle_distance_xy(env: "ManagerBasedRLEnv", command_name: str = "ee_goal") -> torch.Tensor:
    h = cabinet_handle_in_base_frame(env, command_name=command_name)
    return torch.norm(h[:, :2], dim=1, keepdim=True)


def cabinet_phase(env: "ManagerBasedRLEnv", command_name: str = "ee_goal") -> torch.Tensor:
    cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term(command_name)
    return cmd_term.phase.to(dtype=torch.float).unsqueeze(1)


def cabinet_open_progress(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
) -> torch.Tensor:
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = env.scene[asset_cfg.name].find_joints(asset_cfg.joint_names)[0]
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, joint_ids[0]]
    return torch.clamp(drawer_pos, min=0.0).unsqueeze(1)


def gripper_to_cabinet_handle(
    env: "ManagerBasedRLEnv",
    ee_body_name: str = "gripper_link",
    command_name: str = "ee_goal",
) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    ee_id = asset.find_bodies(ee_body_name)[0][0]
    ee_pos_w = asset.data.body_pos_w[:, ee_id]
    cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term(command_name)
    yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    err_w = cmd_term.handle_pos_w - ee_pos_w
    return torch.nan_to_num(math_utils.quat_apply_inverse(yaw_quat, err_w), nan=0.0)


def base_progress_to_handle(env: "ManagerBasedRLEnv", command_name: str = "ee_goal") -> torch.Tensor:
    cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term(command_name)
    root_pos_w = env.scene["robot"].data.root_pos_w
    direction = cmd_term.handle_pos_w[:, :2] - root_pos_w[:, :2]
    dist = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-3)
    unit_dir = direction / dist
    base_vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]
    progress = torch.sum(unit_dir * base_vel_w, dim=1)
    return torch.clamp(progress, min=0.0)


def heading_alignment_to_handle(env: "ManagerBasedRLEnv", command_name: str = "ee_goal") -> torch.Tensor:
    cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term(command_name)
    root_pos_w = env.scene["robot"].data.root_pos_w
    yaw_quat = math_utils.yaw_quat(env.scene["robot"].data.root_quat_w)

    direction = cmd_term.handle_pos_w[:, :2] - root_pos_w[:, :2]
    dist = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-3)
    unit_dir = direction / dist

    forward_b = torch.zeros(env.num_envs, 3, device=env.device)
    forward_b[:, 0] = 1.0
    forward_w = math_utils.quat_apply(yaw_quat, forward_b)[:, :2]
    return torch.clamp(torch.sum(forward_w * unit_dir, dim=1), min=0.0)


def ee_to_handle_exp(env: "ManagerBasedRLEnv", sigma: float = 0.03, command_name: str = "ee_goal") -> torch.Tensor:
    cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene["robot"]
    ee_id = asset.find_bodies("gripper_link")[0][0]
    ee_pos_w = asset.data.body_pos_w[:, ee_id]
    d2 = torch.sum(torch.square(cmd_term.handle_pos_w - ee_pos_w), dim=1)
    return torch.exp(-d2 / max(sigma, 1e-6))


def align_ee_to_handle(
    env: "ManagerBasedRLEnv",
    ee_body_name: str = "gripper_link",
    handle_body_name: str = "drawer_handle_top",
) -> torch.Tensor:
    """Alignment reward of EE orientation to handle frame (cabinet-style)."""
    # Use frame-transformer frames if available to match original cabinet setup.
    if "ee_frame" in env.scene.keys():
        ee_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    else:
        robot: Articulation = env.scene["robot"]
        ee_id = robot.find_bodies(ee_body_name)[0][0]
        ee_quat = robot.data.body_quat_w[:, ee_id]

    if "cabinet_frame" in env.scene.keys():
        handle_quat = env.scene["cabinet_frame"].data.target_quat_w[..., 0, :]
    else:
        cabinet: Articulation = env.scene["cabinet"]
        handle_id = cabinet.find_bodies(handle_body_name)[0][0]
        handle_quat = cabinet.data.body_quat_w[:, handle_id]

    ee_rot = matrix_from_quat(ee_quat)
    handle_rot = matrix_from_quat(handle_quat)

    handle_x = handle_rot[..., 0]
    handle_y = handle_rot[..., 1]
    ee_x = ee_rot[..., 0]
    ee_z = ee_rot[..., 2]

    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def align_grasp_around_handle(
    env: "ManagerBasedRLEnv",
    left_finger_body_name: str = "robotiq_85_left_finger_tip_link",
    right_finger_body_name: str = "robotiq_85_right_finger_tip_link",
    handle_body_name: str = "drawer_handle_top",
) -> torch.Tensor:
    """Cabinet-style grasp pose: left finger above handle, right below."""
    robot: Articulation = env.scene["robot"]
    cabinet: Articulation = env.scene["cabinet"]
    l_id = robot.find_bodies(left_finger_body_name)[0][0]
    r_id = robot.find_bodies(right_finger_body_name)[0][0]
    h_id = cabinet.find_bodies(handle_body_name)[0][0]

    l_pos = robot.data.body_pos_w[:, l_id]
    r_pos = robot.data.body_pos_w[:, r_id]
    h_pos = cabinet.data.body_pos_w[:, h_id]
    is_graspable = (r_pos[:, 2] < h_pos[:, 2]) & (l_pos[:, 2] > h_pos[:, 2])
    return is_graspable.to(dtype=torch.float)


def approach_gripper_handle(
    env: "ManagerBasedRLEnv",
    offset: float = 0.04,
    left_finger_body_name: str = "robotiq_85_left_finger_tip_link",
    right_finger_body_name: str = "robotiq_85_right_finger_tip_link",
    handle_body_name: str = "drawer_handle_top",
) -> torch.Tensor:
    """Cabinet-style finger approach reward when fingers are in graspable arrangement."""
    robot: Articulation = env.scene["robot"]
    cabinet: Articulation = env.scene["cabinet"]
    l_id = robot.find_bodies(left_finger_body_name)[0][0]
    r_id = robot.find_bodies(right_finger_body_name)[0][0]
    h_id = cabinet.find_bodies(handle_body_name)[0][0]

    l_pos = robot.data.body_pos_w[:, l_id]
    r_pos = robot.data.body_pos_w[:, r_id]
    h_pos = cabinet.data.body_pos_w[:, h_id]

    l_dist = torch.abs(l_pos[:, 2] - h_pos[:, 2])
    r_dist = torch.abs(r_pos[:, 2] - h_pos[:, 2])
    is_graspable = (r_pos[:, 2] < h_pos[:, 2]) & (l_pos[:, 2] > h_pos[:, 2])
    return is_graspable.to(dtype=torch.float) * ((offset - l_dist) + (offset - r_dist))


def gripper_knuckle_pos(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"],
    ),
) -> torch.Tensor:
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = env.scene[asset_cfg.name].find_joints(asset_cfg.joint_names)[0]
    return env.scene[asset_cfg.name].data.joint_pos[:, joint_ids]


def grasp_gate_near_handle(
    env: "ManagerBasedRLEnv",
    close_joint_threshold: float = 0.55,
    near_distance_m: float = 0.08,
    command_name: str = "ee_goal",
) -> torch.Tensor:
    knuckles = gripper_knuckle_pos(env)
    is_closed = torch.mean(knuckles, dim=1) > close_joint_threshold

    cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene["robot"]
    ee_id = asset.find_bodies("gripper_link")[0][0]
    ee_pos_w = asset.data.body_pos_w[:, ee_id]
    dist = torch.norm(cmd_term.handle_pos_w - ee_pos_w, dim=1)
    near_handle = dist < near_distance_m
    return (is_closed & near_handle).to(dtype=torch.float)


def grasp_handle_bonus(
    env: "ManagerBasedRLEnv",
    close_joint_threshold: float = 0.55,
    near_distance_m: float = 0.08,
    command_name: str = "ee_goal",
) -> torch.Tensor:
    """Bonus for closing gripper when EE is near handle (cabinet-style grasp term)."""
    return grasp_gate_near_handle(
        env,
        close_joint_threshold=close_joint_threshold,
        near_distance_m=near_distance_m,
        command_name=command_name,
    )


def open_drawer_bonus_cabinet_style(
    env: "ManagerBasedRLEnv",
    drawer_joint_name: str = "drawer_top_joint",
) -> torch.Tensor:
    drawer_id = env.scene["cabinet"].find_joints([drawer_joint_name])[0][0]
    drawer_pos = env.scene["cabinet"].data.joint_pos[:, drawer_id]
    return (1.0 + align_grasp_around_handle(env)) * torch.clamp(drawer_pos, min=0.0)


def multi_stage_open_drawer_cabinet_style(
    env: "ManagerBasedRLEnv",
    drawer_joint_name: str = "drawer_top_joint",
) -> torch.Tensor:
    drawer_id = env.scene["cabinet"].find_joints([drawer_joint_name])[0][0]
    drawer_pos = env.scene["cabinet"].data.joint_pos[:, drawer_id]
    graspable = align_grasp_around_handle(env)
    open_easy = (drawer_pos > 0.01).to(dtype=torch.float) * 0.5
    open_medium = (drawer_pos > 0.2).to(dtype=torch.float) * graspable
    open_hard = (drawer_pos > 0.3).to(dtype=torch.float) * graspable
    return open_easy + open_medium + open_hard


def cabinet_open_success(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.22,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
) -> torch.Tensor:
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = env.scene[asset_cfg.name].find_joints(asset_cfg.joint_names)[0]
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, joint_ids[0]]
    return drawer_pos > threshold


def husky_world_ik_cabinet_total_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term("ee_goal")
    phase = cmd_term.phase.to(dtype=torch.float)
    grasp_gate = grasp_gate_near_handle(env)

    # Approach + open shaping.
    r_ee = 2.5 * ee_to_handle_exp(env, sigma=0.05)
    # Encourage cabinet-style wrist orientation before grasp/open.
    r_align_ee = 0.8 * align_ee_to_handle(env)
    r_approach_gripper = 1.25 * approach_gripper_handle(env, offset=0.045)
    r_align_grasp = 0.3 * align_grasp_around_handle(env)
    r_base = 2.5 * base_progress_to_handle(env)
    r_heading = 0.75 * heading_alignment_to_handle(env)
    drawer_pos = env.scene["cabinet"].data.joint_pos[:, env.scene["cabinet"].find_joints(["drawer_top_joint"])[0][0]]
    # Open reward only when the policy closes gripper near handle (cabinet-like gating).
    r_open = 12.0 * torch.clamp(drawer_pos, min=0.0) * phase * grasp_gate
    r_open_bonus = 3.0 * open_drawer_bonus_cabinet_style(env) * phase
    r_open_stage = 0.8 * multi_stage_open_drawer_cabinet_style(env) * phase
    r_phase = 0.5 * phase
    r_grasp_gate = 0.8 * grasp_gate
    r_grasp_close = 0.5 * grasp_handle_bonus(env)

    # Keep same safety regularizers as world IK task.
    r_reverse = -0.8 * hk_mdp.base_reverse_velocity_penalty(env)
    r_yaw_w = -0.10 * hk_mdp.base_yaw_rate_w_sq(env)
    r_lin_d = -0.03 * hk_mdp.diff_drive_linear_command_delta_sq(env)
    r_vel_d = -0.015 * hk_mdp.base_lin_vel_xy_delta_sq(env)
    r_self_geom = -1.0 * hk_mdp.gripper_chassis_proximity_penalty_sq(env)
    r_orientation = -2.5 * hk_mdp.orientation_l2(env)
    r_ang_vel_xy = -0.6 * hk_mdp.ang_vel_xy_l2(env)
    r_torques = -1e-8 * hk_mdp.joint_torques_l2(env)
    r_joint_acc = -1e-9 * hk_mdp.joint_acc_l2(env)
    r_action_rate = -0.003 * hk_mdp.action_rate_l2(env)
    # Cabinet-style joint velocity regularization.
    r_joint_vel = -1e-4 * torch.sum(torch.square(env.scene["robot"].data.joint_vel), dim=1)
    r_joint_lim = -3.0 * hk_mdp.joint_pos_limits(env, margin=0.0)

    # Per-step debug metrics in extras/log for train & play dashboards.
    if "log" not in env.extras:
        env.extras["log"] = {}
    log = env.extras["log"]
    with torch.no_grad():
        cmd_term: HuskyCabinetEeGoalCommand = env.command_manager.get_term("ee_goal")
        ee_to_handle = torch.norm(
            cmd_term.handle_pos_w - env.scene["robot"].data.body_pos_w[:, env.scene["robot"].find_bodies("gripper_link")[0][0]],
            dim=1,
        )
        align_ee_raw = align_ee_to_handle(env)
        align_grasp_raw = align_grasp_around_handle(env)
        quat_err = math_utils.quat_box_minus(cmd_term.handle_quat_w, env.scene["robot"].data.body_quat_w[:, env.scene["robot"].find_bodies("gripper_link")[0][0]])
        quat_err_deg = torch.norm(quat_err, dim=1) * (180.0 / torch.pi)

        log["metric/grasp_gate"] = float(grasp_gate.mean().item())
        log["metric/align_ee"] = float(align_ee_raw.mean().item())
        log["metric/align_grasp"] = float(align_grasp_raw.mean().item())
        log["metric/ee_to_handle"] = float(ee_to_handle.mean().item())
        log["metric/quat_err_deg"] = float(quat_err_deg.mean().item())

        # "Initial moments" diagnostic window: first 30 env-steps after reset.
        init_mask = env.episode_length_buf < 30
        if bool(torch.any(init_mask)):
            log["metric/init/align_ee"] = float(align_ee_raw[init_mask].mean().item())
            log["metric/init/align_grasp"] = float(align_grasp_raw[init_mask].mean().item())
            log["metric/init/ee_to_handle"] = float(ee_to_handle[init_mask].mean().item())
            log["metric/init/quat_err_deg"] = float(quat_err_deg[init_mask].mean().item())
        else:
            log["metric/init/align_ee"] = float(align_ee_raw.mean().item())
            log["metric/init/align_grasp"] = float(align_grasp_raw.mean().item())
            log["metric/init/ee_to_handle"] = float(ee_to_handle.mean().item())
            log["metric/init/quat_err_deg"] = float(quat_err_deg.mean().item())

    r = (
        r_ee
        + r_align_ee
        + r_approach_gripper
        + r_align_grasp
        + r_base
        + r_heading
        + r_open
        + r_open_bonus
        + r_open_stage
        + r_phase
        + r_grasp_gate
        + r_grasp_close
        + r_reverse
        + r_yaw_w
        + r_lin_d
        + r_vel_d
        + r_self_geom
        + r_orientation
        + r_ang_vel_xy
        + r_torques
        + r_joint_acc
        + r_action_rate
        + r_joint_vel
        + r_joint_lim
    )
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    r = torch.clamp(r, min=0.0)

    terminated = env.termination_manager.terminated
    time_out = env.termination_manager.time_outs
    r += -5.0 * (terminated & ~time_out).to(dtype=r.dtype)
    return r
