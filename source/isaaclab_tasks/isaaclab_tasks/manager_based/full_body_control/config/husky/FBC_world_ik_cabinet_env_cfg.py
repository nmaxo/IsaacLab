"""Husky UR5 cabinet task: approach drawer handle + open with IK-assisted arm control."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.full_body_control.mdp as mdp
from isaaclab_tasks.manager_based.full_body_control.mdp import husky_ik_cabinet_mdp as hkc_mdp
from isaaclab_tasks.manager_based.full_body_control.mdp import husky_ik_mdp as hk_mdp
from isaaclab_tasks.manager_based.full_body_control.mdp.custom_mdp import DiffDriveVelocityActionCfg
from isaaclab_assets.robots import ur5_husky

UR5M_CFG = ur5_husky.UR5M_CFG
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class HuskyWorldIkCabinetSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.6,
                restitution=0.0,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = UR5M_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.20, 0.0, 0.4),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # Visualize gripper-link frame (orientation + position) in world.
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FINAL_HUSKY/husky_with_sensors/base_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/HuskyEeFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/FINAL_HUSKY/ur5_api/ur5/gripper_link",
                name="gripper_link",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(0.5, 0.5, -0.5, -0.5),
                ),
            ),
        ],
    )

    # Handle frame visualizer (same idea as cabinet task): this is what draws the handle frame in the scene.
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/HuskyCabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.5, 0.5, -0.5, -0.5),
                ),
            ),
        ],
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class HuskyWorldIkCabinetEventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Fixed spawn pose: shelf is directly in front of the robot each episode.
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        # Keep default nominal pose at reset (no randomization).
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reset_cabinet_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "cabinet",
                joint_names=["door_left_joint", "door_right_joint", "drawer_bottom_joint", "drawer_top_joint"],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_task_goal = EventTerm(
        func=hkc_mdp.reset_cabinet_ee_goal,
        mode="reset",
        params={"command_name": "ee_goal"},
    )


@configclass
class HuskyWorldIkCabinetActionsCfg:
    vel_actions = DiffDriveVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ],
        wheel_radius=0.17775,
        wheel_base=0.5708,
        max_linear_speed=2.0,
        max_angular_speed=5.0,
        scale=1.0,
        linear_velocity_sign=-1.0,
    )
    arm_ik_delta = hk_mdp.HuskyArmIkDeltaJointPositionActionCfg(
        asset_name="robot",
        ee_body_name="gripper_link",
        arm_joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        action_scale=0.3,
        dls_lambda=0.1,
        ik_pos_weight=1.0,
        ik_rot_weight=0.45,
        command_name="ee_goal",
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"],
        # Robotiq-85 on this Husky USD: near 0.0 is open-ish, larger is closed.
        open_command_expr={"robotiq_85_.*_knuckle_joint": 0.0},
        close_command_expr={"robotiq_85_.*_knuckle_joint": 0.8},
    )


@configclass
class HuskyWorldIkCabinetCommandsCfg:
    ee_goal = hkc_mdp.HuskyCabinetEeGoalCommandCfg(
        asset_name="robot",
        ee_body_name="gripper_link",
        cabinet_asset_name="cabinet",
        handle_body_name="drawer_handle_top",
        local_axis_z_offset=0.3,
        approach_offset_xyz=(-0.12, 0.0, 0.06),
        ee_target_quat_in_handle=(0.298426121, 0.040156964, 0.952406585, 0.047446080),
        open_pull_direction_xyz=(1.0, 0.0, 0.0),
        open_pull_distance=0.25,
        switch_distance=0.10,
        # Use frame visualizers (ee_frame + cabinet_frame) instead of sphere markers.
        debug_vis=False,
        visualize_env_index=-1,
    )


@configclass
class HuskyWorldIkCabinetObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.02, n_max=0.02))
        gripper_knuckle_pos = ObsTerm(
            func=hkc_mdp.gripper_knuckle_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"],
                )
            },
        )
        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )

        handle_in_base = ObsTerm(func=hkc_mdp.cabinet_handle_in_base_frame)
        handle_distance_xy = ObsTerm(func=hkc_mdp.cabinet_handle_distance_xy)
        phase = ObsTerm(func=hkc_mdp.cabinet_phase)
        open_progress = ObsTerm(
            func=hkc_mdp.cabinet_open_progress,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )

        local_gripper_pos = ObsTerm(func=hk_mdp.local_gripper_pos)
        curr_ee_goal_local = ObsTerm(func=hk_mdp.curr_ee_goal_cart)
        gripper_to_handle = ObsTerm(func=hkc_mdp.gripper_to_cabinet_handle)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HuskyWorldIkCabinetRewardsCfg:
    total = RewTerm(func=hkc_mdp.husky_world_ik_cabinet_total_reward, weight=1.0)


@configclass
class HuskyWorldIkCabinetTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success_open = DoneTerm(
        func=hkc_mdp.cabinet_open_success,
        params={"threshold": 0.23, "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )
    base_flipped = DoneTerm(
        func=hk_mdp.base_flipped,
        params={"threshold": 0.5},
    )
    bad_state = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.2, "asset_cfg": SceneEntityCfg("robot")},
    )
    too_high = DoneTerm(
        func=hk_mdp.root_height_above_maximum,
        params={"maximum_height": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class HuskyWorldIkCabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Husky cabinet environment: base approach + IK-assisted opening."""

    scene: HuskyWorldIkCabinetSceneCfg = HuskyWorldIkCabinetSceneCfg(num_envs=2048, env_spacing=12.0)
    actions: HuskyWorldIkCabinetActionsCfg = HuskyWorldIkCabinetActionsCfg()
    observations: HuskyWorldIkCabinetObservationsCfg = HuskyWorldIkCabinetObservationsCfg()
    events: HuskyWorldIkCabinetEventCfg = HuskyWorldIkCabinetEventCfg()
    commands: HuskyWorldIkCabinetCommandsCfg = HuskyWorldIkCabinetCommandsCfg()
    rewards: HuskyWorldIkCabinetRewardsCfg = HuskyWorldIkCabinetRewardsCfg()
    terminations: HuskyWorldIkCabinetTerminationsCfg = HuskyWorldIkCabinetTerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 60.0

        self.viewer.eye = (8.0, 8.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class HuskyWorldIkCabinetEnvCfg_PLAY(HuskyWorldIkCabinetEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
