"""Go2-ARX LeggedGym-style environment (Isaac Lab).

This environment mirrors `legged-robots-manipulation` (legged_gym-style) Go2-ARX:
- 18 actions (12 legs + 6 arm)
- 75 policy observations
- 75 + 187 privileged observations (height scan), for asymmetric actor-critic
- EE-goal tracking with simple trajectory interpolation and collision checking
- Arm action uses joint-position delta plus DLS IK correction toward the current EE goal
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.go2_arx import GO2_ARX_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from .mdp import go2_arx_leggedgym as lg_mdp


@configclass
class Go2ArxLeggedGymSceneCfg(InteractiveSceneCfg):
    """Scene: rough terrains + ray height scanner + contact sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = GO2_ARX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),  # 17 x 11 = 187
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class Go2ArxLeggedGymEventCfg:
    """Events: resets matching legged_gym defaults (tight ranges)."""

    # reset base pose (xy small, yaw uniform)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # reset legs and arm joints with scale around default
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
        },
    )

    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_joint.*"]),
        },
    )

    # initialize EE-goal state buffers on reset
    reset_ee_goal = EventTerm(
        func=lg_mdp.reset_ee_goal,
        mode="reset",
        params={
            "command_name": "ee_goal",
        },
    )


@configclass
class Go2ArxLeggedGymActionsCfg:
    """Actions: legs via joint-pos PD, arm via custom IK+delta joint pos."""

    joint_pos_legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,
        use_default_offset=True,
    )

    arm_ik_delta = lg_mdp.Go2ArxArmIkDeltaJointPositionActionCfg(
        asset_name="robot",
        ee_body_name="arm_link6",
        arm_joint_names=["arm_joint.*"],
        action_scale=1.0,
        dls_lambda=0.05,
        command_name="ee_goal",
    )


@configclass
class Go2ArxLeggedGymCommandsCfg:
    """Commands: velocity command + EE goal command."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 1.0),
            heading=None,
        ),
        debug_vis=False,
    )

    ee_goal = lg_mdp.Go2ArxEeGoalCommandCfg(
        asset_name="robot",
        ee_body_name="arm_link6",
        resampling_time_range=(0.8, 1.6),  # traj+hold combined; internal timing matches legged_gym ranges
        debug_vis=False,
        local_axis_z_offset=0.3,
        traj_time_range_s=(0.6, 1.2),
        hold_time_range_s=(0.2, 0.4),
        sphere_ranges=lg_mdp.Go2ArxEeGoalCommandCfg.SphereRanges(
            pos_l=(0.4, 0.8),
            pos_p=(-3.141592653589793 / 6, 3.141592653589793 / 3),
            pos_y=(-3.141592653589793 / 4, 3.141592653589793 / 4),
        ),
        collision_upper_limits=(0.3, 0.15, 0.05 - 0.165),
        collision_lower_limits=(-0.2, -0.15, -0.35 - 0.165),
        underground_limit=-0.57,
        num_collision_check_samples=10,
    )


@configclass
class Go2ArxLeggedGymObservationsCfg:
    """Observations: policy (75) and privileged (75+187)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=lg_mdp.scaled_velocity_commands)
        dof_err = ObsTerm(func=lg_mdp.dof_err_rel_default)
        dof_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))

        local_gripper_pos = ObsTerm(func=lg_mdp.local_gripper_pos)
        curr_ee_goal_cart = ObsTerm(func=lg_mdp.curr_ee_goal_cart)
        gripper_to_goal = ObsTerm(func=lg_mdp.gripper_to_goal)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        velocity_commands = ObsTerm(func=lg_mdp.scaled_velocity_commands)
        dof_err = ObsTerm(func=lg_mdp.dof_err_rel_default)
        dof_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)

        local_gripper_pos = ObsTerm(func=lg_mdp.local_gripper_pos)
        curr_ee_goal_cart = ObsTerm(func=lg_mdp.curr_ee_goal_cart)
        gripper_to_goal = ObsTerm(func=lg_mdp.gripper_to_goal)

        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 20.5},
            clip=(-1.0, 1.0),
            scale=5.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class Go2ArxLeggedGymRewardsCfg:
    """Single total reward term (matches legged_gym only-positive + termination-after-clip)."""

    total = RewTerm(func=lg_mdp.leggedgym_total_reward, weight=1.0)


@configclass
class Go2ArxLeggedGymTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    hip_contact = DoneTerm(
        func=lg_mdp.illegal_contact_hip,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip"]),
            "threshold": 1.0,
        },
    )


@configclass
class Go2ArxLeggedGymEnvCfg(ManagerBasedRLEnvCfg):
    """LeggedGym-style Go2-ARX (rough terrain)."""

    scene: Go2ArxLeggedGymSceneCfg = Go2ArxLeggedGymSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: Go2ArxLeggedGymActionsCfg = Go2ArxLeggedGymActionsCfg()
    observations: Go2ArxLeggedGymObservationsCfg = Go2ArxLeggedGymObservationsCfg()
    events: Go2ArxLeggedGymEventCfg = Go2ArxLeggedGymEventCfg()
    commands: Go2ArxLeggedGymCommandsCfg = Go2ArxLeggedGymCommandsCfg()
    rewards: Go2ArxLeggedGymRewardsCfg = Go2ArxLeggedGymRewardsCfg()
    terminations: Go2ArxLeggedGymTerminationsCfg = Go2ArxLeggedGymTerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 0.0025
        self.decimation = 8
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # Make sure the viewport covers a wider area of the env grid.
        # Otherwise you may only see markers on the terrain features near env 0.
        self.viewer.eye = (10.5, 10.5, 4.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # NOTE:
        # Isaac Lab RewardManager already multiplies each reward term by both:
        #   (term.weight) * dt
        # Therefore we must NOT pre-scale weights by dt here, otherwise we get dt^2.

        # Enable EE trajectory visualization during training as well.
        # Note: it will only be visible when running with viewer (i.e. not headless).
        self.commands.ee_goal.debug_vis = True
        # visualize for all envs (0..num_envs-1)
        self.commands.ee_goal.visualize_env_index = -1
        self.commands.ee_goal.track_points = 10


@configclass
class Go2ArxLeggedGymEnvCfg_PLAY(Go2ArxLeggedGymEnvCfg):
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
        # disable only non-essential randomization during play
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # already enabled in base env cfg
