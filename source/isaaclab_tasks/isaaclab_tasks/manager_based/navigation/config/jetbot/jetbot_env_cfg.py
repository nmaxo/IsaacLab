# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_tasks.manager_based.navigation.mdp.custom_mdp import DiffDriveVelocityAction, DiffDriveVelocityActionCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# Jetbot Configuration
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

@configclass
class JetbotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Jetbot navigation."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robot
    robot: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
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


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    vel_actions: ActionTerm = DiffDriveVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        wheel_radius=0.0325,  # Jetbot wheel radius: ~32.5mm
        wheel_base=0.1143,    # Jetbot wheelbase: ~114.3mm (distance between wheels)
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.8,
        params={"std": 1.0, "command_name": "pose_command"},  # Меньше std для маленького робота
    )
    
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.1, "command_name": "pose_command"},
    )
    
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )
    
    stability_at_goal = RewTerm(
        func=mdp.stability_reward,
        weight=0.3,
        params={
            "command_name": "pose_command",
            "position_threshold": 0.05,      # Меньше для маленького робота
            "orientation_threshold": 0.1,
            "lin_velocity_threshold": 0.05  # Jetbot медленнее
            "ang_velocity_threshold": 0.2
        }
    )
    
    # vel_pen = RewTerm(
    #     func=mdp.distance_based_velocity_penalty,
    #     weight=-0.15,
    #     params={"command_name": "pose_command"}
    # )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(12.0, 12.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-2.0, 2.0),      # Меньший диапазон для маленького робота
            pos_y=(-2.0, 2.0),      # Jetbot не нужен большой полигон
            heading=(-math.pi, math.pi)
        ),
    )

    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class JetbotNavEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Jetbot navigation environment."""

    # environment settings
    scene: JetbotSceneCfg = JetbotSceneCfg(num_envs=4096, env_spacing=5.0)  # Меньше spacing
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        self.viewer.eye = (2.5, 2.5, 2.5)  # Ближе камера для маленького робота
        
        # simulation settings
        self.sim.dt = 1.0 / 60.0


class JetbotNavEnvCfg_PLAY(JetbotNavEnvCfg):
    """Configuration for playing with trained Jetbot policy."""
    
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0  # Еще меньше для визуализации
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False