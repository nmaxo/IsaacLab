# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import CurriculumTermCfg as CurrTerm

import isaaclab_tasks.manager_based.full_body_control.mdp as mdp
from isaaclab_tasks.manager_based.full_body_control.mdp.custom_mdp import DiffDriveVelocityActionCfg
from isaaclab_assets.robots import ur5_husky

UR5M_CFG = ur5_husky.UR5M_CFG

@configclass
class FBCSceneCfg(InteractiveSceneCfg):
    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: ArticulationCfg = UR5M_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (0.0, 0.0)
            },
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
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    vel_actions: ActionTerm = DiffDriveVelocityActionCfg(
        asset_name="robot",
        joint_names=["front_left_wheel_joint", "front_right_wheel_joint",
                     "rear_left_wheel_joint", "rear_right_wheel_joint"],
        wheel_radius=0.17775,
        wheel_base=0.5708,
        max_linear_speed=3.0,
        max_angular_speed=2.5,
        scale=1.0,
        linear_velocity_sign=-1.0,  # ось X базы в USD Husky смотрит назад
    )
    arm_actions: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        scale=0.4,
        use_default_offset=False
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"}
        )
        # distance = ObsTerm(func = mdp.position_command_error,params={
        #     "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
        #     "command_name": "ee_pose"
        # })
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     noise=Unoise(n_min=-0.01, n_max=0.01)
        # )
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP - оптимизированная структура без конфликтов."""
    
    # ==========================================
    # ОСНОВНЫЕ НАГРАДЫ ЗА ВЫПОЛНЕНИЕ ЗАДАЧИ
    # ==========================================
    
    # Главная награда: экспоненциальная награда за точность позиции
    # Использует tanh для smooth gradient и избегает дублирования с L2
    position_tracking_global = RewTerm(
        func=mdp.position_command_error,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose"
        }
    )
    position_tracking_1 = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "std": 0.12,  # Более агрессивный std для четкого gradient
            "command_name": "ee_pose"
        },
    )
    position_tracking_2 = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "std": 0.3,  # Более агрессивный std для четкого gradient
            "command_name": "ee_pose"
        },
    )
    
    # Награда за ориентацию (отдельно, не конфликтует с позицией)
    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-2.0,  # Негативная награда (штраф) за ошибку ориентации
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose"
        },
    )
    
    # Бонус за успешное достижение цели
    # Даётся только когда робот действительно у цели
    goal_reached_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=10.0,  # Большой бонус за успех
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
            "position_threshold": 0.08,
            "orientation_threshold": 0.3,
        }
    )
    
    # ==========================================
    # НАГРАДЫ ЗА ЭФФЕКТИВНОЕ ПОВЕДЕНИЕ
    # ==========================================
    
    # Награда за стабильность около цели
    # Мотивирует остановиться, когда достиг цели
    stability_bonus = RewTerm(
        func=mdp.stability_reward,
        weight=3.0,
        params={
            "command_name": "ee_pose",
            "position_threshold": 0.15,
            "orientation_threshold": 0.4,
            "lin_velocity_threshold": 0.15,
            "ang_velocity_threshold": 0.3
        }
    )
    
    # Прогрессивная награда: штраф за скорость только вблизи цели
    # Далеко от цели - можно двигаться быстро, близко - нужно замедлиться
    approach_penalty = RewTerm(
        func=mdp.distance_based_velocity_penalty,
        weight=-0.3,
        params={"command_name": "ee_pose"}
    )
    
    # ==========================================
    # ШТРАФЫ ЗА НЕЭФФЕКТИВНОСТЬ
    # ==========================================
    
    # Мягкий штраф за время (мотивирует не затягивать)
    time_penalty = RewTerm(
        func=mdp.time_penalty,
        weight=-0.01,
        params={}
    )
    
    # ==========================================
    # РЕГУЛЯРИЗАЦИЯ (МИНИМАЛЬНЫЕ ШТРАФЫ)
    # ==========================================
    
    # Штраф за резкие изменения действий
    # Делаем очень маленьким, чтобы не мешать основной задаче
    action_smoothness = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.008
    )
    
    # Штраф за высокие скорости суставов
    # Предотвращает слишком резкие движения
    joint_velocity_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc_penalty = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.000008,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    
    # Штраф за большие крутящие моменты
    # Защита оборудования и энергоэффективность
    torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-6,
        params={}
    )
    
    # ==========================================
    # ДОПОЛНИТЕЛЬНЫЕ НАГРАДЫ (ОПЦИОНАЛЬНО)
    # ==========================================
    
    # Если хотите добавить награду за движение в правильном направлении:
    # progress_reward = RewTerm(
    #     func=mdp.progress_to_goal,
    #     weight=0.5,
    #     params={"command_name": "ee_pose"}
    # )
    
    # Если хотите штрафовать за столкновения:
    # collision_penalty = RewTerm(
    #     func=mdp.collision_penalty,
    #     weight=-1.0,
    #     params={}
    # )


# ==========================================
# ПОЯСНЕНИЕ СТРУКТУРЫ НАГРАД
# ==========================================
"""
Приоритеты наград (от большего к меньшему весу):

1. goal_reached_bonus (10.0) - самая важная награда
   ↓ Мотивирует успешное завершение задачи

2. position_tracking (5.0) - основная награда
   ↓ Непрерывно направляет к цели

3. orientation_tracking (-2.0) - вторичная награда
   ↓ Обеспечивает правильную ориентацию

4. stability_bonus (1.0) - награда за качество
   ↓ Учит останавливаться у цели

5. approach_penalty (-0.3) - штраф за неаккуратность
   ↓ Учит замедляться при приближении

6. time_penalty (-0.01) - мотивация эффективности
   ↓ Не даёт медлить

7. Регуляризация (< 0.001) - минимальные штрафы
   ↓ Не мешают основной задаче, но улучшают качество

Ключевые принципы:
- Нет дублирования: position используется только через tanh
- Четкая иерархия: главные награды на 2+ порядка больше регуляризации
- Баланс: положительные награды (16.0) > отрицательные (~2.3)
- Масштабирование: все веса подобраны так, чтобы не заглушать друг друга
"""

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    ee_pose = mdp.UniformPoseFixedCommandCfg(
        asset_name="robot",
        body_name='gripper_link',
        resampling_time_range=(10.0, 18.0),  # Curriculum: начинаем с более частой смены целей
        debug_vis=True,
        ranges=mdp.UniformPoseFixedCommandCfg.Ranges(
            pos_x=(-1.5, 1.5),
            pos_y=(-2.2, 2.2),
            pos_z=(0.6, 0.95),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    goal_reached = DoneTerm(
        func=mdp.goal_reached_bool,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "command_name": "ee_pose",
            "position_threshold": 0.05,
            "orientation_threshold": 0.2,
        }
    )

@configclass
class FBCEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the full body control environment."""
    
    # Scene settings
    scene: FBCSceneCfg = FBCSceneCfg(num_envs=4096, env_spacing=10.5)
    
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 18.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0

class FBCEnvCfg_PLAY(FBCEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False