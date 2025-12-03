# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--enable_logging", action="store_true", default=False, help="Enable TensorBoard logging during play.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for loading checkpoint
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # ==================== СОЗДАНИЕ ДИРЕКТОРИИ ДЛЯ PLAY ЛОГОВ ====================
    # specify directory for logging play runs
    play_log_root = os.path.join("logs", "play", agent_cfg.experiment_name)
    play_log_root = os.path.abspath(play_log_root)
    
    # create timestamped directory for this play session
    play_log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        play_log_dir += f"_{agent_cfg.run_name}_play"
    else:
        play_log_dir += "_play"
    play_log_dir = os.path.join(play_log_root, play_log_dir)
    
    os.makedirs(play_log_dir, exist_ok=True)
    print(f"[INFO] Play logs will be saved to: {play_log_dir}")
    
    # Initialize TensorBoard writer if logging is enabled
    writer = None
    if args_cli.enable_logging:
        writer = SummaryWriter(log_dir=os.path.join(play_log_dir, "summaries"))
        print(f"[INFO] TensorBoard logging enabled at: {os.path.join(play_log_dir, 'summaries')}")
    # ============================================================================

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = play_log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(play_log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(play_log_dir, "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # Episode tracking variables
    episode_reward = torch.zeros(env.num_envs, device=env.device)
    episode_length = torch.zeros(env.num_envs, device=env.device)
    episode_count = 0
    
    # Tracking for custom metrics
    total_episodes = 0
    successful_episodes = 0
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
#             command_manager = env.unwrapped.command_manager
#             command_term = env.unwrapped.command_manager.get_term("ee_pose")

# # Теперь можно логировать метрики
#             pos_error = command_term.metrics["position_error"]
#             ori_error = command_term.metrics["orientation_error"]
#             print(pos_error,ori_error)
#             print(dir(command_manager))
            # ==================== ЛОГИРОВАНИЕ METRIC И SUCCESS RATE ====================
                # Получаем CommandTerm (например, ee_pose)
        
            # ==================== ЛОГИРОВАНИЕ МЕТРИК ====================
            if writer is not None:

                # Accumulate episode stats
                episode_reward += rewards
                episode_length += 1
                
                # ========== 1. БАЗОВЫЕ МЕТРИКИ ==========
                writer.add_scalar("play/step_reward_mean", rewards.mean().item(), timestep)
                writer.add_scalar("play/step_reward_std", rewards.std().item(), timestep)
                writer.add_scalar("play/action_mean", actions.mean().item(), timestep)
                writer.add_scalar("play/action_std", actions.std().item(), timestep)
                writer.add_scalar("play/action_min", actions.min().item(), timestep)
                writer.add_scalar("play/action_max", actions.max().item(), timestep)
                
                # ========== 2. МЕТРИКИ ИЗ INFOS (как при тренировке) ==========
                # Эти метрики обычно предоставляет окружение
                if "extras" in infos:
                    extras = infos["extras"]
                    print(extras)
                    
                    # Логируем все метрики из extras
                    for key, value in extras.items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() == 1:
                                # Скалярное значение
                                writer.add_scalar(f"play/extras/{key}", value.item(), timestep)
                            else:
                                # Вектор значений - логируем статистики
                                writer.add_scalar(f"play/extras/{key}_mean", value.mean().item(), timestep)
                                writer.add_scalar(f"play/extras/{key}_std", value.std().item(), timestep)
                                writer.add_scalar(f"play/extras/{key}_min", value.min().item(), timestep)
                                writer.add_scalar(f"play/extras/{key}_max", value.max().item(), timestep)
                
                # ========== 3. МЕТРИКИ ПО ЗАВЕРШЕНИЮ ЭПИЗОДОВ ==========
                if dones.any():
                    # Индексы завершенных эпизодов
                    done_indices = dones.nonzero(as_tuple=False).flatten()
                    
                    for idx in done_indices:
                        total_episodes += 1
                        
                        # Логируем награду и длину эпизода
                        ep_reward = episode_reward[idx].item()
                        ep_length = episode_length[idx].item()
                        
                        writer.add_scalar("play/episode_reward", ep_reward, total_episodes)
                        writer.add_scalar("play/episode_length", ep_length, total_episodes)
                        
                        # ========== 4. КАСТОМНЫЕ МЕТРИКИ ==========
                        # Пример: считаем успешность (если награда выше порога)
                        success_threshold = 100.0  # Настройте под вашу задачу
                        if ep_reward > success_threshold:
                            successful_episodes += 1
                        
                        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
                        writer.add_scalar("play/success_rate", success_rate, total_episodes)
                        
                        # Пример: метрики из episode_infos если они есть
                        if "episode" in infos:
                            episode_info = infos["episode"]
                            if isinstance(episode_info, dict):
                                for key, value in episode_info.items():
                                    if torch.is_tensor(value):
                                        writer.add_scalar(f"play/episode/{key}", value[idx].item(), total_episodes)
                        
                        # Пример: кастомная метрика - средняя награда за последние N эпизодов
                        if total_episodes % 10 == 0:  # Каждые 10 эпизодов
                            writer.add_scalar("play/episode_reward_avg_last_10", ep_reward, total_episodes)
                    
                    # Сбросить счетчики для завершенных эпизодов
                    episode_reward[done_indices] = 0.0
                    episode_length[done_indices] = 0
                
                # ========== 5. МЕТРИКИ ОКРУЖЕНИЯ (если доступны) ==========
                # Доступ к внутренним переменным окружения
                if hasattr(env.unwrapped, "scene"):
                    scene = env.unwrapped.scene
                    
                    # Пример: логируем позиции/скорости роботов (для Manager-based env)
                    if hasattr(scene, "robot") and hasattr(scene.robot, "data"):
                        robot_data = scene.robot.data
                        
                        # Позиция
                        if hasattr(robot_data, "root_pos_w"):
                            pos = robot_data.root_pos_w
                            writer.add_scalar("play/robot/pos_x_mean", pos[:, 0].mean().item(), timestep)
                            writer.add_scalar("play/robot/pos_y_mean", pos[:, 1].mean().item(), timestep)
                            writer.add_scalar("play/robot/pos_z_mean", pos[:, 2].mean().item(), timestep)
                        
                        # Скорость
                        if hasattr(robot_data, "root_lin_vel_w"):
                            vel = robot_data.root_lin_vel_w
                            speed = torch.norm(vel, dim=1)
                            writer.add_scalar("play/robot/speed_mean", speed.mean().item(), timestep)
                            writer.add_scalar("play/robot/speed_max", speed.max().item(), timestep)
                        
                        # Углы сочленений
                        if hasattr(robot_data, "joint_pos"):
                            joint_pos = robot_data.joint_pos
                            writer.add_scalar("play/robot/joint_pos_mean", joint_pos.mean().item(), timestep)
                            writer.add_scalar("play/robot/joint_pos_std", joint_pos.std().item(), timestep)
                
                # ========== 6. КАСТОМНЫЕ ВЫЧИСЛЕНИЯ ==========
                # Пример: энтропия действий (для непрерывных действий)
                action_entropy = -0.5 * torch.log(2 * torch.pi * actions.var(dim=0)).mean()
                writer.add_scalar("play/action_entropy", action_entropy.item(), timestep)
                
                # Пример: процент клиппинга действий
                clipped_actions = torch.clamp(actions, -1.0, 1.0)
                clipping_ratio = (actions != clipped_actions).float().mean()
                writer.add_scalar("play/action_clipping_ratio", clipping_ratio.item(), timestep)
                
                # Пример: среднее абсолютное значение действий
                writer.add_scalar("play/action_abs_mean", actions.abs().mean().item(), timestep)
            # ============================================================
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close writer if it was created
    if writer is not None:
        writer.close()
        print(f"[INFO] TensorBoard logs saved. View with: tensorboard --logdir {play_log_dir}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


# ==================== ПРИМЕРЫ ДОПОЛНИТЕЛЬНЫХ МЕТРИК ====================
# Вот еще примеры метрик, которые можно добавить в секцию логирования:

# 7. МЕТРИКИ ДЛЯ СПЕЦИФИЧНЫХ ЗАДАЧ:
# 
# Для задач навигации:
# - writer.add_scalar("play/distance_to_goal", distance.mean().item(), timestep)
# - writer.add_scalar("play/goal_reached_ratio", goal_reached.float().mean().item(), timestep)
#
# Для задач манипуляции:
# - writer.add_scalar("play/object_height", object_pos[:, 2].mean().item(), timestep)
# - writer.add_scalar("play/grasp_success", grasp_success.float().mean().item(), timestep)
# - writer.add_scalar("play/end_effector_to_object", distance_ee_obj.mean().item(), timestep)
#
# Для задач локомоции:
# - writer.add_scalar("play/forward_velocity", forward_vel.mean().item(), timestep)
# - writer.add_scalar("play/energy_consumption", power.sum().item(), timestep)
# - writer.add_scalar("play/feet_air_time", air_time.mean().item(), timestep)
# - writer.add_scalar("play/stumble_count", stumbles.sum().item(), timestep)

# 8. МЕТРИКИ РАСПРЕДЕЛЕНИЙ (гистограммы):
#
# if timestep % 100 == 0:  # Логируем каждые 100 шагов
#     writer.add_histogram("play/actions_distribution", actions, timestep)
#     writer.add_histogram("play/rewards_distribution", rewards, timestep)
#     writer.add_histogram("play/observations_distribution", obs, timestep)

# 9. МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:
#
# inference_time = time.time() - start_time
# writer.add_scalar("play/inference_fps", 1.0 / inference_time, timestep)
# writer.add_scalar("play/inference_time_ms", inference_time * 1000, timestep)

# 10. КУМУЛЯТИВНЫЕ МЕТРИКИ:
#
# cumulative_reward += rewards.sum().item()
# writer.add_scalar("play/cumulative_reward", cumulative_reward, timestep)
#
# cumulative_episodes = total_episodes
# writer.add_scalar("play/total_episodes", cumulative_episodes, timestep)

# 11. МЕТРИКИ СРАВНЕНИЯ С BASELINE:
#
# baseline_reward = 50.0  # Определите ваш baseline
# performance_ratio = ep_reward / baseline_reward
# writer.add_scalar("play/performance_vs_baseline", performance_ratio, timestep)

# 12. МЕТРИКИ ИЗ OBSERVATION SPACE:
#
# if hasattr(env.unwrapped, "observation_manager"):
#     obs_manager = env.unwrapped.observation_manager
#     for obs_term_name in obs_manager.active_terms:
#         obs_term = obs_manager.get_term(obs_term_name)
#         if isinstance(obs_term, torch.Tensor):
#             writer.add_scalar(f"play/obs/{obs_term_name}_mean", obs_term.mean().item(), timestep)

# 13. СТАТИСТИЧЕСКИЕ МЕТРИКИ:
#
# # Скользящее среднее наград
# if len(reward_buffer) > 100:
#     writer.add_scalar("play/reward_moving_avg", np.mean(reward_buffer[-100:]), timestep)
#
# # Дисперсия и квантили
# writer.add_scalar("play/reward_variance", rewards.var().item(), timestep)
# writer.add_scalar("play/reward_median", rewards.median().item(), timestep)
# writer.add_scalar("play/reward_25th_percentile", torch.quantile(rewards, 0.25).item(), timestep)
# writer.add_scalar("play/reward_75th_percentile", torch.quantile(rewards, 0.75).item(), timestep)

# 14. МЕТРИКИ СТАБИЛЬНОСТИ ПОЛИТИКИ:
#
# # Изменение действий между шагами
# if previous_actions is not None:
#     action_change = (actions - previous_actions).abs().mean()
#     writer.add_scalar("play/action_smoothness", action_change.item(), timestep)
# previous_actions = actions.clone()

# 15. УСЛОВНЫЕ МЕТРИКИ:
#
# # Логируем только когда происходит что-то интересное
# if rewards.max() > 10.0:
#     writer.add_scalar("play/high_reward_event", rewards.max().item(), timestep)
#
# if dones.sum() > env.num_envs * 0.5:
#     writer.add_scalar("play/mass_termination_event", dones.sum().item(), timestep)
# ========================================================================