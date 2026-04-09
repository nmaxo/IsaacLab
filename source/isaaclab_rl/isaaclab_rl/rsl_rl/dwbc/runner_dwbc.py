"""On-policy runner for DWBC-style dual-reward PPO."""

from __future__ import annotations

import os
import statistics
import time

import torch
from collections import deque

from .actor_critic_dwbc import ActorCriticDWBC
from .ppo_dwbc import PPO_DWBC
from .wrapper_dwbc import DwbcVecEnvWrapper


class DwbcOnPolicyRunner:
    """Runner for dual-head actor/critic PPO with leg/arm reward streams."""

    def __init__(self, env: DwbcVecEnvWrapper, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        obs_prop, _ = self.env.reset()
        obs_prop = obs_prop.to(self.device)

        num_prop = obs_prop.shape[-1]
        num_actions = self.env.num_actions

        pol_cfg = dict(self.policy_cfg)
        pol_cfg.pop("class_name", None)
        for k in ("num_priv", "priv_encoder_dims", "history_len"):
            pol_cfg.pop(k, None)

        self.actor_critic = ActorCriticDWBC(
            num_prop=num_prop,
            **pol_cfg,
        ).to(self.device)

        alg_cfg = dict(self.alg_cfg)
        alg_cfg.pop("class_name", None)

        self.alg = PPO_DWBC(
            actor_critic=self.actor_critic,
            device=self.device,
            **alg_cfg,
        )

        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            num_prop=num_prop,
            action_shape=[num_actions],
        )

        self.num_prop = num_prop

        self.log_dir = log_dir
        self.writer = None
        self.logger_type = "tensorboard"
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = []

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs_prop = self.env.get_observations().to(self.device)

        self.alg.train_mode()

        ep_infos: list = []
        rewbuffer: deque = deque(maxlen=100)
        armrewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_arm_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mixing_ratio = 0.0

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_prop)
                    obs_prop, leg_rew, arm_rew, dones, infos = self.env.step(actions.to(self.env.device))
                    obs_prop = obs_prop.to(self.device)
                    leg_rew, arm_rew, dones = leg_rew.to(self.device), arm_rew.to(self.device), dones.to(self.device)

                    self.alg.process_env_step(leg_rew, arm_rew, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += leg_rew
                        cur_arm_reward_sum += arm_rew
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        armrewbuffer.extend(cur_arm_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_arm_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                self.alg.compute_returns(obs_prop)

            loss_info = self.alg.update()
            mean_value_loss = loss_info["value_loss"]
            mean_surrogate_loss = loss_info["surrogate_loss"]
            mixing_ratio = loss_info["mixing_ratio"]

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 40):
        collection_size = self.num_steps_per_env * self.env.num_envs
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        leg_mean_std = self.alg.actor_critic.std[: self.alg.actor_critic.num_leg_actions].mean()
        arm_mean_std = self.alg.actor_critic.std[self.alg.actor_critic.num_leg_actions :].mean()
        fps = int(collection_size / max(iteration_time, 1e-6))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/mixing_ratio", locs["mixing_ratio"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/leg_mean_noise_std", leg_mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/arm_mean_noise_std", arm_mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_leg_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_arm_reward", statistics.mean(locs["armrewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f""" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m \n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mixing ratio:':>{pad}} {locs['mixing_ratio']:.4f}\n"""
                f"""{'Leg mean noise std:':>{pad}} {leg_mean_std.item():.2f}\n"""
                f"""{'Arm mean noise std:':>{pad}} {arm_mean_std.item():.2f}\n"""
                f"""{'Mean leg reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean arm reward:':>{pad}} {statistics.mean(locs['armrewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f""" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m \n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / max(locs['it'] - locs['start_iter'] + 1, 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        state = loaded_dict["model_state_dict"]
        missing, unexpected = self.alg.actor_critic.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[DwbcOnPolicyRunner] Ignored unexpected keys in checkpoint: {len(unexpected)} keys")
        if missing:
            print(f"[DwbcOnPolicyRunner] Missing keys (new weights random-init): {missing}")
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)

        def policy(obs_prop):
            return self.alg.actor_critic.act_inference(obs_prop)

        return policy

    def _prepare_logging_writer(self):
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            else:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
