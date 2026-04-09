"""Multi-critic PPO implementation for loco-manipulation.

Based on: "Multi-critic Learning for Whole-body End-effector Twist Tracking"
(Vijayan et al., 2025, https://arxiv.org/abs/2507.08656)
"""

from .actor_multi_critic import ActorMultiCritic
from .multi_critic_storage import MultiCriticRolloutStorage
from .multi_critic_ppo import MultiCriticPPO
from .multi_critic_runner import MultiCriticOnPolicyRunner
from .multi_critic_wrapper import MultiCriticVecEnvWrapper
from .multi_critic_cfg import RslRlMultiCriticPpoActorCriticCfg, RslRlMultiCriticPpoAlgorithmCfg, RslRlMultiCriticRunnerCfg
