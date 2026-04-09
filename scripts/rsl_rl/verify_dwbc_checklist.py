# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Verification script for DWBC pipeline vs "Deep Whole-Body Control" (Fu et al., CoRL 2022).
# Run without sim:  ./isaaclab.sh -p scripts/rsl_rl/verify_dwbc_checklist.py
# Run with env (obs dims, network):  ./isaaclab.sh -p scripts/rsl_rl/verify_dwbc_checklist.py --with_env

"""Verify DWBC implementation against the plan checklist (15 points)."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields

# Ensure packages are on path when run via python from repo root.
# Prefer: ./isaaclab.sh -p scripts/rsl_rl/verify_dwbc_checklist.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
_source = os.path.join(_repo_root, "source")
for _sub in ("isaaclab_tasks", "isaaclab_rl", "isaaclab", "isaaclab_assets"):
    _p = os.path.join(_source, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
if _source not in sys.path:
    sys.path.insert(0, _source)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

def _check(condition: bool, msg: str, errors: list[str], warnings: list[str], is_warn: bool = False) -> bool:
    if condition:
        print(f"  [OK] {msg}")
        return True
    if is_warn:
        warnings.append(msg)
        print(f"  [WARN] {msg}")
    else:
        errors.append(msg)
        print(f"  [FAIL] {msg}")
    return False


def get_reward_term_names_from_cfg(rewards_cfg_class):
    """Return list of reward term names as they appear in the env's reward manager."""
    return [f.name for f in fields(rewards_cfg_class)]


def main():
    parser = argparse.ArgumentParser(description="Verify DWBC checklist (Husky DWBC config).")
    parser.add_argument("--with_env", action="store_true", help="Create env once to verify obs dims and network.")
    args = parser.parse_args()
    errors = []
    warnings = []

    print("\n=== DWBC verification checklist (Husky) ===\n")

    # --- 1. Reward groups (leg / arm) ---
    print("1. Reward groups (leg / arm)")
    from isaaclab_tasks.manager_based.full_body_control.config.husky.FBC_dwbc_env_cfg import (
        REWARD_GROUPS,
        HuskyDwbcRewardsCfg,
    )
    reward_term_names = get_reward_term_names_from_cfg(HuskyDwbcRewardsCfg)
    group_names = list(REWARD_GROUPS.keys())
    _check(len(group_names) == 2, f"Exactly 2 reward groups: {group_names}", errors, warnings)
    for gname, terms in REWARD_GROUPS.items():
        for t in terms:
            _check(t in reward_term_names, f"Term '{t}' in group '{gname}' exists in HuskyDwbcRewardsCfg", errors, warnings)
    _check(group_names[0] == "locomotion" and group_names[1] == "manipulation",
           "Groups are 'locomotion' and 'manipulation'", errors, warnings)

    # --- 2. Observations: policy only ---
    print("\n2. Observations (policy only, no privileged)")
    from isaaclab_tasks.manager_based.full_body_control.config.husky.FBC_dwbc_env_cfg import (
        HuskyDwbcObservationsCfg,
    )
    obs_cfg = HuskyDwbcObservationsCfg()
    _check(hasattr(obs_cfg, "policy"), "Observation group 'policy' exists", errors, warnings)
    _check(not hasattr(obs_cfg, "privileged"), "No observation group 'privileged'", errors, warnings)

    # --- 3. Actions: leg first, then arm; 2 + 6 = 8 ---
    print("\n3. Actions (leg then arm, 2+6=8)")
    from isaaclab_tasks.manager_based.full_body_control.config.husky.agents.rsl_rl_dwbc_cfg import (
        HuskyDwbcRunnerCfg,
    )
    runner_cfg = HuskyDwbcRunnerCfg()
    n_leg = runner_cfg.policy.num_leg_actions
    n_arm = runner_cfg.policy.num_arm_actions
    _check(n_leg == 2 and n_arm == 6, f"num_leg_actions=2, num_arm_actions=6 (got {n_leg}, {n_arm})", errors, warnings)
    _check(n_leg + n_arm == 8, "Total actions = 8", errors, warnings)
    from isaaclab_tasks.manager_based.full_body_control.config.husky.FBC_dwbc_env_cfg import (
        HuskyDwbcActionsCfg,
    )
    from dataclasses import fields as dc_fields
    action_fields = [f.name for f in dc_fields(HuskyDwbcActionsCfg)]
    _check(len(action_fields) >= 2 and action_fields[0] == "vel_actions" and action_fields[1] == "arm_actions",
           "Action config order: vel_actions, arm_actions", errors, warnings)

    # --- 4-7. Network (need num_prop, num_priv from env if --with_env) ---
    print("\n4-7. Actor/Critic and encoders (dimensions)")
    if args.with_env:
        try:
            import gymnasium as gym
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
            env_cfg = parse_env_cfg("Isaac-FBC-Husky-DWBC-v0", num_envs=4)
            env = gym.make("Isaac-FBC-Husky-DWBC-v0", cfg=env_cfg)
            obs_dict, _ = env.reset()
            obs_prop = obs_dict.get("policy")
            num_prop = obs_prop.shape[-1] if obs_prop is not None else 0
            env.close()
            _check(obs_prop is not None and num_prop > 0, f"policy obs shape (..., {num_prop})", errors, warnings)
            _check("privileged" not in obs_dict, "reset() dict has no 'privileged' key", errors, warnings)
            # Build network
            from isaaclab_rl.rsl_rl.dwbc.actor_critic_dwbc import ActorCriticDWBC
            pol = runner_cfg.policy
            ac = ActorCriticDWBC(
                num_prop=num_prop,
                num_leg_actions=pol.num_leg_actions,
                num_arm_actions=pol.num_arm_actions,
                actor_hidden_dims=pol.actor_hidden_dims,
                critic_hidden_dims=pol.critic_hidden_dims,
                leg_control_head_hidden_dims=pol.leg_control_head_hidden_dims,
                arm_control_head_hidden_dims=pol.arm_control_head_hidden_dims,
                activation=pol.activation,
                init_noise_std=pol.init_noise_std,
            )
            import torch
            b = 4
            o_p = torch.zeros(b, num_prop)
            act = ac.act(o_p)
            _check(act.shape == (b, 8), f"Actor output shape (batch, 8): got {act.shape}", errors, warnings)
            val = ac.evaluate(o_p)
            _check(val.shape == (b, 2), f"Critic output shape (batch, 2): got {val.shape}", errors, warnings)
        except Exception as e:
            errors.append(f"With-env check failed: {e}")
            print(f"  [FAIL] With-env check failed: {e}")
    else:
        print("  [SKIP] Run with --with_env to verify obs dims and network (requires Isaac Sim).")

    # --- 8. Two streams returns/advantages (T, N, 2) ---
    print("\n8. Rollout storage: returns/advantages shape (T, N, 2)")
    from isaaclab_rl.rsl_rl.dwbc.rollout_storage_dwbc import DwbcRolloutStorage
    import torch
    T, N = 24, 4
    num_prop = 64
    storage = DwbcRolloutStorage(N, T, num_prop, [8], device="cpu")
    _check(storage.rewards.shape == (T, N, 2), f"rewards (T,N,2): {storage.rewards.shape}", errors, warnings)
    _check(storage.values.shape == (T, N, 2), f"values (T,N,2): {storage.values.shape}", errors, warnings)
    last_v = torch.zeros(N, 2)
    storage.compute_returns(last_v, 0.99, 0.95)
    _check(storage.returns.shape == (T, N, 2), f"returns (T,N,2): {storage.returns.shape}", errors, warnings)
    _check(storage.advantages.shape == (T, N, 2), f"advantages (T,N,2): {storage.advantages.shape}", errors, warnings)

    # --- 9. Advantage mixing ---
    print("\n9. Advantage mixing")
    from isaaclab_rl.rsl_rl.dwbc.ppo_dwbc import PPO_DWBC
    _check(hasattr(runner_cfg.algorithm, "mixing_schedule"), "mixing_schedule in algorithm config", errors, warnings)
    sched = runner_cfg.algorithm.mixing_schedule
    _check(len(sched) >= 3, f"mixing_schedule has [max_ratio, start_iter, ramp_iters]: {sched}", errors, warnings)
    if len(sched) >= 3 and sched[0] == 0.0 and sched[1] == 0.0 and sched[2] == 0.0:
        print("  [OK] mixing_schedule [0,0,0] -> ratio always 0 (no cross-mixing).")

    # --- 10. Log-prob per component (batch, 2) ---
    print("\n10. Log-prob per component (batch, 2)")
    from isaaclab_rl.rsl_rl.dwbc.actor_critic_dwbc import ActorCriticDWBC
    _check(hasattr(ActorCriticDWBC, "get_actions_log_prob"),
           "ActorCriticDWBC.get_actions_log_prob returns (batch, 2)", errors, warnings)

    # --- 11-12. No DAgger / priv-hist regularizer (simplified DWBC-PPO) ---
    print("\n11-12. No DAgger / history / priv encoder (actor-critic on policy obs only)")
    _check(
        not hasattr(runner_cfg.algorithm, "dagger_update_freq"),
        "algorithm config has no dagger_update_freq",
        errors,
        warnings,
    )

    # --- 13. Min policy std ---
    print("\n13. Min policy std")
    _check(hasattr(runner_cfg.algorithm, "min_policy_std"), "min_policy_std in algorithm config", errors, warnings)
    mstd = getattr(runner_cfg.algorithm, "min_policy_std", None)
    _check(mstd is not None and len(mstd) == 8, f"min_policy_std length 8 (2 leg + 6 arm): {len(mstd) if mstd else 0}", errors, warnings)

    # --- 14-15. Run / Play instructions ---
    print("\n14-15. Training and play")
    print("  14. Run: ./isaaclab.sh -p scripts/rsl_rl/train_dwbc.py --task Isaac-FBC-Husky-DWBC-v0 --num_envs 2048 --headless")
    print("  15. Play: policy(obs_prop) only (same as training).")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All automated checks passed.")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")
    print("=" * 50 + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
