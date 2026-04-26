"""
SOVEREIGN Agent: PPO-based Deep Reinforcement Learning agent.

Implements:
  - ActorCritic network for joint (political, military) action space
  - PPO training loop with GAE advantage estimation
  - Logging utilities for the core experimental protocol
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import time
import argparse
import json
from collections import Counter

from Sovereign_env import (
    SovereignEnv,
    N_POL_ACTIONS, N_MIL_ACTIONS,
    POL_NEGOTIATE, POL_SEEK_ALLIANCE,
    MIL_ADVANCE, MIL_HOLD, MIL_WITHDRAW, MIL_STRIKE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Neural network
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic for SOVEREIGN.
    Outputs logits for political actions, military actions, and a state value.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.pol_head = nn.Linear(hidden_dim, N_POL_ACTIONS)
        self.mil_head = nn.Linear(hidden_dim, N_MIL_ACTIONS)
        self.val_head = nn.Linear(hidden_dim, 1)

        # Orthogonal init (PPO best practice)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.pol_head.weight, gain=0.01)
        nn.init.orthogonal_(self.mil_head.weight, gain=0.01)
        nn.init.orthogonal_(self.val_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.pol_head(h), self.mil_head(h), self.val_head(h).squeeze(-1)

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pol_logits, mil_logits, value = self(obs)
        pol_dist = Categorical(logits=pol_logits)
        mil_dist = Categorical(logits=mil_logits)
        a_pol = pol_dist.sample()
        a_mil = mil_dist.sample()
        log_prob = pol_dist.log_prob(a_pol) + mil_dist.log_prob(a_mil)
        return torch.stack([a_pol, a_mil], dim=-1), log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pol_logits, mil_logits, value = self(obs)
        pol_dist = Categorical(logits=pol_logits)
        mil_dist = Categorical(logits=mil_logits)
        a_pol, a_mil = actions[:, 0], actions[:, 1]
        log_prob = pol_dist.log_prob(a_pol) + mil_dist.log_prob(a_mil)
        entropy  = pol_dist.entropy() + mil_dist.entropy()
        return log_prob, entropy, value


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.obs:      List[np.ndarray] = []
        self.actions:  List[np.ndarray] = []
        self.rewards:  List[float]      = []
        self.values:   List[float]      = []
        self.log_probs:List[float]      = []
        self.dones:    List[bool]       = []

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalised Advantage Estimation (GAE)."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0
        values_arr = np.array(self.values + [last_value], dtype=np.float32)

        for t in reversed(range(n)):
            mask       = 1.0 - float(self.dones[t])
            delta      = self.rewards[t] + gamma * values_arr[t + 1] * mask - values_arr[t]
            last_gae   = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return torch.tensor(advantages), torch.tensor(returns)


# ─────────────────────────────────────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    """PPO agent for SOVEREIGN."""

    def __init__(
        self,
        obs_dim:       int,
        hidden_dim:    int   = 128,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        clip_eps:      float = 0.2,
        entropy_coef:  float = 0.01,
        value_coef:    float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs:      int   = 4,
        batch_size:    int   = 64,
        device:        str   = "cpu",
    ):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.max_grad_norm= max_grad_norm
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.device       = torch.device(device)

        self.network = ActorCritic(obs_dim, hidden_dim).to(self.device)
        self.optimizer= optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer   = RolloutBuffer()

    def set_entropy_coef(self, value: float) -> None:
        self.entropy_coef = float(max(0.0, value))

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                pol_logits, mil_logits, value = self.network(obs_t)
                a_pol = torch.argmax(pol_logits, dim=-1)
                a_mil = torch.argmax(mil_logits, dim=-1)
                action = torch.stack([a_pol, a_mil], dim=-1)
                pol_dist = Categorical(logits=pol_logits)
                mil_dist = Categorical(logits=mil_logits)
                log_prob = pol_dist.log_prob(a_pol) + mil_dist.log_prob(a_mil)
            else:
                action, log_prob, value = self.network.get_action(obs_t)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def store(self, obs, action, reward, value, log_prob, done):
        self.buffer.add(obs, action, reward, value, log_prob, done)

    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        obs_t = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, last_value = self.network(obs_t)
        last_value = last_value.item()

        advantages, returns = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Tensors
        obs_arr     = torch.tensor(np.array(self.buffer.obs),     dtype=torch.float32)
        actions_arr = torch.tensor(np.array(self.buffer.actions), dtype=torch.long)
        old_lp_arr  = torch.tensor(np.array(self.buffer.log_probs),dtype=torch.float32)

        n = len(self.buffer.rewards)
        metrics = defaultdict(list)

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                obs_b     = obs_arr[mb].to(self.device)
                actions_b = actions_arr[mb].to(self.device)
                old_lp_b  = old_lp_arr[mb].to(self.device)
                adv_b     = advantages[mb].to(self.device)
                ret_b     = returns[mb].to(self.device)

                log_prob, entropy, value = self.network.evaluate_actions(obs_b, actions_b)

                ratio = torch.exp(log_prob - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(value, ret_b)
                entropy_loss= -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(-entropy_loss.item())

        self.buffer.clear()
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location=self.device))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    env_kwargs:      Optional[Dict] = None,
    total_steps:     int   = 200_000,
    rollout_len:     int   = 512,
    hidden_dim:      int   = 128,
    lr:              float = 3e-4,
    gamma:           float = 0.99,
    entropy_start:   float = 0.05,
    entropy_end:     float = 0.005,
    log_interval:    int   = 10,
    seed:            int   = 42,
    verbose:         bool  = True,
) -> Tuple[PPOAgent, List[Dict], Dict[str, float]]:
    """
    Train a PPO agent on SOVEREIGN.

    Returns the trained agent and a list of episode-level logs.
    """
    env_kwargs = env_kwargs or {}
    env = SovereignEnv(seed=seed, **env_kwargs)
    obs_dim = env.observation_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, hidden_dim=hidden_dim, lr=lr, gamma=gamma)

    obs, _ = env.reset(seed=seed)
    ep_reward  = 0.0
    ep_len     = 0
    ep_count   = 0
    ep_logs:   List[Dict] = []
    # Action frequency counters
    pol_counts = np.zeros(N_POL_ACTIONS, dtype=int)
    mil_counts = np.zeros(N_MIL_ACTIONS, dtype=int)

    start_time = time.time()
    global_step = 0
    last_update_metrics: Dict[str, float] = {}

    while global_step < total_steps:
        # Entropy annealing: high exploration early, more deterministic policy later.
        frac = min(1.0, global_step / max(total_steps, 1))
        entropy_coef = entropy_start + frac * (entropy_end - entropy_start)
        agent.set_entropy_coef(entropy_coef)

        # ── Collect rollout ───────────────────────────────────────────────
        for _ in range(rollout_len):
            action, log_prob, value = agent.select_action(obs)
            pol_counts[action[0]] += 1
            mil_counts[action[1]] += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, value, log_prob, done)
            obs = next_obs
            ep_reward += reward
            ep_len    += 1
            global_step += 1

            if done:
                total_pol = max(1, int(pol_counts.sum()))
                total_mil = max(1, int(mil_counts.sum()))
                invasion_rate = float((mil_counts[MIL_ADVANCE] + mil_counts[MIL_STRIKE]) / total_mil)
                diplomacy_rate = float((pol_counts[POL_NEGOTIATE] + pol_counts[POL_SEEK_ALLIANCE]) / total_pol)
                ep_logs.append({
                    "episode":    ep_count,
                    "reward":     ep_reward,
                    "length":     ep_len,
                    "L_final":    info["L"],
                    "theta_final":info["theta"],
                    "t_occ":      info["t_occ"],
                    "invader_non_home_territories": info.get("invader_non_home_territories", 0),
                    "invasion_rate": invasion_rate,
                    "diplomacy_rate": diplomacy_rate,
                    "invader_units": info["invader_units"],
                    "sanction":   info["sanction_active"],
                    "pol_freq":   pol_counts.copy(),
                    "mil_freq":   mil_counts.copy(),
                    "termination": info["termination_reason"],
                    "last_action": action.copy(),
                })
                ep_count  += 1
                ep_reward  = 0.0
                ep_len     = 0

                if verbose and ep_count % log_interval == 0:
                    recent = ep_logs[-log_interval:]
                    avg_r  = np.mean([e["reward"]  for e in recent])
                    avg_L  = np.mean([e["L_final"] for e in recent])
                    avg_th = np.mean([e["theta_final"] for e in recent])
                    avg_len = np.mean([e["length"] for e in recent])
                    avg_inv = np.mean([e["invasion_rate"] for e in recent])
                    avg_dip = np.mean([e["diplomacy_rate"] for e in recent])
                    avg_exp = np.mean([e["invader_non_home_territories"] for e in recent])
                    elapsed = time.time() - start_time
                    print(
                        f"Ep {ep_count:5d} | Steps {global_step:7d} | "
                        f"AvgR {avg_r:7.2f} | L {avg_L:.3f} | θ {avg_th:+.3f} | "
                        f"Inv {avg_inv:.2f} | Dip {avg_dip:.2f} | OccTerr {avg_exp:.2f} | "
                        f"t {elapsed:.0f}s | AvgLen {avg_len:.1f} |"
                    )

                obs, _ = env.reset()
                pol_counts[:] = 0
                mil_counts[:] = 0

        # ── PPO update ────────────────────────────────────────────────────
        last_update_metrics = agent.update(obs)

    env.close()
    if not ep_logs:
        training_summary = {"avg_reward": 0.0, "avg_invasion_rate": 0.0, "avg_diplomacy_rate": 0.0}
    else:
        training_summary = {
            "avg_reward": float(np.mean([e["reward"] for e in ep_logs])),
            "avg_invasion_rate": float(np.mean([e["invasion_rate"] for e in ep_logs])),
            "avg_diplomacy_rate": float(np.mean([e["diplomacy_rate"] for e in ep_logs])),
            "avg_non_home_territories": float(np.mean([e["invader_non_home_territories"] for e in ep_logs])),
        }
    training_summary.update({f"last_{k}": float(v) for k, v in last_update_metrics.items()})
    training_summary["final_entropy_coef"] = float(agent.entropy_coef)
    return agent, ep_logs, training_summary


# ─────────────────────────────────────────────────────────────────────────────
# Experimental protocol
# ─────────────────────────────────────────────────────────────────────────────

CONDITION_CONFIGS = {
    "full_model": {
        "env_kwargs": dict(legitimacy_active=True, occupation_active=True, posture_active=True),
        "train_kwargs": {},
        "expected_policy": "Negotiate or deter",
    },
    "no_legitimacy": {
        "env_kwargs": dict(legitimacy_active=False, occupation_active=True, posture_active=True),
        "train_kwargs": {},
        "expected_policy": "Slower invasion",
    },
    "no_occupation": {
        "env_kwargs": dict(legitimacy_active=True, occupation_active=False, posture_active=True),
        "train_kwargs": {},
        "expected_policy": "Partial invasion",
    },
    "no_posture": {
        "env_kwargs": dict(legitimacy_active=True, occupation_active=True, posture_active=False),
        "train_kwargs": {},
        "expected_policy": "Invasion",
    },
    "baseline_all_off": {
        "env_kwargs": dict(
            legitimacy_active=False,
            occupation_active=False,
            posture_active=False,
        ),
        "train_kwargs": {},
        "expected_policy": "Always invade",
    },
    # Optional extension variant (not strict rulebook baseline).
    "baseline_all_off_shaped": {
        "env_kwargs": dict(
            legitimacy_active=False,
            occupation_active=False,
            posture_active=False,
            weights={"w_R": 0.30},
            hold_penalty=0.01,
        ),
        "train_kwargs": {},
        "expected_policy": "Always invade",
    },
}


RULEBOOK_EXPERIMENT_ORDER = [
    "full_model",
    "no_legitimacy",
    "no_occupation",
    "no_posture",
    "baseline_all_off",
]


def evaluate_policy(
    agent: PPOAgent,
    env_kwargs: Optional[Dict] = None,
    n_episodes: int = 200,
    base_seed: int = 100_000,
) -> Dict[str, Any]:
    """Evaluate a policy with greedy actions for reproducible comparison."""
    env = SovereignEnv(seed=base_seed, **(env_kwargs or {}))
    rewards: List[float] = []
    lengths: List[int] = []
    finals_L: List[float] = []
    finals_theta: List[float] = []
    invasion_rates: List[float] = []
    diplomacy_rates: List[float] = []
    non_home_territories: List[float] = []
    term_counts: Counter = Counter()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        pol_counts = np.zeros(N_POL_ACTIONS, dtype=np.int32)
        mil_counts = np.zeros(N_MIL_ACTIONS, dtype=np.int32)
        info: Dict[str, Any] = {}

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            pol_counts[action[0]] += 1
            mil_counts[action[1]] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

        total_pol = max(1, int(pol_counts.sum()))
        total_mil = max(1, int(mil_counts.sum()))
        invasion_rates.append(float((mil_counts[MIL_ADVANCE] + mil_counts[MIL_STRIKE]) / total_mil))
        diplomacy_rates.append(float((pol_counts[POL_NEGOTIATE] + pol_counts[POL_SEEK_ALLIANCE]) / total_pol))
        rewards.append(float(ep_reward))
        lengths.append(ep_len)
        finals_L.append(float(info.get("L", 0.0)))
        finals_theta.append(float(info.get("theta", 0.0)))
        non_home_territories.append(float(info.get("invader_non_home_territories", 0)))
        term_counts[info.get("termination_reason", "unknown")] += 1

    env.close()
    return {
        "episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_L_final": float(np.mean(finals_L)),
        "mean_theta_final": float(np.mean(finals_theta)),
        "mean_invasion_rate": float(np.mean(invasion_rates)),
        "mean_diplomacy_rate": float(np.mean(diplomacy_rates)),
        "mean_non_home_territories": float(np.mean(non_home_territories)),
        "termination_counts": dict(term_counts),
    }


def _aggregate_seed_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate scalar metrics across multiple training seeds."""
    scalar_keys = [
        "mean_reward", "mean_length", "mean_L_final", "mean_theta_final",
        "mean_invasion_rate", "mean_diplomacy_rate", "mean_non_home_territories",
    ]
    summary: Dict[str, float] = {}
    for key in scalar_keys:
        values = np.array([m[key] for m in metrics], dtype=np.float32)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


def _print_protocol_summary(condition: str, aggregate_eval: Dict[str, float]) -> None:
    print(
        f"  Eval summary [{condition}] "
        f"R {aggregate_eval['mean_reward_mean']:.2f}±{aggregate_eval['mean_reward_std']:.2f} | "
        f"Inv {aggregate_eval['mean_invasion_rate_mean']:.2f}±{aggregate_eval['mean_invasion_rate_std']:.2f} | "
        f"Dip {aggregate_eval['mean_diplomacy_rate_mean']:.2f}±{aggregate_eval['mean_diplomacy_rate_std']:.2f} | "
        f"OccTerr {aggregate_eval['mean_non_home_territories_mean']:.2f}±{aggregate_eval['mean_non_home_territories_std']:.2f}"
    )


def _print_experiment_header(name: str, expected_policy: str) -> None:
    print(f"  Condition: {name}")
    print(f"  Expected optimal policy: {expected_policy}")


def run_protocol(
    total_steps: int = 500_000,
    seeds: Optional[List[int]] = None,
    eval_episodes: int = 200,
    condition_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run the full experimental protocol across all ablation conditions."""
    seeds = seeds or [42]
    results: Dict[str, Dict[str, Any]] = {}

    if condition_names is None:
        selected_names = RULEBOOK_EXPERIMENT_ORDER.copy()
    else:
        unknown = sorted(set(condition_names) - set(CONDITION_CONFIGS.keys()))
        if unknown:
            raise ValueError(
                f"Unknown conditions: {unknown}. "
                f"Available: {sorted(CONDITION_CONFIGS.keys())}"
            )
        selected_names = condition_names

    for name in selected_names:
        config = CONDITION_CONFIGS[name]
        if verbose:
            print(f"\n{'='*60}")
            _print_experiment_header(name, config.get("expected_policy", "N/A"))
            print(f"{'='*60}")

        env_kwargs = dict(config["env_kwargs"])
        train_kwargs = dict(config.get("train_kwargs", {}))

        seed_runs = []
        eval_metrics_by_seed = []
        for seed in seeds:
            if verbose:
                print(f"  -> seed={seed}")
            agent, logs, train_summary = train(
                env_kwargs=env_kwargs,
                total_steps=total_steps,
                seed=seed,
                verbose=verbose,
                **train_kwargs,
            )
            _print_termination_summary(f"{name} seed={seed}", logs)
            eval_metrics = evaluate_policy(
                agent,
                env_kwargs=env_kwargs,
                n_episodes=eval_episodes,
                base_seed=100_000 + seed,
            )
            eval_metrics_by_seed.append(eval_metrics)
            train_term_counts = Counter(e["termination"] for e in logs)
            seed_runs.append({
                "seed": seed,
                "train_summary": train_summary,
                "train_episodes": len(logs),
                "train_termination_counts": dict(train_term_counts),
                "eval": eval_metrics,
            })

        aggregate_eval = _aggregate_seed_metrics(eval_metrics_by_seed)
        if verbose:
            _print_protocol_summary(name, aggregate_eval)

        results[name] = {
            "config": {
                "env_kwargs": env_kwargs,
                "train_kwargs": train_kwargs,
                "expected_policy": config.get("expected_policy", "N/A"),
            },
            "runs": seed_runs,
            "aggregate_eval": aggregate_eval,
        }
    return results


def load_condition_selection(path: str) -> List[str]:
    """Load selected condition names from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "conditions" not in payload:
        raise ValueError("Condition file must be a JSON object with key 'conditions'.")
    conditions = payload["conditions"]
    if not isinstance(conditions, list) or not all(isinstance(c, str) for c in conditions):
        raise ValueError("'conditions' must be a list of strings.")
    return conditions

def _print_termination_summary(name: str, logs: List[Dict]) -> None:
    """Print termination reason breakdown for a training condition."""
    counts = Counter(e["termination"] for e in logs)
    total  = len(logs)
    print(f"\n  Termination summary for '{name}' ({total} episodes):")
    for reason in ["negotiated_settlement", "conquest", "legitimacy_collapse",
                   "military_defeat", "timeout"]:
        n = counts.get(reason, 0)
        pct = 100.0 * n / total if total > 0 else 0.0
        bar = "█" * int(pct / 2)
        print(f"    {reason:<25s} {n:5d}  ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────

def demo_random(n_steps: int = 20):
    """Run a random policy for a quick sanity check."""
    env = SovereignEnv(render_mode="ansi", seed=0)
    obs, info = env.reset()
    total_r = 0.0
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, r, done, _, info = env.step(action)
        total_r += r
        print(env.render())
        print(f"  action=pol:{action[0]} mil:{action[1]}  r={r:.3f}  cumR={total_r:.3f}")
        if done:
            print("  [EPISODE DONE]")
            break
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["protocol", "demo"], default="protocol")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of condition names to run.",
    )
    parser.add_argument(
        "--condition_file",
        type=str,
        default=None,
        help="Optional JSON file with key 'conditions' listing condition names.",
    )
    args = parser.parse_args()

    if args.mode == "protocol":
        selected_conditions = args.conditions
        if args.condition_file:
            selected_conditions = load_condition_selection(args.condition_file)
        run_protocol(
            total_steps=args.steps,
            seeds=args.seeds,
            eval_episodes=args.eval_episodes,
            condition_names=selected_conditions,
        )
    elif args.mode == "demo":
        demo_random(30)
