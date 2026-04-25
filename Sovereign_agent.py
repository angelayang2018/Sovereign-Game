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
from typing import Dict, List, Tuple, Optional
import time
import argparse
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

    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
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
    log_interval:    int   = 10,
    seed:            int   = 42,
    verbose:         bool  = True,
) -> Tuple[PPOAgent, List[Dict]]:
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
    ep_infos:  List[Dict] = []

    # Action frequency counters
    pol_counts = np.zeros(N_POL_ACTIONS, dtype=int)
    mil_counts = np.zeros(N_MIL_ACTIONS, dtype=int)

    start_time = time.time()
    global_step = 0

    while global_step < total_steps:
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
                ep_infos.append(info)
                ep_logs.append({
                    "episode":    ep_count,
                    "reward":     ep_reward,
                    "length":     ep_len,
                    "L_final":    info["L"],
                    "theta_final":info["theta"],
                    "t_occ":      info["t_occ"],
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
                    elapsed = time.time() - start_time
                    print(
                        f"Ep {ep_count:5d} | Steps {global_step:7d} | "
                        f"AvgR {avg_r:7.2f} | L {avg_L:.3f} | θ {avg_th:+.3f} | "
                        f"t {elapsed:.0f}s | AvgLen {avg_len:.1f} |"
                    )

                obs, _ = env.reset()
                pol_counts[:] = 0
                mil_counts[:] = 0

        # ── PPO update ────────────────────────────────────────────────────
        update_metrics = agent.update(obs)

    env.close()
    return agent, ep_logs


# ─────────────────────────────────────────────────────────────────────────────
# Experimental protocol
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONDITIONS = {
    "full_model":        dict(legitimacy_active=True,  occupation_active=True,  posture_active=True),
    "no_legitimacy":     dict(legitimacy_active=False, occupation_active=True,  posture_active=True),
    "no_occupation":     dict(legitimacy_active=True,  occupation_active=False, posture_active=True),
    "no_posture":        dict(legitimacy_active=True,  occupation_active=True,  posture_active=False),
    "baseline_all_off":  dict(legitimacy_active=False, occupation_active=False, posture_active=False),
}


def run_protocol(
    total_steps: int = 500_000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """Run the full experimental protocol across all ablation conditions."""
    results = {}
    for name, flags in ABLATION_CONDITIONS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Condition: {name}")
            print(f"{'='*60}")
        _, logs = train(
            env_kwargs=flags,
            total_steps=total_steps,
            seed=seed,
            verbose=verbose,
        )
        results[name] = logs
        _print_termination_summary(name, logs)
    return results

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
    args = parser.parse_args()

    if args.mode == "protocol":
        run_protocol(total_steps=args.steps)
    elif args.mode == "demo":
        demo_random(30)
