"""
SOVEREIGN: Strategic Simulation Environment for Deep Reinforcement Learning
Gymnasium-compatible environment implementing the SOVEREIGN rulebook.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Nations
INVADER  = 0
DEFENDER = 1
NEUTRAL  = 2
CONTESTED = 3

# Political actions
POL_SEEK_ALLIANCE  = 0
POL_IMPOSE_SANCTION = 1
POL_ISSUE_THREAT   = 2
POL_NEGOTIATE      = 3
POL_DO_NOTHING     = 4
N_POL_ACTIONS = 5

# Military actions
MIL_ADVANCE  = 0
MIL_HOLD     = 1
MIL_WITHDRAW = 2
MIL_STRIKE   = 3
N_MIL_ACTIONS = 4


# ─────────────────────────────────────────────────────────────────────────────
# Default Map: 9 territories
# Layout: I_home(0), D_home(1), N_home(2), contested(3-8)
#
#   [0:I_home] -- [3] -- [4] -- [1:D_home]
#                  |      |
#                 [5] -- [6]
#                  |      |
#   [2:N_home] -- [7] -- [8]
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EDGES = [
    (0, 3), (3, 4), (4, 1),
    (3, 5), (4, 6),
    (5, 6), (5, 7), (6, 8),
    (7, 2), (7, 8), (8, 2),
]

DEFAULT_RESOURCE_VALUES = np.array([0.3, 0.3, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
DEFAULT_STRATEGIC_VALUES = np.array([0.4, 0.4, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
HOME_TERRITORY = {INVADER: 0, DEFENDER: 1, NEUTRAL: 2}

# Initial control: home territories owned, rest contested
INITIAL_CONTROL = np.array([INVADER, DEFENDER, NEUTRAL,
                              CONTESTED, CONTESTED, CONTESTED,
                              CONTESTED, CONTESTED, CONTESTED], dtype=np.int32)

# Initial units per territory [INVADER, DEFENDER, NEUTRAL]
INITIAL_UNITS_I = np.array([8, 0, 0, 2, 1, 1, 0, 0, 0], dtype=np.int32)   # 12 total
INITIAL_UNITS_D = np.array([0, 5, 0, 0, 1, 0, 0, 0, 0], dtype=np.int32)   # 6 total
INITIAL_UNITS_N = np.array([0, 0, 4, 0, 0, 0, 0, 0, 0], dtype=np.int32)   # 4 total (non-combatant)


# ─────────────────────────────────────────────────────────────────────────────
# Reward weights (default)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = dict(
    w_T=0.30,   # territory control
    w_R=0.20,   # resource capture
    w_O=0.25,   # occupation cost
    w_L=0.15,   # legitimacy loss
    w_S=0.20,   # sanction penalty
    w_I=0.10,   # insurgency event
)

# Terminal rewards
TERMINAL_L_COLLAPSE   = -50.0
TERMINAL_MIL_DEFEAT   = -30.0
TERMINAL_NEGOTIATION  = +40.0
TERMINAL_TIMEOUT      =   0.0
TERMINAL_CONQUEST     = +10.0


# ─────────────────────────────────────────────────────────────────────────────
# Neutral posture drift coefficients (default)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_POSTURE_COEFFS = dict(
    alpha=0.04,   # legitimacy coupling
    beta=0.05,    # advance shock
    gamma=0.10,   # strike shock
    delta=0.04,   # negotiate pull
    epsilon=0.03, # alliance-seeking pull
    zeta=0.03,    # occupation drift
    sigma=0.02,   # noise std
)


# ─────────────────────────────────────────────────────────────────────────────
# SOVEREIGN Environment
# ─────────────────────────────────────────────────────────────────────────────

class SovereignEnv(gym.Env):
    """
    SOVEREIGN Gymnasium environment.

    Observation space: flat float32 vector of all state variables.
    Action space: MultiDiscrete([N_POL_ACTIONS, N_MIL_ACTIONS])
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"]}

    def __init__(
        self,
        T_max: int = 200,
        edges: Optional[List[Tuple[int, int]]] = None,
        resource_values: Optional[np.ndarray] = None,
        strategic_values: Optional[np.ndarray] = None,
        weights: Optional[Dict] = None,
        posture_coeffs: Optional[Dict] = None,
        # Ablation flags
        legitimacy_active: bool = True,
        occupation_active: bool = True,
        posture_active: bool = True,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.T_max = T_max
        self.render_mode = render_mode
        self.legitimacy_active = legitimacy_active
        self.occupation_active = occupation_active
        self.posture_active = posture_active

        # Build map graph
        self.edges = edges if edges is not None else DEFAULT_EDGES
        self.G = nx.Graph()
        self.n_territories = 9
        self.G.add_nodes_from(range(self.n_territories))
        self.G.add_edges_from(self.edges)

        self.resource_values  = resource_values  if resource_values  is not None else DEFAULT_RESOURCE_VALUES.copy()
        self.strategic_values = strategic_values if strategic_values is not None else DEFAULT_STRATEGIC_VALUES.copy()

        self.weights = {**DEFAULT_WEIGHTS,       **(weights        or {})}
        self.pc      = {**DEFAULT_POSTURE_COEFFS, **(posture_coeffs or {})}

        # Sanction hysteresis tracker
        self._sanction_active = False
        self._sanction_below_threshold_steps = 0
        self._neutral_ally_active = False
        self._invader_ally_active = False

        # Observation: M (9×3) + U_I (9) + U_D (9) + L + E + E_D + θ + t_occ (normalised) = 27+9+9+5 = 50
        obs_dim = self.n_territories * 3 + self.n_territories + self.n_territories + 5
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Joint action: (political, military)
        self.action_space = spaces.MultiDiscrete([N_POL_ACTIONS, N_MIL_ACTIONS])

        self.np_random = np.random.default_rng(seed)
        self._state: Dict = {}
        self._step_count = 0

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._state = {
            "control":  INITIAL_CONTROL.copy(),        # (9,) int
            "units_I":  INITIAL_UNITS_I.copy(),        # (9,) int – Invader units per territory
            "units_D":  INITIAL_UNITS_D.copy(),        # (9,) int – Defender units per territory
            "L":        1.0,                           # legitimacy
            "E":        1.0,                           # Invader supply index
            "E_D":      1.0,                           # Defender supply index
            "theta":    0.0,                           # neutral posture
            "t_occ":    0,                             # occupation duration
        }
        self._step_count = 0
        self._sanction_active = False
        self._sanction_below_threshold_steps = 0
        self._neutral_ally_active = False
        self._invader_ally_active = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        a_pol, a_mil = int(action[0]), int(action[1])
        s = self._state
        done = False
        terminal_reward = 0.0
        termination_reason = None

        prev_controlled_resources = self._invader_controlled_resources()

        # ── 2-3. Apply political action ──────────────────────────────────────
        self._apply_political(a_pol, s)

        # ── 4-6. Apply military action & Defender response ───────────────────
        self._apply_military(a_mil, s)
        self._defender_response(s)

        # ── 7. Update L, E, t_occ ────────────────────────────────────────────
        self._update_derived_state(a_mil, s)

        # ── 8. Neutral posture shift ──────────────────────────────────────────
        if self.posture_active:
            self._update_posture(a_pol, a_mil, s)

        # ── 9. Threshold events ───────────────────────────────────────────────
        self._check_thresholds(s)

        # ── Negotiated settlement check ───────────────────────────────────────
        if self._check_negotiated_settlement(a_pol, s):
            done = True
            terminal_reward = TERMINAL_NEGOTIATION
            termination_reason = "negotiated_settlement"

        # ── 10. Terminal conditions ───────────────────────────────────────────
        if not done:
            if self.legitimacy_active and s["L"] <= 0.0:
                done = True
                terminal_reward = TERMINAL_L_COLLAPSE
                termination_reason = "legitimacy_collapse"
            elif s["units_I"].sum() == 0:
                done = True
                terminal_reward = TERMINAL_MIL_DEFEAT
                termination_reason = "military_defeat"
            elif np.all(s["control"] == INVADER):
                done = True
                terminal_reward = TERMINAL_CONQUEST
                termination_reason = "conquest"

        self._step_count += 1
        if not done and self._step_count >= self.T_max:
            done = True
            terminal_reward = TERMINAL_TIMEOUT
            termination_reason = "timeout"

        # ── 11. Compute step reward ───────────────────────────────────────────
        step_reward = self._compute_reward(s, a_mil, prev_controlled_resources)
        total_reward = step_reward + (terminal_reward if done else 0.0)

        obs  = self._get_obs()
        info = self._get_info()
        info["terminal_reward"] = terminal_reward
        info["step_reward"]     = step_reward
        info["termination_reason"] = termination_reason

        return obs, total_reward, done, False, info

    # ─── Political actions ────────────────────────────────────────────────────

    def _apply_political(self, a_pol, s):
        # Direct θ effects from table 6.1 (applied before drift model in _update_posture)
        if self.posture_active:
            if   a_pol == POL_SEEK_ALLIANCE:
                s["theta"] = float(np.clip(s["theta"] - 0.05, -1.0, 1.0))
            elif a_pol == POL_IMPOSE_SANCTION:
                s["theta"] = float(np.clip(s["theta"] + 0.04, -1.0, 1.0))
            elif a_pol == POL_ISSUE_THREAT:
                s["theta"] = float(np.clip(s["theta"] + 0.03, -1.0, 1.0))
            elif a_pol == POL_NEGOTIATE:
                s["theta"] = float(np.clip(s["theta"] - 0.04, -1.0, 1.0))
            elif a_pol == POL_DO_NOTHING:
                if s["t_occ"] > 0:
                    s["theta"] = float(np.clip(s["theta"] + 0.01, -1.0, 1.0))  # slow drift

        if not self.legitimacy_active:
            return
        if   a_pol == POL_SEEK_ALLIANCE:
            s["L"] = np.clip(s["L"] + 0.01, 0.0, 1.0)
        elif a_pol == POL_IMPOSE_SANCTION:
            s["L"] = np.clip(s["L"] - 0.02, 0.0, 1.0)
            s["E"] = np.clip(s["E"] - 0.03, 0.0, 1.0)  # cost to self too
        elif a_pol == POL_ISSUE_THREAT:
            s["L"] = np.clip(s["L"] - 0.03, 0.0, 1.0)
        elif a_pol == POL_NEGOTIATE:
            s["L"] = np.clip(s["L"] + 0.03, 0.0, 1.0)
        elif a_pol == POL_DO_NOTHING:
            if s["L"] < 0.5:
                s["L"] = np.clip(s["L"] - 0.005, 0.0, 1.0)

    # ─── Military actions ─────────────────────────────────────────────────────

    def _apply_military(self, a_mil, s):
        if a_mil == MIL_ADVANCE:
            self._do_advance(s)
        elif a_mil == MIL_HOLD:
            pass  # no movement
        elif a_mil == MIL_WITHDRAW:
            self._do_withdraw(s)
        elif a_mil == MIL_STRIKE:
            self._do_strike(s)

    def _do_advance(self, s):
        """
        Invader advances into one adjacent contested territory or Defender territory.
        Picks the territory with the weakest Defender presence adjacent to Invader-held land.
        Combat is probabilistic (Bernoulli) based on relative strength.
        Cannot advance into Neutral home territory.
        """
        candidates = self._advance_candidates(s)
        if not candidates:
            return
        # Choose candidate with fewest Defender units
        target = min(candidates, key=lambda v: s["units_D"][v])
        inv_units = s["units_I"][s["control"] == INVADER].sum()  # attacking from adjacent
        def_units = s["units_D"][target]
        inv_strength = max(inv_units, 1)
        home_bonus = (1.0 + 0.2 * s["E_D"]) if target == HOME_TERRITORY[DEFENDER] else 1.0
        def_strength = max(def_units, 1) * home_bonus

        p_invader_wins = inv_strength / (inv_strength + def_strength)
        if self.np_random.random() < p_invader_wins:
            # Invader captures territory
            s["units_D"][target] = max(0, s["units_D"][target] - 1)
            s["control"][target] = INVADER
            # Move one unit forward
            adjacent_invader = [n for n in self.G.neighbors(target) if s["control"][n] == INVADER]
            if adjacent_invader:
                src = adjacent_invader[0]
                if s["units_I"][src] > 1:
                    s["units_I"][src] -= 1
                    s["units_I"][target] += 1
        else:
            # Defender holds; Invader loses one unit from the front
            front = self._invader_front_territories(s)
            if front:
                t = front[np.argmax([s["units_I"][f] for f in front])]
                s["units_I"][t] = max(0, s["units_I"][t] - 1)
            if s["control"][target] != NEUTRAL:
                s["control"][target] = CONTESTED

    def _do_withdraw(self, s):
        """Cede one contested (non-home) Invader territory."""
        non_home = [v for v in range(self.n_territories)
                    if s["control"][v] == INVADER and v != HOME_TERRITORY[INVADER]]
        if non_home:
            # Withdraw from the furthest (most costly) territory
            target = non_home[0]
            s["units_I"][target] = 0
            s["control"][target] = CONTESTED

    def _do_strike(self, s):
        """
        Destroy one Defender unit on the front line (any territory adjacent to
        an Invader-held territory). Probabilistic: always succeeds if there is
        a valid target, but costs legitimacy.
        """
        targets = self._strike_candidates(s)
        if not targets:
            return
        target = min(targets, key=lambda v: s["units_D"][v])
        if s["units_D"][target] > 0:
            s["units_D"][target] -= 1
        if self.legitimacy_active:
            s["L"] = np.clip(s["L"] - 0.08, 0.0, 1.0)

    def _advance_candidates(self, s) -> List[int]:
        """Territories adjacent to Invader that are not Invader-controlled and not Neutral home."""
        invader_held = [v for v in range(self.n_territories) if s["control"][v] == INVADER]
        candidates = set()
        for v in invader_held:
            for nb in self.G.neighbors(v):
                if s["control"][nb] != INVADER and nb != HOME_TERRITORY[NEUTRAL]:
                    candidates.add(nb)
        return list(candidates)

    def _strike_candidates(self, s) -> List[int]:
        """Defender territories adjacent to Invader-held territory."""
        invader_held = [v for v in range(self.n_territories) if s["control"][v] == INVADER]
        candidates = set()
        for v in invader_held:
            for nb in self.G.neighbors(v):
                if s["units_D"][nb] > 0:
                    candidates.add(nb)
        return list(candidates)

    def _invader_front_territories(self, s) -> List[int]:
        """Invader territories adjacent to non-Invader territories."""
        front = []
        for v in range(self.n_territories):
            if s["control"][v] == INVADER:
                for nb in self.G.neighbors(v):
                    if s["control"][nb] != INVADER:
                        front.append(v)
                        break
        return front

    # ─── Defender rule-based response ────────────────────────────────────────

    def _defender_response(self, s):
        """
        Simple rule-based Defender:
        - If any Invader unit is adjacent to Defender territory, counterattack.
        - Otherwise, reinforce the most threatened territory.
        """
        # Find Defender territories adjacent to Invader
        threatened = []
        for v in range(self.n_territories):
            if s["control"][v] == DEFENDER or v == HOME_TERRITORY[DEFENDER]:
                for nb in self.G.neighbors(v):
                    if s["control"][nb] == INVADER and s["units_I"][nb] > 0:
                        threatened.append(v)
                        break

        if threatened:
            # Counterattack the most accessible threatened territory
            target_v = threatened[0]
            # Find adjacent Invader territory
            invader_adj = [nb for nb in self.G.neighbors(target_v)
                           if s["control"][nb] == INVADER and s["units_I"][nb] > 0]
            if invader_adj:
                attack_src = invader_adj[0]
                def_strength = s["units_D"][target_v] * (1.0 + 0.2 * s["E_D"]) if target_v == HOME_TERRITORY[DEFENDER] else s["units_D"][target_v]
                inv_strength = max(s["units_I"][attack_src], 1)
                p_def_wins = def_strength / (def_strength + inv_strength + 1e-9)
                if self.np_random.random() < p_def_wins:
                    s["units_I"][attack_src] = max(0, s["units_I"][attack_src] - 1)
                    if s["units_I"][attack_src] == 0:
                        s["control"][attack_src] = CONTESTED

    # ─── State updates ────────────────────────────────────────────────────────

    def _update_derived_state(self, a_mil, s):
        # Occupation duration
        if self.occupation_active:
            non_home_invader = any(
                s["control"][v] == INVADER and v != HOME_TERRITORY[INVADER]
                for v in range(self.n_territories)
            )
            if non_home_invader:
                s["t_occ"] += 1
            else:
                s["t_occ"] = 0  # full withdrawal resets counter

        # Occupation cost → supply index
        if self.occupation_active:
            occ_drain = 0.005 * (s["t_occ"] / self.T_max)
            s["E"] = np.clip(s["E"] - occ_drain, 0.0, 1.0)

        # Insurgency
        if self.occupation_active and s["t_occ"] > 0:
            p_ins = 1.0 - np.exp(-0.05 * s["t_occ"])
            if self.np_random.random() < p_ins:
                # Destroy one Invader unit in a non-home territory
                non_home = [v for v in range(self.n_territories)
                            if s["control"][v] == INVADER and v != HOME_TERRITORY[INVADER]
                            and s["units_I"][v] > 0]
                if non_home:
                    victim = self.np_random.choice(non_home)
                    s["units_I"][victim] = max(0, s["units_I"][victim] - 1)
                    if s["units_I"][victim] == 0:
                        s["control"][victim] = CONTESTED

    def _update_posture(self, a_pol, a_mil, s):
        pc = self.pc
        mu = (
            + pc["alpha"]   * (1.0 - s["L"])               # low legitimacy alienates neutral
            + pc["beta"]    * float(a_mil == MIL_ADVANCE)  # territorial advance shocks posture
            + pc["gamma"]   * float(a_mil == MIL_STRIKE)   # strikes carry heavier cost
            - pc["delta"]   * float(a_pol == POL_NEGOTIATE)        # diplomacy pulls posture toward center
            - pc["epsilon"] * float(a_pol == POL_SEEK_ALLIANCE)    # alliance-seeking reduces drift
            + pc["zeta"]    * (s["t_occ"] / self.T_max)    # prolonged occupation steadily alienates
        )
        noise = self.np_random.normal(0.0, pc["sigma"])
        s["theta"] = float(np.clip(s["theta"] + mu + noise, -1.0, 1.0))

    def _check_thresholds(self, s):
        theta = s["theta"]

        # ── Sanctions (θ > 0.60) ──────────────────────────────────────────
        if not self._sanction_active and theta > 0.60:
            self._sanction_active = True
            self._sanction_below_threshold_steps = 0

        if self._sanction_active:
            if theta <= 0.50:
                self._sanction_below_threshold_steps += 1
                if self._sanction_below_threshold_steps >= 5:
                    self._sanction_active = False
                    self._sanction_below_threshold_steps = 0
            else:
                self._sanction_below_threshold_steps = 0
            # Apply sanction drain
            s["E"] = np.clip(s["E"] - 0.01, 0.0, 1.0)

        # ── Neutral joins Defender (θ > 0.85) ────────────────────────────
        if not self._neutral_ally_active and theta > 0.85:
            self._neutral_ally_active = True
            # +2 units to Defender home
            s["units_D"][HOME_TERRITORY[DEFENDER]] += 2
            if self.legitimacy_active:
                s["L"] = np.clip(s["L"] - 0.10, 0.0, 1.0)

        # ── Neutral opens supply routes (θ < -0.60) ───────────────────────
        # Handled in reward function (30% occupation cost reduction)

        # ── Neutral formally allies Invader (θ < -0.85) ───────────────────
        if not self._invader_ally_active and theta < -0.85:
            self._invader_ally_active = True
            if self.legitimacy_active:
                s["L"] = np.clip(s["L"] - 0.05, 0.0, 1.0)
            s["E_D"] = np.clip(s["E_D"] - 0.05, 0.0, 1.0)  # Defender supply hit

    def _check_negotiated_settlement(self, a_pol, s) -> bool:
        """
        Settlement triggered with p=0.5 when L > 0.6 AND θ < 0 AND
        the Invader chose NEGOTIATE this step.
        Requires both legitimacy and posture to be active — if either is
        ablated, the conditions are meaningless and settlement cannot occur.
        """
        if a_pol != POL_NEGOTIATE:
            return False
        if not self.legitimacy_active or not self.posture_active:
            return False
        if s["L"] > 0.6 and s["theta"] < 0.0:
            return self.np_random.random() < 0.5
        return False

    # ─── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(self, s, a_mil, prev_resources) -> float:
        w = self.weights
        t_occ = s["t_occ"]
        T_max  = self.T_max

        # Positive
        curr_resources = self._invader_controlled_resources()
        r_territory = w["w_T"] * curr_resources
        r_capture   = w["w_R"] * max(0.0, curr_resources - prev_resources)

        # Negative
        occ_mult = 0.70 if s["theta"] < -0.60 else 1.0  # supply route bonus
        r_occ  = w["w_O"] * (t_occ / T_max) * occ_mult if self.occupation_active else 0.0
        r_leg  = w["w_L"] * (1.0 - s["L"]) if self.legitimacy_active else 0.0
        sanction_pen = w["w_S"] * (1.0 - s["E"]) if self._sanction_active else 0.0
        # Insurgency handled inside _update_derived_state (unit loss); no extra penalty here
        # but add a small signal
        r_ins = 0.0  # already paid via unit loss

        reward = r_territory + r_capture - r_occ - r_leg - sanction_pen - r_ins
        return float(reward)

    def _invader_controlled_resources(self) -> float:
        s = self._state
        return float(sum(
            self.resource_values[v]
            for v in range(self.n_territories)
            if s["control"][v] == INVADER
        ))

    # ─── Observations ─────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        s = self._state
        n = self.n_territories

        # Territory control: one-hot (9×3)
        M = np.zeros((n, 3), dtype=np.float32)
        for v in range(n):
            c = s["control"][v]
            if c in (INVADER, DEFENDER, NEUTRAL):
                M[v, c] = 1.0
            # CONTESTED → all zeros

        units_I_norm = s["units_I"].astype(np.float32) / 12.0
        units_D_norm = s["units_D"].astype(np.float32) / 6.0

        scalars = np.array([
            s["L"],
            s["E"],
            s["E_D"],
            s["theta"],
            s["t_occ"] / self.T_max,
        ], dtype=np.float32)

        return np.concatenate([M.flatten(), units_I_norm, units_D_norm, scalars])

    def _get_info(self) -> Dict:
        s = self._state
        return {
            "step":            self._step_count,
            "L":               s["L"],
            "E":               s["E"],
            "E_D":             s["E_D"],
            "theta":           s["theta"],
            "t_occ":           s["t_occ"],
            "invader_units":   int(s["units_I"].sum()),
            "defender_units":  int(s["units_D"].sum()),
            "sanction_active": self._sanction_active,
            "neutral_allied_defender": self._neutral_ally_active,
            "neutral_allied_invader":  self._invader_ally_active,
            "control":         s["control"].tolist(),
        }

    # ─── Render ───────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()

    def _render_ansi(self) -> str:
        s = self._state
        ctrl_sym = {INVADER: "I", DEFENDER: "D", NEUTRAL: "N", CONTESTED: "."}
        lines = [
            f"Step {self._step_count:3d} | L={s['L']:.2f}  E={s['E']:.2f}  E_D={s['E_D']:.2f}  "
            f"θ={s['theta']:+.2f}  t_occ={s['t_occ']}",
            "Territory: " + " ".join(f"{v}:{ctrl_sym[s['control'][v]]}" for v in range(self.n_territories)),
            f"Units I: {s['units_I'].tolist()}  D: {s['units_D'].tolist()}",
        ]
        return "\n".join(lines)

    def close(self):
        pass