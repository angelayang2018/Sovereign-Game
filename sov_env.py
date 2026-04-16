"""
SOVEREIGN: A deterministic-core, stochastic-politics strategic simulation.
Gymnasium-compatible environment for deep reinforcement learning research.

Central research question:
    Can a militarily superior agent learn, through experience alone,
    that invasion is a strategically dominated strategy?
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import IntEnum
import networkx as nx


# ─────────────────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────────────────

class Nation(IntEnum):
    INVADER  = 0
    DEFENDER = 1
    NEUTRAL  = 2
    CONTESTED = 3


class PolAction(IntEnum):
    SEEK_ALLIANCE   = 0
    IMPOSE_SANCTION = 1
    ISSUE_THREAT    = 2
    NEGOTIATE       = 3
    DO_NOTHING      = 4


class MilAction(IntEnum):
    ADVANCE  = 0
    HOLD     = 1
    WITHDRAW = 2
    STRIKE   = 3


# ─────────────────────────────────────────────────────────
#  Territory dataclass
# ─────────────────────────────────────────────────────────

@dataclass
class Territory:
    name:            str
    controller:      Nation
    resource_value:  float   # [0, 1]
    strategic_value: float   # [0, 1]
    home_nation:     Optional[Nation] = None  # None if neutral/contested

    def is_home_of(self, nation: Nation) -> bool:
        return self.home_nation == nation


# ─────────────────────────────────────────────────────────
#  Default map factory  (9 territories)
# ─────────────────────────────────────────────────────────

def make_default_map() -> Tuple[List[Territory], nx.Graph]:
    """
    Default 9-territory map.

    Layout (indices):
        Invader home cluster  : 0(I-Home), 1(I-Border), 2(I-Flank)
        Neutral home cluster  : 3(N-Home)
        Contested buffer      : 4(C-North), 5(C-Center), 6(C-South)
        Defender home cluster : 7(D-Border), 8(D-Home)

    Visual topology:
        [0]──[1]──[4]──[7]──[8]
               │    │    │
              [2]──[5]──[6]
                    │
                   [3]
    """
    territories = [
        Territory("I-Home",    Nation.INVADER,  resource_value=0.8, strategic_value=0.9, home_nation=Nation.INVADER),
        Territory("I-Border",  Nation.INVADER,  resource_value=0.4, strategic_value=0.6, home_nation=Nation.INVADER),
        Territory("I-Flank",   Nation.INVADER,  resource_value=0.3, strategic_value=0.4, home_nation=Nation.INVADER),
        Territory("N-Home",    Nation.NEUTRAL,  resource_value=0.5, strategic_value=0.3, home_nation=Nation.NEUTRAL),
        Territory("C-North",   Nation.CONTESTED,resource_value=0.5, strategic_value=0.7),
        Territory("C-Center",  Nation.CONTESTED,resource_value=0.6, strategic_value=0.8),
        Territory("C-South",   Nation.CONTESTED,resource_value=0.4, strategic_value=0.5),
        Territory("D-Border",  Nation.DEFENDER, resource_value=0.4, strategic_value=0.6, home_nation=Nation.DEFENDER),
        Territory("D-Home",    Nation.DEFENDER, resource_value=0.8, strategic_value=0.9, home_nation=Nation.DEFENDER),
    ]

    G = nx.Graph()
    G.add_nodes_from(range(len(territories)))
    edges = [
        (0, 1), (1, 2), (1, 4), (2, 5),
        (4, 5), (4, 7), (5, 6), (5, 3),
        (6, 7), (7, 8),
    ]
    G.add_edges_from(edges)
    return territories, G


def make_linear_map() -> Tuple[List[Territory], nx.Graph]:
    """
    Alternative: pure linear chain (easier topology for ablation studies).
    I-Home ─ I-Border ─ C-West ─ C-Center ─ C-East ─ D-Border ─ D-Home
              N-Home attached to C-Center
    """
    territories = [
        Territory("I-Home",   Nation.INVADER,   resource_value=0.8, strategic_value=0.9, home_nation=Nation.INVADER),
        Territory("I-Border", Nation.INVADER,   resource_value=0.4, strategic_value=0.6, home_nation=Nation.INVADER),
        Territory("C-West",   Nation.CONTESTED, resource_value=0.4, strategic_value=0.5),
        Territory("C-Center", Nation.CONTESTED, resource_value=0.6, strategic_value=0.8),
        Territory("N-Home",   Nation.NEUTRAL,   resource_value=0.5, strategic_value=0.3, home_nation=Nation.NEUTRAL),
        Territory("C-East",   Nation.CONTESTED, resource_value=0.4, strategic_value=0.5),
        Territory("D-Border", Nation.DEFENDER,  resource_value=0.4, strategic_value=0.6, home_nation=Nation.DEFENDER),
        Territory("D-Home",   Nation.DEFENDER,  resource_value=0.8, strategic_value=0.9, home_nation=Nation.DEFENDER),
    ]

    G = nx.Graph()
    G.add_nodes_from(range(len(territories)))
    edges = [(0,1),(1,2),(2,3),(3,4),(3,5),(5,6),(6,7)]
    G.add_edges_from(edges)
    return territories, G


# ─────────────────────────────────────────────────────────
#  Config dataclass  (all tunable hyper-parameters)
# ─────────────────────────────────────────────────────────

@dataclass
class SovereignConfig:
    # Episode length
    T_max: int = 100

    # --- Initial military strengths ---
    invader_ground_init: int = 12
    invader_strike_init: int = 3
    defender_ground_init: int = 6
    defender_strike_init: int = 1
    neutral_ground_init: int = 4

    # --- Neutral posture model ---
    theta_noise_sigma: float = 0.02
    alpha_legitimacy: float  = 0.04   # legitimacy coupling
    beta_advance:     float  = 0.05   # advance shock
    gamma_strike:     float  = 0.10   # strike shock
    delta_negotiate:  float  = 0.04   # negotiate pull
    epsilon_alliance: float  = 0.03   # alliance pull
    zeta_occupation:  float  = 0.03   # occupation drift

    # --- Threshold events ---
    sanction_threshold:      float = 0.60
    coalition_threshold:     float = 0.85
    supply_route_threshold:  float = -0.60
    formal_ally_threshold:   float = -0.85
    sanction_lift_threshold: float = 0.50
    sanction_hysteresis:     int   = 5    # steps below threshold before lifted

    # --- Legitimacy ---
    legitimacy_slow_decay: float = 0.005

    # --- Political action effects on L ---
    pol_L_seek_alliance:   float =  0.01
    pol_L_impose_sanction: float = -0.02
    pol_L_issue_threat:    float = -0.03
    pol_L_negotiate:       float =  0.03
    pol_L_do_nothing:      float =  0.00

    # --- Political action effects on theta ---
    pol_theta_seek_alliance:   float = -0.05
    pol_theta_impose_sanction: float =  0.04
    pol_theta_issue_threat:    float =  0.03
    pol_theta_negotiate:       float = -0.04
    pol_theta_do_nothing:      float =  0.00

    # --- Military action effects on L ---
    mil_L_advance:  float = -0.05
    mil_L_hold:     float =  0.00
    mil_L_withdraw: float =  0.02
    mil_L_strike:   float = -0.08

    # --- Sanction economic penalty per step ---
    sanction_economic_drain: float = 0.01

    # --- Occupation cost ---
    occupation_cost_linear: float = 1.0   # multiplier for w_O formula

    # --- Insurgency hazard ---
    insurgency_lambda: float = 0.05

    # --- Reward weights ---
    w_T: float = 0.30   # territory yield
    w_R: float = 0.20   # resource capture bonus
    w_O: float = 0.25   # occupation cost
    w_L: float = 0.15   # legitimacy penalty
    w_S: float = 0.20   # sanction penalty
    w_I: float = 0.10   # insurgency event

    # --- Terminal rewards ---
    terminal_political_collapse: float = -50.0
    terminal_military_defeat:    float = -30.0
    terminal_negotiated:         float =  40.0
    terminal_conquest:           float =  10.0
    terminal_timeout:            float =   0.0

    # --- Ablation flags ---
    legitimacy_active:     bool = True
    occupation_active:     bool = True
    neutral_posture_active:bool = True

    # --- Defender home-turf bonus ---
    defender_home_bonus:  float = 0.20


# ─────────────────────────────────────────────────────────
#  SOVEREIGN Environment
# ─────────────────────────────────────────────────────────

class SovereignEnv(gym.Env):
    """
    SOVEREIGN Gymnasium environment.

    Observation space: flat float32 vector (see _build_obs_space).
    Action space:      MultiDiscrete([5, 4])  →  (pol_action, mil_action)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: Optional[SovereignConfig] = None,
        map_fn=None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.cfg         = config or SovereignConfig()
        self.render_mode = render_mode
        self._map_fn     = map_fn or make_default_map
        self._rng        = np.random.default_rng(seed)

        # Build spaces (depends on map size)
        territories, graph = self._map_fn()
        self._n_territories = len(territories)
        self.observation_space = self._build_obs_space()
        self.action_space      = spaces.MultiDiscrete([5, 4])  # (pol, mil)

        # Will be fully initialised in reset()
        self.territories: List[Territory] = []
        self.graph: nx.Graph               = None
        self.t: int                        = 0

    # ── Gymnasium API ──────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        super().reset(seed=seed)

        self.territories, self.graph = self._map_fn()
        self.t = 0

        cfg = self.cfg

        # Military units: dict nation → per-territory array
        n = len(self.territories)
        self.ground_units = {
            Nation.INVADER:  np.zeros(n, dtype=int),
            Nation.DEFENDER: np.zeros(n, dtype=int),
            Nation.NEUTRAL:  np.zeros(n, dtype=int),
        }
        self.strike_units = {
            Nation.INVADER:  cfg.invader_strike_init,
            Nation.DEFENDER: cfg.defender_strike_init,
        }

        # Place starting units on home territories
        for tid, terr in enumerate(self.territories):
            if terr.home_nation == Nation.INVADER:
                self.ground_units[Nation.INVADER][tid] = (
                    cfg.invader_ground_init // 3  # spread across 3 home territories
                )
            elif terr.home_nation == Nation.DEFENDER:
                self.ground_units[Nation.DEFENDER][tid] = (
                    cfg.defender_ground_init // 2
                )
            elif terr.home_nation == Nation.NEUTRAL:
                self.ground_units[Nation.NEUTRAL][tid] = cfg.neutral_ground_init

        # Political / economic state
        self.L     = 1.0    # legitimacy
        self.E     = 1.0    # economic supply index
        self.theta = 0.0    # neutral posture

        # Occupation
        self.t_occ = 0

        # Sanction state
        self.sanctions_active    = False
        self.steps_below_lift    = 0

        # Coalition state
        self.coalition_active    = False
        self.supply_routes_open  = False
        self.formal_ally_active  = False

        # Episode tracking
        self.episode_reward = 0.0
        self.done           = False
        self.info: Dict     = {}

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        assert not self.done, "Environment is done. Call reset()."
        a_pol = PolAction(int(action[0]))
        a_mil = MilAction(int(action[1]))
        cfg   = self.cfg

        # ── 2-3. Apply Invader political & military actions ────
        self._apply_political_action(a_pol)
        self._apply_military_action(a_mil)

        # ── 4. Defender responds ───────────────────────────────
        self._defender_respond()

        # ── 5-6. Resolve combat, update control map ───────────
        self._resolve_combat()
        self._update_territory_control()

        # ── 7. Update L, E, t_occ ─────────────────────────────
        self._update_economy()
        self._update_occupation(a_mil)
        self.L = float(np.clip(self.L, 0.0, 1.0))
        self.E = float(np.clip(self.E, 0.0, 1.0))

        # ── 8. Neutral posture shift ───────────────────────────
        if cfg.neutral_posture_active:
            self._update_neutral_posture(a_pol, a_mil)

        # ── 9. Threshold events ────────────────────────────────
        self._check_threshold_events()

        # ── 10. Terminal conditions ────────────────────────────
        reward, self.done, term_info = self._check_terminals()

        # ── 11. Step reward (if not already terminal) ──────────
        if not self.done:
            reward = self._step_reward(a_pol, a_mil)

        self.t += 1
        if self.t >= cfg.T_max and not self.done:
            self.done  = True
            reward    += cfg.terminal_timeout
            term_info["reason"] = "timeout"

        self.episode_reward += reward
        self.info = {
            "t":               self.t,
            "L":               self.L,
            "E":               self.E,
            "theta":           self.theta,
            "t_occ":           self.t_occ,
            "sanctions":       self.sanctions_active,
            "coalition":       self.coalition_active,
            "supply_routes":   self.supply_routes_open,
            "episode_reward":  self.episode_reward,
            **term_info,
        }

        obs = self._get_obs()
        return obs, float(reward), self.done, False, self.info

    def render(self):
        if self.render_mode in ("human", "ansi"):
            return self._render_ansi()

    def close(self):
        pass

    # ── Internal mechanics ─────────────────────────────────

    def _apply_political_action(self, a_pol: PolAction):
        cfg = self.cfg
        if not cfg.legitimacy_active:
            return
        L_delta = {
            PolAction.SEEK_ALLIANCE:   cfg.pol_L_seek_alliance,
            PolAction.IMPOSE_SANCTION: cfg.pol_L_impose_sanction,
            PolAction.ISSUE_THREAT:    cfg.pol_L_issue_threat,
            PolAction.NEGOTIATE:       cfg.pol_L_negotiate,
            PolAction.DO_NOTHING:      cfg.pol_L_do_nothing,
        }[a_pol]

        # Slow decay when L < 0.5 and DO_NOTHING
        if a_pol == PolAction.DO_NOTHING and self.L < 0.5:
            L_delta -= cfg.legitimacy_slow_decay

        self.L += L_delta

    def _apply_military_action(self, a_mil: MilAction):
        cfg = self.cfg
        if cfg.legitimacy_active:
            L_delta = {
                MilAction.ADVANCE:  cfg.mil_L_advance,
                MilAction.HOLD:     cfg.mil_L_hold,
                MilAction.WITHDRAW: cfg.mil_L_withdraw,
                MilAction.STRIKE:   cfg.mil_L_strike,
            }[a_mil]
            self.L += L_delta

        if a_mil == MilAction.ADVANCE:
            self._do_advance()
        elif a_mil == MilAction.WITHDRAW:
            self._do_withdraw()
        elif a_mil == MilAction.STRIKE:
            self._do_strike()
        # HOLD: no territorial change

    def _do_advance(self):
        """Move the Invader's front line one step toward Defender territory."""
        # Find Invader-controlled territories adjacent to non-Invader territories
        candidates = []
        for tid, terr in enumerate(self.territories):
            if terr.controller == Nation.INVADER:
                for nbr in self.graph.neighbors(tid):
                    if self.territories[nbr].controller != Nation.INVADER:
                        candidates.append(nbr)
        if not candidates:
            return
        # Advance into the highest-resource adjacent territory
        target = max(candidates, key=lambda t: self.territories[t].resource_value)
        # Move units from the best adjacent Invader territory
        sources = [
            s for s in self.graph.neighbors(target)
            if self.territories[s].controller == Nation.INVADER
            and self.ground_units[Nation.INVADER][s] > 0
        ]
        if not sources:
            return
        src = max(sources, key=lambda s: self.ground_units[Nation.INVADER][s])
        units_moved = max(1, self.ground_units[Nation.INVADER][src] // 2)
        self.ground_units[Nation.INVADER][src]   -= units_moved
        self.ground_units[Nation.INVADER][target] += units_moved
        self.territories[target].controller = Nation.CONTESTED

    def _do_withdraw(self):
        """Cede the least-strategically-valuable non-home Invader territory."""
        non_home = [
            tid for tid, t in enumerate(self.territories)
            if t.controller == Nation.INVADER and not t.is_home_of(Nation.INVADER)
        ]
        if not non_home:
            return
        # Withdraw from the lowest-strategic-value territory
        target = min(non_home, key=lambda t: self.territories[t].strategic_value)
        units = self.ground_units[Nation.INVADER][target]
        self.ground_units[Nation.INVADER][target] = 0

        # Retreat units to nearest Invader home territory
        home_ids = [
            tid for tid, t in enumerate(self.territories)
            if t.is_home_of(Nation.INVADER)
        ]
        if home_ids:
            dest = home_ids[0]
            self.ground_units[Nation.INVADER][dest] += units
        self.territories[target].controller = Nation.DEFENDER  # reverts to Defender

    def _do_strike(self):
        """Destroy one Defender unit in the territory with the most Defender units."""
        best = max(
            range(len(self.territories)),
            key=lambda t: self.ground_units[Nation.DEFENDER][t],
        )
        if self.ground_units[Nation.DEFENDER][best] > 0:
            self.ground_units[Nation.DEFENDER][best] -= 1
            self.strike_units[Nation.INVADER]        -= 0  # strikes are capability, not consumed

    def _defender_respond(self):
        """
        Rule-based Defender policy:
        - If Invader is advancing, reinforce the most-threatened border territory.
        - If safe, slowly reclaim contested territories adjacent to Defender home.
        """
        # Find contested territories adjacent to Defender-held territories
        contested_adj = []
        for tid, terr in enumerate(self.territories):
            if terr.controller in (Nation.CONTESTED, Nation.INVADER):
                for nbr in self.graph.neighbors(tid):
                    if self.territories[nbr].controller == Nation.DEFENDER:
                        contested_adj.append(tid)
                        break

        if contested_adj:
            target = max(contested_adj, key=lambda t: self.territories[t].strategic_value)
            sources = [
                s for s in self.graph.neighbors(target)
                if self.territories[s].controller == Nation.DEFENDER
                and self.ground_units[Nation.DEFENDER][s] > 1
            ]
            if sources:
                src = max(sources, key=lambda s: self.ground_units[Nation.DEFENDER][s])
                units_moved = max(1, self.ground_units[Nation.DEFENDER][src] // 3)
                self.ground_units[Nation.DEFENDER][src]   -= units_moved
                self.ground_units[Nation.DEFENDER][target] += units_moved

    def _resolve_combat(self):
        """
        Deterministic combat resolution.
        In contested territories: compare unit counts, apply attrition, determine winner.
        """
        for tid, terr in enumerate(self.territories):
            inv = self.ground_units[Nation.INVADER][tid]
            dfn = self.ground_units[Nation.DEFENDER][tid]

            if inv == 0 and dfn == 0:
                continue
            if inv == 0 or dfn == 0:
                continue  # no contested combat

            # Home-turf bonus for Defender
            effective_dfn = dfn
            if terr.home_nation == Nation.DEFENDER:
                effective_dfn = int(dfn * (1 + self.cfg.defender_home_bonus))

            # Attrition: each side loses proportionally to the other's strength
            inv_losses = max(1, int(round(effective_dfn * 0.25)))
            dfn_losses = max(1, int(round(inv           * 0.25)))

            self.ground_units[Nation.INVADER][tid]  = max(0, inv - inv_losses)
            self.ground_units[Nation.DEFENDER][tid] = max(0, dfn - dfn_losses)

    def _update_territory_control(self):
        """Assign control based on unit presence after combat."""
        for tid, terr in enumerate(self.territories):
            inv = self.ground_units[Nation.INVADER][tid]
            dfn = self.ground_units[Nation.DEFENDER][tid]

            if inv > dfn and inv > 0:
                terr.controller = Nation.INVADER
            elif dfn > inv and dfn > 0:
                if terr.home_nation == Nation.DEFENDER:
                    terr.controller = Nation.DEFENDER
                else:
                    terr.controller = Nation.DEFENDER
            elif inv == dfn and inv > 0:
                terr.controller = Nation.CONTESTED
            elif inv == 0 and dfn == 0:
                # territory holds its last assigned controller
                pass

    def _update_economy(self):
        """Update supply index E based on sanctions and occupation."""
        cfg = self.cfg
        if self.sanctions_active:
            self.E -= cfg.sanction_economic_drain
        # Supply routes (Neutral ally)
        if self.supply_routes_open:
            self.E = min(1.0, self.E + 0.005)

    def _update_occupation(self, a_mil: MilAction):
        """Update occupation counter."""
        if not self.cfg.occupation_active:
            return

        non_home_controlled = any(
            t.controller == Nation.INVADER and not t.is_home_of(Nation.INVADER)
            for t in self.territories
        )
        if non_home_controlled:
            self.t_occ += 1
        elif a_mil == MilAction.WITHDRAW:
            # Full withdrawal check
            still_outside = any(
                self.ground_units[Nation.INVADER][tid] > 0
                and not self.territories[tid].is_home_of(Nation.INVADER)
                for tid in range(len(self.territories))
            )
            if not still_outside:
                self.t_occ = 0

    def _update_neutral_posture(self, a_pol: PolAction, a_mil: MilAction):
        """Drift-diffusion model for Neutral posture θ."""
        cfg = self.cfg
        L   = self.L

        mu  = 0.0
        mu += cfg.alpha_legitimacy * (1 - L)
        mu += cfg.beta_advance    * float(a_mil == MilAction.ADVANCE)
        mu += cfg.gamma_strike    * float(a_mil == MilAction.STRIKE)
        mu -= cfg.delta_negotiate * float(a_pol == PolAction.NEGOTIATE)
        mu -= cfg.epsilon_alliance* float(a_pol == PolAction.SEEK_ALLIANCE)
        mu += cfg.zeta_occupation * (self.t_occ / cfg.T_max)

        epsilon = self._rng.normal(0.0, cfg.theta_noise_sigma)
        self.theta = float(np.clip(self.theta + mu + epsilon, -1.0, 1.0))

    def _check_threshold_events(self):
        """Irreversible geopolitical threshold events."""
        cfg = self.cfg
        theta = self.theta

        # ── Sanction ───────────────────────────────────────────
        if not self.sanctions_active and theta > cfg.sanction_threshold:
            self.sanctions_active = True
            self.steps_below_lift = 0

        if self.sanctions_active:
            if theta < cfg.sanction_lift_threshold:
                self.steps_below_lift += 1
                if self.steps_below_lift >= cfg.sanction_hysteresis:
                    self.sanctions_active = False
                    self.steps_below_lift = 0
            else:
                self.steps_below_lift = 0

        # ── Coalition (Neutral joins Defender) ─────────────────
        if not self.coalition_active and theta > cfg.coalition_threshold:
            self.coalition_active = True
            self._add_units(Nation.DEFENDER, 2)
            if cfg.legitimacy_active:
                self.L -= 0.10

        # ── Supply routes (Neutral opens to Invader) ───────────
        if not self.supply_routes_open and theta < cfg.supply_route_threshold:
            self.supply_routes_open = True

        # ── Formal ally (Neutral fully sides with Invader) ─────
        if not self.formal_ally_active and theta < cfg.formal_ally_threshold:
            self.formal_ally_active = True
            # Defender economy hurt
            self.E = max(0.0, self.E - 0.05)  # reflected as Invader advantage

    def _add_units(self, nation: Nation, count: int):
        """Add units to a nation's strongest home territory."""
        home_ids = [
            tid for tid, t in enumerate(self.territories)
            if t.home_nation == nation
        ]
        if home_ids:
            tid = max(home_ids, key=lambda i: self.ground_units[nation][i])
            self.ground_units[nation][tid] += count

    def _check_terminals(self) -> Tuple[float, bool, Dict]:
        cfg   = self.cfg
        info  = {"reason": "ongoing"}

        # Political collapse
        if cfg.legitimacy_active and self.L <= 0.0:
            return cfg.terminal_political_collapse, True, {"reason": "political_collapse"}

        # Military defeat (all Invader units destroyed)
        total_inv = int(self.ground_units[Nation.INVADER].sum())
        if total_inv == 0:
            return cfg.terminal_military_defeat, True, {"reason": "military_defeat"}

        # Negotiated settlement (agent played NEGOTIATE 3 consecutive steps with θ < 0)
        # Simplified: NEGOTIATE while L > 0.7 and theta < 0 triggers settlement
        if self.L > 0.7 and self.theta < 0.0:
            if hasattr(self, "_negotiate_streak") and self._negotiate_streak >= 3:
                self._negotiate_streak = 0
                return cfg.terminal_negotiated, True, {"reason": "negotiated_settlement"}

        # Total conquest
        defender_controlled = any(
            t.controller == Nation.DEFENDER for t in self.territories
        )
        if not defender_controlled:
            return cfg.terminal_conquest, True, {"reason": "total_conquest"}

        return 0.0, False, info

    def _step_reward(self, a_pol: PolAction, a_mil: MilAction) -> float:
        cfg = self.cfg

        # ── Positive: territory yield ──────────────────────────
        controlled_rv = sum(
            t.resource_value
            for t in self.territories
            if t.controller == Nation.INVADER
        )
        r_territory = cfg.w_T * controlled_rv

        # ── Positive: resource capture bonus ──────────────────
        # Approximated by advance action (captured territory this step)
        r_capture = cfg.w_R if a_mil == MilAction.ADVANCE else 0.0

        # ── Negative: occupation cost ─────────────────────────
        r_occupation = 0.0
        if cfg.occupation_active:
            r_occupation = cfg.w_O * (self.t_occ / cfg.T_max)

        # ── Negative: legitimacy penalty ──────────────────────
        r_legitimacy = 0.0
        if cfg.legitimacy_active:
            r_legitimacy = cfg.w_L * (1.0 - self.L)

        # ── Negative: sanction penalty ─────────────────────────
        r_sanction = 0.0
        if self.sanctions_active:
            r_sanction = cfg.w_S * (1.0 - self.E)

        # ── Negative: insurgency ───────────────────────────────
        r_insurgency = 0.0
        if cfg.occupation_active and self.t_occ > 0:
            p_ins = 1.0 - np.exp(-cfg.insurgency_lambda * self.t_occ)
            if self._rng.random() < p_ins:
                # Destroy one Invader unit in a random non-home territory
                non_home_with_units = [
                    tid for tid, t in enumerate(self.territories)
                    if t.controller == Nation.INVADER
                    and not t.is_home_of(Nation.INVADER)
                    and self.ground_units[Nation.INVADER][tid] > 0
                ]
                if non_home_with_units:
                    victim = self._rng.choice(non_home_with_units)
                    self.ground_units[Nation.INVADER][victim] -= 1
                    r_insurgency = cfg.w_I

        reward = (r_territory + r_capture) - (r_occupation + r_legitimacy + r_sanction + r_insurgency)
        return float(reward)

    # ── Negotiate streak tracking ──────────────────────────

    def _apply_political_action(self, a_pol: PolAction):
        """Extended: also track negotiate streak for settlement condition."""
        cfg = self.cfg
        if a_pol == PolAction.NEGOTIATE:
            self._negotiate_streak = getattr(self, "_negotiate_streak", 0) + 1
        else:
            self._negotiate_streak = 0

        if not cfg.legitimacy_active:
            return
        L_delta = {
            PolAction.SEEK_ALLIANCE:   cfg.pol_L_seek_alliance,
            PolAction.IMPOSE_SANCTION: cfg.pol_L_impose_sanction,
            PolAction.ISSUE_THREAT:    cfg.pol_L_issue_threat,
            PolAction.NEGOTIATE:       cfg.pol_L_negotiate,
            PolAction.DO_NOTHING:      cfg.pol_L_do_nothing,
        }[a_pol]
        if a_pol == PolAction.DO_NOTHING and self.L < 0.5:
            L_delta -= cfg.legitimacy_slow_decay
        self.L += L_delta

        # Theta effect from political action
        if cfg.neutral_posture_active:
            theta_delta = {
                PolAction.SEEK_ALLIANCE:   cfg.pol_theta_seek_alliance,
                PolAction.IMPOSE_SANCTION: cfg.pol_theta_impose_sanction,
                PolAction.ISSUE_THREAT:    cfg.pol_theta_issue_threat,
                PolAction.NEGOTIATE:       cfg.pol_theta_negotiate,
                PolAction.DO_NOTHING:      cfg.pol_theta_do_nothing,
            }[a_pol]
            self.theta = float(np.clip(self.theta + theta_delta, -1.0, 1.0))

    # ── Observation builder ────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Flat float32 observation vector.
        Layout:
          [0 : 3*n]      Territory control one-hot (3 per territory: I/D/N+Contested)
          [3n : 5n]      Invader ground units per territory (normalised)
          [5n : 7n]      Defender ground units per territory (normalised)
          [7n]           L  (legitimacy)
          [7n+1]         E  (economic supply)
          [7n+2]         theta (neutral posture, normalised to [0,1])
          [7n+3]         t_occ / T_max
          [7n+4]         t / T_max
          [7n+5]         sanctions_active
          [7n+6]         coalition_active
        """
        n    = len(self.territories)
        Tmax = self.cfg.T_max

        control = np.zeros((n, 3), dtype=np.float32)
        for tid, terr in enumerate(self.territories):
            if terr.controller == Nation.INVADER:
                control[tid, 0] = 1.0
            elif terr.controller == Nation.DEFENDER:
                control[tid, 1] = 1.0
            else:  # NEUTRAL or CONTESTED
                control[tid, 2] = 1.0

        max_units = max(self.cfg.invader_ground_init * 2, 1)
        inv_units = self.ground_units[Nation.INVADER].astype(np.float32) / max_units
        dfn_units = self.ground_units[Nation.DEFENDER].astype(np.float32) / max_units

        scalars = np.array([
            self.L,
            self.E,
            (self.theta + 1.0) / 2.0,
            self.t_occ / Tmax,
            self.t / Tmax,
            float(self.sanctions_active),
            float(self.coalition_active),
        ], dtype=np.float32)

        obs = np.concatenate([control.flatten(), inv_units, dfn_units, scalars])
        return obs

    def _build_obs_space(self) -> spaces.Box:
        n   = self._n_territories
        dim = 3*n + n + n + 7
        return spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

    # ── Renderer ───────────────────────────────────────────

    def _render_ansi(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"  SOVEREIGN  |  Step {self.t:03d}/{self.cfg.T_max}")
        lines.append("=" * 60)
        lines.append(f"  Legitimacy  L = {self.L:.3f}    Supply E = {self.E:.3f}")
        lines.append(f"  Neutral θ     = {self.theta:+.3f}   t_occ = {self.t_occ}")
        lines.append(f"  Sanctions: {'ON ' if self.sanctions_active else 'OFF'}   "
                     f"Coalition: {'ON' if self.coalition_active else 'OFF'}")
        lines.append("-" * 60)
        lines.append(f"  {'Territory':<12} {'Controller':<12} {'I-units':>7} {'D-units':>7}")
        lines.append(f"  {'-'*12} {'-'*12} {'-'*7} {'-'*7}")
        ctrl_sym = {
            Nation.INVADER:   "INVADER  ",
            Nation.DEFENDER:  "DEFENDER ",
            Nation.NEUTRAL:   "NEUTRAL  ",
            Nation.CONTESTED: "CONTESTED",
        }
        for tid, terr in enumerate(self.territories):
            lines.append(
                f"  {terr.name:<12} {ctrl_sym[terr.controller]:<12} "
                f"{self.ground_units[Nation.INVADER][tid]:>7} "
                f"{self.ground_units[Nation.DEFENDER][tid]:>7}"
            )
        lines.append("=" * 60)
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text

    # ── Utility: action names ──────────────────────────────

    @staticmethod
    def action_name(action) -> str:
        return f"{PolAction(action[0]).name} + {MilAction(action[1]).name}"

    # ── Utility: ablation constructor ─────────────────────

    @classmethod
    def make_ablation(cls, variant: str, **kwargs) -> "SovereignEnv":
        """
        Convenience constructor for ablation experiments.
        variant ∈ {"full", "no_legitimacy", "no_occupation",
                   "no_neutral", "baseline"}
        """
        env_kwargs = {k: v for k, v in kwargs.items() if k in ("map_fn", "render_mode", "seed")}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k not in env_kwargs}
        cfg = SovereignConfig(**cfg_kwargs)
        variants = {
            "full":          dict(legitimacy_active=True,  occupation_active=True,  neutral_posture_active=True),
            "no_legitimacy": dict(legitimacy_active=False, occupation_active=True,  neutral_posture_active=True),
            "no_occupation": dict(legitimacy_active=True,  occupation_active=False, neutral_posture_active=True),
            "no_neutral":    dict(legitimacy_active=True,  occupation_active=True,  neutral_posture_active=False),
            "baseline":      dict(legitimacy_active=False, occupation_active=False, neutral_posture_active=False),
        }
        if variant not in variants:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(variants)}")
        for k, v in variants[variant].items():
            setattr(cfg, k, v)
        return cls(config=cfg, **env_kwargs)