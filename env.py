import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SovereignEnv(gym.Env):
    """
    SOVEREIGN: Three-nation geopolitical RL environment
    Deterministic core + stochastic political dynamics
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, T_max=200, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # -------------------------
        # MAP CONFIG (9 territories)
        # -------------------------
        self.num_territories = 9
        self.num_nations = 3  # I, D, N

        # adjacency list (simple graph)
        self.adj = {
            0: [1], 1: [0, 2], 2: [1, 3],
            3: [2, 4], 4: [3, 5],
            5: [4, 6], 6: [5, 7],
            7: [6, 8], 8: [7]
        }

        # home territories
        self.home_I = [0]
        self.home_D = [8]
        self.home_N = [4]

        self.T_max = T_max
        self.t = 0

        # -------------------------
        # ACTION SPACE
        # -------------------------
        self.a_pol_space = 5   # 0..4
        self.a_mil_space = 4   # 0..3

        self.action_space = spaces.MultiDiscrete([self.a_pol_space, self.a_mil_space])

        # -------------------------
        # OBSERVATION SPACE
        # -------------------------
        self.observation_space = spaces.Dict({
            "M": spaces.Box(0, 1, (self.num_territories, 3), dtype=np.float32),
            "U_I": spaces.Box(0, 100, (self.num_territories,), dtype=np.float32),
            "U_D": spaces.Box(0, 100, (self.num_territories,), dtype=np.float32),
            "L": spaces.Box(0, 1, (1,), dtype=np.float32),
            "E": spaces.Box(0, 1, (1,), dtype=np.float32),
            "theta": spaces.Box(-1, 1, (1,), dtype=np.float32),
            "t_occ": spaces.Box(0, 1000, (1,), dtype=np.float32),
        })

        # -------------------------
        # COEFFICIENTS
        # -------------------------
        self.alpha = 0.04
        self.beta = 0.05
        self.gamma = 0.10
        self.delta = 0.04
        self.eps_pol = 0.03
        self.zeta = 0.03

        self.sigma = 0.02
        self.lambda_insurgency = 0.05

        # rewards
        self.w_T = 0.30 # territory yield
        self.w_R = 0.20 # resource capture bonus
        self.w_O = 0.25 # occupation cost
        self.w_L = 0.15 # legitimacy penalty
        self.w_S = 0.20 # sanction penalty 
        self.w_I = 0.10 # insurgency event 

        self.reset()

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0

        # territory control: 0=I,1=D,2=N
        self.M = np.zeros((self.num_territories, 3), dtype=np.float32)
        self.M[self.home_I, 0] = 1
        self.M[self.home_D, 1] = 1
        self.M[self.home_N, 2] = 1

        # unit distributions
        self.U_I = np.zeros(self.num_territories)
        self.U_D = np.zeros(self.num_territories)

        self.U_I[self.home_I] = 12
        self.U_D[self.home_D] = 6

        # state variables
        self.L = 1.0
        self.E = 1.0
        self.theta = 0.0
        self.t_occ = 0

        return self._obs(), {}

    # -------------------------
    # OBSERVATION
    # -------------------------
    def _obs(self):
        return {
            "M": self.M.copy(),
            "U_I": self.U_I.copy(),
            "U_D": self.U_D.copy(),
            "L": np.array([self.L], dtype=np.float32),
            "E": np.array([self.E], dtype=np.float32),
            "theta": np.array([self.theta], dtype=np.float32),
            "t_occ": np.array([self.t_occ], dtype=np.float32),
        }

    # -------------------------
    # POLITICAL ACTIONS
    # -------------------------
    def _apply_political(self, a_pol):
        if a_pol == 0:  # SEEK_ALLIANCE
            self.L += 0.01
            self.theta -= 0.05

        elif a_pol == 1:  # IMPOSE_SANCTION
            self.L -= 0.02
            self.theta += 0.04
            self.E -= 0.03

        elif a_pol == 2:  # ISSUE_THREAT
            self.L -= 0.03
            self.theta += 0.03

        elif a_pol == 3:  # NEGOTIATE
            self.L += 0.03
            self.theta -= 0.04

        elif a_pol == 4:  # DO_NOTHING
            if self.L < 0.5:
                self.L -= 0.005

    # -------------------------
    # MILITARY ACTIONS
    # -------------------------
    def _apply_military(self, a_mil):
        if a_mil == 0:  # ADVANCE
            self.t_occ += 1

            # simple territory flip logic
            target = self._find_attack_target()
            if target is not None:
                self.M[target, :] = 0
                self.M[target, 0] = 1

            self.L -= 0.05

        elif a_mil == 1:  # HOLD
            self.t_occ += 1

        elif a_mil == 2:  # WITHDRAW
            self.t_occ = 0
            self.L += 0.02

        elif a_mil == 3:  # STRIKE
            self.L -= 0.08
            target = self._find_enemy_unit_tile()
            if target is not None:
                self.U_D[target] = max(0, self.U_D[target] - 1)

            self.t_occ += 1

    # -------------------------
    # HELPERS
    # -------------------------
    def _find_attack_target(self):
        # naive: pick first non-home neighbor
        for t in range(self.num_territories):
            if self.M[t, 0] == 0:
                return t
        return None

    def _find_enemy_unit_tile(self):
        idx = np.where(self.U_D > 0)[0]
        return self.rng.choice(idx) if len(idx) > 0 else None

    # -------------------------
    # NEUTRAL DYNAMICS
    # -------------------------
    def _update_theta(self, a_pol, a_mil):

        # legitimacy coupling
        mu = self.alpha * (1 - self.L)

        # military effects
        if a_mil == 0:
            mu += self.beta
        if a_mil == 3:
            mu += self.gamma

        # political effects
        if a_pol == 3:
            mu -= self.delta
        if a_pol == 0:
            mu -= self.eps_pol

        # occupation drift
        mu += self.zeta * (self.t_occ / self.T_max)

        noise = self.rng.normal(0, self.sigma)

        self.theta = np.clip(self.theta + mu + noise, -1, 1)

    # -------------------------
    # INSURGENCY
    # -------------------------
    def _insurgency(self):
        p = 1 - np.exp(-self.lambda_insurgency * self.t_occ)
        if self.rng.random() < p:
            idx = self._find_enemy_unit_tile()
            if idx is not None:
                self.U_I[idx] = max(0, self.U_I[idx] - 1)

    # -------------------------
    # REWARD
    # -------------------------
    def _reward(self):
        territory_value = np.sum(self.M[:, 0]) * self.w_T
        resource_gain = np.sum(self.M[:, 0]) * self.w_R

        occ_cost = self.w_O * (self.t_occ / self.T_max)
        legit_cost = self.w_L * (1 - self.L)
        sanction_cost = self.w_S * (1 if self.theta > 0.6 else 0) * (1 - self.E)

        insurgency_cost = self.w_I * (np.sum(self.U_I == 0) / self.num_territories)

        return territory_value + resource_gain - occ_cost - legit_cost - sanction_cost - insurgency_cost

    # -------------------------
    # TERMINATION
    # -------------------------
    def _done(self):
        if self.L <= 0:
            return True, -50
        if np.sum(self.U_I) == 0:
            return True, -30
        if self.t >= self.T_max:
            return True, 0
        return False, None

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):
        a_pol, a_mil = action

        self._apply_political(a_pol)
        self._apply_military(a_mil)

        self._update_theta(a_pol, a_mil)
        self._insurgency()

        self.t += 1

        reward = self._reward()
        done, terminal_reward = self._done()

        if done and terminal_reward is not None:
            reward += terminal_reward

        return self._obs(), reward, done, False, {}

    # -------------------------
    # RENDER (minimal)
    # -------------------------
    def render(self):
        print(f"T={self.t} L={self.L:.2f} θ={self.theta:.2f} E={self.E:.2f} t_occ={self.t_occ}")