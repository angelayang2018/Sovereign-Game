# Sovereign-Game

Sovereign Game is a strategic simulation environment for deep reinforcement learning research. It models a three-nation geopolitical conflict in which a militarily superior Invader must decide how to pursue strategic objectives through a joint military–political action space. The central research question: **can a militarily superior agent learn, through experience alone, that invasion is a strategically dominated strategy?**

## Required Environment

- OS: Windows, macOS, or Linux
- Python: 3.12
- Recommended virtual environment: venv
- Required Python packages are listed in requirements.txt:
	- numpy>=1.26.4
	- gymnasium==1.2.2
	- networkx==3.6
	- torch==2.9.1

### Setup

1. Create and activate a virtual environment.
2. Install dependencies:

pip install -r requirements.txt

## Extensions

The project runs without any mandatory VS Code extension, but these are recommended:

- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Jupyter (ms-toolsai.jupyter), optional for analysis notebooks

## Environment Overview

### Nations

| Nation | Role | Starting condition |
|--------|------|--------------------|
| Invader (I) | Learning agent (DRL policy) | Hard-power advantage: 15 units |
| Defender (D) | Rule-based opponent | Home-turf bonus: +20% unit effectiveness |
| Neutral (N) | Stochastic observer | Begins at θ = 0 (true neutral) |

### State Variables

| Variable | Symbol | Description |
|----------|--------|-------------|
| Territory control map | M | One-hot encoding of controller per territory |
| Military strength | U_I, U_D | Unit counts per territory |
| International legitimacy | L | Invader's standing; collapse at L = 0 |
| Economic supply index | E | Drained by sanctions |
| Neutral posture | θ | Neutral alignment [-1, +1] |
| Occupation duration | t_occ | Drives occupation cost and insurgency |

### Action Space

Joint action `a = (a_pol, a_mil)` selected each step.

**Political actions:** `SEEK_ALLIANCE`, `IMPOSE_SANCTION`, `ISSUE_THREAT`, `NEGOTIATE`, `DO_NOTHING`

**Military actions:** `ADVANCE`, `HOLD`, `WITHDRAW`, `STRIKE`

### Terminal Conditions

| Condition | Terminal reward |
|-----------|----------------|
| L ≤ 0 (legitimacy collapse) | −50 |
| All Invader units destroyed | −30 |
| Negotiated settlement | +40 |
| Timeout (t ≥ T_max) | −10 |
| Total conquest | +10 |

## How To Run

### Standard protocol run

python Sovereign_agent.py --mode protocol --steps 500000

### Multi-seed protocol with deterministic evaluation

python Sovereign_agent.py --mode protocol --steps 500000 --seeds 42 43 44 --eval_episodes 200

### Run specific ablation conditions only

```bash
python Sovereign_agent.py --mode protocol --conditions full_model baseline_all_off
```

### Load conditions from a JSON file

```bash
python Sovereign_agent.py --mode protocol --condition_file conditions.json
```

Where `conditions.json` is:

```json
{
  "conditions": ["full_model", "no_legitimacy", "no_occupation"]
}
```

---

## Experimental Protocol

Five ablation conditions test which mechanisms drive the agent toward peaceful policy:

| Condition | L active | t_occ active | θ active | Expected policy |
|-----------|----------|--------------|----------|-----------------|
| `full_model` | ✓ | ✓ | ✓ | Negotiate or deter |
| `no_legitimacy` | ✗ | ✓ | ✓ | Slower invasion |
| `no_occupation` | ✓ | ✗ | ✓ | Partial invasion |
| `no_posture` | ✓ | ✓ | ✗ | Invasion |
| `baseline_all_off` | ✗ | ✗ | ✗ | Always invade |

## Compliance Notes

### Compliant with rulebook design

- Added info metrics such as invader_non_home_territories are observational only.
- These metrics do not alter transitions, action space, or terminal conditions.

### Non-compliant or extension behavior

- hold_penalty is an extra shaping term when non-zero.
- `time_pressure` on `r_territory` is an extension to discourage territory farming; not in the rulebook.
- baseline_all_off currently includes extra shaping in Sovereign_agent.py:
	- weights={"w_R": 0.30}
	- hold_penalty=0.01

This means baseline_all_off is currently a modified baseline, not a strict rulebook baseline.

### Recommended strict protocol setting

For strict rulebook experiments, keep baseline_all_off as:

- legitimacy_active=False
- occupation_active=False
- posture_active=False
- hold_penalty=0.0
- default reward weights