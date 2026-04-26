# Sovereign-Game

Sovereign Game is a strategic simulation environment for deep reinforcement learning research.

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

## How To Run

### Standard protocol run

python Sovereign_agent.py --mode protocol --steps 500000

### Multi-seed protocol with deterministic evaluation

python Sovereign_agent.py --mode protocol --steps 500000 --seeds 42 43 44 --eval_episodes 200

## Compliance Notes

### Compliant with rulebook design

- Added info metrics such as invader_non_home_territories are observational only.
- These metrics do not alter transitions, action space, or terminal conditions.

### Non-compliant or extension behavior

- hold_penalty is an extra shaping term when non-zero.
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