# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent reinforcement learning system investigating whether complex social behaviors (cooperation, altruism, tribalism, policing) can emerge from simple survival instincts. Agents live in a GridWorld, compete for food, and maintain a shared **Ledger** of objective interaction history. Built from scratch in PyTorch (CleanRL-style PPO), no RL framework dependencies.

## Project Structure

```
core/           Foundation modules (Config, Ledger, utils)
env/            Environment implementations (GridWorld, BatchedGridWorld, VecEnv)
agents/         Networks + RL algorithms (ActorCritic, PPO, SAC, buffers)
training/       Training entry-point scripts (train.py, train_vec.py, train_sac.py, scenarios)
analysis/       Diagnostics, evaluation, visualization
experiments/    YAML configs (configs/) + experiment runners
tests/          Correctness tests + benchmarks
docs/           PROJECT_SPEC.md, ARCHITECTURE.md
results/        All outputs (runs/, ablation/, replays/, plots/, journal/)
```

## Commands

```bash
# Single-environment training
python -m training.train
python -m training.train --visualize                    # with live rendering
python -m training.train --visualize --render-every 50  # render every N updates

# Vectorized multi-environment training (faster)
python -m training.train_vec --n_envs 8 --pretrain
python -m training.train_vec --n_envs 16 --batch_size 1024

# SAC training
python -m training.train_sac --experiment experiments/configs/sac.yaml --visualize

# Visualize a saved checkpoint
python -m analysis.visualize_checkpoint --checkpoint path/to/checkpoint.pt

# Diagnostics
python -m analysis.diagnose_social
python -m analysis.diagnose_spatial
python -m analysis.diagnose_friend_foe

# Run experiments
python -m experiments.run_experiment experiments/configs/my_experiment.yaml
```

Dependencies: `pip install -r requirements.txt` (torch>=2.0.0, numpy, matplotlib)

## Architecture

### Core Pipeline

```
Config → Environment(GridWorld) → Observations → Network(ActorCritic) → Actions → PPO Update
              ↕                                        ↑
           Ledger ──────── social history ──────────────┘
```

### Key Modules

- **`env/environment.py`** — Fully vectorized GridWorld with PyTorch tensors (no Python loops). Single-occupancy grid, HP decay, poor food (solo) vs rich food (requires 2+ agents). Movement uses "Heavyweight Rule" (highest HP wins contested cells).

- **`core/ledger.py`** — `[n_agents × n_agents × 4]` tensor storing objective facts: damage dealt, food given, coop count, defense score. Core design principle: store actions, not relationships — agents derive feelings from facts.

- **`agents/network.py`** — Two main classes:
  - `ObservationEncoder`: Fourier-encoded entity tokens (agents + food) processed through multi-head attention (4 heads, 64 dim), plus signal encoder and self-state encoder. Outputs 144-dim feature vector.
  - `ActorCritic`: Shared trunk → factored dual-action heads (direction: 5, action type: 5) + value head + auxiliary value heads for decomposed rewards (survival/resource/social).

- **`agents/ppo.py`** — Three variants: `PPO` (single agent), `IndependentPPO` (N independent networks, supports clone mode for curriculum), `VmapPPO` (vectorized across parallel envs).

- **`agents/buffer.py`** — `RolloutBuffer` / `VecBuffer` for trajectory storage and GAE advantage computation.

- **`training/scenarios.py`** — Curriculum system with progressive phases: solo food-finding (phases 1-5), then cooperative rich-food scenarios (phases 6+). Auto-advances based on return thresholds.

- **`core/config.py`** — Centralized `Config` dataclass with all hyperparameters.

### Import Convention

All packages have `__init__.py` re-exports. Use either style:
```python
from core.config import Config          # explicit
from core import Config                 # via __init__.py
from agents import ActorCritic, PPO     # via __init__.py
```

### Factored Action Space

Agents choose two things per step:
1. **Direction** (5): UP, DOWN, LEFT, RIGHT, STAY
2. **Action Type** (5): MOVE, ATTACK, GIVE, SIGNAL, COOPERATE

This factoring reduces complexity vs a flat 25-action space and allows independent masking per head.

### Entity Token System

All observable entities (agents + food) are represented as uniform tokens with:
- Fourier-encoded relative positions (4 frequency bands × 4 = 16 dims)
- Velocity (2), type one-hot (2), value/HP (1), social history (4 ledger channels)
- Processed through masked multi-head attention with residual connections

## Coding Conventions

- **Vectorize everything** — use PyTorch tensor ops, scatter, broadcasting, advanced indexing. Avoid Python loops for env/training logic.
- **Device-aware** — always pass `device=self.device` when creating tensors.
- **Masked operations** — use `masked_fill` for invalid actions/tokens, not conditional logic.
- **Constants** — ALL_CAPS at module level (e.g., `DIR_UP`, `ACT_ATTACK`).
- **Config via dataclass** — all hyperparameters live in `Config`, never hardcoded in logic.
- **Orthogonal init** — policy heads use 0.01 gain, value heads use 1.0 gain.

## Key Design Decisions

- **No RL framework** — raw PyTorch for full control over ledger integration and learning.
- **Objective ledger, not subjective memory** — the ledger records what happened, not how agents feel about it.
- **Curriculum learning** — agents must master solo survival before attempting cooperation.
- **Transparent world** — all actions are globally visible; reputation emerges naturally.
