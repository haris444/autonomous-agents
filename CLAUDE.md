# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent reinforcement learning system investigating whether complex social behaviors (cooperation, altruism, tribalism, policing) can emerge from simple survival instincts. Agents live in a GridWorld, compete for food, and maintain a shared **Ledger** of objective interaction history. Built from scratch in PyTorch (CleanRL-style PPO), no RL framework dependencies.

## Commands

```bash
# Single-environment training
python train.py
python train.py --visualize                    # with live rendering
python train.py --visualize --render-every 50  # render every N updates

# Vectorized multi-environment training (faster)
python train_vec.py --n_envs 8 --pretrain
python train_vec.py --n_envs 16 --batch_size 1024

# Visualize a saved checkpoint
python visualize_checkpoint.py --checkpoint path/to/checkpoint.pt

# Diagnostics
python diagnose_social.py
python diagnose_spatial.py
python diagnose_friend_foe.py
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

- **`environment.py`** — Fully vectorized GridWorld with PyTorch tensors (no Python loops). Single-occupancy grid, HP decay, poor food (solo) vs rich food (requires 2+ agents). Movement uses "Heavyweight Rule" (highest HP wins contested cells).

- **`ledger.py`** — `[n_agents × n_agents × 4]` tensor storing objective facts: damage dealt, food given, coop count, defense score. Core design principle: store actions, not relationships — agents derive feelings from facts.

- **`network.py`** — Two main classes:
  - `ObservationEncoder`: Fourier-encoded entity tokens (agents + food) processed through multi-head attention (4 heads, 64 dim), plus signal encoder and self-state encoder. Outputs 144-dim feature vector.
  - `ActorCritic`: Shared trunk → factored dual-action heads (direction: 5, action type: 5) + value head + auxiliary value heads for decomposed rewards (survival/resource/social).

- **`ppo.py`** — Three variants: `PPO` (single agent), `IndependentPPO` (N independent networks, supports clone mode for curriculum), `VmapPPO` (vectorized across parallel envs).

- **`buffer.py`** — `RolloutBuffer` / `VecBuffer` for trajectory storage and GAE advantage computation.

- **`scenarios.py`** — Curriculum system with progressive phases: solo food-finding (phases 1-5), then cooperative rich-food scenarios (phases 6+). Auto-advances based on return thresholds.

- **`config.py`** — Centralized `Config` dataclass with all hyperparameters.

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
