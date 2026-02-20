# Project Specification v2.3

## Project Title: Emergent Morality & Social Dynamics in Multi-Agent RL

**Core Investigation:** Can complex social behaviors (policing, tribalism, altruism) emerge from simple survival instincts when agents have social memory (Ledger) and social intuition (pre-computed social features)?

---

## 1. Environment (The World)

### Physics

| Feature | Rule |
|---------|------|
| **Grid Type** | Single Occupancy (Hard Collision). Only 1 agent per cell. |
| **Obstacles** | Agents act as physical walls to each other. |
| **Food Layer** | Agents walk over food. Food does not block movement. |
| **Food (Poor)** | Single-agent harvestable. Low HP restore. |
| **Food (Rich)** | Requires 2+ agents adjacent to the food cell to unlock/eat. High HP restore. |
| **Social Visibility** | All actions are globally observed by all agents instantly. No hidden social information. |
| **HP Decay** | Agents lose `X%` of max HP per tick (configurable). Forces food-seeking. |

### Movement Resolution: "The Heavyweight Rule"

When multiple agents try to enter the same cell `(x,y)`:
- **Priority:** Agent with highest current HP wins the spot
- **Losers:** All other agents "bounce" (stay in previous cell)
- **Ties:** Random coin flip
- **Blocking:** Cannot displace a stationary agent

### Combat

- **Damage:** `dmg = attacker_HP / 2`
- Stronger agents hit harder, weaker agents hit softer

---

## 2. Agent Architecture (The Brain)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Algorithm** | PPO or DQN | Standard, well-documented |
| **Memory** | Global Ledger Tensor | Objective action history, agents derive feelings |
| **Vision** | NxN grid centered on agent | Spatial awareness, CNN-friendly |
| **Network** | Feedforward (or CNN + MLP) | No recurrence — Ledger provides memory |

---

## 3. The Ledger (Global Interaction Tensor)

### Structure: `[N_agents × N_agents × 4]`

| Axis | Meaning |
|------|---------|
| **X (Source)** | The Actor — who performed the action |
| **Y (Target)** | The Subject — who received the action |
| **Z (Channels)** | 4 objective facts about their history |

### The 4 Channels

| Channel | Name | Records | Why Distinct |
|---------|------|---------|--------------|
| 0 | `Damage_Dealt` | Total HP removed by Actor from Target | Tracks aggression/bullying |
| 1 | `Food_Given` | Total food transferred Actor → Target | Tracks altruism/bribes |
| 2 | `Coop_Count` | Times they simultaneously unlocked Rich Food | Tracks competence/trust ("we work well together" ≠ "he gave me lunch") |
| 3 | `Defense_Score` | Damage Actor dealt to someone attacking Target | Tracks loyalty ("he saved my life" ≠ "he gave me lunch") |

### Key Principle

> **Store objective ACTIONS, not subjective relationships.**
> Agents derive feelings from facts. The Ledger is the courtroom transcript, not the verdict.

---

## 4. Observation Space

### A. Spatial Vision (NxN Grid)

Physical surroundings only — what's nearby.

| Channel | Description |
|---------|-------------|
| `Empty` | 1 if nothing |
| `Food_Poor` | 1 if solo food |
| `Food_Rich` | 1 if coop food |
| `Agent_Present` | 1 if agent here |
| `Agent_Health` | Their health (0 if empty) |

**Shape:** `(N, N, num_channels)`

### B. Social Information (Global)

Agents receive the full Ledger directly. All social history is globally visible.

**Shape:** `[N_agents × N_agents × 4]` (the Ledger itself)

### C. Signals (Global)

Per-agent flag indicating who is currently signaling "follow me".

**Shape:** `[N_agents]` (1 if signaling, 0 otherwise)

### D. Self-State Vector

| Feature | Description |
|---------|-------------|
| `My_Health` | Current HP |

---

## 5. Action Space

| Head | Options |
|------|---------|
| **Move** | `[Up, Down, Left, Right, Stay]` |
| **Interact** | `[Attack, Give_Food, Signal, Idle]` |

### Signal Action

- **Meaning:** "Follow me"
- **Broadcast:** Sent to all agents globally
- **Reception:** Agents receive it in their observation input
- **Interpretation:** Agents decide freely how to respond

---

## 6. Reward Structure

| Event | Reward | Ledger Update |
|-------|--------|---------------|
| Eat poor food | `+R_small` | — |
| Eat rich food (coop) | `+R_large` | `Coop_Count[A][B]++`, `Coop_Count[B][A]++` |
| Deal damage to B | `+dmg` | `Damage_Dealt[me][B] += dmg` |
| Receive damage from B | `-R_damage` | `Damage_Dealt[B][me] += dmg` |
| Give food to B | `-opportunity_cost` | `Food_Given[me][B] += amt` |
| Receive food from B | `+amt` | `Food_Given[B][me] += amt` |
| Defend B (attack B's attacker) | — | `Defense_Score[me][B] += dmg_dealt` |

**Give Food Cost:** `opportunity_cost = min(food_value, max_HP - my_HP)`
Giving food costs what you would have gained from eating it. At 90/100 HP with 20-value food, cost = 10.

---

## 7. Experiments

| Scenario | Config | Expected Emergence |
|----------|--------|-------------------|
| **Abundance** | Mostly poor food | Individualists |
| **Scarcity** | Mostly rich food | Forced partnerships, trust dynamics |

---

## 8. Open Design Questions

1. Grid size / Vision size?
2. Agent count?

---

## 9. Implementation Phases

| Phase | Deliverable |
|-------|-------------|
| 1 | Core environment (grid, movement, food) |
| 2 | Ledger tensor + update logic |
| 3 | Observation builder (NxN grid + Ledger) |
| 4 | RL training pipeline |
| 5 | Visualization + experiments |
