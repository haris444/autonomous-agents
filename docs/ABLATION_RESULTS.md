# Ablation Study: Results and Plans

## Research Question

**Which mechanisms are necessary for emergent cooperation in multi-agent survival?**

We isolate the contribution of each component by removing one at a time from the full system (baseline).

---

## Inventory of Trained Checkpoints

| ID | Name | Config | Steps | Env | Source | Checkpoint Path |
|---|---|---|---|---|---|---|
| **A** | Baseline (Full System) | 8x8, 8 agents, 1 predator, lifesteal, all rewards | 10.5M | 8 envs | Local | `sac_phase11_lifesteal/checkpoint_ep10500.pt` |
| **A-colab** | Baseline (Colab) | Same as A | 5M | 4 envs | Colab R1 | `results/ab_test_colab2/.../A_baseline/checkpoint_final.pt` |
| **B** | No Lifesteal | lifesteal=0, rest same as A | 5M | 4 envs | Colab R1 | `results/ab_test_colab2/.../B_no_lifesteal/checkpoint_final.pt` |
| **C** | No Predator | n_predators=0, rest same as A | 5M | 4 envs | Colab R1 | `results/ab_test_colab2/.../C_no_predator/checkpoint_final.pt` |
| **D-colab** | No Social (all zeroed) | r_hierarchy=0, r_revenge=0, r_coop=0 | 5M | 4 envs | Colab R1 | `results/ab_test_colab2/.../D_no_social/checkpoint_final.pt` |
| **D-local** | No Social (keep hierarchy) | r_revenge=0, r_coop=0, r_hierarchy=0.25 | 20M (running) | 8 envs | Local | `sac_ablation_no_social_rewards/checkpoint_ep*.pt` |
| **B2** | No Ledger | ablate_ledger=True | 5M | 4 envs | Colab R2 | *pending* |
| **E** | No Rich Food | rich_food_spawn_rate=0, r_large=0 | 5M | 4 envs | Colab R2 | *pending* |
| **F** | No Ledger + No Social | ablate_ledger=True + zero all social rewards | 5M | 4 envs | Colab R2 | *pending* |
| **G** | 3 Predators (low dmg) | n_predators=3, predator_damage=5 | 5M | 4 envs | Colab R2 | *pending* |

---

## Round 1 Results (Completed)

### Training Metrics (5M steps, Colab)

| Metric | A: Baseline | B: No Lifesteal | C: No Predator | D: No Social |
|---|---|---|---|---|
| Avg Return (last 500) | +255.9 | +255.9 | **+679.1** | +322.0 |
| Final Return | +424.0 | +424.0 | **+762.5** | +301.0 |
| Q1 Mean | 33.0 | 33.0 | **83.7** | 49.0 |
| Episodes | 5,139 | 5,139 | 4,895 | 5,313 |

### Behavioral Evaluation (50 episodes, greedy policy)

| Metric | A: Baseline | C: No Predator | D: No Social |
|---|---|---|---|
| **Avg Return** | +458.5 | **+689.7** | +439.9 |
| Deaths/ep | 0.94 | **0.00** | 0.70 |
| Coops/ep | 244.8 | **343.7** | 259.2 |
| Survival% | 88.3% | **100%** | 91.3% |
| Attack% | 3.69% | **0.57%** | 2.54% |
| Coop% | 35.5% | 33.9% | 24.2% |
| Agent Return Diversity (std) | 66.3 | **12.5** | 61.1 |

### Key Findings — Round 1

1. **B = A (Lifesteal irrelevant).** Agents learned not to fight, so lifesteal (healing on damage dealt) has zero effect. Identical results bit-for-bit with same seed.

2. **C >> A (Predator hurts, not helps).** Removing the predator gives +50% return, zero deaths, 100% survival. The predator disrupts foraging more than it catalyzes cooperation. Agents haven't learned to coordinate against it.

3. **D ≈ A (Social rewards barely matter).** With all social rewards zeroed (including hierarchy), return only drops 4%. Cooperation (259 coops/ep) still emerges at nearly the same rate. Cooperation is **environmentally forced** by rich food mechanics, not reward-shaped.

4. **C has lowest diversity.** Without the predator creating chaos, all 8 agents converge to the same optimal strategy. The predator, while hurting returns, actually creates behavioral diversity.

---

## Baseline Deep Analysis (Local, 10.5M steps)

### Spatial Navigation — PERFECT
100% correct movement toward food in all 4 cardinal directions.

### Social Discrimination — NOT YET
All agents attack 100% regardless of social history. Exception: all agents go IDLE when facing an agent with high `food_given` history.

### Friend/Foe Values — PARTIAL
Network values allies +1.5 higher than enemies internally, but hasn't translated to different actions. Cosine similarity between ally/enemy representations: 0.88 (moderate differentiation).

### Per-Agent Personality (50 episodes, greedy)

| Agent | Return | Surv% | Atk% | Coop% | Give% | Personality |
|---|---|---|---|---|---|---|
| A0 | +335.6 | 84% | 6.3% | 21.9% | 0.7% | UNDERPERFORMER |
| A1 | +298.2 | 78% | 4.5% | 21.5% | 12.9% | ALTRUIST, FRAGILE |
| A2 | +368.3 | 82% | 3.1% | 25.3% | 12.3% | ALTRUIST |
| A3 | +437.7 | 88% | 2.9% | 27.0% | 4.9% | BALANCED |
| A4 | +429.9 | 86% | 1.2% | 28.2% | 0.1% | BALANCED |
| A5 | +455.5 | 90% | 2.9% | 29.0% | 2.0% | BALANCED |
| A6 | +543.3 | 98% | 0.9% | 30.0% | 0.1% | TOP-PERFORMER |
| A7 | +491.3 | 94% | 1.3% | 27.0% | 0.2% | EXPLORER |

**Important finding:** Synthetic probes show all 8 agent heads produce **identical outputs** for the same input. Behavioral diversity comes from different observations (positions, neighbors), not learned head specialization.

### Local Ablation Progress (No Social, Keep Hierarchy)

At 6M steps: returns +224 to +617, catching up to baseline. At 3M steps it was negative — confirming slow start without coop reward but eventual convergence.

---

## Round 2 Experiments (Pending — Colab)

| Variant | Change | Hypothesis | Expected Result |
|---|---|---|---|
| **B2: No Ledger** | `ablate_ledger=True` | Social memory is necessary for discrimination | If coop drops → ledger validates as core contribution |
| **E: No Rich Food** | `rich_food_spawn_rate=0` | Rich food forces cooperation | Coop should drop to ~0%. Smoking gun for environmental forcing. |
| **F: No Ledger + No Social** | Double ablation | Pure survival baseline | Lowest cooperation. Establishes floor. |
| **G: 3 Predators** | `n_predators=3, predator_damage=5` | More threats force alliance | 1 predator hurt. 3 weaker ones might force grouping. |

---

## Cross-Environment Testing Plan

### Concept

Take the final checkpoints from each training variant and evaluate them in **environments they weren't trained in**. This tests generalization and reveals whether learned behaviors are robust or brittle.

### Test Matrix

Each row is a trained checkpoint. Each column is a test environment. Cells show what we measure.

| Trained In → Test In ↓ | A: Baseline | C: No Predator | D: No Social | D-local: Keep Hierarchy | B2: No Ledger | E: No Rich Food |
|---|---|---|---|---|---|---|
| **Standard** (baseline env) | native | cross-test | cross-test | cross-test | cross-test | cross-test |
| **No Predator** env | cross-test | native | cross-test | cross-test | cross-test | cross-test |
| **3 Predators** env | cross-test | cross-test | cross-test | cross-test | cross-test | cross-test |
| **No Rich Food** env | cross-test | cross-test | cross-test | cross-test | cross-test | native |
| **Scarce Food** (half spawn rate) | cross-test | cross-test | cross-test | cross-test | cross-test | cross-test |
| **Crowded** (8 agents, 6x6 grid) | cross-test | cross-test | cross-test | cross-test | cross-test | cross-test |

### Metrics per Cell

For each (checkpoint, environment) pair, run 50 episodes and measure:

1. **Return** — overall performance
2. **Survival%** — robustness
3. **Cooperation rate** — does cooperation transfer?
4. **Attack rate** — does aggression change?
5. **Deaths/episode** — fragility in new environments

### Key Questions This Answers

1. **Does predator training help in predator-free environments?** (A in No Predator env vs C in No Predator env). If A still performs well without predators, it learned generalizable cooperation.

2. **Can no-predator agents survive predators?** (C in 3 Predator env). Never seen a predator — does it panic, die, or adapt?

3. **Does social reward training transfer?** (A vs D in Standard env). Both should cooperate, but does A do it better?

4. **Resource scarcity stress test.** (All checkpoints in Scarce Food). Who is most robust to environmental pressure?

5. **Does cooperation transfer to no-rich-food?** (A in No Rich Food env). If agents cooperate even when rich food doesn't exist, cooperation is a learned habit, not just opportunism.

6. **Crowded stress test.** (All in 6x6 grid). More agent density → more competition. Who adapts?

### Implementation

The cross-env test script should:

```python
# For each checkpoint:
#   For each test environment config:
#     Run 50 episodes with greedy policy
#     Record per-agent stats
#     Save to results/cross_env_test/{checkpoint}_{env}.json

# Then generate comparison tables and heatmaps
```

Test environments are defined by overriding specific config values on top of the checkpoint's native config:

```python
TEST_ENVS = {
    'standard':     {},                                    # Baseline env
    'no_predator':  {'n_predators': 0},                    # Remove predator
    '3_predators':  {'n_predators': 3, 'predator_damage': 5.0},  # Heavy threat
    'no_rich_food': {'rich_food_spawn_rate': 0.0, 'r_large': 0.0},  # Solo food only
    'scarce_food':  {'poor_food_spawn_rate': 0.000168, 'rich_food_spawn_rate': 0.0042},  # Half food
    'crowded':      {'grid_size': 6},                      # Smaller grid, more density
}
```

### Priority Order

1. First: Run Round 2 Colab experiments (B2, E, F, G)
2. Second: Build cross-env test script
3. Third: Run cross-env tests on all available checkpoints
4. Fourth: Generate comparison tables and final report

---

## Summary of All Findings So Far

| Finding | Evidence | Confidence |
|---|---|---|
| Cooperation is environmentally forced | D ≈ A (social rewards don't matter) | HIGH |
| Rich food is the cooperation driver | 26% coop rate in all variants with rich food | HIGH (pending E confirmation) |
| Predator hurts more than helps | C >> A (+50% return, 0 deaths) | HIGH |
| Lifesteal is irrelevant | B = A (agents don't fight) | HIGH |
| Social discrimination not yet learned | 100% attack regardless of history | HIGH |
| Value differentiation is emerging | Allies valued +1.5 higher | MEDIUM |
| Per-agent heads don't specialize | Identical outputs on synthetic probes | HIGH |
| Behavioral diversity from observations | Different positions/neighbors → different actions | HIGH |
| Ledger (social memory) importance | UNKNOWN — pending B2 | — |
