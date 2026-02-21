# Final Experiments: The Ingredients of Emergent Social Behavior

## Research Question

**Can complex social behaviors (cooperation, altruism, discrimination) emerge from simple survival instincts — and which mechanisms are necessary?**

## Experimental Design

Four experiments, each removing one mechanism from the full system. This isolates the contribution of each component.

### Variants

| Variant | Change from Baseline | Hypothesis |
|---|---|---|
| **A: Full System** | Nothing — all mechanisms enabled | Social behaviors emerge when agents have survival pressure, social memory, external threats, and reward shaping |
| **B: No Social Memory** | `ablate_ledger=True` (zero ledger channels in observations) | Without interaction history, agents cannot discriminate friend from foe → no stable relationships form |
| **C: No External Threat** | `n_predators=0` | Without a common enemy, agents have less incentive to cooperate → reduced alliance formation |
| **D: No Reward Shaping** | Zero all social rewards (`r_hierarchy`, `r_revenge`, `r_coop_attempt`, `r_reciprocity`, `r_defense`, `r_betrayal`, `r_food_share`) | Without explicit social incentives, cooperation can/cannot emerge from pure survival pressure alone |

### Why These Four

1. **The Ledger (B)** is the project's core contribution — an objective record of interactions that agents observe. If ablating it kills social behavior, that validates the entire design. If it doesn't matter, we learn something unexpected.

2. **Predators (C)** test the **common enemy hypothesis** from evolutionary biology: external threats drive in-group cooperation. This is directly testable in our environment.

3. **Social rewards (D)** test the deepest question: is reward shaping necessary, or can cooperation emerge from environmental pressure alone? If D still produces cooperation, that's the strongest possible finding.

4. **Full system (A)** is the control — it establishes that social behavior CAN emerge when all pieces are in place.

## Training Configuration

All variants share the same base config, differing only in the ablated mechanism.

```yaml
# Environment
grid_size: 15              # Room for agents to choose who to interact with
n_agents: 8
max_hp: 100.0
hp_decay_rate: 0.01
attack_damage_fraction: 0.125
lifesteal_fraction: 0.25   # Combat heals attacker — makes aggression viable
max_steps_per_episode: 128

# Food
poor_food_value: 10.0
rich_food_value: 30.0      # Rich food requires 2+ agents → forces cooperation
food_coverage_cap: 0.40

# Social Rewards (all enabled in A/B/C, all zero in D)
r_hierarchy: 0.25
r_revenge: 0.5
r_coop_attempt: 2.0
r_reciprocity: 0.5
r_defense: 1.0
r_betrayal: -2.0
r_food_share: 1.0

# Predator (enabled in A/B/D, disabled in C)
n_predators: 1
predator_hp_mult: 0.5
predator_damage: 10.0
predator_kill_reward: 5.0
predator_respawn_steps: 20

# SAC Hyperparameters
sac_learning_rate: 0.001
sac_tau: 0.01
sac_alpha_init: 0.8
sac_auto_alpha: true
sac_target_entropy_scale: 0.1
sac_gamma: 0.95
sac_buffer_size: 500000
sac_batch_size: 2048
sac_learning_starts: 50000
sac_update_frequency: 4

# Training
total_timesteps: 10000000   # 10M steps — enough for full curriculum
n_envs: 4                   # 4 per run × 4 runs = 16 total on GPU
seed: 42
pretrain_mode: true          # Curriculum ON
curriculum_enabled: true
```

## Curriculum Progression

Agents progress through 12 phases. Early phases are shared across all variants (predators disabled in phases 1-9 by design). Differences emerge in phases 10-12 where social dynamics activate.

| Phases | Focus | What Agents Learn |
|---|---|---|
| 1-5 | Solo food-finding | Basic navigation, food collection |
| 6-8 | Cooperative food | Move toward partner, collect rich food together |
| 9 | Low-HP cooperation | Cooperate under resource pressure |
| 10-12 | Social dynamics | Discrimination, alliance, defense (predators active) |

## Measurements

### Training Metrics (from CSV logs)

- **Episode return** over time — learning speed and final performance
- **Curriculum phase** over time — how fast each variant progresses
- **Q-values** — what agents expect to earn (value estimation)
- **Entropy temperatures** (alpha) — exploration vs exploitation balance

### Behavioral Metrics (post-training evaluation)

Load each variant's final checkpoint and run evaluation episodes. Measure:

| Metric | What It Captures |
|---|---|
| **Cooperation rate** | Fraction of rich food collected (requires 2+ agents) |
| **Attack frequency** | How often agents attack each other |
| **Food sharing rate** | GIVE actions directed at other agents |
| **Predator kill rate** | Coordinated predator kills (where applicable) |
| **Average survival time** | Steps alive per episode |
| **Discrimination index** | Difference in behavior toward allies vs strangers (from ledger history) |
| **Curriculum phase reached** | How far each variant progresses |

### Expected Results

| Metric | A (Full) | B (No Ledger) | C (No Predator) | D (No Social) |
|---|---|---|---|---|
| Cooperation | High | Low (can't identify partners) | Medium (less pressure) | Medium-Low (no incentive) |
| Discrimination | High | None (no social info) | Medium | Medium-High (can see ledger) |
| Attack frequency | Medium | High (can't tell friend/foe) | Medium | High (no punishment) |
| Predator kills | High | Low (uncoordinated) | N/A | Medium |
| Curriculum speed | Fast | Slower (phases 10+ harder) | Similar to A | Slower (no social rewards) |

## Compute Budget

- **Platform:** Google Colab T4 GPU (15.6 GB VRAM)
- **4 runs in parallel**, 4 envs each
- **10M timesteps per run**
- **Estimated time:** 4-6 hours total
- **Output:** ~4 checkpoints + 4 CSV logs + comparison plots

## File Structure

```
results/ab_test/
├── ab_comparison.png          # Side-by-side training curves
├── ab_behavioral.png          # Behavioral metrics comparison
├── A_full_system/
│   ├── config.yaml
│   ├── log.csv
│   ├── stdout.log
│   └── checkpoint_*.pt
├── B_no_ledger/
│   ├── ...
├── C_no_predator/
│   ├── ...
└── D_no_social/
    ├── ...
```
