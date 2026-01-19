"""
Configuration dataclass for Multi-Agent RL system.
All hyperparameters are centralized here.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # Environment
    grid_size: int = 15
    n_agents: int = 8
    max_hp: float = 100.0
    hp_decay_rate: float = 0.01   # Fraction of max HP lost per tick (1 HP/tick)
    max_steps_per_episode: int = 128

    # Actions - unified action space (move OR interact, not both)
    n_actions: int = 15          # 5 move + 10 interact (no IDLE)
    n_move_actions: int = 5      # UP, DOWN, LEFT, RIGHT, STAY (actions 0-4)
    n_interact_actions: int = 10 # ATTACK x4, GIVE x4, SIGNAL, COOPERATE (actions 5-14, no IDLE)
    n_directions: int = 4        # 4 cardinal directions for ATTACK/GIVE

    # Entity tokens (unified agents + food representation)
    max_food_tokens: int = 16    # Limit food tokens to K nearest
    entity_token_dim: int = 8    # dx, dy, type, quality/hp, social[4]

    # Attention/Transformer parameters
    attention_embed_dim: int = 64
    attention_num_heads: int = 4

    # Food
    poor_food_value: float = 10.0
    rich_food_value: float = 30.0
    poor_food_spawn_rate: float = 0.000336  # Probability per empty cell per tick (80% less)
    rich_food_spawn_rate: float = 0.0084
    food_coverage_cap: float = 0.40  # Max fraction of grid each food type can cover

    # Rewards (eating >> approaching to incentivize actually eating)
    r_small: float = 5.0        # Eat poor food (5x approach reward for 10 steps)
    r_large: float = 20.0       # Eat rich food (coop)
    r_attack_mult: float = 0.1  # Attack reward = 10% of ALL damage dealt (5x reduced)
    r_damage_taken: float = -1.0  # Penalty per HP lost (scaled by HP ratio)
    r_low_hp: float = -0.0625   # Per-tick penalty when HP is low (minimal signal)
    r_food_share: float = 0.0   # No bonus - giving food already transfers HP naturally
    r_defense: float = 0.0      # No bonus - defense value is in the OBSERVATION, not reward
    r_revenge: float = 0.5      # Bonus for retaliating against attackers (50% of damage dealt)
    r_survival: float = 0.0     # No survival bonus
    r_death: float = 0.0        # No death penalty (survival incentive from HP decay)
    r_coop_attempt: float = 2.0 # Intrinsic reward for choosing COOP when near rich food + ally

    # PPO Hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 0.95         # High gamma for multi-agent long-term planning
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.001     # Very low entropy for solo curriculum (confident actions)
    ent_coef_coop: float = 0.05 # Higher entropy for coop phases (explore COOP action)
    vf_coef: float = 0.5        # Value function loss coefficient
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 2_000_000
    num_steps: int = 128        # Steps per rollout before update (batch_size = 8 * 128 = 1024)
    num_minibatches: int = 1
    update_epochs: int = 4
    seed: int = 42

    # Pretraining (curriculum learning)
    pretrain_mode: bool = False          # Single-agent pretraining mode
    pretrain_spawn_agents: int = 1       # How many agents to spawn (others start dead)

    # Curriculum learning (food distance progression)
    curriculum_enabled: bool = True      # Enable curriculum learning for food spawning
    curriculum_phase: int = 1            # Starting phase (1-8)
    curriculum_thresholds: tuple = (50, 40, 30)  # Avg return thresholds to advance phases

    # Cooperation curriculum (phases 6-8: 2 agents + rich food only)
    # Phase 6: 2 agents spawn adjacent to rich food (distance 1)
    # Phase 7: 2 agents spawn near rich food (distance 2)
    # Phase 8: 2 agents spawn near rich food (distance 3)
    # Both agents use agent 0's network (clone) for parameter sharing

    # Derived values (computed in __post_init__)
    batch_size: int = field(init=False)
    minibatch_size: int = field(init=False)
    max_entities: int = field(init=False)  # n_agents + max_food_tokens
    ledger_channels: int = field(init=False)

    def __post_init__(self):
        self.batch_size = self.n_agents * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.max_entities = self.n_agents + self.max_food_tokens  # Total tokens in sequence
        self.ledger_channels = 4  # Damage_Dealt, Food_Given, Coop_Count, Defense_Score
