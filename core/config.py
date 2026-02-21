"""
Configuration dataclass for Multi-Agent RL system.
All hyperparameters are centralized here.
"""
from dataclasses import dataclass, field, fields


@dataclass
class Config:
    # Environment
    grid_size: int = 15
    n_agents: int = 8
    max_hp: float = 100.0
    hp_decay_rate: float = 0.01   # Fraction of max HP lost per tick (1 HP/tick)
    heal_per_tick: float = 5.0    # Max HP healed from inventory per tick
    attack_damage_fraction: float = 0.5  # Damage = attacker_hp * this fraction
    lifesteal_fraction: float = 0.25     # Attacker heals this fraction of damage dealt
    max_steps_per_episode: int = 128

    # Factored action space - direction + action type heads
    n_directions: int = 5        # UP=0, DOWN=1, LEFT=2, RIGHT=3, STAY=4
    n_action_types: int = 5      # MOVE=0, ATTACK=1, GIVE=2, SIGNAL=3, COOPERATE=4

    # Entity tokens (unified agents + food representation)
    max_food_tokens: int = 16    # Limit food tokens to K nearest
    entity_token_dim: int = field(init=False)  # Derived: fourier_bands*4 + velocity[2] + type[2] + value[1] + social[4]
    fourier_bands: int = 4       # Number of frequency octaves (1, 2, 4, 8)

    # Attention/Transformer parameters
    attention_embed_dim: int = 64
    attention_num_heads: int = 4
    group_embed_dim: int = 16       # Per-group projection dim (position, velocity, type, value, social)

    # Food
    poor_food_value: float = 10.0
    rich_food_value: float = 30.0
    poor_food_spawn_rate: float = 0.000336  # Probability per empty cell per tick (80% less)
    rich_food_spawn_rate: float = 0.0084
    food_coverage_cap: float = 0.40  # Max fraction of grid each food type can cover

    # Rewards (eating >> approaching to incentivize actually eating)
    # Rewards (eating >> approaching to incentivize actually eating)
    r_small: float = 10.0       # (Was 5.0) Eat poor food (matches legacy working behavior)
    r_large: float = 20.0       # (Reverted)
    r_attack_mult: float = 0.1  # Attack reward = 10% of ALL damage dealt
    r_damage_taken: float = -1.0  # Penalty per HP lost (scaled by HP ratio)
    r_low_hp: float = 0.0       # (Disabled) No starvation penalty (fixes -150 return)
    r_food_share: float = 1.0   # Reward for giving food to others
    r_betrayal: float = -2.0    # Penalty for attacking agents who helped you
    r_reciprocity: float = 0.5  # Bonus for cooperating with past helpers
    r_defense: float = 1.0      # Reward for attacking someone who attacked your ally
    r_revenge: float = 0.5      # Bonus for retaliating
    r_survival: float = 0.0     # (Disabled/Reverted)
    r_death: float = 0.0        # (Disabled/Reverted)
    r_coop_attempt: float = 2.0 # Intrinsic reward for choosing COOP

    # Social hierarchy reward (competitive pressure)
    r_hierarchy: float = 0.1              # Per-step bonus for top-ranked agent (0 = disabled)
    hierarchy_food_weight: float = 1.0    # Weight for cumulative food eaten
    hierarchy_damage_weight: float = 1.0  # Weight for total damage dealt
    hierarchy_hp_weight: float = 1.0      # Weight for current HP ratio
    hierarchy_kill_weight: float = 0.0    # Weight for predator kills (0 = disabled)

    # Reward shaping (for faster learning)
    r_approach_food: float = 0.1  # (Reverted) Stronger scent towards food
    r_ally_proximity: float = 0.0 # (Disabled/Reverted)

    # PPO Hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 0.95         # High gamma for multi-agent long-term planning
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.001     # Very low entropy for solo curriculum (confident actions)
    ent_coef_coop: float = 0.05 # Higher entropy for coop phases (explore COOP action)
    vf_coef: float = 1.0        # Value function loss coefficient (higher = stronger value learning)
    aux_vf_coef: float = 0.5    # Weight for auxiliary value losses (survival, resource, social)
    max_grad_norm: float = 0.5

    # SAC Hyperparameters (Discrete Soft Actor-Critic)
    sac_learning_rate: float = 3e-4       # LR for actor, critic, and alpha
    sac_tau: float = 0.005                # Polyak averaging for target networks
    sac_alpha_init: float = 0.2           # Initial entropy coefficient
    sac_auto_alpha: bool = True           # Learn alpha automatically
    sac_target_entropy_scale: float = 0.1 # target_entropy = -scale * log(1/|A|) per head
    sac_gamma: float = 0.99              # Discount (SAC typically uses higher gamma)
    sac_buffer_size: int = 500_000        # Replay buffer capacity
    sac_batch_size: int = 256             # Minibatch size for SAC updates
    sac_learning_starts: int = 5000       # Random actions before training starts
    sac_update_frequency: int = 1         # Do updates every N env steps
    sac_utd_ratio: int = 1               # Gradient steps per update (UTD=utd_ratio/update_frequency)
    sac_target_update_interval: int = 1   # Steps between target network Polyak updates

    # Training
    total_timesteps: int = 2_000_000
    num_steps: int = 128        # Steps per rollout before update (batch_size = 8 * 128 = 1024)
    num_minibatches: int = 1
    update_epochs: int = 4
    seed: int = 42
    n_envs: int = 8             # Number of parallel environments for vectorized training

    # Predator (environmental enemy)
    n_predators: int = 0               # 0 = disabled (backward compat)
    predator_hp_mult: float = 0.5      # Predator HP = max_hp * this (50 HP)
    predator_damage: float = 10.0      # Fixed damage per attack
    predator_kill_reward: float = 5.0  # Reward for landing killing blow
    predator_respawn_steps: int = 20   # Steps before dead predator respawns

    # Ablation
    ablate_ledger: bool = False  # Zero out social features in observations (for control experiments)

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

    # Set of field names that are derived (computed in __post_init__, not settable)
    _DERIVED_FIELDS = frozenset({
        'entity_token_dim', 'batch_size', 'minibatch_size', 'max_entities', 'ledger_channels'
    })

    def __post_init__(self):
        self.entity_token_dim = self.fourier_bands * 4 + 9  # fourier + velocity[2] + type[2] + value[1] + social[4]
        self.batch_size = self.n_agents * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.max_entities = self.n_agents + self.max_food_tokens + self.n_predators  # Total tokens in sequence
        self.ledger_channels = 4  # Damage_Dealt, Food_Given, Coop_Count, Defense_Score

    def to_dict(self) -> dict:
        """Serialize all fields to a plain dict (tuples -> lists for YAML compat)."""
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, tuple):
                val = list(val)
            d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        """Create Config from dict, filtering derived fields and converting lists -> tuples."""
        # Get the set of init-able field names and their types
        init_fields = {f.name: f for f in fields(cls) if f.init}
        filtered = {}
        for key, val in d.items():
            if key in cls._DERIVED_FIELDS or key.startswith('_'):
                continue
            if key not in init_fields:
                continue
            # Convert lists back to tuples for tuple-typed fields
            if isinstance(val, list):
                val = tuple(val)
            filtered[key] = val
        return cls(**filtered)
