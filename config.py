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
    vision_size: int = 7  # NxN observation window (must be odd)
    max_hp: float = 100.0
    hp_decay_rate: float = 0.01  # Fraction of max HP lost per tick
    max_steps_per_episode: int = 5000

    # Actions
    n_move_actions: int = 5      # UP, DOWN, LEFT, RIGHT, STAY
    n_interact_actions: int = 11  # ATTACK x4, GIVE x4, SIGNAL, COOPERATE, IDLE

    # Food
    poor_food_value: float = 10.0
    rich_food_value: float = 30.0
    poor_food_spawn_rate: float = 0.03  # Probability per empty cell per tick
    rich_food_spawn_rate: float = 0.01

    # Rewards
    r_small: float = 10.0       # Eat poor food
    r_large: float = 50.0       # Eat rich food (coop)
    r_attack_mult: float = 0.5  # Attack reward = 50% of damage dealt
    r_damage_taken: float = -0.5  # Penalty per HP lost
    r_food_share: float = 0.0   # No reward for giving food
    r_survival: float = 0.0     # No survival bonus
    r_death: float = -100.0     # Penalty for dying

    # PPO Hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01      # Entropy bonus coefficient
    vf_coef: float = 0.5        # Value function loss coefficient
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 2_000_000
    num_steps: int = 128        # Steps per rollout before update (batch_size = 8 * 128 = 1024)
    num_minibatches: int = 4
    update_epochs: int = 4
    seed: int = 42

    # Derived values (computed in __post_init__)
    batch_size: int = field(init=False)
    minibatch_size: int = field(init=False)
    vision_channels: int = field(init=False)
    ledger_channels: int = field(init=False)

    def __post_init__(self):
        self.batch_size = self.n_agents * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.vision_channels = 5 + self.n_agents  # Empty, Food_Poor, Food_Rich, Agent_Present, Agent_Health + one-hot Agent_ID
        self.ledger_channels = 4  # Damage_Dealt, Food_Given, Coop_Count, Defense_Score
