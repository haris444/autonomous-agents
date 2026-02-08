"""
GridWorld Environment for Multi-Agent RL - VECTORIZED VERSION.

Features:
- Single occupancy grid (hard collision)
- Poor food (solo) and Rich food (requires 2+ adjacent agents)
- HP decay per tick
- Heavyweight movement resolution
- Combat system (dmg = attacker_HP / 2)

All operations vectorized with PyTorch for GPU acceleration.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from config import Config
from ledger import Ledger

# Import scenarios module (lazy import to avoid circular dependency)
# Scenarios are applied via apply_scenario() after reset()


# Factored action space: Direction + Action Type
# Direction head (5 outputs)
DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT, DIR_STAY = 0, 1, 2, 3, 4

# Action type head (5 outputs)
ACT_MOVE, ACT_ATTACK, ACT_GIVE, ACT_SIGNAL, ACT_COOPERATE = 0, 1, 2, 3, 4

# Direction deltas: UP, DOWN, LEFT, RIGHT, STAY
DIR_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]


class GridWorld:
    """
    Multi-agent grid environment with social dynamics.
    Fully vectorized for GPU acceleration.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.grid_size = config.grid_size
        self.n_agents = config.n_agents

        # Ledger for social memory
        self.ledger = Ledger(config.n_agents, device)

        # Direction deltas tensor [5, 2] for movement (4 cardinal directions + stay)
        self.direction_deltas = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1],  # Cardinal: UP, DOWN, LEFT, RIGHT
             [0, 0]],                            # STAY
            device=device, dtype=torch.long
        )

        # Direction deltas tensor [4, 2] for attacks/give (cached for reuse)
        self.dir_deltas_4 = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]],
            device=device, dtype=torch.long
        )

        # State tensors (initialized in reset)
        self.agent_positions: torch.Tensor = None  # [n_agents, 2] (row, col)
        self.prev_agent_positions: torch.Tensor = None  # [n_agents, 2] previous positions for velocity
        self.agent_hp: torch.Tensor = None         # [n_agents]
        self.agent_alive: torch.Tensor = None      # [n_agents] bool
        self.agent_inventory: torch.Tensor = None  # [n_agents] stored food units
        self.poor_food: torch.Tensor = None        # [grid_size, grid_size] bool
        self.rich_food: torch.Tensor = None        # [grid_size, grid_size] bool
        self.signals: torch.Tensor = None          # [n_agents] bool
        self.occupancy: torch.Tensor = None        # [grid_size, grid_size] -> agent_id or -1

        # Episode tracking for warmup food boost
        self.episode_count = 0
        self.warmup_episodes = 0  # Disabled: was 100x food spawn/reward, 10x episode length

        self.step_count = 0

        # Curriculum learning tracking
        self.curriculum_phase = config.curriculum_phase if config.curriculum_enabled else 4
        self.episode_returns = []  # Track returns for phase advancement
        self.current_episode_reward = 0.0  # Accumulator for current episode

        # Attack history buffer for extended defense tracking (last 3 steps)
        # Shape: [3, n_agents, n_agents] - recent_attacks[t, attacker, victim] = damage
        self.recent_attacks = torch.zeros((3, config.n_agents, config.n_agents), device=device)

        # Predator state (environmental enemy)
        self.n_predators = config.n_predators
        self.predator_positions: torch.Tensor = None   # [n_predators, 2]
        self.predator_hp: torch.Tensor = None          # [n_predators]
        self.predator_alive: torch.Tensor = None       # [n_predators] bool
        self.predator_prev_positions: torch.Tensor = None  # [n_predators, 2]
        self.predator_respawn_timer: torch.Tensor = None   # [n_predators]
        self.predator_ledger: torch.Tensor = None   # [n_agents, n_predators, 2]

        # Cooperation tracking for diagnostics
        self.last_coop_success_count = 0  # Number of successful cooperations in last step

        # Current scenario (set via apply_scenario() after reset())
        self._current_scenario = None

    def _update_occupancy(self) -> None:
        """Update occupancy grid from agent and predator positions - FULLY VECTORIZED."""
        self.occupancy.fill_(-1)
        alive_mask = self.agent_alive

        if alive_mask.any():
            alive_indices = torch.where(alive_mask)[0]
            alive_positions = self.agent_positions[alive_indices]
            rows = alive_positions[:, 0]
            cols = alive_positions[:, 1]
            # Advanced indexing: write all at once
            self.occupancy[rows, cols] = alive_indices

        # Predators occupy cells as n_agents + pred_idx (>= n_agents means predator)
        if self.n_predators > 0 and self.predator_alive is not None:
            alive_pred = torch.where(self.predator_alive)[0]
            if alive_pred.numel() > 0:
                pred_pos = self.predator_positions[alive_pred]
                pred_rows = pred_pos[:, 0]
                pred_cols = pred_pos[:, 1]
                self.occupancy[pred_rows, pred_cols] = self.n_agents + alive_pred

    def _init_predators(self) -> None:
        """Initialize all predators at random grid edges with full HP."""
        n_pred = self.n_predators
        self.predator_hp = torch.full((n_pred,), self.config.max_hp * self.config.predator_hp_mult, device=self.device)
        self.predator_alive = torch.ones(n_pred, device=self.device, dtype=torch.bool)
        self.predator_respawn_timer = torch.zeros(n_pred, device=self.device, dtype=torch.long)
        self.predator_positions = torch.zeros((n_pred, 2), device=self.device, dtype=torch.long)
        for p in range(n_pred):
            self._respawn_predator(p)
        self.predator_prev_positions = self.predator_positions.clone()
        self.predator_ledger = torch.zeros(self.n_agents, n_pred, 2, device=self.device)

    def _respawn_predator(self, p: int) -> None:
        """Respawn predator at a random grid edge cell that's unoccupied."""
        gs = self.grid_size
        self.predator_hp[p] = self.config.max_hp * self.config.predator_hp_mult
        self.predator_alive[p] = True
        self.predator_respawn_timer[p] = 0

        # Collect all edge cells
        edge_cells = []
        for c in range(gs):
            edge_cells.extend([(0, c), (gs - 1, c)])  # top and bottom rows
        for r in range(1, gs - 1):
            edge_cells.extend([(r, 0), (r, gs - 1)])  # left and right cols

        # Shuffle and find first unoccupied
        perm = torch.randperm(len(edge_cells))
        for idx in perm:
            r, c = edge_cells[idx.item()]
            if self.occupancy[r, c] == -1:
                self.predator_positions[p] = torch.tensor([r, c], device=self.device)
                return

        # Fallback: place at corner even if occupied
        self.predator_positions[p] = torch.tensor([0, 0], device=self.device)

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment and return initial observations for all agents."""
        self.ledger.reset()
        self.recent_attacks.zero_()  # Clear attack history
        self.step_count = 0
        self.episode_count += 1
        self.current_episode_reward = 0.0
        self.partner_relationship = None  # Set by scenario if scripted partners

        # Initialize occupancy grid
        self.occupancy = torch.full(
            (self.grid_size, self.grid_size), -1,
            device=self.device, dtype=torch.long
        )

        # Initialize agent positions
        if self.config.pretrain_mode and self.config.curriculum_enabled and self.curriculum_phase <= 5:
            # Solo curriculum (phases 1-5): place agent in center of grid
            center = self.grid_size // 2
            self.agent_positions = torch.zeros((self.n_agents, 2), device=self.device, dtype=torch.long)
            self.agent_positions[0] = torch.tensor([center, center], device=self.device)
            # Other agents get random positions (they'll be dead in pretrain mode anyway)
            if self.n_agents > 1:
                other_positions = torch.randperm(self.grid_size * self.grid_size, device=self.device)[:self.n_agents - 1]
                rows = other_positions // self.grid_size
                cols = other_positions % self.grid_size
                self.agent_positions[1:] = torch.stack([rows, cols], dim=1)
        elif self.config.pretrain_mode and self.config.curriculum_enabled and self.curriculum_phase >= 6:
            # Coop curriculum (phases 6-8): place 2 agents near rich food
            # Rich food spawns in center, agents spawn adjacent to it
            self.agent_positions = torch.zeros((self.n_agents, 2), device=self.device, dtype=torch.long)
            center = self.grid_size // 2
            # Agent 0 and 1 will be positioned near the food (set after food spawns)
            self.agent_positions[0] = torch.tensor([center, center], device=self.device)
            self.agent_positions[1] = torch.tensor([center, center + 1], device=self.device)
            # Other agents get random positions (they'll be dead anyway)
            if self.n_agents > 2:
                other_positions = torch.randperm(self.grid_size * self.grid_size, device=self.device)[:self.n_agents - 2]
                rows = other_positions // self.grid_size
                cols = other_positions % self.grid_size
                self.agent_positions[2:] = torch.stack([rows, cols], dim=1)
        else:
            # Normal: random positions, no overlap
            all_positions = torch.randperm(self.grid_size * self.grid_size, device=self.device)[:self.n_agents]
            rows = all_positions // self.grid_size
            cols = all_positions % self.grid_size
            self.agent_positions = torch.stack([rows, cols], dim=1)

        # Initialize HP, alive status, and inventory
        self.agent_hp = torch.full((self.n_agents,), self.config.max_hp, device=self.device)
        self.agent_alive = torch.ones(self.n_agents, device=self.device, dtype=torch.bool)
        self.agent_inventory = torch.zeros(self.n_agents, device=self.device)  # Start with no stored food
        self.food_eaten_total = torch.zeros(self.n_agents, device=self.device)

        # Pretraining mode: only keep appropriate number of agents alive
        if self.config.pretrain_mode:
            if self.curriculum_phase >= 6:
                # Cooperation phases: 2 agents active
                n_active = 2
            else:
                # Solo phases: 1 agent active
                n_active = self.config.pretrain_spawn_agents
            self.agent_alive[n_active:] = False
            self.agent_hp[n_active:] = 0
            if self.episode_count == 1:  # Print once on first episode
                print(f"[Pretrain] Active agents: {self.agent_alive.sum().item()}/{self.n_agents}")

        # Initialize food grids
        self.poor_food = torch.zeros((self.grid_size, self.grid_size),
                                      device=self.device, dtype=torch.bool)
        self.rich_food = torch.zeros((self.grid_size, self.grid_size),
                                      device=self.device, dtype=torch.bool)

        # Update occupancy and spawn food
        self._update_occupancy()
        if self.config.pretrain_mode and self.config.curriculum_enabled:
            self._spawn_food_curriculum()
        else:
            self._initialize_food_at_capacity()

        # Initialize signals (none active)
        self.signals = torch.zeros(self.n_agents, device=self.device, dtype=torch.bool)

        # Initialize previous positions for velocity computation (same as current at reset)
        self.prev_agent_positions = self.agent_positions.clone()

        # Initialize predator state
        if self.n_predators > 0:
            self._init_predators()
            self._update_occupancy()

        return self._get_all_observations()

    def apply_scenario(self, scenario: 'Scenario') -> None:
        """
        Apply a scenario to configure the environment after reset().

        The scenario controls agent positioning, activation, and food spawning.
        This should be called immediately after reset() in the training loop.

        Args:
            scenario: The scenario to apply (from scenarios.py)
        """
        self._current_scenario = scenario

        # Let scenario setup agent positions and activation
        scenario.setup(self)

        # Clear existing food and let scenario spawn fresh
        self.poor_food.zero_()
        self.rich_food.zero_()
        scenario.spawn_food(self)

    def step(self, directions: torch.Tensor, action_types: torch.Tensor) -> Tuple[
        Dict[str, torch.Tensor],  # observations (batched)
        torch.Tensor,              # rewards [n_agents]
        torch.Tensor,              # dones [n_agents]
        dict                       # infos
    ]:
        """
        Execute one environment step with factored actions.

        Factored action space:
            directions: [n_agents] int tensor (0-4: UP, DOWN, LEFT, RIGHT, STAY)
            action_types: [n_agents] int tensor (0-4: MOVE, ATTACK, GIVE, SIGNAL, COOPERATE)

        Direction is used for MOVE, ATTACK, GIVE. Ignored for SIGNAL and COOPERATE.

        Args:
            directions: [n_agents] direction (0-4)
            action_types: [n_agents] action type (0-4)

        Returns:
            observations, rewards, dones, infos
        """
        self.step_count += 1

        # Store previous positions for velocity calculation (before any movement)
        self.prev_agent_positions = self.agent_positions.clone()

        # === Decompose factored actions ===
        is_move = (action_types == ACT_MOVE)
        is_attack = (action_types == ACT_ATTACK)
        is_give = (action_types == ACT_GIVE)
        is_signal = (action_types == ACT_SIGNAL)
        is_coop = (action_types == ACT_COOPERATE)

        # === Track HP at start ===
        hp_before = self.agent_hp.clone()

        # === Track distance to food BEFORE movement (for approach reward) ===
        dist_to_food_before, _ = self._compute_nearest_food_info(self.agent_positions)

        # Clear signals from last step
        self.signals.zero_()

        # 1. Process interactions FIRST (attack, give food, signal)
        #    This happens from CURRENT position before movement
        #    So agent attacks what they SEE in their observation
        #    Now also handles agent-vs-predator attacks
        attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, give_rewards, predator_kill_rewards = self._resolve_interactions(
            directions, action_types, is_attack, is_give, is_signal
        )

        # 2. Resolve movement (Heavyweight Rule)
        #    Only process if agent chose MOVE action
        #    Predators block cells (treated as stationary)
        self._resolve_movement(directions, is_move)
        self._update_occupancy()

        # 3. Predator movement + attack (after agent movement)
        predator_damage = self._predator_step()

        # 4. Apply HP decay (agents only, predators don't decay)
        self._apply_hp_decay()

        # 5. Process food eating (agents on food cells after movement)
        food_rewards = self._process_food_eating(action_types)

        # 5a. Compute intrinsic reward for attempting COOP near rich food + ally
        intrinsic_coop_rewards = self._compute_intrinsic_coop_rewards(action_types)

        # 5b. Auto-consume inventory to heal (if HP < max and have inventory)
        self._consume_inventory()

        # 5c. Hierarchy rewards (competitive ranking bonus)
        hierarchy_rewards = self._compute_hierarchy_rewards()

        # 6. Spawn new food
        self._spawn_food()

        # 7. Check deaths (agents + predators)
        death_rewards = self._check_deaths()
        self._update_occupancy()

        # === EXPLICIT REWARD COMPUTATION ===
        hp_after = self.agent_hp.clone()

        # Include predator damage in total damage taken
        total_damage_taken = damage_taken + predator_damage

        # Non-linear damage pain (lower HP = hurts more)
        # At full HP: multiplier ≈ 1, at 10% HP: multiplier ≈ 10
        hp_ratio = hp_before / self.config.max_hp
        pain_multiplier = 1.0 / (hp_ratio + 0.1)
        damage_pain = total_damage_taken * pain_multiplier * self.config.r_damage_taken

        # Low HP penalty (constant per tick, scales with how low HP is)
        # At full HP: penalty ≈ 0, at 10% HP: penalty ≈ 0.9 * r_low_hp
        hp_ratio_after = hp_after / self.config.max_hp
        low_hp_penalty = (1.0 - hp_ratio_after) * self.config.r_low_hp * self.agent_alive.float()

        # === APPROACH REWARD (reward shaping for faster learning) ===
        # Compute distance to food AFTER movement
        dist_to_food_after, _ = self._compute_nearest_food_info(self.agent_positions)
        # Reward for getting closer (positive when dist decreases)
        # Handle inf values (no food) by setting approach_delta to 0
        approach_delta = dist_to_food_before - dist_to_food_after
        approach_delta = torch.where(
            torch.isinf(dist_to_food_before) | torch.isinf(dist_to_food_after),
            torch.zeros_like(approach_delta),
            approach_delta
        )
        approach_reward = approach_delta * self.config.r_approach_food * self.agent_alive.float()

        # Survival bonus (constant positive reward for staying alive)
        survival_bonus = self.config.r_survival * self.agent_alive.float()

        # Combine all rewards
        rewards = (
            death_rewards           # Death penalty (-50.0)
            + food_rewards          # Eat food (+5.0)
            + intrinsic_coop_rewards  # Intrinsic reward for COOP attempt near food + ally
            + attack_rewards        # Attack bonus
            + defense_rewards       # Defense bonus (for protecting allies)
            + revenge_rewards       # Revenge bonus (for retaliating against attackers)
            + betrayal_rewards      # Betrayal penalty (for attacking benefactors)
            + give_rewards          # Food sharing bonus
            + predator_kill_rewards # Reward for killing predators
            + hierarchy_rewards     # Social hierarchy ranking bonus
            + damage_pain           # Damage pain (non-linear)
            + low_hp_penalty        # Low HP penalty (constant per tick)
            + approach_reward       # Reward for moving closer to food
            + survival_bonus        # Bonus for staying alive
        )

        # Build outputs
        observations = self._get_all_observations()
        dones = ~self.agent_alive

        # Episode ends if all agents dead or max steps reached
        # Warmup episodes are 10x longer
        all_dead = ~self.agent_alive.any()
        episode_length = self.config.max_steps_per_episode * (10 if self.episode_count <= self.warmup_episodes else 1)
        max_steps = self.step_count >= episode_length
        episode_done = all_dead or max_steps
        if episode_done:
            dones = torch.ones(self.n_agents, device=self.device, dtype=torch.bool)

        # Curriculum tracking: accumulate agent 0's rewards and check phase advancement
        self.current_episode_reward += rewards[0].item()
        if episode_done and self.config.pretrain_mode and self.config.curriculum_enabled:
            self._check_curriculum_advance(self.current_episode_reward)
            self.current_episode_reward = 0.0

        # Group reward components for auxiliary value heads
        survival_rewards = damage_pain + low_hp_penalty + death_rewards
        resource_rewards = food_rewards + approach_reward
        social_rewards = attack_rewards + defense_rewards + revenge_rewards + betrayal_rewards + intrinsic_coop_rewards + give_rewards + predator_kill_rewards + hierarchy_rewards

        infos = {
            'reward_survival': survival_rewards,   # [n_agents] tensor
            'reward_resource': resource_rewards,   # [n_agents] tensor
            'reward_social': social_rewards,       # [n_agents] tensor
        }

        return observations, rewards, dones, infos

    def _compute_adjacency_matrix(self) -> torch.Tensor:
        """
        Compute which agents are adjacent to which.
        Returns [n_agents, n_agents] bool tensor.
        """
        pos = self.agent_positions.float()  # [n, 2]

        # Manhattan distance between all pairs
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [n, n, 2]
        manhattan = diff.abs().sum(dim=-1)  # [n, n]

        # Adjacent = distance 1, both alive, not self
        adjacent = (manhattan == 1)
        alive_matrix = self.agent_alive.unsqueeze(0) & self.agent_alive.unsqueeze(1)
        adjacent = adjacent & alive_matrix
        adjacent.fill_diagonal_(False)

        return adjacent

    def _fourier_encode(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Encode relative positions using Fourier features (multi-frequency sin/cos).

        Args:
            rel_pos: [..., 2] tensor of (dx, dy) normalized to [-1, 1]

        Returns:
            [..., fourier_bands*4] tensor of Fourier features
        """
        # Frequency bands: powers of 2 (coarse to fine)
        bands = [2.0 ** i for i in range(self.config.fourier_bands)]

        dx = rel_pos[..., 0:1]  # [..., 1]
        dy = rel_pos[..., 1:2]  # [..., 1]

        features = []
        for freq in bands:
            # Each band produces 4 features: sin(dx), cos(dx), sin(dy), cos(dy)
            features.append(torch.sin(freq * torch.pi * dx))
            features.append(torch.cos(freq * torch.pi * dx))
            features.append(torch.sin(freq * torch.pi * dy))
            features.append(torch.cos(freq * torch.pi * dy))

        return torch.cat(features, dim=-1)  # [..., fourier_bands*4]

    def _compute_nearest_food_info(self, positions: torch.Tensor, poor_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Manhattan distance and value of nearest food for each agent.

        Args:
            positions: [n_agents, 2] agent positions
            poor_only: If True, only consider poor food (for smell rewards)

        Returns:
            distances: [n_agents] distance to nearest food, or inf if no food
            values: [n_agents] value of nearest food, or 0 if no food
        """
        # Get food coordinates and their values
        poor_coords = torch.nonzero(self.poor_food, as_tuple=False)  # [P, 2]
        rich_coords = torch.nonzero(self.rich_food, as_tuple=False) if not poor_only else torch.empty((0, 2), device=self.device, dtype=torch.long)

        n_poor = poor_coords.size(0)
        n_rich = rich_coords.size(0)

        if n_poor == 0 and n_rich == 0:
            return (
                torch.full((self.n_agents,), float('inf'), device=self.device),
                torch.zeros(self.n_agents, device=self.device)
            )

        agent_pos = positions.float()  # [N, 2]

        # Initialize with inf distance
        min_dist = torch.full((self.n_agents,), float('inf'), device=self.device)
        nearest_value = torch.zeros(self.n_agents, device=self.device)

        # Check poor food
        if n_poor > 0:
            diff = agent_pos.unsqueeze(1) - poor_coords.float().unsqueeze(0)  # [N, P, 2]
            poor_dist = diff.abs().sum(dim=-1)  # [N, P]
            poor_min_dist, _ = poor_dist.min(dim=1)  # [N]

            closer_mask = poor_min_dist < min_dist
            min_dist = torch.where(closer_mask, poor_min_dist, min_dist)
            nearest_value = torch.where(closer_mask, self.config.poor_food_value, nearest_value)

        # Check rich food (skip if poor_only)
        if n_rich > 0:
            diff = agent_pos.unsqueeze(1) - rich_coords.float().unsqueeze(0)  # [N, R, 2]
            rich_dist = diff.abs().sum(dim=-1)  # [N, R]
            rich_min_dist, _ = rich_dist.min(dim=1)  # [N]

            closer_mask = rich_min_dist < min_dist
            min_dist = torch.where(closer_mask, rich_min_dist, min_dist)
            nearest_value = torch.where(closer_mask, self.config.rich_food_value, nearest_value)

        return min_dist, nearest_value

    def _resolve_movement(self, move_actions: torch.Tensor, is_move: torch.Tensor = None) -> None:
        """
        Fully vectorized movement resolution using the Heavyweight Rule.
        No loops, no .item() calls.

        Args:
            move_actions: [n_agents] movement action indices (0-4)
            is_move: [n_agents] bool mask - only move agents where True
                     If None, all alive agents can move (legacy behavior)
        """
        n = self.n_agents
        gs = self.grid_size

        # Only process alive agents who chose to move
        alive_mask = self.agent_alive
        if is_move is not None:
            alive_mask = alive_mask & is_move

        # Compute intended destinations for all agents
        intended = self.agent_positions + self.direction_deltas[move_actions]
        intended = intended.clamp(0, gs - 1)

        # Agents not moving (dead or chose interact) stay in place
        intended = torch.where(alive_mask.unsqueeze(1), intended, self.agent_positions)

        # Convert to flat indices for conflict detection
        flat_intended = intended[:, 0] * gs + intended[:, 1]
        flat_current = self.agent_positions[:, 0] * gs + self.agent_positions[:, 1]

        # Identify staying vs moving agents
        is_staying = (flat_intended == flat_current)

        # Get occupant at each agent's intended destination
        dest_rows = intended[:, 0]
        dest_cols = intended[:, 1]
        occupant_at_dest = self.occupancy[dest_rows, dest_cols]  # [n_agents]

        # Check if occupant exists and is staying
        has_occupant = occupant_at_dest >= 0
        safe_occupant_idx = occupant_at_dest.clamp(min=0, max=n - 1)  # Clamp to agent range for indexing
        is_predator_occupant = occupant_at_dest >= n  # Predators are stored as n_agents + pred_idx
        # For agent occupants, check if they're staying; predators always block
        agent_occupant_staying = is_staying[safe_occupant_idx] & has_occupant & ~is_predator_occupant
        occupant_is_staying = agent_occupant_staying | (is_predator_occupant & has_occupant)

        # Agent cannot block itself
        agent_ids = torch.arange(n, device=self.device)
        is_self_occupant = (occupant_at_dest == agent_ids)

        # Blocked = has occupant AND occupant is staying AND it's not self
        blocked_by_stationary = has_occupant & occupant_is_staying & ~is_self_occupant

        # Valid contender = alive AND not blocked
        valid_contender = alive_mask & ~blocked_by_stationary

        # Add random noise for tiebreaking (small relative to HP)
        tiebreaker = torch.rand(n, device=self.device) * 0.001
        hp_with_tiebreak = self.agent_hp + tiebreaker

        # Set HP to -inf for non-contenders so they can't win
        hp_contest = torch.where(
            valid_contender,
            hp_with_tiebreak,
            torch.full((n,), float('-inf'), device=self.device)
        )

        # Find max HP per destination using scatter_reduce
        max_hp_per_dest = torch.full((gs * gs,), float('-inf'), device=self.device)
        max_hp_per_dest.scatter_reduce_(
            0, flat_intended, hp_contest, reduce='amax', include_self=True
        )

        # An agent wins if their hp_contest equals max at their destination
        max_hp_at_my_dest = max_hp_per_dest[flat_intended]
        is_winner = valid_contender & (hp_contest == max_hp_at_my_dest)

        # Apply movement: winners move, others stay
        self.agent_positions = torch.where(
            is_winner.unsqueeze(1),
            intended,
            self.agent_positions
        )

    def _resolve_interactions(
        self,
        directions: torch.Tensor,
        action_types: torch.Tensor,
        is_attack: torch.Tensor,
        is_give: torch.Tensor,
        is_signal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process attack, give food, and signal actions with direction targeting.

        Args:
            directions: [n_agents] direction indices (0-4: UP, DOWN, LEFT, RIGHT, STAY)
            action_types: [n_agents] action type indices (0-4: MOVE, ATTACK, GIVE, SIGNAL, COOPERATE)
            is_attack: [n_agents] bool mask for agents choosing ATTACK
            is_give: [n_agents] bool mask for agents choosing GIVE
            is_signal: [n_agents] bool mask for agents choosing SIGNAL

        Returns:
            attack_rewards: [n_agents] reward for damage dealt
            damage_taken: [n_agents] damage received by each agent
            defense_rewards: [n_agents] reward for defending allies
            revenge_rewards: [n_agents] reward for retaliating against attackers
            betrayal_rewards: [n_agents] penalty for attacking benefactors
            give_rewards: [n_agents] reward for giving food
            predator_kill_rewards: [n_agents] reward for killing predators
        """
        # Signal actions - only process alive agents who chose SIGNAL
        signal_mask = is_signal & self.agent_alive
        self.signals = signal_mask

        # Attack actions - only process alive agents who chose ATTACK
        attack_mask = is_attack & self.agent_alive
        attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, predator_kill_rewards = self._resolve_attacks(directions, attack_mask)

        # Give food actions - only process alive agents who chose GIVE
        give_mask = is_give & self.agent_alive
        give_rewards = self._resolve_give_food(directions, give_mask)

        return attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, give_rewards, predator_kill_rewards

    def _resolve_attacks(self, directions: torch.Tensor, attack_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resolve attacks with direction targeting and defense tracking - FULLY VECTORIZED.

        Each attacker targets ONE cell in their chosen direction.
        Defense: if C attacks A who attacked B in last 3 steps, C defended B.
        Revenge: if C attacks A who attacked C recently, C gets revenge reward.
        Betrayal: if C attacks A who helped C before (food/coop), C gets betrayal penalty.
        Attack costs 5% of max HP regardless of whether it hits.

        Args:
            directions: [n_agents] direction indices (0-4: UP, DOWN, LEFT, RIGHT, STAY)
            attack_mask: [n_agents] bool mask for agents choosing ATTACK

        Returns:
            attack_rewards: [n_agents] reward for damage dealt
            damage_taken: [n_agents] damage received by each agent
            defense_rewards: [n_agents] reward for defending allies
            revenge_rewards: [n_agents] reward for retaliating
            betrayal_rewards: [n_agents] penalty for attacking benefactors
            predator_kill_rewards: [n_agents] reward for killing predators
        """
        # Initialize return tensors
        attack_rewards = torch.zeros(self.n_agents, device=self.device)
        damage_taken = torch.zeros(self.n_agents, device=self.device)
        defense_rewards = torch.zeros(self.n_agents, device=self.device)
        revenge_rewards = torch.zeros(self.n_agents, device=self.device)
        betrayal_rewards = torch.zeros(self.n_agents, device=self.device)
        predator_kill_rewards = torch.zeros(self.n_agents, device=self.device)

        # Early exit if no attackers
        if not attack_mask.any():
            return attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, predator_kill_rewards

        # === STEP 1: Apply attack cost to ALL attackers at once ===
        attack_cost = self.config.max_hp * 0.05
        self.agent_hp = self.agent_hp - attack_cost * attack_mask.float()

        # === STEP 2: Compute target positions for ALL agents ===
        # Use only the first 4 directions (UP, DOWN, LEFT, RIGHT) - clamp to handle STAY
        attack_dirs = directions.clamp(0, 3)
        deltas = self.dir_deltas_4[attack_dirs]  # [n_agents, 2]
        target_positions = self.agent_positions + deltas
        target_rows = target_positions[:, 0]
        target_cols = target_positions[:, 1]

        # === STEP 4: Check bounds ===
        in_bounds = (
            (target_rows >= 0) & (target_rows < self.grid_size) &
            (target_cols >= 0) & (target_cols < self.grid_size)
        )
        valid_attack = attack_mask & in_bounds

        # Clamp for safe indexing
        safe_rows = target_rows.clamp(0, self.grid_size - 1)
        safe_cols = target_cols.clamp(0, self.grid_size - 1)

        # === STEP 5: Look up targets from occupancy grid ===
        target_ids = self.occupancy[safe_rows, safe_cols]  # [n_agents]

        # Separate agent targets from predator targets
        has_target = target_ids >= 0
        is_predator_target = target_ids >= self.n_agents  # Predator IDs are n_agents + pred_idx
        is_agent_target = has_target & ~is_predator_target

        # === Agent-vs-Agent attacks ===
        valid_target_ids = target_ids.clamp(min=0, max=self.n_agents - 1)
        target_alive = self.agent_alive[valid_target_ids] & is_agent_target
        valid_hit = valid_attack & is_agent_target & target_alive

        # === STEP 6: Compute damage ===
        damage = (self.agent_hp * self.config.attack_damage_fraction) * valid_hit.float()

        # === STEP 7: Build damage matrix and apply damage ===
        damage_matrix = torch.zeros((self.n_agents, self.n_agents), device=self.device)
        hit_indices = torch.where(valid_hit)[0]

        if hit_indices.numel() > 0:
            hit_targets = target_ids[hit_indices]
            hit_damage = damage[hit_indices]

            # Set damage matrix entries
            damage_matrix[hit_indices, hit_targets] = hit_damage

            # Aggregate damage per target using scatter_add
            damage_per_target = torch.zeros(self.n_agents, device=self.device)
            damage_per_target.scatter_add_(0, hit_targets, hit_damage)
            self.agent_hp = self.agent_hp - damage_per_target

        # === Agent-vs-Predator attacks ===
        if self.n_predators > 0:
            valid_pred_hit = valid_attack & is_predator_target
            pred_hit_indices = torch.where(valid_pred_hit)[0]
            if pred_hit_indices.numel() > 0:
                pred_target_ids = target_ids[pred_hit_indices] - self.n_agents  # Convert to predator index
                pred_damage = (self.agent_hp[pred_hit_indices] * self.config.attack_damage_fraction)

                # Apply damage to each predator
                for i in range(pred_hit_indices.numel()):
                    p_idx = pred_target_ids[i].item()
                    if self.predator_alive[p_idx]:
                        self.predator_hp[p_idx] -= pred_damage[i]
                        # Record in predator ledger: agent dealt damage to predator
                        if self.predator_ledger is not None:
                            self.predator_ledger[pred_hit_indices[i], p_idx, 0] += pred_damage[i]
                        # Attack reward for hitting predator
                        attack_rewards[pred_hit_indices[i]] += pred_damage[i] * self.config.r_attack_mult
                        # Check if predator died
                        if self.predator_hp[p_idx] <= 0:
                            self.predator_alive[p_idx] = False
                            self.predator_respawn_timer[p_idx] = self.config.predator_respawn_steps
                            predator_kill_rewards[pred_hit_indices[i]] += self.config.predator_kill_reward

        # === STEP 8: Update ledger ===
        self.ledger.tensor[:, :, Ledger.DAMAGE_DEALT] += damage_matrix

        # === STEP 9: Update attack history buffer ===
        # Roll history: [0,1,2] -> [1,2,0], then overwrite slot 0 with current
        self.recent_attacks = torch.roll(self.recent_attacks, 1, dims=0)
        self.recent_attacks[0] = damage_matrix  # Current step attacks

        # === STEP 10: Extended defense tracking (last 3 steps) ===
        # C attacks A → check if A attacked anyone B in last 3 steps
        # If so, C defended B by attacking their attacker A
        if hit_indices.numel() > 0:
            # Who has A (the victim of C's attack) attacked recently?
            # recent_victims[attacker, victim] = True if attacker hit victim in last 3 steps
            recent_victims = (self.recent_attacks > 0).any(dim=0)  # [n_agents, n_agents]

            # For each C who hit someone (A), check A's recent victims (B)
            c_indices = hit_indices
            a_indices = target_ids[c_indices]  # Who C attacked (A)

            # For each (C, A) pair, find all B that A attacked recently
            for i in range(c_indices.numel()):
                c = c_indices[i]
                a = a_indices[i]
                dmg = damage[c]

                # === Revenge: Did A attack ME (C) recently? ===
                if recent_victims[a, c]:  # A attacked C recently
                    revenge_rewards[c] += dmg * self.config.r_revenge

                # === Betrayal: Did A help ME (C) before? ===
                food_from_a = self.ledger.tensor[a, c, Ledger.FOOD_GIVEN]
                coop_with_a = self.ledger.tensor[a, c, Ledger.COOP_COUNT]
                help_from_a = food_from_a + coop_with_a
                if help_from_a > 0:
                    # Penalty for attacking someone who helped you
                    betrayal_rewards[c] += help_from_a * self.config.r_betrayal

                # === Defense: Get all B that A attacked recently (excluding C) ===
                victims_of_a = recent_victims[a].clone()
                victims_of_a[c] = False  # Exclude self (revenge handled separately)
                b_indices = torch.where(victims_of_a)[0]

                # C defended each of these victims B
                if b_indices.numel() > 0:
                    # Update defense scores and rewards for all (C, B) pairs
                    for b in b_indices:
                        self.ledger.tensor[c, b, Ledger.DEFENSE_SCORE] += dmg
                        defense_rewards[c] += dmg * self.config.r_defense

        # === STEP 11: Compute rewards and damage taken ===
        # Attack rewards for agent-vs-agent (predator rewards already added above)
        total_damage_dealt = damage_matrix.sum(dim=1)  # Sum over all targets
        attack_rewards += total_damage_dealt * self.config.r_attack_mult

        # Damage taken per agent (sum over attackers) - still tracks all damage for pain calculation
        damage_taken = damage_matrix.sum(dim=0)

        return attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, predator_kill_rewards

    def _resolve_give_food(self, directions: torch.Tensor, give_mask: torch.Tensor) -> torch.Tensor:
        """
        Resolve food giving from INVENTORY - FULLY VECTORIZED.

        Giver transfers from their INVENTORY (no HP cost!).
        Target receives to their INVENTORY.
        Only transfers if giver has inventory AND valid target exists.

        Args:
            directions: [n_agents] direction indices (0-4: UP, DOWN, LEFT, RIGHT, STAY)
            give_mask: [n_agents] bool mask for agents choosing GIVE

        Returns:
            give_rewards: [n_agents] reward for giving food
        """
        give_rewards = torch.zeros(self.n_agents, device=self.device)
        food_value = self.config.poor_food_value

        # Early exit if no givers
        if not give_mask.any():
            return give_rewards

        # === STEP 1: Compute transfer amount for each giver ===
        # Can only give what you have in inventory (up to food_value per action)
        give_amount = torch.minimum(
            torch.full((self.n_agents,), food_value, device=self.device),
            self.agent_inventory
        ) * give_mask.float()

        # Valid givers have inventory to give
        valid_giver = give_mask & (give_amount > 0)

        # === STEP 2: Compute target positions ===
        # Use direction directly - clamp to handle STAY (would be invalid anyway)
        give_dirs = directions.clamp(0, 3)
        deltas = self.dir_deltas_4[give_dirs]
        target_positions = self.agent_positions + deltas
        target_rows = target_positions[:, 0]
        target_cols = target_positions[:, 1]

        # === STEP 3: Check bounds ===
        in_bounds = (
            (target_rows >= 0) & (target_rows < self.grid_size) &
            (target_cols >= 0) & (target_cols < self.grid_size)
        )

        # Clamp for safe indexing
        safe_rows = target_rows.clamp(0, self.grid_size - 1)
        safe_cols = target_cols.clamp(0, self.grid_size - 1)

        # === STEP 4: Look up targets ===
        target_ids = self.occupancy[safe_rows, safe_cols]

        has_target = target_ids >= 0
        valid_target_ids = target_ids.clamp(min=0)
        target_alive = self.agent_alive[valid_target_ids] & has_target
        valid_transfer = valid_giver & in_bounds & has_target & target_alive

        # === STEP 5: Transfer INVENTORY (only if valid target) ===
        transfer_indices = torch.where(valid_transfer)[0]

        if transfer_indices.numel() > 0:
            transfer_targets = target_ids[transfer_indices]
            transfer_values = give_amount[transfer_indices]

            # Deduct from giver's inventory
            self.agent_inventory[transfer_indices] -= transfer_values

            # Add to target's inventory using scatter_add
            inventory_gain = torch.zeros(self.n_agents, device=self.device)
            inventory_gain.scatter_add_(0, transfer_targets, transfer_values)
            self.agent_inventory = self.agent_inventory + inventory_gain

            # === STEP 6: Reward givers ===
            give_rewards[transfer_indices] = transfer_values * self.config.r_food_share

            # === STEP 7: Update ledger ===
            flat_indices = transfer_indices * self.n_agents + transfer_targets
            ledger_food = self.ledger.tensor[:, :, Ledger.FOOD_GIVEN].view(-1)
            ledger_food.scatter_add_(0, flat_indices, transfer_values)

        return give_rewards

    def _predator_step(self) -> torch.Tensor:
        """Move predators toward closest agent, attack adjacent agents, handle respawn.

        Returns:
            predator_damage_to_agents: [n_agents] damage dealt by predators this step
        """
        predator_damage_to_agents = torch.zeros(self.n_agents, device=self.device)
        if self.n_predators == 0:
            return predator_damage_to_agents

        for p in range(self.n_predators):
            # Handle respawn timer for dead predators
            if not self.predator_alive[p]:
                self.predator_respawn_timer[p] -= 1
                if self.predator_respawn_timer[p] <= 0:
                    self._update_occupancy()
                    self._respawn_predator(p)
                    self._update_occupancy()
                continue

            # Store previous position for velocity
            self.predator_prev_positions[p] = self.predator_positions[p].clone()

            # Find closest alive agent (Manhattan distance)
            alive_indices = torch.where(self.agent_alive)[0]
            if alive_indices.numel() == 0:
                continue

            alive_pos = self.agent_positions[alive_indices].float()  # [K, 2]
            pred_pos = self.predator_positions[p].float()  # [2]
            dists = (alive_pos - pred_pos.unsqueeze(0)).abs().sum(dim=1)  # [K]
            closest_idx = alive_indices[dists.argmin()]
            closest_pos = self.agent_positions[closest_idx]

            # Move one step toward closest agent (primary axis first)
            diff = closest_pos.float() - pred_pos
            abs_diff = diff.abs()

            # Choose primary axis (larger absolute difference)
            new_pos = self.predator_positions[p].clone()
            if abs_diff[0] >= abs_diff[1] and abs_diff[0] > 0:
                # Move along row axis
                step = 1 if diff[0] > 0 else -1
                candidate = new_pos.clone()
                candidate[0] += step
            elif abs_diff[1] > 0:
                # Move along col axis
                step = 1 if diff[1] > 0 else -1
                candidate = new_pos.clone()
                candidate[1] += step
            else:
                candidate = new_pos  # Already at target

            # Check bounds and occupancy before moving
            candidate = candidate.clamp(0, self.grid_size - 1)
            cr, cc = candidate[0].item(), candidate[1].item()
            if self.occupancy[cr, cc] == -1:
                # Clear old occupancy
                old_r, old_c = self.predator_positions[p][0].item(), self.predator_positions[p][1].item()
                if self.occupancy[old_r, old_c] == self.n_agents + p:
                    self.occupancy[old_r, old_c] = -1
                # Move predator
                self.predator_positions[p] = candidate
                self.occupancy[cr, cc] = self.n_agents + p

            # Attack closest adjacent alive agent (fixed damage)
            pred_pos_now = self.predator_positions[p]
            for ai in range(self.n_agents):
                if not self.agent_alive[ai]:
                    continue
                agent_pos = self.agent_positions[ai]
                manhattan = (pred_pos_now.float() - agent_pos.float()).abs().sum()
                if manhattan <= 1.0:
                    dmg = self.config.predator_damage
                    self.agent_hp[ai] -= dmg
                    predator_damage_to_agents[ai] += dmg
                    # Record in predator ledger: predator dealt damage to agent
                    if self.predator_ledger is not None:
                        self.predator_ledger[ai, p, 1] += dmg
                    break  # Only attack one agent per step

        return predator_damage_to_agents

    def _process_food_eating(self, action_types: torch.Tensor) -> torch.Tensor:
        """Process agents eating food on their cells.

        Poor food: picked up automatically by stepping on it -> goes to INVENTORY
        Rich food: requires 2+ adjacent agents who both chose COOPERATE action -> goes to INVENTORY

        Args:
            action_types: [n_agents] action type indices (0-4: MOVE, ATTACK, GIVE, SIGNAL, COOPERATE)

        Returns:
            food_rewards: [n_agents] explicit reward for picking up food
        """
        food_rewards = torch.zeros(self.n_agents, device=self.device)

        # Get agent positions
        rows = self.agent_positions[:, 0]
        cols = self.agent_positions[:, 1]

        # Poor food - check which alive agents are on poor food
        on_poor = self.poor_food[rows, cols] & self.agent_alive

        # Explicit reward for picking up poor food (100x during warmup)
        reward_multiplier = 100.0 if self.episode_count <= self.warmup_episodes else 1.0
        food_rewards += on_poor.float() * self.config.r_small * reward_multiplier

        # Add poor food to INVENTORY (not HP directly)
        self.agent_inventory = self.agent_inventory + on_poor.float() * self.config.poor_food_value
        self.food_eaten_total = self.food_eaten_total + on_poor.float()

        # Remove eaten poor food
        poor_food_flat = self.poor_food.view(-1)
        eat_indices = rows[on_poor] * self.grid_size + cols[on_poor]
        poor_food_flat[eat_indices] = False

        # Rich food - requires 2+ adjacent agents who BOTH chose COOPERATE (vectorized)
        coop_mask = (action_types == ACT_COOPERATE) & self.agent_alive

        rich_rewards = self._process_rich_food_vectorized(coop_mask)
        food_rewards += rich_rewards

        return food_rewards

    def _process_rich_food_vectorized(self, coop_mask: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized rich food cooperation resolution.
        No loops, no .item() calls.

        Returns:
            rich_rewards: [n_agents] explicit reward for eating rich food
        """
        # Get all rich food positions [F, 2]
        rich_food_coords = torch.nonzero(self.rich_food, as_tuple=False)
        n_food = rich_food_coords.size(0)

        # Early exit if no rich food or fewer than 2 cooperators
        if n_food == 0 or coop_mask.sum() < 2:
            self.last_coop_success_count = 0
            return torch.zeros(self.n_agents, device=self.device)

        # Compute distance matrix [N, F] - Manhattan distance
        agent_pos = self.agent_positions.float()  # [N, 2]
        food_pos = rich_food_coords.float()       # [F, 2]
        diff = agent_pos.unsqueeze(1) - food_pos.unsqueeze(0)  # [N, F, 2]
        manhattan_dist = diff.abs().sum(dim=-1)  # [N, F]

        # Eligibility: within range AND chose cooperate
        within_range = manhattan_dist <= 1  # [N, F]
        eligible = within_range & coop_mask.unsqueeze(1)  # [N, F]

        # Find consumed foods (2+ cooperators)
        cooperators_per_food = eligible.sum(dim=0)  # [F]
        consumed_mask = cooperators_per_food >= 2   # [F] bool

        if not consumed_mask.any():
            self.last_coop_success_count = 0
            return torch.zeros(self.n_agents, device=self.device)

        # Track successful cooperations for diagnostics
        self.last_coop_success_count = consumed_mask.sum().item()

        # INVENTORY update: count consumed foods each agent participated in
        foods_per_agent = eligible.float() @ consumed_mask.float()  # [N]
        inventory_gain = foods_per_agent * self.config.rich_food_value
        self.agent_inventory = self.agent_inventory + inventory_gain
        self.food_eaten_total = self.food_eaten_total + (foods_per_agent > 0).float()

        # Ledger update: cooperation pairs via outer product
        consumed_food_indices = torch.where(consumed_mask)[0]  # [C]
        eligible_consumed = eligible[:, consumed_food_indices].float()  # [N, C]
        coop_pairs = eligible_consumed @ eligible_consumed.T  # [N, N]
        coop_pairs.fill_diagonal_(0)  # No self-cooperation

        # === Reciprocity bonus: reward cooperating with agents who helped you before ===
        # Prior help from each agent BEFORE updating ledger (food given + prior coop count)
        prior_food = self.ledger.tensor[:, :, Ledger.FOOD_GIVEN]  # [N, N]
        prior_coop = self.ledger.tensor[:, :, Ledger.COOP_COUNT]  # [N, N]
        prior_help = prior_food + prior_coop  # prior_help[i,j] = help agent i received from j

        # For each agent, sum over partners: coop_pairs[i,j] * prior_help[i,j]
        # This rewards cooperating with agents who previously helped you
        reciprocity_bonus = (coop_pairs * prior_help).sum(dim=1) * self.config.r_reciprocity

        self.ledger.tensor[:, :, Ledger.COOP_COUNT] += coop_pairs

        # Remove consumed food
        consumed_rows = rich_food_coords[consumed_food_indices, 0]
        consumed_cols = rich_food_coords[consumed_food_indices, 1]
        self.rich_food[consumed_rows, consumed_cols] = False

        # Return explicit reward for participating in rich food consumption (100x during warmup)
        # Plus reciprocity bonus for cooperating with those who helped you
        reward_multiplier = 100.0 if self.episode_count <= self.warmup_episodes else 1.0
        participated = (foods_per_agent > 0).float()
        food_reward = participated * self.config.r_large * reward_multiplier
        return food_reward + reciprocity_bonus

    def _compute_intrinsic_coop_rewards(self, action_types: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward for attempting COOP when conditions are right.

        An agent gets intrinsic reward if:
        1. They chose the COOP action
        2. They are within distance 1 of rich food
        3. Another alive agent is also within distance 1 of the same rich food

        This encourages agents to try COOP even before they coordinate successfully.

        Args:
            action_types: [n_agents] action type indices (0-4)
        """
        intrinsic_rewards = torch.zeros(self.n_agents, device=self.device)

        # Get agents who chose COOP and are alive
        coop_mask = (action_types == ACT_COOPERATE) & self.agent_alive

        # Early exit if no cooperators or no rich food
        if not coop_mask.any():
            return intrinsic_rewards

        rich_food_coords = torch.nonzero(self.rich_food, as_tuple=False)
        n_food = rich_food_coords.size(0)

        if n_food == 0:
            return intrinsic_rewards

        # Compute distance from all agents to all rich food [N, F]
        agent_pos = self.agent_positions.float()
        food_pos = rich_food_coords.float()
        diff = agent_pos.unsqueeze(1) - food_pos.unsqueeze(0)  # [N, F, 2]
        manhattan_dist = diff.abs().sum(dim=-1)  # [N, F]

        # Find agents within distance 1 of each food [N, F]
        within_range = (manhattan_dist <= 1) & self.agent_alive.unsqueeze(1)

        # For each food, count how many alive agents are nearby
        agents_per_food = within_range.sum(dim=0)  # [F]

        # Foods with 2+ agents nearby are "cooperation opportunities"
        coop_opportunity = agents_per_food >= 2  # [F]

        # An agent gets intrinsic reward if:
        # - They chose COOP
        # - They are within range of a food with 2+ agents
        # Check: for each agent who chose COOP, is there any food they're near that has 2+ agents?
        near_coop_opportunity = (within_range & coop_opportunity.unsqueeze(0)).any(dim=1)  # [N]

        # Reward agents who chose COOP and are near a coop opportunity
        reward_mask = coop_mask & near_coop_opportunity
        intrinsic_rewards = reward_mask.float() * self.config.r_coop_attempt

        return intrinsic_rewards

    def _compute_hierarchy_rewards(self) -> torch.Tensor:
        """Compute per-step hierarchy bonus based on agent ranking.

        Agents ranked by weighted sum of: cumulative food eaten, total damage
        dealt, and current HP ratio. Top agent gets r_hierarchy, bottom gets 0.
        """
        cfg = self.config
        if cfg.r_hierarchy == 0.0:
            return torch.zeros(self.n_agents, device=self.device)

        alive = self.agent_alive.float()
        n_active = alive.sum()
        if n_active < 2:
            return torch.zeros(self.n_agents, device=self.device)

        eps = 1e-8

        # Component scores (normalize each by max across agents)
        food = self.food_eaten_total * alive
        food_max = food.max().clamp(min=eps)
        food_norm = food / food_max

        # Total damage dealt (sum over all targets from ledger)
        damage = self.ledger.tensor[:, :, Ledger.DAMAGE_DEALT].sum(dim=1) * alive
        damage_max = damage.max().clamp(min=eps)
        damage_norm = damage / damage_max

        hp_ratio = (self.agent_hp / cfg.max_hp) * alive
        hp_max = hp_ratio.max().clamp(min=eps)
        hp_norm = hp_ratio / hp_max

        # Weighted hierarchy score
        score = (cfg.hierarchy_food_weight * food_norm
                 + cfg.hierarchy_damage_weight * damage_norm
                 + cfg.hierarchy_hp_weight * hp_norm) * alive

        # Rank via argsort of argsort (rank 0 = lowest score)
        ranks = score.argsort().argsort().float()

        # Reward: linear from -r_hierarchy (bottom) to +r_hierarchy (top)
        # Bottom 50% get negative rewards, top 50% get positive
        denom = (n_active - 1).clamp(min=1.0)
        rewards = (2.0 * ranks / denom - 1.0) * cfg.r_hierarchy * alive

        return rewards

    def _apply_hp_decay(self) -> None:
        """Apply HP decay to all alive agents - fully vectorized."""
        decay = self.config.max_hp * self.config.hp_decay_rate
        self.agent_hp = self.agent_hp - decay * self.agent_alive.float()

    def _consume_inventory(self) -> None:
        """Auto-consume inventory to heal, capped at heal_per_tick."""
        hp_missing = self.config.max_hp - self.agent_hp
        cap = torch.full_like(self.agent_inventory, self.config.heal_per_tick)
        heal_amount = torch.minimum(torch.minimum(hp_missing, self.agent_inventory), cap)
        heal_amount = heal_amount * self.agent_alive.float() * (hp_missing > 0).float()
        self.agent_hp = self.agent_hp + heal_amount
        self.agent_inventory = self.agent_inventory - heal_amount

    def _check_deaths(self) -> torch.Tensor:
        """Mark agents with HP <= 0 as dead and return death penalties."""
        # Track who was alive before
        was_alive = self.agent_alive.clone()

        # Update alive status
        self.agent_alive = self.agent_hp > 0

        # Newly dead agents get death penalty
        newly_dead = was_alive & ~self.agent_alive
        death_rewards = newly_dead.float() * self.config.r_death

        return death_rewards

    def _initialize_food_at_capacity(self) -> None:
        """Initialize food at full capacity on empty cells."""
        total_cells = self.grid_size * self.grid_size
        cap_cells = int(total_cells * self.config.food_coverage_cap)

        # Get empty cells (no agent)
        empty = (self.occupancy == -1)
        empty_indices = empty.flatten().nonzero(as_tuple=True)[0]

        # Shuffle empty indices for random placement
        perm = torch.randperm(len(empty_indices), device=self.device)
        shuffled_indices = empty_indices[perm]

        # Place poor food on first cap_cells positions
        poor_count = min(cap_cells, len(shuffled_indices))
        for i in range(poor_count):
            idx = shuffled_indices[i].item()
            row, col = idx // self.grid_size, idx % self.grid_size
            self.poor_food[row, col] = True

        # Place rich food on next cap_cells positions (non-overlapping)
        # Skip rich food in pretrain mode
        if not self.config.pretrain_mode:
            rich_start = poor_count
            rich_count = min(cap_cells, len(shuffled_indices) - rich_start)
            for i in range(rich_count):
                idx = shuffled_indices[rich_start + i].item()
                row, col = idx // self.grid_size, idx % self.grid_size
                self.rich_food[row, col] = True

    def _spawn_food_curriculum(self) -> None:
        """Spawn food for curriculum learning - progressive difficulty."""
        if self.curriculum_phase >= 6:
            # Cooperation phases 6-8: spawn rich food, position agents nearby
            self._spawn_food_coop_curriculum()
            return

        # Solo phases 1-5: Only use agent 0
        r, c = self.agent_positions[0][0].item(), self.agent_positions[0][1].item()

        # Phase configs: (distance, cardinal_only)
        # Phase 1: dist 1, cardinal | Phase 2: dist 2, cardinal
        # Phase 3: dist 3, any      | Phase 4: dist 7, any
        # Phase 5: max dist, any (final solo phase)
        # All distances are bounded by grid_size - 1 to support small grids
        max_dist = self.grid_size - 1
        phase_configs = {
            1: (min(1, max_dist), True),
            2: (min(2, max_dist), True),
            3: (min(3, max_dist), False),
            4: (min(7, max_dist), False),
            5: (max_dist, False),
        }

        phase = min(self.curriculum_phase, 5)  # Cap at phase 5 for solo
        distance, cardinal_only = phase_configs[phase]

        # Find valid positions at the target distance
        valid_positions = self._get_positions_at_distance(r, c, distance, cardinal_only)

        # Pick one random position and spawn exactly one food
        if valid_positions:
            idx = torch.randint(len(valid_positions), (1,)).item()
            nr, nc = valid_positions[idx]
            self.poor_food[nr, nc] = True

    def _spawn_food_coop_curriculum(self) -> None:
        """Spawn RICH food for cooperation curriculum (phases 6-8).

        Places one rich food and positions both agents ADJACENT to it.
        Phase 6: agents adjacent (distance 1) - just need to learn COOP
        Phase 7: agents at distance 2 from food
        Phase 8: agents at distance 3 from food

        Key: Phase 6 makes both agents adjacent so they ONLY need to learn
        the COOP action, not movement + coordination simultaneously.
        """
        # Bound distances by grid_size - 1 to support small grids
        max_dist = self.grid_size - 1
        phase_distances = {6: min(1, max_dist), 7: min(2, max_dist), 8: min(3, max_dist)}
        distance = phase_distances.get(self.curriculum_phase, 1)

        # Place rich food in center of grid
        center = self.grid_size // 2
        self.rich_food[center, center] = True

        # Find valid positions for agents at the specified distance from food
        valid_positions = self._get_positions_at_distance(center, center, distance, cardinal_only=False)

        # Need at least 2 positions for both agents
        if len(valid_positions) >= 2:
            # Randomly pick 2 positions for the agents
            perm = torch.randperm(len(valid_positions))[:2]
            r0, c0 = valid_positions[perm[0].item()]
            r1, c1 = valid_positions[perm[1].item()]
            self.agent_positions[0] = torch.tensor([r0, c0], device=self.device)
            self.agent_positions[1] = torch.tensor([r1, c1], device=self.device)
        else:
            # Fallback: place agents adjacent to food
            self.agent_positions[0] = torch.tensor([center - 1, center], device=self.device)
            self.agent_positions[1] = torch.tensor([center, center - 1], device=self.device)

        # Update occupancy after repositioning
        self._update_occupancy()

    def _get_positions_at_distance(self, r: int, c: int, distance: int, cardinal_only: bool) -> list:
        """Get all valid grid positions at exact Manhattan distance from (r, c)."""
        positions = []

        if cardinal_only:
            # Only 4 cardinal directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr * distance, c + dc * distance
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    positions.append((nr, nc))
        else:
            # All positions at exact Manhattan distance
            for dr in range(-distance, distance + 1):
                dc_abs = distance - abs(dr)  # Remaining distance for column
                for dc in ([-dc_abs, dc_abs] if dc_abs > 0 else [0]):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        positions.append((nr, nc))

        return positions

    def _spawn_food_near(self, agent_pos: torch.Tensor, min_dist: int, max_dist: int, count: int) -> None:
        """Spawn 'count' food items within distance range of agent."""
        r, c = agent_pos[0].item(), agent_pos[1].item()
        candidates = []

        for dr in range(-max_dist, max_dist + 1):
            for dc in range(-max_dist, max_dist + 1):
                nr, nc = r + dr, c + dc
                dist = abs(dr) + abs(dc)  # Manhattan distance
                if min_dist <= dist <= max_dist:
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.occupancy[nr, nc] == -1 and not self.poor_food[nr, nc]:
                            candidates.append((nr, nc))

        # Randomly select 'count' cells
        if candidates:
            perm = torch.randperm(len(candidates))[:count]
            for idx in perm:
                nr, nc = candidates[idx]
                self.poor_food[nr, nc] = True

    def _check_curriculum_advance(self, episode_return: float) -> None:
        """Track episode returns for curriculum advancement based on thresholds."""
        # Import thresholds from scenarios module
        from scenarios import THRESHOLDS

        # Max phase is the highest defined in THRESHOLDS
        max_phase = max(THRESHOLDS.keys())
        if self.curriculum_phase >= max_phase:
            return

        self.episode_returns.append(episode_return)

        # Check advancement every 10 episodes (was 20, reduced for faster progression)
        if len(self.episode_returns) >= 10:
            avg_return = sum(self.episode_returns[-10:]) / 10

            threshold = THRESHOLDS.get(self.curriculum_phase, 999999)

            if avg_return >= threshold and self.curriculum_phase < max_phase:
                self.curriculum_phase += 1
                self.episode_returns = []
                print(f"[Curriculum] Avg return {avg_return:.1f} >= {threshold} -> Advanced to phase {self.curriculum_phase}")

    def advance_curriculum_phase(self) -> bool:
        """Manually advance curriculum phase. Returns True if advanced, False if already at max."""
        from scenarios import THRESHOLDS
        max_phase = max(THRESHOLDS.keys())
        if self.curriculum_phase >= max_phase:
            return False
        self.curriculum_phase += 1
        self.episode_returns = []
        print(f"[Curriculum] Advanced to phase {self.curriculum_phase}")
        return True

    def get_curriculum_phase(self) -> int:
        """Get current curriculum phase."""
        return self.curriculum_phase

    def _spawn_food(self) -> None:
        """Spawn new food on empty cells - fully vectorized with caps."""
        # In curriculum pretrain mode, use scenario-based or legacy curriculum spawning
        if self.config.pretrain_mode and self.config.curriculum_enabled:
            self._spawn_food_curriculum_continuous()
            return

        # Mask of empty cells (no agent, no food)
        empty = (self.occupancy == -1) & ~self.poor_food & ~self.rich_food

        total_cells = self.grid_size * self.grid_size
        cap_cells = int(total_cells * self.config.food_coverage_cap)

        # Count current food
        poor_count = self.poor_food.sum().item()
        rich_count = self.rich_food.sum().item()

        # Spawn poor food (2x rate if below cap, skip if at cap)
        if poor_count < cap_cells:
            poor_multiplier = 2.0
            poor_rate = min(1.0, self.config.poor_food_spawn_rate * poor_multiplier)
            rand = torch.rand((self.grid_size, self.grid_size), device=self.device)
            spawn_poor = empty & (rand < poor_rate)
            self.poor_food = self.poor_food | spawn_poor
        else:
            spawn_poor = torch.zeros_like(empty)

        # Spawn rich food (2x rate if below cap, skip if at cap)
        # Skip rich food in pretrain mode
        if not self.config.pretrain_mode and rich_count < cap_cells:
            rich_multiplier = 2.0
            rich_rate = min(1.0, self.config.rich_food_spawn_rate * rich_multiplier)
            rand2 = torch.rand((self.grid_size, self.grid_size), device=self.device)
            spawn_rich = empty & ~spawn_poor & (rand2 < rich_rate)
            self.rich_food = self.rich_food | spawn_rich

    def _spawn_food_curriculum_continuous(self) -> None:
        """Respawn food when eaten during curriculum - delegates to scenario if set."""
        # If scenario is active, use its respawn logic
        if self._current_scenario is not None:
            self._current_scenario.respawn_food(self)
            return

        # Legacy fallback: cooperation phases 6-8: respawn rich food
        if self.curriculum_phase >= 6:
            self._spawn_food_coop_curriculum_continuous()
            return

        # Legacy fallback: Solo phases 1-5: respawn poor food
        if self.poor_food.any():
            return  # Food still exists, don't spawn more

        # Only use agent 0 (the single active agent in pretrain mode)
        r, c = self.agent_positions[0][0].item(), self.agent_positions[0][1].item()

        # Phase configs: (distance, cardinal_only)
        # All distances are bounded by grid_size - 1 to support small grids
        max_dist = self.grid_size - 1
        phase_configs = {
            1: (min(1, max_dist), True),
            2: (min(2, max_dist), True),
            3: (min(3, max_dist), False),
            4: (min(7, max_dist), False),
            5: (max_dist, False),
        }

        phase = min(self.curriculum_phase, 5)  # Cap at phase 5
        distance, cardinal_only = phase_configs[phase]

        # Find valid positions and spawn one food
        valid_positions = self._get_positions_at_distance(r, c, distance, cardinal_only)
        if valid_positions:
            idx = torch.randint(len(valid_positions), (1,)).item()
            nr, nc = valid_positions[idx]
            self.poor_food[nr, nc] = True

    def _spawn_food_coop_curriculum_continuous(self) -> None:
        """Respawn rich food for cooperation curriculum when eaten."""
        if self.rich_food.any():
            return  # Rich food still exists, don't spawn more

        # Bound distances by grid_size - 1 to support small grids
        max_dist = self.grid_size - 1
        phase_distances = {6: min(1, max_dist), 7: min(2, max_dist), 8: min(3, max_dist)}
        distance = phase_distances.get(self.curriculum_phase, 1)

        # Get positions of both agents
        r0, c0 = self.agent_positions[0][0].item(), self.agent_positions[0][1].item()
        r1, c1 = self.agent_positions[1][0].item(), self.agent_positions[1][1].item()

        # Find center point between agents
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2

        # Find positions within 'distance' of BOTH agents (intersection)
        valid_positions = []
        for dr in range(-distance * 2, distance * 2 + 1):
            for dc in range(-distance * 2, distance * 2 + 1):
                nr, nc = center_r + dr, center_c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    # Check distance to both agents
                    dist0 = abs(nr - r0) + abs(nc - c0)
                    dist1 = abs(nr - r1) + abs(nc - c1)
                    # Both agents must be within reach
                    if dist0 <= distance and dist1 <= distance:
                        # Don't spawn on agents
                        if self.occupancy[nr, nc] == -1:
                            valid_positions.append((nr, nc))

        if valid_positions:
            idx = torch.randint(len(valid_positions), (1,)).item()
            nr, nc = valid_positions[idx]
            self.rich_food[nr, nc] = True
        else:
            # Fallback: spawn between the agents if possible
            if 0 <= center_r < self.grid_size and 0 <= center_c < self.grid_size:
                if self.occupancy[center_r, center_c] == -1:
                    self.rich_food[center_r, center_c] = True

    def _get_all_observations(self) -> Dict[str, torch.Tensor]:
        """Build batched observations for all agents."""
        entity_tokens, entity_mask = self._get_entity_tokens()
        return {
            'entity_tokens': entity_tokens,    # [n_agents, max_entities, 25]
            'entity_mask': entity_mask,        # [n_agents, max_entities]
            'signals': self.signals.float().unsqueeze(0).expand(self.n_agents, -1),
            'self_hp': (self.agent_hp / self.config.max_hp).unsqueeze(1),
            'self_inventory': (self.agent_inventory / self.config.max_hp).unsqueeze(1),  # Normalized by max_hp
            'agent_id': torch.arange(self.n_agents, device=self.device)
        }

    def get_action_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute valid action masks for factored action space.

        Returns:
            direction_mask: [n_agents, 5] bool - which directions are valid
            action_type_mask: [n_agents, 5] bool - which action types are valid

        Direction mask (5 outputs): UP, DOWN, LEFT, RIGHT, STAY
            - For MOVE: direction valid if in bounds
            - For ATTACK/GIVE: direction valid if adjacent agent exists
            - STAY (direction 4) is always valid

        Action type mask (5 outputs): MOVE, ATTACK, GIVE, SIGNAL, COOPERATE
            - MOVE: always valid for alive agents
            - ATTACK: valid if any adjacent agent exists
            - GIVE: valid if any adjacent agent exists
            - SIGNAL: always valid for alive agents
            - COOPERATE: valid if rich food within distance 1
        """
        n = self.n_agents
        gs = self.grid_size

        # === DIRECTION MASK [n, 5] ===
        # Check each direction for movement bounds
        intended = self.agent_positions.unsqueeze(1) + self.direction_deltas.unsqueeze(0)  # [n, 5, 2]
        in_bounds = (
            (intended[..., 0] >= 0) & (intended[..., 0] < gs) &
            (intended[..., 1] >= 0) & (intended[..., 1] < gs)
        )
        direction_mask = in_bounds & self.agent_alive.unsqueeze(1)  # [n, 5]

        # === Check for adjacent agents (for ATTACK/GIVE validity) ===
        target_pos = self.agent_positions.unsqueeze(1) + self.dir_deltas_4.unsqueeze(0)  # [n, 4, 2]
        target_in_bounds = (
            (target_pos[..., 0] >= 0) & (target_pos[..., 0] < gs) &
            (target_pos[..., 1] >= 0) & (target_pos[..., 1] < gs)
        )
        safe_rows = target_pos[..., 0].clamp(0, gs - 1)
        safe_cols = target_pos[..., 1].clamp(0, gs - 1)
        occupants = self.occupancy[safe_rows, safe_cols]  # [n, 4]

        has_target = (occupants >= 0)
        is_agent_occ = has_target & (occupants < n)
        target_alive = self.agent_alive[occupants.clamp(min=0, max=n-1)] & is_agent_occ
        direction_has_agent = target_in_bounds & target_alive  # [n, 4] - which directions have adjacent agent

        # Check for adjacent predators too (for ATTACK validity)
        direction_has_predator = target_in_bounds & has_target & (occupants >= n)  # [n, 4]

        # Any adjacent attackable entity? (agent or predator)
        any_adjacent_agent = direction_has_agent.any(dim=1)  # [n]
        any_adjacent_target = (direction_has_agent | direction_has_predator).any(dim=1)  # [n]

        # === ACTION TYPE MASK [n, 5] ===
        action_type_mask = torch.zeros(n, 5, dtype=torch.bool, device=self.device)

        # MOVE (type 0) - always valid for alive agents
        action_type_mask[:, ACT_MOVE] = self.agent_alive

        # ATTACK (type 1) - valid if any adjacent agent OR predator exists
        action_type_mask[:, ACT_ATTACK] = any_adjacent_target & self.agent_alive

        # GIVE (type 2) - valid if any adjacent agent exists (can't give to predator)
        action_type_mask[:, ACT_GIVE] = any_adjacent_agent & self.agent_alive

        # SIGNAL (type 3) - always valid for alive agents
        action_type_mask[:, ACT_SIGNAL] = self.agent_alive

        # COOPERATE (type 4) - valid if rich food within distance 1
        rich_coords = torch.nonzero(self.rich_food, as_tuple=False)  # [F, 2]
        if rich_coords.numel() > 0:
            pos = self.agent_positions.float()
            food_pos = rich_coords.float()
            dist = (pos.unsqueeze(1) - food_pos.unsqueeze(0)).abs().sum(dim=-1)  # [n, F]
            rich_nearby = (dist <= 1).any(dim=1)  # [n]
        else:
            rich_nearby = torch.zeros(n, dtype=torch.bool, device=self.device)
        action_type_mask[:, ACT_COOPERATE] = rich_nearby & self.agent_alive

        return direction_mask, action_type_mask

    def _get_entity_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create unified entity tokens for agents AND food using Fourier positional encoding.

        Token format (25 features):
            - fourier_spatial[16]: 4 freq bands × (sin_dx, cos_dx, sin_dy, cos_dy)
            - velocity[2]: (dv_x, dv_y) × 10.0 scaling
            - type_onehot[2]: [is_food, is_agent]
            - value[1]: quality (food) or hp (agent)
            - social[4]: interaction history (zeros for food)

        Returns:
            tokens: [n_agents, max_entities, 25] - padded to max_entities
            mask: [n_agents, max_entities] - True if token is valid
        """
        n = self.n_agents
        gs = self.config.grid_size
        max_food = self.config.max_food_tokens
        max_ent = self.config.max_entities  # n_agents + max_food_tokens
        token_dim = self.config.entity_token_dim  # 25

        # Initialize output tensors
        tokens = torch.zeros(n, max_ent, token_dim, device=self.device)
        mask = torch.zeros(n, max_ent, device=self.device, dtype=torch.bool)

        # === AGENT TOKENS (first n_agents slots) ===
        # Current relative positions normalized to [-1, 1]
        rel_pos_curr = (self.agent_positions.unsqueeze(0) - self.agent_positions.unsqueeze(1)).float()
        rel_pos_norm = rel_pos_curr / gs  # [n, n, 2]

        # Previous relative positions (for velocity)
        rel_pos_prev = (self.prev_agent_positions.unsqueeze(0) - self.prev_agent_positions.unsqueeze(1)).float()
        rel_pos_prev_norm = rel_pos_prev / gs  # [n, n, 2]

        # Fourier encode current position [n, n, 16]
        fourier_agents = self._fourier_encode(rel_pos_norm)

        # Velocity = (current - previous) * 10.0 for visibility [n, n, 2]
        velocity_agents = (rel_pos_norm - rel_pos_prev_norm) * 10.0

        # One-hot type: [is_food=0, is_agent=1] for all agent tokens [n, n, 2]
        type_onehot_agents = torch.zeros(n, n, 2, device=self.device)
        type_onehot_agents[:, :, 1] = 1  # is_agent = 1

        # HP normalized [n, n, 1]
        agent_hp = (self.agent_hp / self.config.max_hp).view(1, n, 1).expand(n, n, 1)

        # Social features [n, n, 4] scaled by 5.0 for stronger signal
        if self.config.ablate_ledger:
            social = torch.zeros(n, n, 4, device=self.device)
        else:
            social = self.ledger.get_normalized_tensor() * 5.0

        # Combine: [n, n, 25] = fourier[16] + velocity[2] + type[2] + hp[1] + social[4]
        agent_tokens = torch.cat([fourier_agents, velocity_agents, type_onehot_agents, agent_hp, social], dim=-1)
        tokens[:, :n, :] = agent_tokens

        # Agent mask: alive agents are valid
        mask[:, :n] = self.agent_alive.unsqueeze(0).expand(n, n)

        # === FOOD TOKENS (next max_food_tokens slots) ===
        # Gather all food positions
        poor_coords = torch.nonzero(self.poor_food, as_tuple=False)  # [F1, 2]
        rich_coords = torch.nonzero(self.rich_food, as_tuple=False)  # [F2, 2]

        # Combine with quality indicator (0.5 for poor, 1.0 for rich)
        if poor_coords.numel() > 0:
            poor_quality = torch.full((poor_coords.shape[0], 1), 0.5, device=self.device)
            poor_food = torch.cat([poor_coords.float(), poor_quality], dim=-1)  # [F1, 3]
        else:
            poor_food = torch.zeros(0, 3, device=self.device)

        if rich_coords.numel() > 0:
            rich_quality = torch.full((rich_coords.shape[0], 1), 1.0, device=self.device)
            rich_food = torch.cat([rich_coords.float(), rich_quality], dim=-1)  # [F2, 3]
        else:
            rich_food = torch.zeros(0, 3, device=self.device)

        # Combine all food: [F, 3] where columns are (row, col, quality)
        all_food = torch.cat([poor_food, rich_food], dim=0)
        n_food = all_food.shape[0]

        if n_food > 0:
            # For each agent, compute distance to each food and select K nearest
            agent_pos = self.agent_positions.float()  # [n, 2]
            food_pos = all_food[:, :2]  # [F, 2]
            food_quality = all_food[:, 2]  # [F]

            # Relative positions: food_pos - agent_pos -> [n, F, 2]
            rel_pos_food = food_pos.unsqueeze(0) - agent_pos.unsqueeze(1)

            # Manhattan distance for sorting
            dist = rel_pos_food.abs().sum(dim=-1)  # [n, F]

            # Get K nearest for each agent
            k = min(max_food, n_food)
            _, nearest_idx = dist.topk(k, dim=1, largest=False)  # [n, k]

            # Gather nearest food positions and qualities
            batch_idx = torch.arange(n, device=self.device).unsqueeze(1).expand(n, k)
            nearest_rel_pos = rel_pos_food[batch_idx, nearest_idx]  # [n, k, 2]
            nearest_quality = food_quality[nearest_idx]  # [n, k]

            # Normalize positions for Fourier encoding
            nearest_rel_norm = nearest_rel_pos / gs  # [n, k, 2]

            # Fourier encode food positions [n, k, 16]
            fourier_food = self._fourier_encode(nearest_rel_norm)

            # Velocity for food = 0 (food doesn't move) [n, k, 2]
            velocity_food = torch.zeros(n, k, 2, device=self.device)

            # One-hot type: [is_food=1, is_agent=0] [n, k, 2]
            type_onehot_food = torch.zeros(n, k, 2, device=self.device)
            type_onehot_food[:, :, 0] = 1  # is_food = 1

            # Quality [n, k, 1]
            food_val = nearest_quality.unsqueeze(-1)

            # No social for food [n, k, 4]
            food_social = torch.zeros(n, k, 4, device=self.device)

            # Combine: [n, k, 25] = fourier[16] + velocity[2] + type[2] + value[1] + social[4]
            food_tokens = torch.cat([fourier_food, velocity_food, type_onehot_food, food_val, food_social], dim=-1)

            # Place in output (after agent tokens)
            tokens[:, n:n+k, :] = food_tokens
            mask[:, n:n+k] = True

        # === PREDATOR TOKENS (after food tokens) ===
        if self.n_predators > 0 and self.predator_alive is not None:
            pred_start = n + max_food  # Slot index where predator tokens begin
            for p in range(self.n_predators):
                if not self.predator_alive[p]:
                    continue

                slot = pred_start + p
                if slot >= max_ent:
                    break

                # Relative position normalized to [-1, 1]
                rel_pos_pred = (self.predator_positions[p].float() - self.agent_positions.float()) / gs  # [n, 2]
                fourier_pred = self._fourier_encode(rel_pos_pred)  # [n, 16]

                # Velocity
                rel_prev_pred = (self.predator_prev_positions[p].float() - self.prev_agent_positions.float()) / gs
                velocity_pred = (rel_pos_pred - rel_prev_pred) * 10.0  # [n, 2]

                # Type: agent-like [0, 1]
                type_pred = torch.zeros(n, 2, device=self.device)
                type_pred[:, 1] = 1  # is_agent

                # Value: predator HP normalized by max_hp (>1.0 distinguishes from agents)
                hp_pred = torch.full((n, 1), self.predator_hp[p].item() / self.config.max_hp, device=self.device)

                # Social: from predator ledger (damage dealt/received)
                social_pred = torch.zeros(n, 4, device=self.device)
                if self.predator_ledger is not None:
                    social_pred[:, 0] = (self.predator_ledger[:, p, 0] / 100.0).clamp(0, 1) * 5.0
                    social_pred[:, 1] = (self.predator_ledger[:, p, 1] / 100.0).clamp(0, 1) * 5.0

                # Combine: [n, 25]
                pred_token = torch.cat([fourier_pred, velocity_pred, type_pred, hp_pred, social_pred], dim=-1)
                tokens[:, slot, :] = pred_token
                mask[:, slot] = True

        return tokens, mask


# Wrapper for backward compatibility with dict-based API
class GridWorldCompat(GridWorld):
    """Wrapper providing backward-compatible dict-based API."""

    def step(self, directions: Dict[int, int], action_types: Dict[int, int]) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, dict]
    ]:
        """Dict-based step for backward compatibility.

        Args:
            directions: Dict mapping agent_id -> direction (0-4)
            action_types: Dict mapping agent_id -> action type (0-4)
        """
        # Convert dicts to tensors
        direction_tensor = torch.tensor(
            [directions[i] for i in range(self.n_agents)],
            device=self.device, dtype=torch.long
        )
        action_type_tensor = torch.tensor(
            [action_types[i] for i in range(self.n_agents)],
            device=self.device, dtype=torch.long
        )

        # Call vectorized step
        obs, rewards, dones, infos = super().step(direction_tensor, action_type_tensor)

        # Convert outputs to dicts
        obs_dict = {}
        for i in range(self.n_agents):
            obs_dict[i] = {
                'entity_tokens': obs['entity_tokens'][i],
                'entity_mask': obs['entity_mask'][i],
                'signals': obs['signals'][i],
                'self_hp': obs['self_hp'][i],
                'self_inventory': obs['self_inventory'][i],
                'agent_id': obs['agent_id'][i]
            }

        rewards_dict = {i: rewards[i].item() for i in range(self.n_agents)}
        dones_dict = {i: dones[i].item() for i in range(self.n_agents)}
        infos_dict = {i: {} for i in range(self.n_agents)}

        return obs_dict, rewards_dict, dones_dict, infos_dict

    def reset(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Dict-based reset for backward compatibility."""
        obs = super().reset()

        obs_dict = {}
        for i in range(self.n_agents):
            obs_dict[i] = {
                'entity_tokens': obs['entity_tokens'][i],
                'entity_mask': obs['entity_mask'][i],
                'signals': obs['signals'][i],
                'self_hp': obs['self_hp'][i],
                'self_inventory': obs['self_inventory'][i],
                'agent_id': obs['agent_id'][i]
            }

        return obs_dict
