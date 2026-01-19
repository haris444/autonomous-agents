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


# Unified action space (15 actions total - move OR interact per step)
# Movement actions (0-4)
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
ACTION_STAY = 4

# Interaction actions (5-14)
ACTION_ATTACK_UP, ACTION_ATTACK_DOWN, ACTION_ATTACK_LEFT, ACTION_ATTACK_RIGHT = 5, 6, 7, 8
ACTION_GIVE_UP, ACTION_GIVE_DOWN, ACTION_GIVE_LEFT, ACTION_GIVE_RIGHT = 9, 10, 11, 12
ACTION_SIGNAL = 13
ACTION_COOPERATE = 14

# Legacy constants for internal use (relative to interact action base)
# Used when decomposing unified actions for processing
INTERACT_BASE = 5  # First interact action index
ATTACK_DIR_OFFSET = 0   # ATTACK_UP is at INTERACT_BASE + 0
GIVE_DIR_OFFSET = 4     # GIVE_UP is at INTERACT_BASE + 4

# Direction deltas: UP, DOWN, LEFT, RIGHT
DIR_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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

        # Cooperation tracking for diagnostics
        self.last_coop_success_count = 0  # Number of successful cooperations in last step

        # Current scenario (set via apply_scenario() after reset())
        self._current_scenario = None

    def _update_occupancy(self) -> None:
        """Update occupancy grid from agent positions - FULLY VECTORIZED."""
        self.occupancy.fill_(-1)
        alive_mask = self.agent_alive

        if not alive_mask.any():
            return

        alive_indices = torch.where(alive_mask)[0]
        alive_positions = self.agent_positions[alive_indices]
        rows = alive_positions[:, 0]
        cols = alive_positions[:, 1]

        # Advanced indexing: write all at once
        self.occupancy[rows, cols] = alive_indices

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

    def step(self, actions: torch.Tensor) -> Tuple[
        Dict[str, torch.Tensor],  # observations (batched)
        torch.Tensor,              # rewards [n_agents]
        torch.Tensor,              # dones [n_agents]
        dict                       # infos
    ]:
        """
        Execute one environment step with unified actions.

        Each agent chooses ONE action per step (move OR interact, not both).

        Action space (15 total):
            0-4: Movement (UP, DOWN, LEFT, RIGHT, STAY)
            5-8: ATTACK (UP, DOWN, LEFT, RIGHT)
            9-12: GIVE (UP, DOWN, LEFT, RIGHT)
            13: SIGNAL
            14: COOPERATE

        Args:
            actions: [n_agents] int tensor of unified actions (0-14)

        Returns:
            observations, rewards, dones, infos
        """
        self.step_count += 1

        # === Decompose unified actions into move vs interact ===
        # Actions 0-4 are movement, 5-14 are interactions
        is_move = actions < INTERACT_BASE  # Actions 0-4 are moves
        is_interact = ~is_move

        # For movement: use action directly if moving, else STAY
        move_actions = torch.where(is_move, actions, torch.tensor(ACTION_STAY, device=self.device))

        # For interactions: convert to relative index (0-9)
        # Actions 5-14 map to interact indices 0-9
        interact_actions = torch.where(is_interact, actions - INTERACT_BASE, torch.tensor(-1, device=self.device))

        # === Track HP at start ===
        hp_before = self.agent_hp.clone()

        # Clear signals from last step
        self.signals.zero_()

        # 1. Process interactions FIRST (attack, give food, signal)
        #    This happens from CURRENT position before movement
        #    So agent attacks what they SEE in their observation
        #    Only process if agent chose an interact action
        attack_rewards, damage_taken, defense_rewards, revenge_rewards = self._resolve_interactions(interact_actions, is_interact)

        # 2. Resolve movement (Heavyweight Rule)
        #    Only process if agent chose a move action
        self._resolve_movement(move_actions, is_move)
        self._update_occupancy()

        # 3. Apply HP decay (vectorized)
        self._apply_hp_decay()

        # 4. Process food eating (agents on food cells after movement)
        food_rewards = self._process_food_eating(actions)

        # 4a. Compute intrinsic reward for attempting COOP near rich food + ally
        intrinsic_coop_rewards = self._compute_intrinsic_coop_rewards(actions)

        # 4b. Auto-consume inventory to heal (if HP < max and have inventory)
        self._consume_inventory()

        # 5. Spawn new food
        self._spawn_food()

        # 6. Check deaths (HP <= 0)
        death_rewards = self._check_deaths()
        self._update_occupancy()

        # === EXPLICIT REWARD COMPUTATION ===
        hp_after = self.agent_hp.clone()

        # Non-linear damage pain (lower HP = hurts more)
        # At full HP: multiplier ≈ 1, at 10% HP: multiplier ≈ 10
        hp_ratio = hp_before / self.config.max_hp
        pain_multiplier = 1.0 / (hp_ratio + 0.1)
        damage_pain = damage_taken * pain_multiplier * self.config.r_damage_taken

        # Low HP penalty (constant per tick, scales with how low HP is)
        # At full HP: penalty ≈ 0, at 10% HP: penalty ≈ 0.9 * r_low_hp
        hp_ratio_after = hp_after / self.config.max_hp
        low_hp_penalty = (1.0 - hp_ratio_after) * self.config.r_low_hp * self.agent_alive.float()

        # Combine all rewards
        rewards = (
            death_rewards           # Death penalty (-50.0)
            + food_rewards          # Eat food (+5.0)
            + intrinsic_coop_rewards  # Intrinsic reward for COOP attempt near food + ally
            + attack_rewards        # Attack bonus
            + defense_rewards       # Defense bonus (for protecting allies)
            + revenge_rewards       # Revenge bonus (for retaliating against attackers)
            + damage_pain           # Damage pain (non-linear)
            + low_hp_penalty        # Low HP penalty (constant per tick)
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

        infos = {}

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
        safe_occupant_idx = occupant_at_dest.clamp(min=0)
        occupant_is_staying = is_staying[safe_occupant_idx] & has_occupant

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

    def _resolve_interactions(self, interact_actions: torch.Tensor, is_interact: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process attack, give food, and signal actions with direction targeting.

        Args:
            interact_actions: [n_agents] relative interact action indices (0-9)
                0-3: ATTACK (UP, DOWN, LEFT, RIGHT)
                4-7: GIVE (UP, DOWN, LEFT, RIGHT)
                8: SIGNAL
                9: COOPERATE
                -1: Not interacting (chose movement)
            is_interact: [n_agents] bool mask - only process agents where True

        Returns:
            attack_rewards: [n_agents] reward for damage dealt
            damage_taken: [n_agents] damage received by each agent
            defense_rewards: [n_agents] reward for defending allies
            revenge_rewards: [n_agents] reward for retaliating against attackers
        """
        # Only process agents who chose interact actions and are alive
        active_mask = is_interact & self.agent_alive

        # Signal actions (relative action 8)
        signal_mask = (interact_actions == 8) & active_mask
        self.signals = signal_mask

        # Attack actions (relative 0-3: 4 cardinal directions)
        attack_mask = (interact_actions >= 0) & (interact_actions <= 3) & active_mask
        attack_rewards, damage_taken, defense_rewards, revenge_rewards = self._resolve_attacks(interact_actions, attack_mask)

        # Give food actions (relative 4-7: 4 cardinal directions)
        give_mask = (interact_actions >= 4) & (interact_actions <= 7) & active_mask
        self._resolve_give_food(interact_actions, give_mask)

        return attack_rewards, damage_taken, defense_rewards, revenge_rewards

    def _resolve_attacks(self, interact_actions: torch.Tensor, attack_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resolve attacks with direction targeting and defense tracking - FULLY VECTORIZED.

        Each attacker targets ONE cell in their chosen direction.
        Defense: if C attacks A who attacked B in last 3 steps, C defended B.
        Revenge: if C attacks A who attacked C recently, C gets revenge reward.
        Attack costs 5% of max HP regardless of whether it hits.

        Returns:
            attack_rewards: [n_agents] reward for damage dealt
            damage_taken: [n_agents] damage received by each agent
            defense_rewards: [n_agents] reward for defending allies
            revenge_rewards: [n_agents] reward for retaliating
        """
        # Initialize return tensors
        attack_rewards = torch.zeros(self.n_agents, device=self.device)
        damage_taken = torch.zeros(self.n_agents, device=self.device)
        defense_rewards = torch.zeros(self.n_agents, device=self.device)
        revenge_rewards = torch.zeros(self.n_agents, device=self.device)

        # Early exit if no attackers
        if not attack_mask.any():
            return attack_rewards, damage_taken, defense_rewards, revenge_rewards

        # === STEP 1: Apply attack cost to ALL attackers at once ===
        attack_cost = self.config.max_hp * 0.05
        self.agent_hp = self.agent_hp - attack_cost * attack_mask.float()

        # === STEP 2: Compute target positions for ALL agents ===
        attack_dirs = interact_actions.clamp(0, 3)
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

        # Valid hit: in bounds, attacking, target exists, target alive
        has_target = target_ids >= 0
        valid_target_ids = target_ids.clamp(min=0)
        target_alive = self.agent_alive[valid_target_ids] & has_target
        valid_hit = valid_attack & has_target & target_alive

        # === STEP 6: Compute damage ===
        damage = (self.agent_hp / 2) * valid_hit.float()

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
        # Attack rewards = 50% of ALL damage dealt (incentivizes combat)
        total_damage_dealt = damage_matrix.sum(dim=1)  # Sum over all targets
        attack_rewards = total_damage_dealt * self.config.r_attack_mult

        # Damage taken per agent (sum over attackers) - still tracks all damage for pain calculation
        damage_taken = damage_matrix.sum(dim=0)

        return attack_rewards, damage_taken, defense_rewards, revenge_rewards

    def _resolve_give_food(self, interact_actions: torch.Tensor, give_mask: torch.Tensor) -> None:
        """
        Resolve food giving from INVENTORY - FULLY VECTORIZED.

        Giver transfers from their INVENTORY (no HP cost!).
        Target receives to their INVENTORY.
        Only transfers if giver has inventory AND valid target exists.
        """
        food_value = self.config.poor_food_value

        # Early exit if no givers
        if not give_mask.any():
            return

        # === STEP 1: Compute transfer amount for each giver ===
        # Can only give what you have in inventory (up to food_value per action)
        give_amount = torch.minimum(
            torch.full((self.n_agents,), food_value, device=self.device),
            self.agent_inventory
        ) * give_mask.float()

        # Valid givers have inventory to give
        valid_giver = give_mask & (give_amount > 0)

        # === STEP 2: Compute target positions ===
        # Relative interact actions 4-7 are GIVE directions (UP, DOWN, LEFT, RIGHT)
        give_dirs = (interact_actions - 4).clamp(0, 3)
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

            # === STEP 6: Update ledger ===
            flat_indices = transfer_indices * self.n_agents + transfer_targets
            ledger_food = self.ledger.tensor[:, :, Ledger.FOOD_GIVEN].view(-1)
            ledger_food.scatter_add_(0, flat_indices, transfer_values)

    def _process_food_eating(self, actions: torch.Tensor) -> torch.Tensor:
        """Process agents eating food on their cells.

        Poor food: picked up automatically by stepping on it -> goes to INVENTORY
        Rich food: requires 2+ adjacent agents who both chose COOPERATE action -> goes to INVENTORY

        Args:
            actions: [n_agents] unified action indices (0-14)

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

        # Remove eaten poor food
        poor_food_flat = self.poor_food.view(-1)
        eat_indices = rows[on_poor] * self.grid_size + cols[on_poor]
        poor_food_flat[eat_indices] = False

        # Rich food - requires 2+ adjacent agents who BOTH chose COOPERATE (vectorized)
        # ACTION_COOPERATE = 14 in unified action space
        coop_mask = (actions == ACTION_COOPERATE) & self.agent_alive

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

        # Ledger update: cooperation pairs via outer product
        consumed_food_indices = torch.where(consumed_mask)[0]  # [C]
        eligible_consumed = eligible[:, consumed_food_indices].float()  # [N, C]
        coop_pairs = eligible_consumed @ eligible_consumed.T  # [N, N]
        coop_pairs.fill_diagonal_(0)  # No self-cooperation
        self.ledger.tensor[:, :, Ledger.COOP_COUNT] += coop_pairs

        # Remove consumed food
        consumed_rows = rich_food_coords[consumed_food_indices, 0]
        consumed_cols = rich_food_coords[consumed_food_indices, 1]
        self.rich_food[consumed_rows, consumed_cols] = False

        # Return explicit reward for participating in rich food consumption (100x during warmup)
        reward_multiplier = 100.0 if self.episode_count <= self.warmup_episodes else 1.0
        participated = (foods_per_agent > 0).float()
        return participated * self.config.r_large * reward_multiplier

    def _compute_intrinsic_coop_rewards(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward for attempting COOP when conditions are right.

        An agent gets intrinsic reward if:
        1. They chose the COOP action
        2. They are within distance 1 of rich food
        3. Another alive agent is also within distance 1 of the same rich food

        This encourages agents to try COOP even before they coordinate successfully.

        Args:
            actions: [n_agents] unified action indices (0-14)
        """
        intrinsic_rewards = torch.zeros(self.n_agents, device=self.device)

        # Get agents who chose COOP and are alive
        # ACTION_COOPERATE = 14 in unified action space
        coop_mask = (actions == ACTION_COOPERATE) & self.agent_alive

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

    def _apply_hp_decay(self) -> None:
        """Apply HP decay to all alive agents - fully vectorized."""
        decay = self.config.max_hp * self.config.hp_decay_rate
        self.agent_hp = self.agent_hp - decay * self.agent_alive.float()

    def _consume_inventory(self) -> None:
        """Auto-heal when HP < max. Free healing, no inventory required."""
        # How much HP is missing?
        hp_missing = self.config.max_hp - self.agent_hp

        # Heal up to 10 HP per tick for free (enough to offset HP decay + some)
        heal_amount = torch.minimum(hp_missing, torch.tensor(10.0, device=self.device))

        # Only heal alive agents with missing HP
        heal_amount = heal_amount * self.agent_alive.float() * (hp_missing > 0).float()

        # Apply healing (free, no inventory consumed)
        self.agent_hp = self.agent_hp + heal_amount

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

        # Check advancement every 20 episodes
        if len(self.episode_returns) >= 20:
            avg_return = sum(self.episode_returns[-20:]) / 20

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
            'entity_tokens': entity_tokens,    # [n_agents, max_entities, 8]
            'entity_mask': entity_mask,        # [n_agents, max_entities]
            'signals': self.signals.float().unsqueeze(0).expand(self.n_agents, -1),
            'self_hp': (self.agent_hp / self.config.max_hp).unsqueeze(1),
            'self_inventory': (self.agent_inventory / self.config.max_hp).unsqueeze(1),  # Normalized by max_hp
            'agent_id': torch.arange(self.n_agents, device=self.device)
        }

    def get_action_masks(self) -> torch.Tensor:
        """
        Compute valid action mask for unified action space.

        Returns:
            action_mask: [n_agents, 15] bool - which actions are valid

        Action space:
            0-4: Movement (UP, DOWN, LEFT, RIGHT, STAY)
            5-8: ATTACK (UP, DOWN, LEFT, RIGHT) - needs adjacent agent
            9-12: GIVE (UP, DOWN, LEFT, RIGHT) - needs adjacent agent
            13: SIGNAL - always valid
            14: COOPERATE - needs rich food within distance 1
        """
        n = self.n_agents
        gs = self.grid_size

        # Initialize all actions as valid
        action_mask = torch.ones(n, 15, dtype=torch.bool, device=self.device)

        # === MOVEMENT (0-4) ===
        # Check each direction: in bounds?
        intended = self.agent_positions.unsqueeze(1) + self.direction_deltas.unsqueeze(0)  # [n, 5, 2]
        in_bounds = (
            (intended[..., 0] >= 0) & (intended[..., 0] < gs) &
            (intended[..., 1] >= 0) & (intended[..., 1] < gs)
        )
        move_mask = in_bounds & self.agent_alive.unsqueeze(1)  # [n, 5]
        action_mask[:, 0:5] = move_mask

        # === DIRECTION TARGETS (for attack/give) [n, 4] ===
        # Check each of 4 cardinal directions: is there a living agent?
        target_pos = self.agent_positions.unsqueeze(1) + self.dir_deltas_4.unsqueeze(0)  # [n, 4, 2]
        target_in_bounds = (
            (target_pos[..., 0] >= 0) & (target_pos[..., 0] < gs) &
            (target_pos[..., 1] >= 0) & (target_pos[..., 1] < gs)
        )
        safe_rows = target_pos[..., 0].clamp(0, gs - 1)
        safe_cols = target_pos[..., 1].clamp(0, gs - 1)
        occupants = self.occupancy[safe_rows, safe_cols]  # [n, 4]

        has_target = (occupants >= 0)
        target_alive = self.agent_alive[occupants.clamp(min=0)] & has_target
        direction_valid = target_in_bounds & target_alive & self.agent_alive.unsqueeze(1)  # [n, 4]

        # === ATTACK (5-8) ===
        action_mask[:, 5:9] = direction_valid

        # === GIVE (9-12) ===
        action_mask[:, 9:13] = direction_valid

        # === SIGNAL (13) - always valid for alive agents ===
        action_mask[:, 13] = self.agent_alive

        # === COOPERATE (14) - valid if rich food within distance 1 ===
        rich_coords = torch.nonzero(self.rich_food, as_tuple=False)  # [F, 2]
        if rich_coords.numel() > 0:
            pos = self.agent_positions.float()
            food_pos = rich_coords.float()
            dist = (pos.unsqueeze(1) - food_pos.unsqueeze(0)).abs().sum(dim=-1)  # [n, F]
            rich_nearby = (dist <= 1).any(dim=1)  # [n]
        else:
            rich_nearby = torch.zeros(n, dtype=torch.bool, device=self.device)
        action_mask[:, 14] = rich_nearby & self.agent_alive

        return action_mask

    def _get_entity_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create unified entity tokens for agents AND food.

        Token format (8 features):
            - dx, dy: relative position (normalized by grid size)
            - entity_type: 0=food, 1=agent
            - value: quality (food) or hp (agent)
            - social[4]: interaction history (zeros for food)

        Returns:
            tokens: [n_agents, max_entities, 8] - padded to max_entities
            mask: [n_agents, max_entities] - True if token is valid
        """
        n = self.n_agents
        gs = self.config.grid_size
        max_food = self.config.max_food_tokens
        max_ent = self.config.max_entities  # n_agents + max_food_tokens

        # Initialize output tensors
        tokens = torch.zeros(n, max_ent, 8, device=self.device)
        mask = torch.zeros(n, max_ent, device=self.device, dtype=torch.bool)

        # === AGENT TOKENS (first n_agents slots) ===
        # Relative positions: pos[j] - pos[i] for all pairs
        # Scale by 5x after normalization so positions have stronger signal in embeddings
        rel_pos_agents = (self.agent_positions.unsqueeze(0) - self.agent_positions.unsqueeze(1)).float()
        rel_pos_agents = (rel_pos_agents / gs) * 5.0  # Normalize and scale up

        # Agent features
        agent_type = torch.ones(n, n, 1, device=self.device)  # type=1 for agents
        agent_hp = (self.agent_hp / self.config.max_hp).view(1, n, 1).expand(n, n, 1)
        # Scale social features by 5x to match position feature magnitude
        social = self.ledger.get_normalized_tensor() * 5.0  # [n, n, 4] scaled to [0, 5]

        # Combine: [n, n, 8] = dx, dy, type, hp, social[4]
        agent_tokens = torch.cat([rel_pos_agents, agent_type, agent_hp, social], dim=-1)
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

            # Normalize and scale up positions (5x for stronger signal)
            nearest_rel_pos = (nearest_rel_pos / gs) * 5.0

            # Build food tokens: [dx, dy, type=0, quality, 0, 0, 0, 0]
            food_type = torch.zeros(n, k, 1, device=self.device)  # type=0 for food
            food_val = nearest_quality.unsqueeze(-1)  # [n, k, 1]
            food_social = torch.zeros(n, k, 4, device=self.device)  # No social for food

            food_tokens = torch.cat([nearest_rel_pos, food_type, food_val, food_social], dim=-1)

            # Place in output (after agent tokens)
            tokens[:, n:n+k, :] = food_tokens
            mask[:, n:n+k] = True

        return tokens, mask


# Wrapper for backward compatibility with dict-based API
class GridWorldCompat(GridWorld):
    """Wrapper providing backward-compatible dict-based API."""

    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, dict]
    ]:
        """Dict-based step for backward compatibility.

        Args:
            actions: Dict mapping agent_id -> unified action (0-14)
        """
        # Convert dict to tensor
        action_tensor = torch.tensor(
            [actions[i] for i in range(self.n_agents)],
            device=self.device, dtype=torch.long
        )

        # Call vectorized step
        obs, rewards, dones, infos = super().step(action_tensor)

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
