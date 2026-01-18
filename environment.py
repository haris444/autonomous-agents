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


# Action indices for movement (8 directions + stay)
MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = 0, 1, 2, 3
MOVE_UP_LEFT, MOVE_UP_RIGHT, MOVE_DOWN_LEFT, MOVE_DOWN_RIGHT = 4, 5, 6, 7
MOVE_STAY = 8

# Interact actions with direction: ATTACK (0-7), GIVE (8-15), SIGNAL=16, COOPERATE=17, IDLE=18
ATTACK_UP, ATTACK_DOWN, ATTACK_LEFT, ATTACK_RIGHT = 0, 1, 2, 3
ATTACK_UP_LEFT, ATTACK_UP_RIGHT, ATTACK_DOWN_LEFT, ATTACK_DOWN_RIGHT = 4, 5, 6, 7
GIVE_UP, GIVE_DOWN, GIVE_LEFT, GIVE_RIGHT = 8, 9, 10, 11
GIVE_UP_LEFT, GIVE_UP_RIGHT, GIVE_DOWN_LEFT, GIVE_DOWN_RIGHT = 12, 13, 14, 15
INTERACT_SIGNAL = 16
INTERACT_COOPERATE = 17
INTERACT_IDLE = 18

# Direction deltas for interact actions (indices 0-7 for attack, 8-15 for give)
# Maps to: UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT
INTERACT_DIR_DELTAS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),      # Cardinal
    (-1, -1), (-1, 1), (1, -1), (1, 1)     # Diagonal
]


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

        # Direction deltas tensor [9, 2] for movement (8 directions + stay)
        self.direction_deltas = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1],      # Cardinal: UP, DOWN, LEFT, RIGHT
             [-1, -1], [-1, 1], [1, -1], [1, 1],    # Diagonal: UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT
             [0, 0]],                               # STAY
            device=device, dtype=torch.long
        )

        # Direction deltas tensor [8, 2] for attacks/give (cached for reuse)
        self.dir_deltas_8 = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1],
             [-1, -1], [-1, 1], [1, -1], [1, 1]],
            device=device, dtype=torch.long
        )

        # State tensors (initialized in reset)
        self.agent_positions: torch.Tensor = None  # [n_agents, 2] (row, col)
        self.agent_hp: torch.Tensor = None         # [n_agents]
        self.agent_alive: torch.Tensor = None      # [n_agents] bool
        self.poor_food: torch.Tensor = None        # [grid_size, grid_size] bool
        self.rich_food: torch.Tensor = None        # [grid_size, grid_size] bool
        self.signals: torch.Tensor = None          # [n_agents] bool
        self.occupancy: torch.Tensor = None        # [grid_size, grid_size] -> agent_id or -1

        # Episode tracking for warmup food boost
        self.episode_count = 0
        self.warmup_episodes = 20  # 100x food spawn/reward, 10x episode length

        self.step_count = 0

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
        self.step_count = 0
        self.episode_count += 1

        # Initialize occupancy grid
        self.occupancy = torch.full(
            (self.grid_size, self.grid_size), -1,
            device=self.device, dtype=torch.long
        )

        # Initialize agent positions (random, no overlap)
        # Use torch for random positions
        all_positions = torch.randperm(self.grid_size * self.grid_size, device=self.device)[:self.n_agents]
        rows = all_positions // self.grid_size
        cols = all_positions % self.grid_size
        self.agent_positions = torch.stack([rows, cols], dim=1)

        # Initialize HP and alive status
        self.agent_hp = torch.full((self.n_agents,), self.config.max_hp, device=self.device)
        self.agent_alive = torch.ones(self.n_agents, device=self.device, dtype=torch.bool)

        # Initialize food grids
        self.poor_food = torch.zeros((self.grid_size, self.grid_size),
                                      device=self.device, dtype=torch.bool)
        self.rich_food = torch.zeros((self.grid_size, self.grid_size),
                                      device=self.device, dtype=torch.bool)

        # Update occupancy and spawn food
        self._update_occupancy()
        self._spawn_food()

        # Initialize signals (none active)
        self.signals = torch.zeros(self.n_agents, device=self.device, dtype=torch.bool)

        return self._get_all_observations()

    def step(self, move_actions: torch.Tensor, interact_actions: torch.Tensor) -> Tuple[
        Dict[str, torch.Tensor],  # observations (batched)
        torch.Tensor,              # rewards [n_agents]
        torch.Tensor,              # dones [n_agents]
        dict                       # infos
    ]:
        """
        Execute one environment step with tensor inputs.

        Reward = HP_delta (HP_after - HP_before) + death_penalty

        Args:
            move_actions: [n_agents] int tensor of movement actions
            interact_actions: [n_agents] int tensor of interaction actions

        Returns:
            observations, rewards, dones, infos
        """
        self.step_count += 1

        # === Track HP at start ===
        hp_before = self.agent_hp.clone()

        # Clear signals from last step
        self.signals.zero_()

        # 1. Process interactions FIRST (attack, give food, signal)
        #    This happens from CURRENT position before movement
        #    So agent attacks what they SEE in their observation
        attack_rewards, damage_taken = self._resolve_interactions(interact_actions)

        # 2. Resolve movement (Heavyweight Rule)
        self._resolve_movement(move_actions)
        self._update_occupancy()

        # Compute smell reward: (food_value / distance) / 3
        # Stronger smell when closer, richer food smells more
        dist, food_value = self._compute_nearest_food_info(self.agent_positions)

        # Avoid division by zero (distance=0 means on food, they'll eat it)
        safe_dist = torch.clamp(dist, min=1.0)
        smell_rewards = (food_value / safe_dist) / 3.0

        # Handle no-food case (inf distance)
        smell_rewards = torch.where(
            torch.isinf(dist),
            torch.zeros_like(smell_rewards),
            smell_rewards
        )
        smell_rewards = smell_rewards * self.agent_alive.float()

        # 3. Apply HP decay (vectorized)
        self._apply_hp_decay()

        # 4. Process food eating (agents on food cells after movement)
        food_rewards = self._process_food_eating(interact_actions)

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
            (hp_after - hp_before)    # HP delta (implicit)
            + death_rewards           # Death penalty
            + food_rewards            # Explicit food bonus
            + attack_rewards          # Attack bonus
            + damage_pain             # Damage pain (non-linear)
            + low_hp_penalty          # Low HP penalty (constant per tick)
            + smell_rewards           # Smell reward for moving toward food
        )

        # Build outputs
        observations = self._get_all_observations()
        dones = ~self.agent_alive

        # Episode ends if all agents dead or max steps reached
        # Warmup episodes are 10x longer
        all_dead = ~self.agent_alive.any()
        episode_length = self.config.max_steps_per_episode * (10 if self.episode_count <= self.warmup_episodes else 1)
        max_steps = self.step_count >= episode_length
        if all_dead or max_steps:
            dones = torch.ones(self.n_agents, device=self.device, dtype=torch.bool)

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

    def _compute_nearest_food_info(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Manhattan distance and value of nearest food for each agent.

        Args:
            positions: [n_agents, 2] agent positions

        Returns:
            distances: [n_agents] distance to nearest food, or inf if no food
            values: [n_agents] value of nearest food, or 0 if no food
        """
        # Get food coordinates and their values
        poor_coords = torch.nonzero(self.poor_food, as_tuple=False)  # [P, 2]
        rich_coords = torch.nonzero(self.rich_food, as_tuple=False)  # [R, 2]

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

        # Check rich food
        if n_rich > 0:
            diff = agent_pos.unsqueeze(1) - rich_coords.float().unsqueeze(0)  # [N, R, 2]
            rich_dist = diff.abs().sum(dim=-1)  # [N, R]
            rich_min_dist, _ = rich_dist.min(dim=1)  # [N]

            closer_mask = rich_min_dist < min_dist
            min_dist = torch.where(closer_mask, rich_min_dist, min_dist)
            nearest_value = torch.where(closer_mask, self.config.rich_food_value, nearest_value)

        return min_dist, nearest_value

    def _resolve_movement(self, move_actions: torch.Tensor) -> None:
        """
        Fully vectorized movement resolution using the Heavyweight Rule.
        No loops, no .item() calls.
        """
        n = self.n_agents
        gs = self.grid_size

        # Only process alive agents
        alive_mask = self.agent_alive

        # Compute intended destinations for all agents
        intended = self.agent_positions + self.direction_deltas[move_actions]
        intended = intended.clamp(0, gs - 1)

        # Dead agents stay in place
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

    def _resolve_interactions(self, interact_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process attack, give food, and signal actions with direction targeting.

        Returns:
            attack_rewards: [n_agents] reward for damage dealt
            damage_taken: [n_agents] damage received by each agent
        """
        # Signal actions (action 8)
        signal_mask = (interact_actions == INTERACT_SIGNAL) & self.agent_alive
        self.signals = signal_mask

        # Attack actions (0-7: 8 directions)
        attack_mask = (interact_actions <= ATTACK_DOWN_RIGHT) & self.agent_alive
        attack_rewards, damage_taken = self._resolve_attacks(interact_actions, attack_mask)

        # Give food actions (8-15: 8 directions)
        give_mask = ((interact_actions >= GIVE_UP) & (interact_actions <= GIVE_DOWN_RIGHT)) & self.agent_alive
        self._resolve_give_food(interact_actions, give_mask)

        return attack_rewards, damage_taken

    def _resolve_attacks(self, interact_actions: torch.Tensor, attack_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve attacks with direction targeting and defense tracking - FULLY VECTORIZED.

        Each attacker targets ONE cell in their chosen direction.
        Defense: if C attacks A who is attacking B, C defended B.
        Attack costs 5% of max HP regardless of whether it hits.

        Returns:
            attack_rewards: [n_agents] reward for damage dealt
            damage_taken: [n_agents] damage received by each agent
        """
        # Initialize return tensors
        attack_rewards = torch.zeros(self.n_agents, device=self.device)
        damage_taken = torch.zeros(self.n_agents, device=self.device)

        # Early exit if no attackers
        if not attack_mask.any():
            return attack_rewards, damage_taken

        # === STEP 1: Apply attack cost to ALL attackers at once ===
        attack_cost = self.config.max_hp * 0.05
        self.agent_hp = self.agent_hp - attack_cost * attack_mask.float()

        # === STEP 2: Compute target positions for ALL agents ===
        attack_dirs = interact_actions.clamp(0, 7)
        deltas = self.dir_deltas_8[attack_dirs]  # [n_agents, 2]
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

        # === STEP 9: Defense tracking (vectorized where possible) ===
        # attack_target[i] = target agent id if i hit someone, else -1
        attack_target = torch.where(valid_hit, target_ids,
                                    torch.full_like(target_ids, -1))

        if hit_indices.numel() > 0:
            # C = attackers who hit someone
            c_indices = hit_indices
            a_indices = attack_target[c_indices]  # who C attacked

            # Check if A was attacking someone B
            a_was_hitting = valid_hit[a_indices]
            b_indices = attack_target[a_indices]

            # Defense: C attacked A, A was attacking B
            defense_mask = a_was_hitting & (b_indices >= 0)

            if defense_mask.any():
                defense_c = c_indices[defense_mask]
                defense_b = b_indices[defense_mask]
                defense_damage = damage[defense_c]

                # Update defense scores (vectorized with scatter_add_)
                n = self.n_agents
                flat_idx = defense_c * n + defense_b
                ledger_flat = self.ledger.tensor[:, :, Ledger.DEFENSE_SCORE].view(-1)
                ledger_flat.scatter_add_(0, flat_idx, defense_damage)

        # === STEP 10: Compute rewards and damage taken ===
        # Attack rewards = damage dealt * multiplier
        damage_dealt_per_agent = damage_matrix.sum(dim=1)  # Sum over targets
        attack_rewards = damage_dealt_per_agent * self.config.r_attack_mult

        # Damage taken per agent (sum over attackers)
        damage_taken = damage_matrix.sum(dim=0)

        return attack_rewards, damage_taken

    def _resolve_give_food(self, interact_actions: torch.Tensor, give_mask: torch.Tensor) -> None:
        """
        Resolve food giving with direction targeting and HP transfer - FULLY VECTORIZED.

        Giver LOSES HP regardless of whether there's a valid target (costs HP to the void).
        Target GAINS HP if present.
        If giver HP < food_value, transfers all remaining HP.
        """
        food_value = self.config.poor_food_value

        # Early exit if no givers
        if not give_mask.any():
            return

        # === STEP 1: Compute cost amount for each giver ===
        # cost = min(food_value, giver_hp), but only for givers
        cost_amount = torch.minimum(
            torch.full((self.n_agents,), food_value, device=self.device),
            self.agent_hp
        ) * give_mask.float()

        # Valid givers have HP to give
        valid_giver = give_mask & (cost_amount > 0)

        # === STEP 2: Deduct cost from all givers (even to void) ===
        self.agent_hp = self.agent_hp - cost_amount

        # === STEP 3: Compute target positions ===
        give_dirs = (interact_actions - GIVE_UP).clamp(0, 7)
        deltas = self.dir_deltas_8[give_dirs]
        target_positions = self.agent_positions + deltas
        target_rows = target_positions[:, 0]
        target_cols = target_positions[:, 1]

        # === STEP 4: Check bounds ===
        in_bounds = (
            (target_rows >= 0) & (target_rows < self.grid_size) &
            (target_cols >= 0) & (target_cols < self.grid_size)
        )

        # Clamp for safe indexing
        safe_rows = target_rows.clamp(0, self.grid_size - 1)
        safe_cols = target_cols.clamp(0, self.grid_size - 1)

        # === STEP 5: Look up targets ===
        target_ids = self.occupancy[safe_rows, safe_cols]

        has_target = target_ids >= 0
        valid_target_ids = target_ids.clamp(min=0)
        target_alive = self.agent_alive[valid_target_ids] & has_target
        valid_transfer = valid_giver & in_bounds & has_target & target_alive

        # === STEP 6: Transfer HP to targets ===
        transfer_indices = torch.where(valid_transfer)[0]

        if transfer_indices.numel() > 0:
            transfer_targets = target_ids[transfer_indices]
            transfer_values = cost_amount[transfer_indices]

            # Aggregate HP gains per target using scatter_add
            hp_gain = torch.zeros(self.n_agents, device=self.device)
            hp_gain.scatter_add_(0, transfer_targets, transfer_values)

            # Add HP (clamped to max)
            self.agent_hp = (self.agent_hp + hp_gain).clamp(max=self.config.max_hp)

            # === STEP 7: Update ledger ===
            # Flatten index for scatter_add on 2D ledger slice
            flat_indices = transfer_indices * self.n_agents + transfer_targets
            ledger_food = self.ledger.tensor[:, :, Ledger.FOOD_GIVEN].view(-1)
            ledger_food.scatter_add_(0, flat_indices, transfer_values)

    def _process_food_eating(self, interact_actions: torch.Tensor) -> torch.Tensor:
        """Process agents eating food on their cells.

        Poor food: eaten automatically by stepping on it
        Rich food: requires 2+ adjacent agents who both chose COOPERATE action
        Cooperate action costs 1 HP regardless of success.

        Returns:
            food_rewards: [n_agents] explicit reward for eating food
        """
        food_rewards = torch.zeros(self.n_agents, device=self.device)

        # Get agent positions
        rows = self.agent_positions[:, 0]
        cols = self.agent_positions[:, 1]

        # Poor food - check which alive agents are on poor food
        on_poor = self.poor_food[rows, cols] & self.agent_alive

        # Explicit reward for eating poor food (100x during warmup)
        reward_multiplier = 100.0 if self.episode_count <= self.warmup_episodes else 1.0
        food_rewards += on_poor.float() * self.config.r_small * reward_multiplier

        # Update HP for agents eating poor food
        self.agent_hp = torch.where(
            on_poor,
            (self.agent_hp + self.config.poor_food_value).clamp(max=self.config.max_hp),
            self.agent_hp
        )

        # Remove eaten poor food
        poor_food_flat = self.poor_food.view(-1)
        eat_indices = rows[on_poor] * self.grid_size + cols[on_poor]
        poor_food_flat[eat_indices] = False

        # Rich food - requires 2+ adjacent agents who BOTH chose COOPERATE (vectorized)
        coop_mask = (interact_actions == INTERACT_COOPERATE) & self.agent_alive

        # Cooperate action costs 1 HP (even if no food consumed)
        self.agent_hp = self.agent_hp - coop_mask.float() * 1.0

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
            return torch.zeros(self.n_agents, device=self.device)

        # HP update: count consumed foods each agent participated in
        foods_per_agent = eligible.float() @ consumed_mask.float()  # [N]
        hp_gain = foods_per_agent * self.config.rich_food_value
        self.agent_hp = (self.agent_hp + hp_gain).clamp(max=self.config.max_hp)

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

    def _apply_hp_decay(self) -> None:
        """Apply HP decay to all alive agents - fully vectorized."""
        decay = self.config.max_hp * self.config.hp_decay_rate
        self.agent_hp = self.agent_hp - decay * self.agent_alive.float()

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

    def _spawn_food(self) -> None:
        """Spawn new food on empty cells - fully vectorized."""
        # Mask of empty cells (no agent, no food)
        empty = (self.occupancy == -1) & ~self.poor_food & ~self.rich_food

        # 100x food spawn rate during warmup episodes
        food_multiplier = 100.0 if self.episode_count <= self.warmup_episodes else 1.0
        poor_rate = min(1.0, self.config.poor_food_spawn_rate * food_multiplier)
        rich_rate = min(1.0, self.config.rich_food_spawn_rate * food_multiplier)

        # Random spawn probabilities
        rand = torch.rand((self.grid_size, self.grid_size), device=self.device)

        # Spawn poor food
        spawn_poor = empty & (rand < poor_rate)
        self.poor_food = self.poor_food | spawn_poor

        # Spawn rich food (only where poor didn't spawn)
        rand2 = torch.rand((self.grid_size, self.grid_size), device=self.device)
        spawn_rich = empty & ~spawn_poor & (rand2 < rich_rate)
        self.rich_food = self.rich_food | spawn_rich

    def _get_all_observations(self) -> Dict[str, torch.Tensor]:
        """Build batched observations for all agents."""
        return {
            'spatial': self._get_all_spatial_obs(),
            'ledger': self.ledger.get_normalized_tensor().unsqueeze(0).expand(self.n_agents, -1, -1, -1),
            'signals': self.signals.float().unsqueeze(0).expand(self.n_agents, -1),
            'self_hp': (self.agent_hp / self.config.max_hp).unsqueeze(1),
            'agent_id': torch.arange(self.n_agents, device=self.device)
        }

    def get_action_masks(self) -> Dict[str, torch.Tensor]:
        """
        Compute valid action masks for current state.

        Returns dict with:
            move_mask: [n_agents, 9] bool - which moves are valid
            interact_type_mask: [n_agents, 5] bool - which interact types valid
            direction_mask: [n_agents, 8] bool - which directions have targets
        """
        n = self.n_agents
        gs = self.grid_size

        # === MOVE MASK [n, 9] ===
        # Check each direction: in bounds? (STAY is always valid)
        intended = self.agent_positions.unsqueeze(1) + self.direction_deltas.unsqueeze(0)  # [n, 9, 2]
        in_bounds = (
            (intended[..., 0] >= 0) & (intended[..., 0] < gs) &
            (intended[..., 1] >= 0) & (intended[..., 1] < gs)
        )
        move_mask = in_bounds & self.agent_alive.unsqueeze(1)  # [n, 9]

        # === DIRECTION MASK [n, 8] (for attack/give) ===
        # Check each of 8 directions: is there a living agent?
        target_pos = self.agent_positions.unsqueeze(1) + self.dir_deltas_8.unsqueeze(0)  # [n, 8, 2]
        target_in_bounds = (
            (target_pos[..., 0] >= 0) & (target_pos[..., 0] < gs) &
            (target_pos[..., 1] >= 0) & (target_pos[..., 1] < gs)
        )
        safe_rows = target_pos[..., 0].clamp(0, gs - 1)
        safe_cols = target_pos[..., 1].clamp(0, gs - 1)
        occupants = self.occupancy[safe_rows, safe_cols]  # [n, 8]

        has_target = (occupants >= 0)
        target_alive = self.agent_alive[occupants.clamp(min=0)] & has_target
        direction_mask = target_in_bounds & target_alive  # [n, 8]

        # === INTERACT TYPE MASK [n, 5] ===
        # 0=ATTACK: always valid (direction mask handles targeting)
        # 1=GIVE: always valid (direction mask handles targeting)
        # 2=SIGNAL: always valid
        # 3=COOPERATE: valid if rich food within distance 1
        # 4=IDLE: always valid

        # Check for rich food nearby (distance <= 1)
        rich_coords = torch.nonzero(self.rich_food, as_tuple=False)  # [F, 2]
        if rich_coords.numel() > 0:
            pos = self.agent_positions.float()
            food_pos = rich_coords.float()
            dist = (pos.unsqueeze(1) - food_pos.unsqueeze(0)).abs().sum(dim=-1)  # [n, F]
            rich_nearby = (dist <= 1).any(dim=1)  # [n]
        else:
            rich_nearby = torch.zeros(n, dtype=torch.bool, device=self.device)

        interact_type_mask = torch.ones(n, 5, dtype=torch.bool, device=self.device)
        interact_type_mask[:, 3] = rich_nearby  # COOPERATE only if rich food nearby

        return {
            'move_mask': move_mask,
            'interact_type_mask': interact_type_mask,
            'direction_mask': direction_mask
        }

    def _get_all_spatial_obs(self) -> torch.Tensor:
        """
        Get spatial observations for ALL agents at once - FULLY VECTORIZED.
        Uses unfold for efficient window extraction + F.one_hot for agent IDs.
        Returns [n_agents, vision_size, vision_size, channels]
        """
        vs = self.config.vision_size
        half = vs // 2
        n = self.n_agents
        gs = self.grid_size
        pad = half

        # === STEP 1: Pad all grids ===
        padded_poor = F.pad(self.poor_food.float().unsqueeze(0).unsqueeze(0),
                           (pad, pad, pad, pad), value=0).squeeze()
        padded_rich = F.pad(self.rich_food.float().unsqueeze(0).unsqueeze(0),
                           (pad, pad, pad, pad), value=0).squeeze()
        # Add 1 to occupancy so -1 becomes 0 (for safe one_hot indexing)
        padded_occ = F.pad((self.occupancy + 1).float().unsqueeze(0).unsqueeze(0),
                          (pad, pad, pad, pad), value=0).squeeze()

        # === STEP 2: Use unfold to extract ALL windows at once ===
        # unfold extracts sliding windows: [H, W] -> [H-vs+1, W-vs+1, vs, vs]
        poor_windows = padded_poor.unfold(0, vs, 1).unfold(1, vs, 1)  # [gs, gs, vs, vs]
        rich_windows = padded_rich.unfold(0, vs, 1).unfold(1, vs, 1)
        occ_windows = padded_occ.unfold(0, vs, 1).unfold(1, vs, 1)

        # === STEP 3: Index into windows at agent positions ===
        rows = self.agent_positions[:, 0]
        cols = self.agent_positions[:, 1]

        # Extract windows for each agent: [n_agents, vs, vs]
        agent_poor = poor_windows[rows, cols]
        agent_rich = rich_windows[rows, cols]
        agent_occ = occ_windows[rows, cols]  # values are (agent_id + 1), 0 means empty

        # Restore to original agent_id (-1 for empty, but we'll use 0+ for indexing)
        agent_occ_ids = agent_occ.long() - 1  # [n_agents, vs, vs], -1 = empty

        # === STEP 4: Build observation channels ===
        # Channels: 0=Empty, 1=Poor food, 2=Rich food, 3=Agent health, 4+=One-hot agent ID
        n_channels = 4 + self.n_agents
        obs = torch.zeros((n, vs, vs, n_channels), device=self.device)

        # Channel 1: Poor food
        obs[:, :, :, 1] = agent_poor

        # Channel 2: Rich food
        obs[:, :, :, 2] = agent_rich

        # === STEP 5: One-hot Agent ID channels (4 to 4+n_agents) - VECTORIZED ===
        # F.one_hot on clamped IDs, then mask out empty cells
        agent_present = (agent_occ_ids >= 0)
        safe_ids_for_onehot = agent_occ_ids.clamp(min=0)
        one_hot = F.one_hot(safe_ids_for_onehot, num_classes=self.n_agents).float()
        # Zero out one-hot for empty cells
        one_hot = one_hot * agent_present.unsqueeze(-1).float()
        obs[:, :, :, 4:4+self.n_agents] = one_hot

        # Channel 3: Agent health (normalized) - only where agent present
        hp_values = self.agent_hp[safe_ids_for_onehot] / self.config.max_hp
        obs[:, :, :, 3] = torch.where(agent_present, hp_values, torch.zeros_like(hp_values))

        # Channel 0: Empty (not food and not agent)
        obs[:, :, :, 0] = (~(agent_poor.bool() | agent_rich.bool() | agent_present)).float()

        return obs


# Wrapper for backward compatibility with dict-based API
class GridWorldCompat(GridWorld):
    """Wrapper providing backward-compatible dict-based API."""

    def step(self, actions: Dict[int, Tuple[int, int]]) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, dict]
    ]:
        """Dict-based step for backward compatibility."""
        # Convert dict to tensors
        move_actions = torch.tensor(
            [actions[i][0] for i in range(self.n_agents)],
            device=self.device, dtype=torch.long
        )
        interact_actions = torch.tensor(
            [actions[i][1] for i in range(self.n_agents)],
            device=self.device, dtype=torch.long
        )

        # Call vectorized step
        obs, rewards, dones, infos = super().step(move_actions, interact_actions)

        # Convert outputs to dicts
        obs_dict = {}
        for i in range(self.n_agents):
            obs_dict[i] = {
                'spatial': obs['spatial'][i],
                'ledger': obs['ledger'][i],
                'signals': obs['signals'][i],
                'self_hp': obs['self_hp'][i],
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
                'spatial': obs['spatial'][i],
                'ledger': obs['ledger'][i],
                'signals': obs['signals'][i],
                'self_hp': obs['self_hp'][i],
                'agent_id': obs['agent_id'][i]
            }

        return obs_dict
