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


# Action indices
MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_STAY = 0, 1, 2, 3, 4

# Interact actions with direction: ATTACK_UP=0, ATTACK_DOWN=1, ATTACK_LEFT=2, ATTACK_RIGHT=3
# GIVE_UP=4, GIVE_DOWN=5, GIVE_LEFT=6, GIVE_RIGHT=7, SIGNAL=8, COOPERATE=9, IDLE=10
ATTACK_UP, ATTACK_DOWN, ATTACK_LEFT, ATTACK_RIGHT = 0, 1, 2, 3
GIVE_UP, GIVE_DOWN, GIVE_LEFT, GIVE_RIGHT = 4, 5, 6, 7
INTERACT_SIGNAL = 8
INTERACT_COOPERATE = 9
INTERACT_IDLE = 10

# Direction deltas for interact actions (indices 0-3 for attack, 4-7 for give)
# Maps to: UP=[-1,0], DOWN=[1,0], LEFT=[0,-1], RIGHT=[0,1]
INTERACT_DIR_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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

        # Direction deltas tensor [5, 2] for movement
        self.direction_deltas = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]],
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

        self.step_count = 0

    def _update_occupancy(self) -> None:
        """Update occupancy grid from agent positions. O(n) but simple."""
        self.occupancy.fill_(-1)
        alive_mask = self.agent_alive
        alive_indices = torch.where(alive_mask)[0]
        for idx in alive_indices:
            r, c = self.agent_positions[idx]
            self.occupancy[r, c] = idx

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment and return initial observations for all agents."""
        self.ledger.reset()
        self.step_count = 0

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

        Args:
            move_actions: [n_agents] int tensor of movement actions
            interact_actions: [n_agents] int tensor of interaction actions

        Returns:
            observations, rewards, dones, infos
        """
        self.step_count += 1
        rewards = torch.zeros(self.n_agents, device=self.device)

        # Clear signals from last step
        self.signals.zero_()

        # 1. Resolve movement (Heavyweight Rule)
        self._resolve_movement(move_actions)
        self._update_occupancy()

        # 2. Apply HP decay (vectorized)
        self._apply_hp_decay()

        # 3. Process interactions (attack, give food, signal)
        interaction_rewards = self._resolve_interactions(interact_actions)
        rewards += interaction_rewards

        # 4. Process food eating (agents on food cells)
        # Pass interact_actions so rich food requires COOPERATE action
        food_rewards = self._process_food_eating(interact_actions)
        rewards += food_rewards

        # 5. Survival reward for staying alive
        rewards += self.config.r_survival * self.agent_alive.float()

        # 5. Spawn new food
        self._spawn_food()

        # 6. Check deaths (HP <= 0) and apply death penalty
        death_rewards = self._check_deaths()
        rewards += death_rewards
        self._update_occupancy()

        # Build outputs
        observations = self._get_all_observations()
        dones = ~self.agent_alive

        # Episode ends if all agents dead or max steps reached
        all_dead = ~self.agent_alive.any()
        max_steps = self.step_count >= self.config.max_steps_per_episode
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

    def _resolve_movement(self, move_actions: torch.Tensor) -> None:
        """Apply Heavyweight Rule for movement resolution - vectorized."""
        # Only process alive agents
        alive_mask = self.agent_alive

        # Compute intended destinations for all agents
        intended = self.agent_positions + self.direction_deltas[move_actions]
        intended = intended.clamp(0, self.grid_size - 1)

        # Dead agents stay in place
        intended = torch.where(alive_mask.unsqueeze(1), intended, self.agent_positions)

        # Convert to flat indices for conflict detection
        flat_intended = intended[:, 0] * self.grid_size + intended[:, 1]
        flat_current = self.agent_positions[:, 0] * self.grid_size + self.agent_positions[:, 1]

        # Find unique destinations and count agents wanting each
        unique_dests, inverse_indices = torch.unique(flat_intended, return_inverse=True)

        # For each unique destination, find all agents wanting it
        new_positions = self.agent_positions.clone()

        for dest_idx, dest in enumerate(unique_dests):
            # Agents wanting this destination
            wanting = (inverse_indices == dest_idx) & alive_mask

            if wanting.sum() == 0:
                continue

            wanting_indices = torch.where(wanting)[0]

            # Check if destination is occupied by stationary agent not in wanting group
            dest_row = dest // self.grid_size
            dest_col = dest % self.grid_size
            occupant = self.occupancy[dest_row, dest_col].item()

            if occupant >= 0 and not wanting[occupant]:
                # Blocked by stationary agent - no one moves
                continue

            if len(wanting_indices) == 1:
                # Uncontested - agent moves
                new_positions[wanting_indices[0]] = torch.tensor(
                    [dest_row, dest_col], device=self.device
                )
            else:
                # Contested - highest HP wins
                hps = self.agent_hp[wanting_indices]
                max_hp = hps.max()
                winners = wanting_indices[hps == max_hp]

                # Random tiebreaker
                winner_idx = torch.randint(len(winners), (1,), device=self.device)
                winner = winners[winner_idx]
                new_positions[winner] = torch.tensor(
                    [dest_row, dest_col], device=self.device
                )

        self.agent_positions = new_positions

    def _resolve_interactions(self, interact_actions: torch.Tensor) -> torch.Tensor:
        """Process attack, give food, and signal actions with direction targeting."""
        rewards = torch.zeros(self.n_agents, device=self.device)

        # Signal actions (action 8)
        signal_mask = (interact_actions == INTERACT_SIGNAL) & self.agent_alive
        self.signals = signal_mask

        # Attack actions (0-3: UP, DOWN, LEFT, RIGHT)
        attack_mask = (interact_actions <= ATTACK_RIGHT) & self.agent_alive
        attack_rewards = self._resolve_attacks(interact_actions, attack_mask)
        rewards += attack_rewards

        # Give food actions (4-7: UP, DOWN, LEFT, RIGHT)
        give_mask = ((interact_actions >= GIVE_UP) & (interact_actions <= GIVE_RIGHT)) & self.agent_alive
        give_rewards = self._resolve_give_food(interact_actions, give_mask)
        rewards += give_rewards

        return rewards

    def _resolve_attacks(self, interact_actions: torch.Tensor, attack_mask: torch.Tensor) -> torch.Tensor:
        """
        Resolve attacks with direction targeting and defense tracking.

        Each attacker targets ONE cell in their chosen direction.
        Defense: if C attacks A who is attacking B, C defended B.
        Attack costs 5% of max HP regardless of whether it hits.
        """
        rewards = torch.zeros(self.n_agents, device=self.device)
        damage_matrix = torch.zeros((self.n_agents, self.n_agents), device=self.device)

        # Track who is attacking whom for defense calculation
        # attack_target[attacker_id] = target_id or -1 if no target
        attack_target = torch.full((self.n_agents,), -1, device=self.device, dtype=torch.long)

        # Attack cost: 5% of max HP
        attack_cost = self.config.max_hp * 0.05

        for attacker in torch.where(attack_mask)[0]:
            # Attacker pays 5% HP cost regardless of hit/miss
            self.agent_hp[attacker] -= attack_cost

            # Get direction from action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            direction = interact_actions[attacker].item()
            delta_row, delta_col = INTERACT_DIR_DELTAS[direction]

            # Compute target position
            attacker_pos = self.agent_positions[attacker]
            target_row = attacker_pos[0] + delta_row
            target_col = attacker_pos[1] + delta_col

            # Check bounds
            if target_row < 0 or target_row >= self.grid_size:
                continue
            if target_col < 0 or target_col >= self.grid_size:
                continue

            # Check if there's an agent at target position
            target_id = self.occupancy[target_row, target_col].item()
            if target_id == -1 or not self.agent_alive[target_id]:
                continue

            # Calculate and apply damage
            damage = self.agent_hp[attacker] / 2
            self.agent_hp[target_id] -= damage
            damage_matrix[attacker, target_id] = damage
            attack_target[attacker] = target_id

        # Update ledger with damage dealt
        self.ledger.tensor[:, :, Ledger.DAMAGE_DEALT] += damage_matrix

        # === Defense Tracking ===
        # If C attacks A, and A is attacking B, then C defended B
        for attacker_c in torch.where(attack_mask)[0]:
            target_a = attack_target[attacker_c].item()
            if target_a == -1:
                continue

            # Check if A (the person C attacked) was attacking someone B
            victim_b = attack_target[target_a].item()
            if victim_b == -1:
                continue

            # C defended B by attacking A
            damage_c_dealt = damage_matrix[attacker_c, target_a].item()
            if damage_c_dealt > 0:
                self.ledger.record_defense(attacker_c.item(), victim_b, damage_c_dealt)

        # Rewards: attacker gets 50% of damage dealt
        damage_dealt = damage_matrix.sum(dim=1)
        rewards += damage_dealt * self.config.r_attack_mult

        # Penalty for receiving damage
        damage_received = damage_matrix.sum(dim=0)
        rewards += damage_received * self.config.r_damage_taken

        return rewards

    def _resolve_give_food(self, interact_actions: torch.Tensor, give_mask: torch.Tensor) -> torch.Tensor:
        """
        Resolve food giving with direction targeting and HP transfer.

        Giver LOSES HP regardless of whether there's a valid target (costs HP to the void).
        Target GAINS HP if present.
        If giver HP < food_value, transfers all remaining HP.
        """
        rewards = torch.zeros(self.n_agents, device=self.device)
        food_value = self.config.poor_food_value

        for giver in torch.where(give_mask)[0]:
            # Calculate cost amount = min(food_value, giver's HP)
            cost_amount = min(food_value, self.agent_hp[giver].item())
            if cost_amount <= 0:
                continue

            # Giver ALWAYS pays the cost (even to void)
            self.agent_hp[giver] -= cost_amount

            # Get direction from action (4=UP, 5=DOWN, 6=LEFT, 7=RIGHT -> direction 0-3)
            direction = interact_actions[giver].item() - GIVE_UP  # Convert 4-7 to 0-3
            delta_row, delta_col = INTERACT_DIR_DELTAS[direction]

            # Compute target position
            giver_pos = self.agent_positions[giver]
            target_row = giver_pos[0] + delta_row
            target_col = giver_pos[1] + delta_col

            # Check bounds - if out of bounds, HP is lost to void
            if target_row < 0 or target_row >= self.grid_size:
                continue
            if target_col < 0 or target_col >= self.grid_size:
                continue

            # Check if there's an agent at target position - if not, HP is lost to void
            target_id = self.occupancy[target_row, target_col].item()
            if target_id == -1 or not self.agent_alive[target_id]:
                continue

            # Target receives the HP
            self.agent_hp[target_id] = (self.agent_hp[target_id] + cost_amount).clamp(max=self.config.max_hp)

            # Update ledger
            self.ledger.tensor[giver, target_id, Ledger.FOOD_GIVEN] += cost_amount

            # Reward for sharing
            rewards[giver] = self.config.r_food_share

        return rewards

    def _process_food_eating(self, interact_actions: torch.Tensor) -> torch.Tensor:
        """Process agents eating food on their cells.

        Poor food: eaten automatically by stepping on it
        Rich food: requires 2+ adjacent agents who both chose COOPERATE action
        """
        rewards = torch.zeros(self.n_agents, device=self.device)

        # Get agent positions
        rows = self.agent_positions[:, 0]
        cols = self.agent_positions[:, 1]

        # Poor food - check which alive agents are on poor food
        on_poor = self.poor_food[rows, cols] & self.agent_alive

        # Update HP for agents eating poor food
        self.agent_hp = torch.where(
            on_poor,
            (self.agent_hp + self.config.poor_food_value).clamp(max=self.config.max_hp),
            self.agent_hp
        )

        # Rewards for poor food
        rewards = rewards + on_poor.float() * self.config.r_small

        # Remove eaten poor food
        poor_food_flat = self.poor_food.view(-1)
        eat_indices = rows[on_poor] * self.grid_size + cols[on_poor]
        poor_food_flat[eat_indices] = False

        # Rich food - requires 2+ adjacent agents who BOTH chose COOPERATE
        coop_mask = (interact_actions == INTERACT_COOPERATE) & self.agent_alive
        rich_food_positions = torch.where(self.rich_food)
        for i in range(len(rich_food_positions[0])):
            food_row = rich_food_positions[0][i]
            food_col = rich_food_positions[1][i]

            # Find agents on or adjacent to this food who chose COOPERATE
            cooperators = self._get_rich_food_cooperators(food_row, food_col, coop_mask)

            if cooperators is not None and len(cooperators) >= 2:
                # All cooperators get the reward
                for coop_id in cooperators:
                    self.agent_hp[coop_id] = (
                        self.agent_hp[coop_id] + self.config.rich_food_value
                    ).clamp(max=self.config.max_hp)
                    rewards[coop_id] += self.config.r_large

                # Record cooperation in ledger (all pairs)
                for i_idx, a in enumerate(cooperators):
                    for b in cooperators[i_idx + 1:]:
                        self.ledger.record_cooperation(a.item(), b.item())

                # Remove the food
                self.rich_food[food_row, food_col] = False

        return rewards

    def _get_rich_food_cooperators(self, food_row: torch.Tensor, food_col: torch.Tensor,
                                    coop_mask: torch.Tensor) -> torch.Tensor:
        """Get agents on or adjacent to rich food cell who chose COOPERATE action."""
        food_pos = torch.tensor([food_row, food_col], device=self.device).float()

        # Distance from each agent to food
        agent_pos = self.agent_positions.float()
        dist = (agent_pos - food_pos).abs().sum(dim=1)

        # On or adjacent = distance <= 1, AND chose COOPERATE
        near = (dist <= 1) & coop_mask

        cooperators = torch.where(near)[0]
        return cooperators if len(cooperators) >= 2 else None

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

        # Random spawn probabilities
        rand = torch.rand((self.grid_size, self.grid_size), device=self.device)

        # Spawn poor food
        spawn_poor = empty & (rand < self.config.poor_food_spawn_rate)
        self.poor_food = self.poor_food | spawn_poor

        # Spawn rich food (only where poor didn't spawn)
        rand2 = torch.rand((self.grid_size, self.grid_size), device=self.device)
        spawn_rich = empty & ~spawn_poor & (rand2 < self.config.rich_food_spawn_rate)
        self.rich_food = self.rich_food | spawn_rich

    def _get_all_observations(self) -> Dict[str, torch.Tensor]:
        """Build batched observations for all agents."""
        return {
            'spatial': self._get_all_spatial_obs(),
            'ledger': self.ledger.get_normalized_tensor().unsqueeze(0).expand(self.n_agents, -1, -1, -1),
            'signals': self.signals.float().unsqueeze(0).expand(self.n_agents, -1),
            'self_hp': (self.agent_hp / self.config.max_hp).unsqueeze(1)
        }

    def _get_all_spatial_obs(self) -> torch.Tensor:
        """
        Get spatial observations for ALL agents at once.
        Returns [n_agents, vision_size, vision_size, channels]
        """
        vs = self.config.vision_size
        half = vs // 2
        n = self.n_agents

        # Pad grids for boundary handling
        pad = half

        # Create padded versions of grids
        padded_poor = F.pad(self.poor_food.float().unsqueeze(0).unsqueeze(0),
                           (pad, pad, pad, pad), value=0).squeeze()
        padded_rich = F.pad(self.rich_food.float().unsqueeze(0).unsqueeze(0),
                           (pad, pad, pad, pad), value=0).squeeze()
        padded_occ = F.pad(self.occupancy.float().unsqueeze(0).unsqueeze(0),
                          (pad, pad, pad, pad), value=-1).squeeze()

        # Prepare output tensor (5 + n_agents channels: Empty, PoorFood, RichFood, AgentPresent, AgentHP, + one-hot AgentID)
        n_channels = 5 + self.n_agents
        obs = torch.zeros((n, vs, vs, n_channels), device=self.device)

        # Agent positions offset by padding
        positions = self.agent_positions + pad

        # Extract windows for each agent
        for i in range(n):
            r, c = positions[i]
            r_start = r - half
            r_end = r_start + vs
            c_start = c - half
            c_end = c_start + vs

            window_poor = padded_poor[r_start:r_end, c_start:c_end]
            window_rich = padded_rich[r_start:r_end, c_start:c_end]
            window_occ = padded_occ[r_start:r_end, c_start:c_end]

            # Channel 1: Poor food
            obs[i, :, :, 1] = window_poor

            # Channel 2: Rich food
            obs[i, :, :, 2] = window_rich

            # Channel 3: Agent present
            agent_present = (window_occ >= 0)
            obs[i, :, :, 3] = agent_present.float()

            # Channel 4: Agent health (normalized)
            agent_ids = window_occ.long().clamp(min=0)
            hp_values = self.agent_hp[agent_ids] / self.config.max_hp
            obs[i, :, :, 4] = torch.where(agent_present, hp_values, torch.zeros_like(hp_values))

            # Channels 5 to 5+n_agents: One-hot Agent ID
            # For each position, set channel 5+agent_id to 1 if that agent is there
            for agent_id in range(self.n_agents):
                is_this_agent = (window_occ == agent_id)
                obs[i, :, :, 5 + agent_id] = is_this_agent.float()

            # Channel 0: Empty (not food and not agent)
            obs[i, :, :, 0] = (~(window_poor.bool() | window_rich.bool() | agent_present)).float()

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
                'self_hp': obs['self_hp'][i]
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
                'self_hp': obs['self_hp'][i]
            }

        return obs_dict
