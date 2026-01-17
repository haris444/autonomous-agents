"""
GridWorld Environment for Multi-Agent RL.

Features:
- Single occupancy grid (hard collision)
- Poor food (solo) and Rich food (requires 2+ adjacent agents)
- HP decay per tick
- Heavyweight movement resolution
- Combat system (dmg = attacker_HP / 2)
"""
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np

from config import Config
from ledger import Ledger


# Action indices
MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_STAY = 0, 1, 2, 3, 4
INTERACT_ATTACK, INTERACT_GIVE, INTERACT_SIGNAL, INTERACT_IDLE = 0, 1, 2, 3


class GridWorld:
    """
    Multi-agent grid environment with social dynamics.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.grid_size = config.grid_size
        self.n_agents = config.n_agents

        # Ledger for social memory
        self.ledger = Ledger(config.n_agents, device)

        # State tensors (initialized in reset)
        self.agent_positions: torch.Tensor = None  # [n_agents, 2] (row, col)
        self.agent_hp: torch.Tensor = None         # [n_agents]
        self.agent_alive: torch.Tensor = None      # [n_agents] bool
        self.poor_food: torch.Tensor = None        # [grid_size, grid_size] bool
        self.rich_food: torch.Tensor = None        # [grid_size, grid_size] bool
        self.signals: torch.Tensor = None          # [n_agents] bool

        self.step_count = 0

    def reset(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Reset environment and return initial observations for all agents."""
        self.ledger.reset()
        self.step_count = 0

        # Initialize agent positions (random, no overlap)
        positions = []
        occupied = set()
        for _ in range(self.n_agents):
            while True:
                pos = (random.randint(0, self.grid_size - 1),
                       random.randint(0, self.grid_size - 1))
                if pos not in occupied:
                    occupied.add(pos)
                    positions.append(pos)
                    break
        self.agent_positions = torch.tensor(positions, device=self.device, dtype=torch.long)

        # Initialize HP and alive status
        self.agent_hp = torch.full((self.n_agents,), self.config.max_hp, device=self.device)
        self.agent_alive = torch.ones(self.n_agents, device=self.device, dtype=torch.bool)

        # Initialize food grids
        self.poor_food = torch.zeros((self.grid_size, self.grid_size),
                                      device=self.device, dtype=torch.bool)
        self.rich_food = torch.zeros((self.grid_size, self.grid_size),
                                      device=self.device, dtype=torch.bool)
        self._spawn_food()

        # Initialize signals (none active)
        self.signals = torch.zeros(self.n_agents, device=self.device, dtype=torch.bool)

        return self._get_all_observations()

    def step(self, actions: Dict[int, Tuple[int, int]]) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],  # observations
        Dict[int, float],                     # rewards
        Dict[int, bool],                      # dones
        Dict[int, dict]                       # infos
    ]:
        """
        Execute one environment step.

        Args:
            actions: Dict mapping agent_id -> (move_action, interact_action)

        Returns:
            observations, rewards, dones, infos
        """
        self.step_count += 1
        rewards = {i: 0.0 for i in range(self.n_agents)}

        # Clear signals from last step
        self.signals.zero_()

        # 1. Resolve movement (Heavyweight Rule)
        move_actions = {i: actions[i][0] for i in range(self.n_agents) if self.agent_alive[i]}
        self._resolve_movement(move_actions)

        # 2. Apply HP decay
        self._apply_hp_decay()

        # 3. Process interactions (attack, give food, signal)
        interact_actions = {i: actions[i][1] for i in range(self.n_agents) if self.agent_alive[i]}
        interaction_rewards = self._resolve_interactions(interact_actions)
        for i, r in interaction_rewards.items():
            rewards[i] += r

        # 4. Process food eating (agents on food cells)
        food_rewards = self._process_food_eating()
        for i, r in food_rewards.items():
            rewards[i] += r

        # 5. Spawn new food
        self._spawn_food()

        # 6. Check deaths (HP <= 0)
        self._check_deaths()

        # Build outputs
        observations = self._get_all_observations()
        dones = {i: not self.agent_alive[i].item() for i in range(self.n_agents)}

        # Episode ends if all agents dead or max steps reached
        all_dead = not self.agent_alive.any().item()
        max_steps = self.step_count >= self.config.max_steps_per_episode
        if all_dead or max_steps:
            dones = {i: True for i in range(self.n_agents)}

        infos = {i: {} for i in range(self.n_agents)}

        return observations, rewards, dones, infos

    def _resolve_movement(self, move_actions: Dict[int, int]) -> None:
        """Apply Heavyweight Rule for movement resolution."""
        # Calculate intended destinations
        direction_deltas = {
            MOVE_UP: (-1, 0),
            MOVE_DOWN: (1, 0),
            MOVE_LEFT: (0, -1),
            MOVE_RIGHT: (0, 1),
            MOVE_STAY: (0, 0)
        }

        intended_dest: Dict[int, Tuple[int, int]] = {}
        for agent_id, move in move_actions.items():
            if not self.agent_alive[agent_id]:
                continue

            current = tuple(self.agent_positions[agent_id].tolist())
            delta = direction_deltas[move]
            new_row = current[0] + delta[0]
            new_col = current[1] + delta[1]

            # Clamp to grid bounds
            new_row = max(0, min(self.grid_size - 1, new_row))
            new_col = max(0, min(self.grid_size - 1, new_col))

            intended_dest[agent_id] = (new_row, new_col)

        # Group by destination
        dest_to_agents: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for agent_id, dest in intended_dest.items():
            dest_to_agents[dest].append(agent_id)

        # Resolve conflicts
        for dest, agents in dest_to_agents.items():
            # Check if destination has a stationary agent not in this group
            stationary_agent = self._get_agent_at(dest)
            if stationary_agent is not None and stationary_agent not in agents:
                # Cell blocked by stationary agent - all bounce
                continue

            if len(agents) == 1:
                # Uncontested move
                self.agent_positions[agents[0]] = torch.tensor(dest, device=self.device)
            else:
                # Contested - highest HP wins
                hps = [(self.agent_hp[a].item(), a) for a in agents]
                max_hp = max(hp for hp, _ in hps)
                winners = [a for hp, a in hps if hp == max_hp]

                # Random tiebreaker
                winner = random.choice(winners)
                self.agent_positions[winner] = torch.tensor(dest, device=self.device)
                # Losers stay in place (bounce)

    def _resolve_interactions(self, interact_actions: Dict[int, int]) -> Dict[int, float]:
        """Process attack, give food, and signal actions."""
        rewards = {}

        for agent_id, action in interact_actions.items():
            if not self.agent_alive[agent_id]:
                continue

            if action == INTERACT_ATTACK:
                # Attack adjacent agents
                r = self._execute_attack(agent_id)
                rewards[agent_id] = rewards.get(agent_id, 0.0) + r

            elif action == INTERACT_GIVE:
                # Give food to adjacent agent
                r = self._execute_give_food(agent_id)
                rewards[agent_id] = rewards.get(agent_id, 0.0) + r

            elif action == INTERACT_SIGNAL:
                # Broadcast signal
                self.signals[agent_id] = True

            # INTERACT_IDLE does nothing

        return rewards

    def _execute_attack(self, attacker: int) -> float:
        """Attack all adjacent agents. Returns reward for attacker."""
        if not self.agent_alive[attacker]:
            return 0.0

        attacker_pos = tuple(self.agent_positions[attacker].tolist())
        total_damage = 0.0

        # Find adjacent agents
        for target in range(self.n_agents):
            if target == attacker or not self.agent_alive[target]:
                continue

            target_pos = tuple(self.agent_positions[target].tolist())
            if self._is_adjacent(attacker_pos, target_pos):
                # Deal damage
                damage = self.agent_hp[attacker].item() / 2
                self.agent_hp[target] -= damage
                total_damage += damage

                # Update ledger
                self.ledger.record_damage(attacker, target, damage)

                # Check for defense (other agents attacking the attacker to defend target)
                # This is recorded when another agent attacks the attacker

        return total_damage  # Attacker gets reward equal to damage dealt

    def _execute_give_food(self, giver: int) -> float:
        """Give food to an adjacent agent. Returns cost for giver."""
        if not self.agent_alive[giver]:
            return 0.0

        giver_pos = tuple(self.agent_positions[giver].tolist())
        food_value = self.config.poor_food_value

        # Find first adjacent alive agent
        for target in range(self.n_agents):
            if target == giver or not self.agent_alive[target]:
                continue

            target_pos = tuple(self.agent_positions[target].tolist())
            if self._is_adjacent(giver_pos, target_pos):
                # Calculate opportunity cost
                hp_gap = self.config.max_hp - self.agent_hp[giver].item()
                opportunity_cost = min(food_value, hp_gap)

                # Transfer "food" (HP) to target
                self.agent_hp[target] = torch.clamp(
                    self.agent_hp[target] + food_value,
                    max=self.config.max_hp
                )

                # Update ledger
                self.ledger.record_food_given(giver, target, food_value)

                return -opportunity_cost  # Negative reward (cost)

        return 0.0  # No adjacent agent to give to

    def _process_food_eating(self) -> Dict[int, float]:
        """Process agents eating food on their cells."""
        rewards = {}

        for agent_id in range(self.n_agents):
            if not self.agent_alive[agent_id]:
                continue

            pos = tuple(self.agent_positions[agent_id].tolist())
            row, col = pos

            # Check poor food
            if self.poor_food[row, col]:
                self.agent_hp[agent_id] = torch.clamp(
                    self.agent_hp[agent_id] + self.config.poor_food_value,
                    max=self.config.max_hp
                )
                self.poor_food[row, col] = False
                rewards[agent_id] = rewards.get(agent_id, 0.0) + self.config.r_small

            # Check rich food (requires 2+ adjacent agents)
            if self.rich_food[row, col]:
                cooperators = self._get_rich_food_cooperators(pos)
                if cooperators is not None and len(cooperators) >= 2:
                    # All cooperators get the reward
                    for coop_id in cooperators:
                        self.agent_hp[coop_id] = torch.clamp(
                            self.agent_hp[coop_id] + self.config.rich_food_value,
                            max=self.config.max_hp
                        )
                        rewards[coop_id] = rewards.get(coop_id, 0.0) + self.config.r_large

                    # Record cooperation in ledger (all pairs)
                    for i, a in enumerate(cooperators):
                        for b in cooperators[i + 1:]:
                            self.ledger.record_cooperation(a, b)

                    self.rich_food[row, col] = False

        return rewards

    def _get_rich_food_cooperators(self, food_pos: Tuple[int, int]) -> Optional[List[int]]:
        """Get list of agents adjacent to rich food cell."""
        adjacent_agents = []

        for agent_id in range(self.n_agents):
            if not self.agent_alive[agent_id]:
                continue

            agent_pos = tuple(self.agent_positions[agent_id].tolist())
            if self._is_adjacent(food_pos, agent_pos) or food_pos == agent_pos:
                adjacent_agents.append(agent_id)

        return adjacent_agents if len(adjacent_agents) >= 2 else None

    def _apply_hp_decay(self) -> None:
        """Apply HP decay to all alive agents."""
        decay = self.config.max_hp * self.config.hp_decay_rate
        for agent_id in range(self.n_agents):
            if self.agent_alive[agent_id]:
                self.agent_hp[agent_id] -= decay

    def _check_deaths(self) -> None:
        """Mark agents with HP <= 0 as dead."""
        self.agent_alive = self.agent_hp > 0

    def _spawn_food(self) -> None:
        """Spawn new food on empty cells based on spawn rates."""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Skip if cell has agent or food
                if self._get_agent_at((row, col)) is not None:
                    continue
                if self.poor_food[row, col] or self.rich_food[row, col]:
                    continue

                # Random spawn
                if random.random() < self.config.poor_food_spawn_rate:
                    self.poor_food[row, col] = True
                elif random.random() < self.config.rich_food_spawn_rate:
                    self.rich_food[row, col] = True

    def _get_agent_at(self, pos: Tuple[int, int]) -> Optional[int]:
        """Get agent ID at position, or None if empty."""
        for agent_id in range(self.n_agents):
            if not self.agent_alive[agent_id]:
                continue
            if tuple(self.agent_positions[agent_id].tolist()) == pos:
                return agent_id
        return None

    def _is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent (4-connected)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _get_all_observations(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Build observations for all agents."""
        return {i: self._build_observation(i) for i in range(self.n_agents)}

    def _build_observation(self, agent_id: int) -> Dict[str, torch.Tensor]:
        """Build observation dict for a single agent."""
        return {
            'spatial': self._get_spatial_obs(agent_id),
            'ledger': self.ledger.get_normalized_tensor(),
            'signals': self.signals.float(),
            'self_hp': self.agent_hp[agent_id:agent_id + 1] / self.config.max_hp
        }

    def _get_spatial_obs(self, agent_id: int) -> torch.Tensor:
        """
        Get NxN vision grid centered on agent.

        Channels: [Empty, Food_Poor, Food_Rich, Agent_Present, Agent_Health]
        """
        vs = self.config.vision_size
        half = vs // 2

        # Initialize observation
        obs = torch.zeros((vs, vs, self.config.vision_channels), device=self.device)

        # Agent position
        agent_pos = self.agent_positions[agent_id].tolist()
        agent_row, agent_col = int(agent_pos[0]), int(agent_pos[1])

        # Fill observation grid
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                world_row = agent_row + dr
                world_col = agent_col + dc
                obs_row = dr + half
                obs_col = dc + half

                # Check bounds
                if not (0 <= world_row < self.grid_size and 0 <= world_col < self.grid_size):
                    # Out of bounds - treat as empty (or could be wall)
                    obs[obs_row, obs_col, 0] = 1.0  # Empty channel
                    continue

                # Check what's at this cell
                cell_agent = self._get_agent_at((world_row, world_col))

                if cell_agent is not None:
                    obs[obs_row, obs_col, 3] = 1.0  # Agent present
                    obs[obs_row, obs_col, 4] = self.agent_hp[cell_agent].item() / self.config.max_hp
                elif self.poor_food[world_row, world_col]:
                    obs[obs_row, obs_col, 1] = 1.0  # Poor food
                elif self.rich_food[world_row, world_col]:
                    obs[obs_row, obs_col, 2] = 1.0  # Rich food
                else:
                    obs[obs_row, obs_col, 0] = 1.0  # Empty

        return obs
