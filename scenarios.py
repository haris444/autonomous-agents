"""
Scenario System for Curriculum Learning.

Modular abstraction where scenarios define training setup and the curriculum sequences them.
Each scenario configures:
- Number of active agents
- Clone mode (share networks)
- Food spawning behavior
- Agent positioning
- Social history injection (for social pretraining)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING
import random

import torch

if TYPE_CHECKING:
    from environment import GridWorld


@dataclass
class ScenarioConfig:
    """Configuration returned by a scenario."""
    n_active_agents: int
    clone_mode: bool = False          # Share networks between agents
    inject_social: bool = False       # Pre-inject social histories
    partner_mode: str = "learning"    # "learning" = normal, "always_coop" = scripted helper


class Scenario(ABC):
    """Abstract base for training scenarios."""

    @abstractmethod
    def setup(self, env: 'GridWorld') -> None:
        """Called after env.reset() to configure the scenario."""
        pass

    @abstractmethod
    def get_config(self) -> ScenarioConfig:
        """Return scenario configuration."""
        pass

    @abstractmethod
    def spawn_food(self, env: 'GridWorld') -> None:
        """Spawn food for this scenario (called on reset)."""
        pass

    @abstractmethod
    def respawn_food(self, env: 'GridWorld') -> None:
        """Respawn food during episode (called when food is eaten)."""
        pass


# =============================================================================
# SOLO PHASES (1-5): Single agent finds food at increasing distances
# =============================================================================

class SoloFoodScenario(Scenario):
    """Phases 1-5: Single agent finds food at increasing distances."""

    def __init__(self, distance: int, cardinal_only: bool = False):
        self.distance = distance
        self.cardinal_only = cardinal_only

    def get_config(self) -> ScenarioConfig:
        return ScenarioConfig(n_active_agents=1, clone_mode=False)

    def setup(self, env: 'GridWorld') -> None:
        """Position agent in center, deactivate others."""
        center = env.grid_size // 2
        env.agent_positions[0] = torch.tensor([center, center], device=env.device)

        # Deactivate all agents except agent 0
        env.agent_alive[1:] = False
        env.agent_hp[1:] = 0

        # Randomize other agent positions (they're dead anyway)
        if env.n_agents > 1:
            other_positions = torch.randperm(env.grid_size * env.grid_size, device=env.device)[:env.n_agents - 1]
            rows = other_positions // env.grid_size
            cols = other_positions % env.grid_size
            env.agent_positions[1:] = torch.stack([rows, cols], dim=1)

        env._update_occupancy()

    def spawn_food(self, env: 'GridWorld') -> None:
        """Spawn one poor food at the target distance from agent 0."""
        r, c = env.agent_positions[0][0].item(), env.agent_positions[0][1].item()

        # Bound distance by grid size
        max_dist = env.grid_size - 1
        distance = min(self.distance, max_dist)

        valid_positions = self._get_positions_at_distance(env, r, c, distance, self.cardinal_only)

        if valid_positions:
            idx = torch.randint(len(valid_positions), (1,)).item()
            nr, nc = valid_positions[idx]
            env.poor_food[nr, nc] = True

    def respawn_food(self, env: 'GridWorld') -> None:
        """Respawn food if none exists."""
        if env.poor_food.any():
            return  # Food still exists

        self.spawn_food(env)

    def _get_positions_at_distance(
        self, env: 'GridWorld', r: int, c: int, distance: int, cardinal_only: bool
    ) -> List[Tuple[int, int]]:
        """Get all valid grid positions at exact Manhattan distance."""
        positions = []

        if cardinal_only:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr * distance, c + dc * distance
                if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size:
                    positions.append((nr, nc))
        else:
            for dr in range(-distance, distance + 1):
                dc_abs = distance - abs(dr)
                for dc in ([-dc_abs, dc_abs] if dc_abs > 0 else [0]):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size:
                        positions.append((nr, nc))

        return positions


# =============================================================================
# COOPERATION PHASES (6-8): Two agents cooperate on rich food
# =============================================================================

class CoopFoodScenario(Scenario):
    """Phases 6-8: Two agents cooperate on rich food.

    Uses a scripted partner (agent 1) that always coops when conditions are right,
    so agent 0 can learn to cooperate with a reliable partner.
    """

    def __init__(self, distance: int):
        self.distance = distance

    def get_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            n_active_agents=2,
            clone_mode=False,           # Independent networks
            partner_mode="always_coop"  # Agent 1 always coops when near food + agent 0
        )

    def setup(self, env: 'GridWorld') -> None:
        """Position 2 agents near where rich food will spawn."""
        from ledger import Ledger

        # Bound distance by grid_size - 1
        max_dist = env.grid_size - 1
        distance = min(self.distance, max_dist)

        center = env.grid_size // 2

        # Find valid positions at the specified distance from center (where food spawns)
        valid_positions = self._get_positions_at_distance(env, center, center, distance)

        if len(valid_positions) >= 2:
            perm = torch.randperm(len(valid_positions))[:2]
            r0, c0 = valid_positions[perm[0].item()]
            r1, c1 = valid_positions[perm[1].item()]
            env.agent_positions[0] = torch.tensor([r0, c0], device=env.device)
            env.agent_positions[1] = torch.tensor([r1, c1], device=env.device)
        else:
            # Fallback: place agents adjacent to center
            env.agent_positions[0] = torch.tensor([center - 1, center], device=env.device)
            env.agent_positions[1] = torch.tensor([center, center - 1], device=env.device)

        # Activate agents 0 and 1, deactivate the rest
        env.agent_alive[:2] = True
        env.agent_hp[:2] = env.config.max_hp
        env.agent_alive[2:] = False
        env.agent_hp[2:] = 0

        # Randomize other agent positions
        if env.n_agents > 2:
            other_positions = torch.randperm(env.grid_size * env.grid_size, device=env.device)[:env.n_agents - 2]
            rows = other_positions // env.grid_size
            cols = other_positions % env.grid_size
            env.agent_positions[2:] = torch.stack([rows, cols], dim=1)

        env._update_occupancy()

        # Inject ally history: make agent 1 appear as a trusted ally to agent 0
        # Agent 1 has "given food" and "defended" agent 0 in the past
        env.ledger.tensor[1, 0, Ledger.FOOD_GIVEN] = 50.0     # Agent 1 gave food to agent 0
        env.ledger.tensor[1, 0, Ledger.DEFENSE_SCORE] = 30.0  # Agent 1 defended agent 0
        env.ledger.tensor[1, 0, Ledger.COOP_COUNT] = 5.0      # They've cooperated before

    def spawn_food(self, env: 'GridWorld') -> None:
        """Spawn one rich food in center of grid."""
        center = env.grid_size // 2
        env.rich_food[center, center] = True

    def respawn_food(self, env: 'GridWorld') -> None:
        """Respawn rich food reachable by both agents."""
        if env.rich_food.any():
            return

        # Bound distance
        max_dist = env.grid_size - 1
        distance = min(self.distance, max_dist)

        # Get positions of both agents
        r0, c0 = env.agent_positions[0][0].item(), env.agent_positions[0][1].item()
        r1, c1 = env.agent_positions[1][0].item(), env.agent_positions[1][1].item()

        # Find center point between agents
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2

        # Find positions within 'distance' of BOTH agents
        valid_positions = []
        for dr in range(-distance * 2, distance * 2 + 1):
            for dc in range(-distance * 2, distance * 2 + 1):
                nr, nc = center_r + dr, center_c + dc
                if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size:
                    dist0 = abs(nr - r0) + abs(nc - c0)
                    dist1 = abs(nr - r1) + abs(nc - c1)
                    if dist0 <= distance and dist1 <= distance:
                        if env.occupancy[nr, nc] == -1:
                            valid_positions.append((nr, nc))

        if valid_positions:
            idx = torch.randint(len(valid_positions), (1,)).item()
            nr, nc = valid_positions[idx]
            env.rich_food[nr, nc] = True
        else:
            # Fallback: spawn between agents
            if 0 <= center_r < env.grid_size and 0 <= center_c < env.grid_size:
                if env.occupancy[center_r, center_c] == -1:
                    env.rich_food[center_r, center_c] = True

    def _get_positions_at_distance(
        self, env: 'GridWorld', r: int, c: int, distance: int
    ) -> List[Tuple[int, int]]:
        """Get all valid grid positions at exact Manhattan distance."""
        positions = []
        for dr in range(-distance, distance + 1):
            dc_abs = distance - abs(dr)
            for dc in ([-dc_abs, dc_abs] if dc_abs > 0 else [0]):
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size:
                    positions.append((nr, nc))
        return positions


class CoopFoodLowHPScenario(CoopFoodScenario):
    """Phase 8.5: Same as CoopFoodScenario but agents start at 50% HP.

    This teaches agents to cooperate even when wounded, preventing the
    "desperation aggression" behavior where low-HP agents attack allies.
    """

    def __init__(self, distance: int, hp_fraction: float = 0.5):
        super().__init__(distance)
        self.hp_fraction = hp_fraction

    def setup(self, env: 'GridWorld') -> None:
        """Position 2 agents near rich food, but at reduced HP."""
        # Call parent setup first
        super().setup(env)

        # Reduce HP for active agents
        for i in range(2):
            env.agent_hp[i] = env.config.max_hp * self.hp_fraction


# =============================================================================
# SOCIAL PHASES (9+): Multiple agents with social dynamics
# =============================================================================

class SocialScenario(Scenario):
    """Phases 9+: Multiple agents with social dynamics."""

    def __init__(
        self,
        n_agents: int,
        inject_histories: bool = True,
        scripted_partners: bool = False,
        clone_weights: bool = False,
        coherent_histories: bool = False,
        neutral_prob: float = 0.2,
        always_one_ally: bool = False,  # Ensure agent 1 is always an ally
    ):
        self.n_agents = n_agents
        self.inject_histories = inject_histories
        self.scripted_partners = scripted_partners
        self.clone_weights = clone_weights
        self.coherent_histories = coherent_histories
        self.neutral_prob = neutral_prob
        self.always_one_ally = always_one_ally

    def get_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            n_active_agents=self.n_agents,
            clone_mode=self.clone_weights,
            inject_social=self.inject_histories,
            # Partner mode is "scripted" when scripted_partners=True, actual behavior set per-episode
            partner_mode="scripted" if self.scripted_partners else "learning"
        )

    def setup(self, env: 'GridWorld') -> None:
        """Position agents in interaction range, optionally inject histories."""
        self._position_agents(env)

        if self.scripted_partners:
            if self.n_agents == 2 and self.neutral_prob > 0:
                # 2-agent mode: partner is either friend (scripted ally) or neutral (learning)
                is_neutral = random.random() < self.neutral_prob
                env.partner_relationship = 'neutral' if is_neutral else 'ally'
                if env.partner_relationship == 'ally':
                    self._inject_ally_histories(env)
                # For neutral: no history injected (stranger), learned policy used
            elif self.always_one_ally and self.n_agents >= 3:
                # Mixed scenario: agent 1 is always ally, agent 2+ can be neutral
                has_neutral = random.random() < self.neutral_prob
                env.partner_relationship = 'mixed' if has_neutral else 'ally'
                self._inject_mixed_histories(env, has_neutral)
            else:
                # Original behavior: all partners are ally OR all are enemy
                env.partner_relationship = 'enemy' if random.random() < self.neutral_prob else 'ally'
                if env.partner_relationship == 'ally':
                    self._inject_ally_histories(env)
                else:
                    self._inject_enemy_histories(env)
        elif self.inject_histories:
            if self.always_one_ally and self.n_agents >= 3:
                # Learning mode with guaranteed ally + possible neutral
                has_neutral = random.random() < self.neutral_prob
                env.partner_relationship = 'mixed' if has_neutral else 'ally'
                self._inject_mixed_histories(env, has_neutral)
            elif self.coherent_histories:
                self._inject_coherent_histories(env)
            else:
                self._inject_random_histories(env)

    def _position_agents(self, env: 'GridWorld') -> None:
        """Place agents close enough to interact."""
        center = env.grid_size // 2

        # Position agents in a cluster around center
        # Use a spiral pattern to place agents adjacent to each other
        positions = self._generate_cluster_positions(center, center, self.n_agents, env.grid_size)

        for i in range(min(self.n_agents, env.n_agents)):
            r, c = positions[i]
            env.agent_positions[i] = torch.tensor([r, c], device=env.device)
            env.agent_alive[i] = True
            env.agent_hp[i] = env.config.max_hp

        # Deactivate remaining agents
        for i in range(self.n_agents, env.n_agents):
            env.agent_alive[i] = False
            env.agent_hp[i] = 0

        env._update_occupancy()

    def _generate_cluster_positions(
        self, center_r: int, center_c: int, n: int, grid_size: int
    ) -> List[Tuple[int, int]]:
        """Generate n positions in a cluster around center."""
        positions = [(center_r, center_c)]

        # Spiral outward from center
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        r, c = center_r, center_c
        dir_idx = 0
        steps_in_dir = 1
        steps_taken = 0
        turns = 0

        while len(positions) < n:
            dr, dc = directions[dir_idx]
            r, c = r + dr, c + dc
            steps_taken += 1

            if 0 <= r < grid_size and 0 <= c < grid_size:
                if (r, c) not in positions:
                    positions.append((r, c))

            if steps_taken >= steps_in_dir:
                steps_taken = 0
                dir_idx = (dir_idx + 1) % 4
                turns += 1
                if turns % 2 == 0:
                    steps_in_dir += 1

            # Safety limit
            if len(positions) + steps_in_dir * 4 > grid_size * grid_size:
                break

        return positions[:n]

    def _inject_ally_histories(self, env: 'GridWorld') -> None:
        """Inject ally histories matching scripted partner behavior.

        Scripted partners (agent 1+) always cooperate, so inject histories
        showing they have been helpful to agent 0. This ensures the signal
        matches actual behavior during the episode.
        """
        from ledger import Ledger

        # Scripted partners (1, 2, ...) have ally history with agent 0
        for partner_id in range(1, self.n_agents):
            # Partner has helped agent 0 (matches always_coop behavior)
            env.ledger.tensor[partner_id, 0, Ledger.FOOD_GIVEN] = random.uniform(40, 60)
            env.ledger.tensor[partner_id, 0, Ledger.COOP_COUNT] = random.uniform(3, 7)
            env.ledger.tensor[partner_id, 0, Ledger.DEFENSE_SCORE] = random.uniform(20, 40)
            # No damage - they're allies
            env.ledger.tensor[partner_id, 0, Ledger.DAMAGE_DEALT] = 0.0

            # Agent 0 has also been cooperative with partner (mutual relationship)
            env.ledger.tensor[0, partner_id, Ledger.FOOD_GIVEN] = random.uniform(20, 40)
            env.ledger.tensor[0, partner_id, Ledger.COOP_COUNT] = random.uniform(3, 7)
            env.ledger.tensor[0, partner_id, Ledger.DEFENSE_SCORE] = random.uniform(10, 30)
            env.ledger.tensor[0, partner_id, Ledger.DAMAGE_DEALT] = 0.0

        # Partners also have ally histories with each other
        for i in range(1, self.n_agents):
            for j in range(1, self.n_agents):
                if i != j:
                    env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = random.uniform(30, 50)
                    env.ledger.tensor[i, j, Ledger.COOP_COUNT] = random.uniform(2, 5)
                    env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = random.uniform(15, 35)
                    env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = 0.0

    def _inject_enemy_histories(self, env: 'GridWorld') -> None:
        """Inject enemy histories matching scripted enemy behavior.

        Scripted enemies (agent 1+) always attack, so inject histories
        showing they have been hostile to agent 0. This ensures the signal
        matches actual behavior during the episode.
        """
        from ledger import Ledger

        # Scripted enemies (1, 2, ...) have enemy history with agent 0
        for enemy_id in range(1, self.n_agents):
            # Enemy has attacked agent 0 (matches always_attack behavior)
            env.ledger.tensor[enemy_id, 0, Ledger.DAMAGE_DEALT] = random.uniform(40, 70)
            # Minimal positive signals - former acquaintance turned hostile
            env.ledger.tensor[enemy_id, 0, Ledger.FOOD_GIVEN] = 0.0
            env.ledger.tensor[enemy_id, 0, Ledger.COOP_COUNT] = random.uniform(0, 1)
            env.ledger.tensor[enemy_id, 0, Ledger.DEFENSE_SCORE] = 0.0

            # Agent 0 has also fought with enemy (mutual hostility)
            env.ledger.tensor[0, enemy_id, Ledger.DAMAGE_DEALT] = random.uniform(20, 50)
            env.ledger.tensor[0, enemy_id, Ledger.FOOD_GIVEN] = 0.0
            env.ledger.tensor[0, enemy_id, Ledger.COOP_COUNT] = random.uniform(0, 1)
            env.ledger.tensor[0, enemy_id, Ledger.DEFENSE_SCORE] = 0.0

    def _inject_mixed_histories(self, env: 'GridWorld', has_neutral: bool) -> None:
        """Inject mixed histories: agent 1 is always ally, agent 2 can be neutral.

        This teaches agent 0 to discriminate between friends and strangers
        when both are present simultaneously.
        """
        from ledger import Ledger

        # Agent 1 is ALWAYS an ally
        env.ledger.tensor[1, 0, Ledger.FOOD_GIVEN] = random.uniform(40, 60)
        env.ledger.tensor[1, 0, Ledger.COOP_COUNT] = random.uniform(3, 7)
        env.ledger.tensor[1, 0, Ledger.DEFENSE_SCORE] = random.uniform(20, 40)
        env.ledger.tensor[1, 0, Ledger.DAMAGE_DEALT] = 0.0

        env.ledger.tensor[0, 1, Ledger.FOOD_GIVEN] = random.uniform(20, 40)
        env.ledger.tensor[0, 1, Ledger.COOP_COUNT] = random.uniform(3, 7)
        env.ledger.tensor[0, 1, Ledger.DEFENSE_SCORE] = random.uniform(10, 30)
        env.ledger.tensor[0, 1, Ledger.DAMAGE_DEALT] = 0.0

        # Agent 2+ depends on has_neutral flag
        for partner_id in range(2, self.n_agents):
            if has_neutral:
                # Neutral: no history at all (stranger)
                env.ledger.tensor[partner_id, 0, :] = 0.0
                env.ledger.tensor[0, partner_id, :] = 0.0
            else:
                # Ally history
                env.ledger.tensor[partner_id, 0, Ledger.FOOD_GIVEN] = random.uniform(40, 60)
                env.ledger.tensor[partner_id, 0, Ledger.COOP_COUNT] = random.uniform(3, 7)
                env.ledger.tensor[partner_id, 0, Ledger.DEFENSE_SCORE] = random.uniform(20, 40)
                env.ledger.tensor[partner_id, 0, Ledger.DAMAGE_DEALT] = 0.0

                env.ledger.tensor[0, partner_id, Ledger.FOOD_GIVEN] = random.uniform(20, 40)
                env.ledger.tensor[0, partner_id, Ledger.COOP_COUNT] = random.uniform(3, 7)
                env.ledger.tensor[0, partner_id, Ledger.DEFENSE_SCORE] = random.uniform(10, 30)
                env.ledger.tensor[0, partner_id, Ledger.DAMAGE_DEALT] = 0.0

        # Allies (agent 1 + non-neutral partners) have positive histories with each other
        ally_ids = [1] + [i for i in range(2, self.n_agents) if not has_neutral]
        for i in ally_ids:
            for j in ally_ids:
                if i != j:
                    env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = random.uniform(30, 50)
                    env.ledger.tensor[i, j, Ledger.COOP_COUNT] = random.uniform(2, 5)
                    env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = random.uniform(15, 35)
                    env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = 0.0

    def _inject_coherent_histories(self, env: 'GridWorld') -> None:
        """Inject coherent friend/foe relationships (not mixed signals).

        Each agent pair is assigned a relationship type:
        - Friend (40%): positive signals, no damage
        - Foe (30%): damage history, no positive signals
        - Neutral (30%): no history

        Relationships are symmetric (if A is friend of B, B is friend of A).
        """
        from ledger import Ledger

        # Determine relationship for each unique pair
        relationships = {}  # (i, j) -> 'friend' | 'foe' | 'neutral'

        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                roll = random.random()
                if roll < 0.4:
                    relationships[(i, j)] = 'friend'
                elif roll < 0.7:
                    relationships[(i, j)] = 'foe'
                else:
                    relationships[(i, j)] = 'neutral'

        # Apply symmetric relationships to ledger
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue

                # Get relationship (order-independent lookup)
                key = (min(i, j), max(i, j))
                rel = relationships[key]

                if rel == 'friend':
                    # Positive history - helped each other
                    env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = random.uniform(30, 60)
                    env.ledger.tensor[i, j, Ledger.COOP_COUNT] = random.uniform(2, 6)
                    env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = random.uniform(15, 40)
                    env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = 0.0

                elif rel == 'foe':
                    # Negative history - attacked each other
                    env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = random.uniform(40, 80)
                    env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = 0.0
                    env.ledger.tensor[i, j, Ledger.COOP_COUNT] = 0.0
                    env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = 0.0

                # else: neutral - leave at 0 (no history)

    def _inject_random_histories(self, env: 'GridWorld') -> None:
        """Inject varied social histories for training diversity.

        Each channel is rolled independently so agents can have mixed histories
        (e.g., former enemy turned ally, helped with food but never defended, etc.)
        """
        from ledger import Ledger

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    # Each channel independent
                    if random.random() < 0.3:  # 30% enemy history
                        env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = random.uniform(30, 80)
                    if random.random() < 0.3:  # 30% food given history
                        env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = random.uniform(20, 50)
                    if random.random() < 0.25:  # 25% coop history
                        env.ledger.tensor[i, j, Ledger.COOP_COUNT] = random.uniform(2, 8)
                    if random.random() < 0.25:  # 25% defense history
                        env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = random.uniform(10, 40)

    def spawn_food(self, env: 'GridWorld') -> None:
        """Spawn both poor and rich food for social scenarios."""
        total_cells = env.grid_size * env.grid_size
        cap_cells = int(total_cells * env.config.food_coverage_cap)

        # Get empty cells
        empty = (env.occupancy == -1)
        empty_indices = empty.flatten().nonzero(as_tuple=True)[0]

        if len(empty_indices) == 0:
            return

        perm = torch.randperm(len(empty_indices), device=env.device)
        shuffled_indices = empty_indices[perm]

        # Place poor food
        poor_count = min(cap_cells // 2, len(shuffled_indices))
        for i in range(poor_count):
            idx = shuffled_indices[i].item()
            row, col = idx // env.grid_size, idx % env.grid_size
            env.poor_food[row, col] = True

        # Place rich food
        rich_start = poor_count
        rich_count = min(cap_cells // 4, len(shuffled_indices) - rich_start)
        for i in range(rich_count):
            idx = shuffled_indices[rich_start + i].item()
            row, col = idx // env.grid_size, idx % env.grid_size
            env.rich_food[row, col] = True

    def respawn_food(self, env: 'GridWorld') -> None:
        """Standard food respawning like normal gameplay."""
        # Use the environment's normal food spawning logic
        empty = (env.occupancy == -1) & ~env.poor_food & ~env.rich_food

        total_cells = env.grid_size * env.grid_size
        cap_cells = int(total_cells * env.config.food_coverage_cap)

        poor_count = env.poor_food.sum().item()
        rich_count = env.rich_food.sum().item()

        # Spawn poor food
        if poor_count < cap_cells:
            poor_rate = min(1.0, env.config.poor_food_spawn_rate * 2.0)
            rand = torch.rand((env.grid_size, env.grid_size), device=env.device)
            spawn_poor = empty & (rand < poor_rate)
            env.poor_food = env.poor_food | spawn_poor

        # Spawn rich food
        if rich_count < cap_cells:
            empty_after_poor = (env.occupancy == -1) & ~env.poor_food & ~env.rich_food
            rich_rate = min(1.0, env.config.rich_food_spawn_rate * 2.0)
            rand2 = torch.rand((env.grid_size, env.grid_size), device=env.device)
            spawn_rich = empty_after_poor & (rand2 < rich_rate)
            env.rich_food = env.rich_food | spawn_rich


# =============================================================================
# MULTI-TEAM SCENARIO: N teams with team-based social dynamics
# =============================================================================

class MultiTeamScenario(Scenario):
    """Multi-team training with team-based social dynamics.

    Splits agents evenly into N teams at episode start. Ledger is initialized:
    - Same team = allies (high coop, food given, defense)
    - Different teams = enemies (high damage dealt)
    """

    def __init__(self, n_agents: int, num_teams: int = 2):
        self.n_agents = n_agents
        self.num_teams = num_teams

    def get_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            n_active_agents=self.n_agents,
            clone_mode=False,
            inject_social=True,
            partner_mode="learning"
        )

    def setup(self, env: 'GridWorld') -> None:
        """Position agents and inject team-based histories."""
        team_assignments = self._assign_teams()
        self._position_agents_by_team(env, team_assignments)
        self._inject_team_histories(env, team_assignments)

    def _assign_teams(self) -> List[int]:
        """Split agents evenly into teams, shuffle assignments.

        For N agents and T teams:
        - Base size: N // T
        - Remainder N % T distributed to first teams

        Examples:
        - 8 agents, 2 teams -> [4, 4]
        - 8 agents, 3 teams -> [3, 3, 2]
        - 8 agents, 4 teams -> [2, 2, 2, 2]
        """
        base_size = self.n_agents // self.num_teams
        remainder = self.n_agents % self.num_teams

        assignments = []
        for team_id in range(self.num_teams):
            team_size = base_size + (1 if team_id < remainder else 0)
            assignments.extend([team_id] * team_size)

        random.shuffle(assignments)
        return assignments

    def _inject_team_histories(self, env: 'GridWorld', teams: List[int]) -> None:
        """Inject ledger histories based on team assignments.

        Allies (same team): positive history (food given, coop, defense)
        Enemies (diff team): negative history (damage dealt)
        """
        from ledger import Ledger

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue

                if teams[i] == teams[j]:
                    # Same team = allies
                    env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = random.uniform(40, 60)
                    env.ledger.tensor[i, j, Ledger.COOP_COUNT] = random.uniform(3, 7)
                    env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = random.uniform(20, 40)
                    env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = 0.0
                else:
                    # Different team = enemies
                    env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = random.uniform(40, 70)
                    env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = 0.0
                    env.ledger.tensor[i, j, Ledger.COOP_COUNT] = 0.0
                    env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = 0.0

    def _position_agents_by_team(self, env: 'GridWorld', teams: List[int]) -> None:
        """Place teams on opposite sides of the grid.

        For 2 teams: left side vs right side
        For 3+ teams: divide grid into sectors
        """
        gs = env.grid_size
        center = gs // 2

        # Group agents by team
        team_agents = [[] for _ in range(self.num_teams)]
        for agent_id, team_id in enumerate(teams):
            team_agents[team_id].append(agent_id)

        # Calculate spawn positions for each team
        edge_margin = 2  # How close to edge teams spawn
        if self.num_teams == 2:
            # Left edge vs right edge (true opposite sides)
            team_centers = [
                (center, edge_margin),          # Team 0: left edge
                (center, gs - 1 - edge_margin), # Team 1: right edge
            ]
        elif self.num_teams == 3:
            # Triangle formation at edges
            team_centers = [
                (edge_margin, center),                  # Team 0: top edge
                (gs - 1 - edge_margin, edge_margin),    # Team 1: bottom-left
                (gs - 1 - edge_margin, gs - 1 - edge_margin),  # Team 2: bottom-right
            ]
        elif self.num_teams == 4:
            # Four corners at edges
            team_centers = [
                (edge_margin, edge_margin),                     # Team 0: top-left
                (edge_margin, gs - 1 - edge_margin),            # Team 1: top-right
                (gs - 1 - edge_margin, edge_margin),            # Team 2: bottom-left
                (gs - 1 - edge_margin, gs - 1 - edge_margin),   # Team 3: bottom-right
            ]
        else:
            # Fallback: distribute around center
            import math
            team_centers = []
            radius = gs // 3
            for t in range(self.num_teams):
                angle = 2 * math.pi * t / self.num_teams
                r = int(center + radius * math.sin(angle))
                c = int(center + radius * math.cos(angle))
                team_centers.append((r, c))

        # Position each team's agents around their team center
        for team_id, agents in enumerate(team_agents):
            tr, tc = team_centers[team_id]
            positions = self._generate_cluster_positions(tr, tc, len(agents), gs)
            for idx, agent_id in enumerate(agents):
                r, c = positions[idx]
                env.agent_positions[agent_id] = torch.tensor([r, c], device=env.device)
                env.agent_alive[agent_id] = True
                env.agent_hp[agent_id] = env.config.max_hp

        # Deactivate remaining agents
        for i in range(self.n_agents, env.n_agents):
            env.agent_alive[i] = False
            env.agent_hp[i] = 0

        env._update_occupancy()

    def _position_agents(self, env: 'GridWorld') -> None:
        """Place agents close enough to interact (reused from SocialScenario)."""
        center = env.grid_size // 2

        # Position agents in a cluster around center using spiral pattern
        positions = self._generate_cluster_positions(center, center, self.n_agents, env.grid_size)

        for i in range(min(self.n_agents, env.n_agents)):
            r, c = positions[i]
            env.agent_positions[i] = torch.tensor([r, c], device=env.device)
            env.agent_alive[i] = True
            env.agent_hp[i] = env.config.max_hp

        # Deactivate remaining agents
        for i in range(self.n_agents, env.n_agents):
            env.agent_alive[i] = False
            env.agent_hp[i] = 0

        env._update_occupancy()

    def _generate_cluster_positions(
        self, center_r: int, center_c: int, n: int, grid_size: int
    ) -> List[Tuple[int, int]]:
        """Generate n positions in a cluster around center (spiral pattern)."""
        positions = [(center_r, center_c)]

        # Spiral outward from center
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        r, c = center_r, center_c
        dir_idx = 0
        steps_in_dir = 1
        steps_taken = 0
        turns = 0

        while len(positions) < n:
            dr, dc = directions[dir_idx]
            r, c = r + dr, c + dc
            steps_taken += 1

            if 0 <= r < grid_size and 0 <= c < grid_size:
                if (r, c) not in positions:
                    positions.append((r, c))

            if steps_taken >= steps_in_dir:
                steps_taken = 0
                dir_idx = (dir_idx + 1) % 4
                turns += 1
                if turns % 2 == 0:
                    steps_in_dir += 1

            # Safety limit
            if len(positions) + steps_in_dir * 4 > grid_size * grid_size:
                break

        return positions[:n]

    def spawn_food(self, env: 'GridWorld') -> None:
        """Spawn both poor and rich food (reused from SocialScenario)."""
        total_cells = env.grid_size * env.grid_size
        cap_cells = int(total_cells * env.config.food_coverage_cap)

        # Get empty cells
        empty = (env.occupancy == -1)
        empty_indices = empty.flatten().nonzero(as_tuple=True)[0]

        if len(empty_indices) == 0:
            return

        perm = torch.randperm(len(empty_indices), device=env.device)
        shuffled_indices = empty_indices[perm]

        # Place poor food
        poor_count = min(cap_cells // 2, len(shuffled_indices))
        for i in range(poor_count):
            idx = shuffled_indices[i].item()
            row, col = idx // env.grid_size, idx % env.grid_size
            env.poor_food[row, col] = True

        # Place rich food
        rich_start = poor_count
        rich_count = min(cap_cells // 4, len(shuffled_indices) - rich_start)
        for i in range(rich_count):
            idx = shuffled_indices[rich_start + i].item()
            row, col = idx // env.grid_size, idx % env.grid_size
            env.rich_food[row, col] = True

    def respawn_food(self, env: 'GridWorld') -> None:
        """Standard food respawning (reused from SocialScenario)."""
        empty = (env.occupancy == -1) & ~env.poor_food & ~env.rich_food

        total_cells = env.grid_size * env.grid_size
        cap_cells = int(total_cells * env.config.food_coverage_cap)

        poor_count = env.poor_food.sum().item()
        rich_count = env.rich_food.sum().item()

        # Spawn poor food
        if poor_count < cap_cells:
            poor_rate = min(1.0, env.config.poor_food_spawn_rate * 2.0)
            rand = torch.rand((env.grid_size, env.grid_size), device=env.device)
            spawn_poor = empty & (rand < poor_rate)
            env.poor_food = env.poor_food | spawn_poor

        # Spawn rich food
        if rich_count < cap_cells:
            empty_after_poor = (env.occupancy == -1) & ~env.poor_food & ~env.rich_food
            rich_rate = min(1.0, env.config.rich_food_spawn_rate * 2.0)
            rand2 = torch.rand((env.grid_size, env.grid_size), device=env.device)
            spawn_rich = empty_after_poor & (rand2 < rich_rate)
            env.rich_food = env.rich_food | spawn_rich


# =============================================================================
# CURRICULUM DEFINITION
# =============================================================================

def get_curriculum() -> dict:
    """Return the curriculum mapping phase -> scenario."""
    return {
        # Solo food-finding (phases 1-5)
        1: SoloFoodScenario(distance=1, cardinal_only=True),
        2: SoloFoodScenario(distance=2, cardinal_only=True),
        3: SoloFoodScenario(distance=3, cardinal_only=False),
        4: SoloFoodScenario(distance=7, cardinal_only=False),
        5: SoloFoodScenario(distance=14, cardinal_only=False),

        # Cooperation (phases 6-8)
        6: CoopFoodScenario(distance=1),
        7: CoopFoodScenario(distance=2),
        8: CoopFoodScenario(distance=3),

        # Low-HP cooperation (phase 9) - teaches cooperation when wounded
        # Prevents "desperation aggression" where low-HP agents attack allies
        9: CoopFoodLowHPScenario(distance=3, hp_fraction=0.5),

        # Social pretraining (phases 10-12)
        # Phase 10: 2 agents - 50% scripted friend (like phase 9), 50% neutral stranger (learning)
        # Teaches agent 0 to handle both cooperative allies and unknown strangers
        10: SocialScenario(n_agents=2, inject_histories=True, scripted_partners=True, neutral_prob=0.5),
        # Phase 11: Cloned weights, coherent friend/foe histories (learn to discriminate)
        11: SocialScenario(n_agents=4, inject_histories=True, clone_weights=True, coherent_histories=True),
        12: SocialScenario(n_agents=8, inject_histories=False),
    }


def get_thresholds() -> dict:
    """Return advancement thresholds for each phase."""
    return {
        # Solo phases: 70% of theoretical max
        # floor(128/distance) * 5 food reward
        1: 448,   # dist 1: 128 foods * 5 = 640, 70% = 448
        2: 224,   # dist 2: 64 foods * 5 = 320, 70% = 224
        3: 147,   # dist 3: 42 foods * 5 = 210, 70% = 147
        4: 63,    # dist 7: 18 foods * 5 = 90, 70% = 63
        5: 32,    # dist 14: 9 foods * 5 = 45, 70% = 32

        # Coop phases: per-agent average
        6: 896,   # 2 agents, rich food (20 reward each)
        7: 448,
        8: 294,

        # Low-HP coop phase: same mission as phase 8 but at 50% HP
        # Slightly lower threshold since survival is harder
        9: 250,   # 2 agents at 50% HP, must cooperate without attacking ally

        # Social phases
        10: 800,  # 3-agent with scripted ally - must learn reliable cooperation
        11: 500,  # 4-agent with cloned weights, coherent friend/foe histories
        12: 60,   # Full population emergent learning
    }


# Convenience exports
CURRICULUM = get_curriculum()
THRESHOLDS = get_thresholds()
