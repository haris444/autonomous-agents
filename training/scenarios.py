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
    from env.environment import GridWorld


# =============================================================================
# Shared utilities
# =============================================================================

@dataclass
class ScenarioConfig:
    """Configuration returned by a scenario."""
    n_active_agents: int
    clone_mode: bool = False          # Share networks between agents
    inject_social: bool = False       # Pre-inject social histories
    partner_mode: str = "learning"    # "learning" = normal, "always_coop" = scripted helper


def get_positions_at_distance(
    grid_size: int, r: int, c: int, distance: int, cardinal_only: bool = False
) -> List[Tuple[int, int]]:
    """Get all valid grid positions at exact Manhattan distance from (r, c)."""
    positions = []
    if cardinal_only:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr * distance, c + dc * distance
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                positions.append((nr, nc))
    else:
        for dr in range(-distance, distance + 1):
            dc_abs = distance - abs(dr)
            for dc in ([-dc_abs, dc_abs] if dc_abs > 0 else [0]):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    positions.append((nr, nc))
    return positions


def generate_cluster_positions(
    center_r: int, center_c: int, n: int, grid_size: int
) -> List[Tuple[int, int]]:
    """Generate n positions in a spiral cluster around (center_r, center_c)."""
    positions = [(center_r, center_c)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
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

        if len(positions) + steps_in_dir * 4 > grid_size * grid_size:
            break

    return positions[:n]


def inject_pair_history(env: 'GridWorld', i: int, j: int, relationship: str) -> None:
    """Inject ledger history for a single directed pair (i -> j).

    relationship: 'ally', 'enemy', or 'neutral'
    """
    from core.ledger import Ledger

    if relationship == 'ally':
        env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = random.uniform(30, 60)
        env.ledger.tensor[i, j, Ledger.COOP_COUNT] = random.uniform(2, 7)
        env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = random.uniform(15, 40)
        env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = 0.0
    elif relationship == 'enemy':
        env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = random.uniform(40, 70)
        env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = 0.0
        env.ledger.tensor[i, j, Ledger.COOP_COUNT] = random.uniform(0, 1)
        env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = 0.0
    # neutral: leave at 0


def inject_symmetric_history(env: 'GridWorld', i: int, j: int, relationship: str) -> None:
    """Inject ledger history for both directions of a pair."""
    inject_pair_history(env, i, j, relationship)
    inject_pair_history(env, j, i, relationship)


def activate_agents(env: 'GridWorld', n_active: int) -> None:
    """Activate first n_active agents, deactivate the rest."""
    env.agent_alive[:n_active] = True
    env.agent_hp[:n_active] = env.config.max_hp
    if n_active < env.n_agents:
        env.agent_alive[n_active:] = False
        env.agent_hp[n_active:] = 0


def randomize_dead_agents(env: 'GridWorld', start_idx: int) -> None:
    """Scatter dead agents to random positions (avoids occupancy conflicts)."""
    n_dead = env.n_agents - start_idx
    if n_dead > 0:
        other_positions = torch.randperm(env.grid_size * env.grid_size, device=env.device)[:n_dead]
        rows = other_positions // env.grid_size
        cols = other_positions % env.grid_size
        env.agent_positions[start_idx:] = torch.stack([rows, cols], dim=1)


def get_rich_food_positions(grid_size: int, n: int) -> List[Tuple[int, int]]:
    """Return up to n fixed rich food positions (center + 4 corners)."""
    center = grid_size // 2
    edge = grid_size - 1
    return [
        (center, center),
        (1, 1),
        (1, edge - 1),
        (edge - 1, 1),
        (edge - 1, edge - 1),
    ][:n]


def spawn_random_food(env: 'GridWorld', n_rich_food: int = 0) -> None:
    """Spawn poor food (+ optional fixed rich food) on the grid."""
    # Place fixed rich foods at known positions
    if n_rich_food > 0:
        for r, c in get_rich_food_positions(env.grid_size, n_rich_food):
            env.rich_food[r, c] = True

    total_cells = env.grid_size * env.grid_size
    cap_cells = int(total_cells * env.config.food_coverage_cap)

    empty = (env.occupancy == -1) & ~env.rich_food
    empty_indices = empty.flatten().nonzero(as_tuple=True)[0]
    if len(empty_indices) == 0:
        return

    perm = torch.randperm(len(empty_indices), device=env.device)
    shuffled = empty_indices[perm]

    # Place poor food
    poor_count = min(cap_cells // 2, len(shuffled))
    for i in range(poor_count):
        idx = shuffled[i].item()
        env.poor_food[idx // env.grid_size, idx % env.grid_size] = True

    # Place random rich food only if no fixed rich food
    if n_rich_food == 0:
        rich_start = poor_count
        rich_count = min(cap_cells // 4, len(shuffled) - rich_start)
        for i in range(rich_count):
            idx = shuffled[rich_start + i].item()
            env.rich_food[idx // env.grid_size, idx % env.grid_size] = True


def respawn_random_food(env: 'GridWorld', n_rich_food: int = 0) -> None:
    """Respawn food during an episode (standard poor + optional delayed rich)."""
    # Delayed rich food respawn
    if n_rich_food > 0:
        current_rich = int(env.rich_food.sum().item())
        missing = n_rich_food - current_rich
        if missing > 0:
            empty = (env.occupancy == -1) & ~env.poor_food & ~env.rich_food
            empty_indices = empty.flatten().nonzero(as_tuple=True)[0]
            if len(empty_indices) > 0:
                for _ in range(missing):
                    if random.random() < 0.05:
                        idx = empty_indices[torch.randint(len(empty_indices), (1,)).item()].item()
                        env.rich_food[idx // env.grid_size, idx % env.grid_size] = True

    # Standard poor food respawn
    empty = (env.occupancy == -1) & ~env.poor_food & ~env.rich_food
    total_cells = env.grid_size * env.grid_size
    cap_cells = int(total_cells * env.config.food_coverage_cap)

    if env.poor_food.sum().item() < cap_cells:
        poor_rate = min(1.0, env.config.poor_food_spawn_rate * 2.0)
        rand = torch.rand((env.grid_size, env.grid_size), device=env.device)
        env.poor_food = env.poor_food | (empty & (rand < poor_rate))

    # Random rich food only if no fixed rich food
    if n_rich_food == 0 and env.rich_food.sum().item() < cap_cells:
        empty_after = (env.occupancy == -1) & ~env.poor_food & ~env.rich_food
        rich_rate = min(1.0, env.config.rich_food_spawn_rate * 2.0)
        rand2 = torch.rand((env.grid_size, env.grid_size), device=env.device)
        env.rich_food = env.rich_food | (empty_after & (rand2 < rich_rate))


# =============================================================================
# Base class
# =============================================================================

class Scenario(ABC):
    """Abstract base for training scenarios."""

    @abstractmethod
    def setup(self, env: 'GridWorld') -> None:
        pass

    @abstractmethod
    def get_config(self) -> ScenarioConfig:
        pass

    @abstractmethod
    def spawn_food(self, env: 'GridWorld') -> None:
        pass

    @abstractmethod
    def respawn_food(self, env: 'GridWorld') -> None:
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
        center = env.grid_size // 2
        env.agent_positions[0] = torch.tensor([center, center], device=env.device)
        env.agent_alive[1:] = False
        env.agent_hp[1:] = 0
        randomize_dead_agents(env, 1)
        env._update_occupancy()

    def spawn_food(self, env: 'GridWorld') -> None:
        r, c = env.agent_positions[0][0].item(), env.agent_positions[0][1].item()
        distance = min(self.distance, env.grid_size - 1)
        valid = get_positions_at_distance(env.grid_size, r, c, distance, self.cardinal_only)
        if valid:
            nr, nc = valid[torch.randint(len(valid), (1,)).item()]
            env.poor_food[nr, nc] = True

    def respawn_food(self, env: 'GridWorld') -> None:
        if not env.poor_food.any():
            self.spawn_food(env)


# =============================================================================
# COOPERATION PHASES (6-8): Two agents cooperate on rich food
# =============================================================================

class CoopFoodScenario(Scenario):
    """Phases 6-8: Two agents cooperate on rich food."""

    def __init__(self, distance: int, n_rich_food: int = 1):
        self.distance = distance
        self.n_rich_food = n_rich_food

    def get_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            n_active_agents=2, clone_mode=False, partner_mode="always_coop"
        )

    def setup(self, env: 'GridWorld') -> None:
        from core.ledger import Ledger

        distance = min(self.distance, env.grid_size - 1)
        center = env.grid_size // 2
        valid = get_positions_at_distance(env.grid_size, center, center, distance)

        if len(valid) >= 2:
            perm = torch.randperm(len(valid))[:2]
            r0, c0 = valid[perm[0].item()]
            r1, c1 = valid[perm[1].item()]
            env.agent_positions[0] = torch.tensor([r0, c0], device=env.device)
            env.agent_positions[1] = torch.tensor([r1, c1], device=env.device)
        else:
            env.agent_positions[0] = torch.tensor([center - 1, center], device=env.device)
            env.agent_positions[1] = torch.tensor([center, center - 1], device=env.device)

        activate_agents(env, 2)
        randomize_dead_agents(env, 2)
        env._update_occupancy()

        # Inject ally history
        env.ledger.tensor[1, 0, Ledger.FOOD_GIVEN] = 50.0
        env.ledger.tensor[1, 0, Ledger.DEFENSE_SCORE] = 30.0
        env.ledger.tensor[1, 0, Ledger.COOP_COUNT] = 5.0

    def spawn_food(self, env: 'GridWorld') -> None:
        for r, c in get_rich_food_positions(env.grid_size, self.n_rich_food):
            env.rich_food[r, c] = True

    def respawn_food(self, env: 'GridWorld') -> None:
        missing = self.n_rich_food - int(env.rich_food.sum().item())
        if missing <= 0:
            return

        distance = min(self.distance, env.grid_size - 1)
        r0, c0 = env.agent_positions[0][0].item(), env.agent_positions[0][1].item()
        r1, c1 = env.agent_positions[1][0].item(), env.agent_positions[1][1].item()
        center_r, center_c = (r0 + r1) // 2, (c0 + c1) // 2

        valid = []
        for dr in range(-distance * 2, distance * 2 + 1):
            for dc in range(-distance * 2, distance * 2 + 1):
                nr, nc = center_r + dr, center_c + dc
                if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size:
                    if (abs(nr - r0) + abs(nc - c0) <= distance and
                            abs(nr - r1) + abs(nc - c1) <= distance and
                            env.occupancy[nr, nc] == -1 and not env.rich_food[nr, nc]):
                        valid.append((nr, nc))

        for _ in range(missing):
            if valid:
                nr, nc = valid.pop(torch.randint(len(valid), (1,)).item())
                env.rich_food[nr, nc] = True
            elif 0 <= center_r < env.grid_size and 0 <= center_c < env.grid_size:
                if env.occupancy[center_r, center_c] == -1:
                    env.rich_food[center_r, center_c] = True


class CoopFoodLowHPScenario(CoopFoodScenario):
    """Phase 9: Same as CoopFoodScenario but agents start at reduced HP."""

    def __init__(self, distance: int, hp_fraction: float = 0.5, n_rich_food: int = 1):
        super().__init__(distance, n_rich_food=n_rich_food)
        self.hp_fraction = hp_fraction

    def setup(self, env: 'GridWorld') -> None:
        super().setup(env)
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
        always_one_ally: bool = False,
        friend_foe: bool = False,
        n_rich_food: int = 0,
    ):
        self.n_agents = n_agents
        self.inject_histories = inject_histories
        self.scripted_partners = scripted_partners
        self.clone_weights = clone_weights
        self.coherent_histories = coherent_histories
        self.neutral_prob = neutral_prob
        self.always_one_ally = always_one_ally
        self.friend_foe = friend_foe
        self.n_rich_food = n_rich_food

    def get_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            n_active_agents=self.n_agents,
            clone_mode=self.clone_weights,
            inject_social=self.inject_histories,
            partner_mode="scripted" if self.scripted_partners else "learning"
        )

    def setup(self, env: 'GridWorld') -> None:
        self._position_agents(env)

        if self.friend_foe:
            self._setup_friend_foe(env)
            return

        if self.scripted_partners:
            self._setup_scripted(env)
        elif self.inject_histories:
            self._setup_injected(env)

    def _position_agents(self, env: 'GridWorld') -> None:
        positions = generate_cluster_positions(
            env.grid_size // 2, env.grid_size // 2, self.n_agents, env.grid_size
        )
        for i in range(min(self.n_agents, env.n_agents)):
            env.agent_positions[i] = torch.tensor(positions[i], device=env.device)
        activate_agents(env, self.n_agents)
        env._update_occupancy()

    def _setup_friend_foe(self, env: 'GridWorld') -> None:
        """Place enemy far away, inject friend (agent 1) + foe (agent 2) histories."""
        pos0 = env.agent_positions[0]
        candidates = []
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if abs(r - pos0[0].item()) + abs(c - pos0[1].item()) >= 5:
                    if env.occupancy[r, c] == -1:
                        candidates.append((r, c))
        if candidates:
            r, c = candidates[random.randint(0, len(candidates) - 1)]
            env.agent_positions[2] = torch.tensor([r, c], device=env.device)
            env._update_occupancy()

        env.partner_relationship = 'mixed'
        inject_symmetric_history(env, 0, 1, 'ally')
        inject_symmetric_history(env, 0, 2, 'enemy')

    def _setup_scripted(self, env: 'GridWorld') -> None:
        if self.n_agents == 2 and self.neutral_prob > 0:
            is_neutral = random.random() < self.neutral_prob
            env.partner_relationship = 'neutral' if is_neutral else 'ally'
            if not is_neutral:
                self._inject_all_allies(env)
        elif self.always_one_ally and self.n_agents >= 3:
            has_neutral = random.random() < self.neutral_prob
            env.partner_relationship = 'mixed' if has_neutral else 'ally'
            self._inject_mixed(env, has_neutral)
        else:
            env.partner_relationship = 'enemy' if random.random() < self.neutral_prob else 'ally'
            if env.partner_relationship == 'ally':
                self._inject_all_allies(env)
            else:
                self._inject_all_enemies(env)

    def _setup_injected(self, env: 'GridWorld') -> None:
        if self.always_one_ally and self.n_agents >= 3:
            has_neutral = random.random() < self.neutral_prob
            env.partner_relationship = 'mixed' if has_neutral else 'ally'
            self._inject_mixed(env, has_neutral)
        elif self.coherent_histories:
            self._inject_coherent(env)
        else:
            self._inject_random(env)

    def _inject_all_allies(self, env: 'GridWorld') -> None:
        """All partners (1+) are allies of agent 0 and each other."""
        for pid in range(1, self.n_agents):
            inject_symmetric_history(env, 0, pid, 'ally')
        for i in range(1, self.n_agents):
            for j in range(1, self.n_agents):
                if i != j:
                    inject_pair_history(env, i, j, 'ally')

    def _inject_all_enemies(self, env: 'GridWorld') -> None:
        """All partners (1+) are enemies of agent 0."""
        for eid in range(1, self.n_agents):
            inject_symmetric_history(env, 0, eid, 'enemy')

    def _inject_mixed(self, env: 'GridWorld', has_neutral: bool) -> None:
        """Agent 1 is always ally; agent 2+ depend on has_neutral flag."""
        inject_symmetric_history(env, 0, 1, 'ally')
        for pid in range(2, self.n_agents):
            rel = 'neutral' if has_neutral else 'ally'
            inject_symmetric_history(env, 0, pid, rel)
        # Allies have positive history with each other
        ally_ids = [1] + [i for i in range(2, self.n_agents) if not has_neutral]
        for i in ally_ids:
            for j in ally_ids:
                if i != j:
                    inject_pair_history(env, i, j, 'ally')

    def _inject_coherent(self, env: 'GridWorld') -> None:
        """Assign each pair a consistent relationship: friend/foe/neutral."""
        relationships = {}
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                roll = random.random()
                if roll < 0.4:
                    relationships[(i, j)] = 'ally'
                elif roll < 0.7:
                    relationships[(i, j)] = 'enemy'
                else:
                    relationships[(i, j)] = 'neutral'

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    key = (min(i, j), max(i, j))
                    inject_pair_history(env, i, j, relationships[key])

    def _inject_random(self, env: 'GridWorld') -> None:
        """Inject varied histories with independently rolled channels."""
        from core.ledger import Ledger

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    if random.random() < 0.3:
                        env.ledger.tensor[i, j, Ledger.DAMAGE_DEALT] = random.uniform(30, 80)
                    if random.random() < 0.3:
                        env.ledger.tensor[i, j, Ledger.FOOD_GIVEN] = random.uniform(20, 50)
                    if random.random() < 0.25:
                        env.ledger.tensor[i, j, Ledger.COOP_COUNT] = random.uniform(2, 8)
                    if random.random() < 0.25:
                        env.ledger.tensor[i, j, Ledger.DEFENSE_SCORE] = random.uniform(10, 40)

    def spawn_food(self, env: 'GridWorld') -> None:
        spawn_random_food(env, self.n_rich_food)

    def respawn_food(self, env: 'GridWorld') -> None:
        respawn_random_food(env, self.n_rich_food)


# =============================================================================
# MULTI-TEAM SCENARIO: N teams with team-based social dynamics
# =============================================================================

class MultiTeamScenario(Scenario):
    """Multi-team training with team-based social dynamics."""

    def __init__(self, n_agents: int, num_teams: int = 2):
        self.n_agents = n_agents
        self.num_teams = num_teams

    def get_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            n_active_agents=self.n_agents, clone_mode=False,
            inject_social=True, partner_mode="learning"
        )

    def setup(self, env: 'GridWorld') -> None:
        teams = self._assign_teams()
        self._position_agents_by_team(env, teams)
        self._inject_team_histories(env, teams)

    def _assign_teams(self) -> List[int]:
        base = self.n_agents // self.num_teams
        remainder = self.n_agents % self.num_teams
        assignments = []
        for t in range(self.num_teams):
            assignments.extend([t] * (base + (1 if t < remainder else 0)))
        random.shuffle(assignments)
        return assignments

    def _inject_team_histories(self, env: 'GridWorld', teams: List[int]) -> None:
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    rel = 'ally' if teams[i] == teams[j] else 'enemy'
                    inject_pair_history(env, i, j, rel)

    def _position_agents_by_team(self, env: 'GridWorld', teams: List[int]) -> None:
        gs = env.grid_size
        center = gs // 2
        edge_margin = 2

        # Group agents by team
        team_agents = [[] for _ in range(self.num_teams)]
        for agent_id, team_id in enumerate(teams):
            team_agents[team_id].append(agent_id)

        # Calculate team spawn centers
        if self.num_teams == 2:
            team_centers = [
                (center, edge_margin),
                (center, gs - 1 - edge_margin),
            ]
        elif self.num_teams == 3:
            team_centers = [
                (edge_margin, center),
                (gs - 1 - edge_margin, edge_margin),
                (gs - 1 - edge_margin, gs - 1 - edge_margin),
            ]
        elif self.num_teams == 4:
            team_centers = [
                (edge_margin, edge_margin),
                (edge_margin, gs - 1 - edge_margin),
                (gs - 1 - edge_margin, edge_margin),
                (gs - 1 - edge_margin, gs - 1 - edge_margin),
            ]
        else:
            import math
            team_centers = []
            radius = gs // 3
            for t in range(self.num_teams):
                angle = 2 * math.pi * t / self.num_teams
                r = int(center + radius * math.sin(angle))
                c = int(center + radius * math.cos(angle))
                team_centers.append((r, c))

        for team_id, agents in enumerate(team_agents):
            tr, tc = team_centers[team_id]
            positions = generate_cluster_positions(tr, tc, len(agents), gs)
            for idx, agent_id in enumerate(agents):
                env.agent_positions[agent_id] = torch.tensor(positions[idx], device=env.device)

        activate_agents(env, self.n_agents)
        env._update_occupancy()

    def spawn_food(self, env: 'GridWorld') -> None:
        spawn_random_food(env)

    def respawn_food(self, env: 'GridWorld') -> None:
        respawn_random_food(env)


# =============================================================================
# CURRICULUM DEFINITION
# =============================================================================

def get_curriculum() -> dict:
    return {
        # Solo food-finding (phases 1-5)
        1: SoloFoodScenario(distance=1, cardinal_only=True),
        2: SoloFoodScenario(distance=2, cardinal_only=True),
        3: SoloFoodScenario(distance=3, cardinal_only=False),
        4: SoloFoodScenario(distance=7, cardinal_only=False),
        5: SoloFoodScenario(distance=14, cardinal_only=False),
        # Cooperation (phases 6-8)
        6: CoopFoodScenario(distance=1),
        7: CoopFoodScenario(distance=2, n_rich_food=2),
        8: CoopFoodScenario(distance=3),
        # Low-HP cooperation (phase 9)
        9: CoopFoodLowHPScenario(distance=3, hp_fraction=0.5, n_rich_food=5),
        # Social pretraining (phases 10-12)
        10: SocialScenario(n_agents=2, inject_histories=True, scripted_partners=True, neutral_prob=0.5, n_rich_food=5),
        11: SocialScenario(n_agents=3, scripted_partners=True, friend_foe=True, n_rich_food=5),
        12: SocialScenario(n_agents=8, inject_histories=False, n_rich_food=5),
    }


def get_thresholds() -> dict:
    return {
        1: 448, 2: 224, 3: 147, 4: 63, 5: 32,
        6: 896, 7: 448, 8: 294, 9: 250,
        10: 800, 11: 500, 12: 60,
    }


CURRICULUM = get_curriculum()
THRESHOLDS = get_thresholds()
