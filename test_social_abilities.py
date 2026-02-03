"""
Social Abilities Test - Comprehensive testing of learned social behaviors.

Runs actual environment episodes with pre-configured social histories (ledger injection)
to verify agents have learned social behaviors like enemy recognition, ally cooperation,
retaliation, and defense.

Usage:
    python test_social_abilities.py --model checkpoint.pt --episodes 50
    python test_social_abilities.py --model checkpoint.pt --episodes 10 --quick
"""

import argparse
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy import stats
from enum import IntEnum

import torch.nn.functional as F

from config import Config
from environment import GridWorld
from network import ActorCritic
from ppo import VmapPPO
from utils import get_device


# =============================================================================
# Constants
# =============================================================================

class MoveAction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

class InteractAction(IntEnum):
    ATTACK_UP = 0
    ATTACK_DOWN = 1
    ATTACK_LEFT = 2
    ATTACK_RIGHT = 3
    GIVE_UP = 4
    GIVE_DOWN = 5
    GIVE_LEFT = 6
    GIVE_RIGHT = 7
    SIGNAL = 8
    COOPERATE = 9
    IDLE = 10

# Direction deltas: UP, DOWN, LEFT, RIGHT
DIR_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Ledger channel indices
LEDGER_DAMAGE_DEALT = 0
LEDGER_FOOD_GIVEN = 1
LEDGER_COOP_COUNT = 2
LEDGER_DEFENSE_SCORE = 3

# Ledger normalization maxes (from ledger.py)
LEDGER_MAX = [100.0, 100.0, 10.0, 100.0]


# =============================================================================
# Social History Presets
# =============================================================================

def make_enemy_history() -> Dict[str, float]:
    """History indicating this agent has been an enemy (dealt damage)."""
    return {'damage_dealt': 0.8}

def make_ally_history() -> Dict[str, float]:
    """History indicating this agent has been a helpful ally."""
    return {'food_given': 0.5, 'defense_score': 0.5, 'coop_count': 0.3}

def make_cooperator_history() -> Dict[str, float]:
    """History indicating past cooperation."""
    return {'coop_count': 0.8}

def make_neutral_history() -> Dict[str, float]:
    """No prior interaction history."""
    return {}


# =============================================================================
# Scripted Agent Behaviors
# =============================================================================

class ScriptedRole:
    ENEMY = "enemy"          # Attacks the test agent
    ALLY = "ally"            # Gives food, doesn't attack
    PASSIVE = "passive"      # IDLE only
    ATTACKER = "attacker"    # Attacks a specific target


class ScriptedAgent:
    """Controlled agent behavior for testing (not using trained policy)."""

    def __init__(self, agent_id: int, role: str, target_id: int = 0):
        self.agent_id = agent_id
        self.role = role
        self.target_id = target_id  # Who to interact with

    def get_action(self, env) -> Tuple[int, int]:
        """Return factored action (direction, action_type) based on role.

        Factored action space:
            direction: 0-4 (UP, DOWN, LEFT, RIGHT, STAY)
            action_type: 0-4 (MOVE, ATTACK, GIVE, SIGNAL, COOPERATE)
        """
        ACT_MOVE, ACT_ATTACK, ACT_GIVE, ACT_SIGNAL, ACT_COOPERATE = 0, 1, 2, 3, 4
        DIR_STAY = 4

        my_pos = env.agent_positions[self.agent_id]
        target_pos = env.agent_positions[self.target_id]

        # Calculate direction to target
        diff = target_pos - my_pos
        direction = self._get_direction_to(diff)

        if self.role == ScriptedRole.ENEMY:
            # Attack toward test agent
            if direction is not None and env.agent_alive[self.target_id]:
                return direction, ACT_ATTACK
            return DIR_STAY, ACT_MOVE  # Stay in place

        elif self.role == ScriptedRole.ALLY:
            # Give food toward test agent
            if direction is not None and env.agent_alive[self.target_id]:
                return direction, ACT_GIVE
            return DIR_STAY, ACT_MOVE  # Stay in place

        elif self.role == ScriptedRole.ATTACKER:
            # Attack the target (used for defense/retaliation tests)
            if direction is not None and env.agent_alive[self.target_id]:
                return direction, ACT_ATTACK
            return DIR_STAY, ACT_MOVE  # Stay in place

        else:  # PASSIVE
            return DIR_STAY, ACT_MOVE  # Stay in place

    def _get_direction_to(self, diff: torch.Tensor) -> Optional[int]:
        """Get direction index (0-3) to reach target, or None if not adjacent."""
        dr, dc = diff[0].item(), diff[1].item()

        # Must be adjacent (Manhattan distance 1)
        if abs(dr) + abs(dc) != 1:
            return None

        if dr == -1:
            return 0  # UP
        elif dr == 1:
            return 1  # DOWN
        elif dc == -1:
            return 2  # LEFT
        elif dc == 1:
            return 3  # RIGHT
        return None


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics collected from a single episode."""
    attacks_by_direction: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(4)})
    gives_by_direction: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(4)})
    coop_attempts: int = 0
    total_actions: int = 0
    attacks_total: int = 0
    gives_total: int = 0

    # For retaliation tracking
    attacks_before_provocation: int = 0
    attacks_after_provocation: int = 0
    steps_before_provocation: int = 0
    steps_after_provocation: int = 0


@dataclass
class ScenarioResult:
    """Results from running a scenario multiple times."""
    name: str
    episodes: int
    metrics: List[EpisodeMetrics]
    passed: bool = False
    p_value: float = 1.0
    ratio: float = 0.0
    description: str = ""

    # Summary stats
    mean_primary: float = 0.0
    std_primary: float = 0.0
    mean_baseline: float = 0.0
    std_baseline: float = 0.0


# =============================================================================
# Social Test Runner
# =============================================================================

class SocialTestRunner:
    """Runs social ability test scenarios."""

    def __init__(self, config: Config, device: torch.device, model_path: str):
        self.config = config
        self.device = device
        self.model_path = model_path

        # Detect checkpoint n_agents on first load
        self._checkpoint_n_agents: Optional[int] = None
        self._checkpoint_config: Optional[Config] = None
        self._detect_checkpoint_config()

        # Will be initialized per scenario
        self.env: Optional[GridWorld] = None
        self.multi_agent: Optional[VmapPPO] = None

    def _fix_state_dict_compatibility(self, state_dict: dict) -> dict:
        """Fix architecture mismatches between checkpoint and current model."""
        fixed = state_dict.copy()

        # Fix self_encoder: older checkpoints have [64, 1] but current is [64, 2]
        key = 'encoder.self_encoder.0.weight'
        if key in fixed and fixed[key].shape[1] == 1:
            old_weight = fixed[key]  # [64, 1]
            # Expand to [64, 2] by repeating the HP weight for inventory
            new_weight = torch.cat([old_weight, old_weight], dim=1)  # [64, 2]
            fixed[key] = new_weight

        return fixed

    def _detect_checkpoint_config(self):
        """Detect the config used to train the checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Check if config is saved in checkpoint
        if 'config' in checkpoint:
            self._checkpoint_config = checkpoint['config']
            self._checkpoint_n_agents = self._checkpoint_config.n_agents
            return

        # Otherwise detect from state dict shapes
        if 'network_state_dicts' in checkpoint:
            state_dict = checkpoint['network_state_dicts'][0]
        else:
            state_dict = checkpoint

        # Signal encoder input size = n_agents
        signal_weight = state_dict.get('encoder.signal_encoder.0.weight')
        if signal_weight is not None:
            self._checkpoint_n_agents = signal_weight.shape[1]
        else:
            self._checkpoint_n_agents = 8  # Default fallback

    def load_model(self, n_agents_for_test: int):
        """Load the trained model for testing.

        Creates environment with checkpoint's n_agents but tests with subset.
        n_agents_for_test specifies how many agents are active in the test.
        """
        # Create test config based on checkpoint's n_agents
        test_config = Config()
        test_config.n_agents = self._checkpoint_n_agents

        # Disable pretraining/curriculum for testing - we want full control
        test_config.pretrain_mode = False
        test_config.curriculum_enabled = False

        # Store the number of agents we'll actually use in tests
        self.active_agents = min(n_agents_for_test, test_config.n_agents)

        # Create environment and policy with checkpoint's n_agents
        self.env = GridWorld(test_config, self.device)
        self.multi_agent = VmapPPO(test_config, self.device)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if 'network_state_dicts' in checkpoint:
            state_dict = checkpoint['network_state_dicts'][0]
        else:
            state_dict = checkpoint

        # Handle architecture mismatches (e.g., older checkpoints)
        state_dict = self._fix_state_dict_compatibility(state_dict)

        for net in self.multi_agent.networks:
            net.load_state_dict(state_dict, strict=False)

        # Set to eval mode
        for net in self.multi_agent.networks:
            net.eval()

    def inject_social_history(self, histories: Dict[Tuple[int, int], Dict[str, float]]):
        """Inject pre-set ledger values after env.reset()."""
        for (src, tgt), values in histories.items():
            self.env.ledger.tensor[src, tgt, LEDGER_DAMAGE_DEALT] = values.get('damage_dealt', 0) * LEDGER_MAX[0]
            self.env.ledger.tensor[src, tgt, LEDGER_FOOD_GIVEN] = values.get('food_given', 0) * LEDGER_MAX[1]
            self.env.ledger.tensor[src, tgt, LEDGER_COOP_COUNT] = values.get('coop_count', 0) * LEDGER_MAX[2]
            self.env.ledger.tensor[src, tgt, LEDGER_DEFENSE_SCORE] = values.get('defense_score', 0) * LEDGER_MAX[3]

    def set_positions(self, positions: Dict[int, Tuple[int, int]]):
        """Override agent positions for controlled setup.

        Agents not in positions dict are placed in corners and marked inactive.
        """
        n_total = self.env.n_agents

        # Place inactive agents in far corners (spread out to avoid collisions)
        corner_positions = [(0, 0), (0, 14), (14, 0), (14, 14), (0, 7), (14, 7), (7, 0), (7, 14)]

        corner_idx = 0
        for agent_id in range(n_total):
            if agent_id in positions:
                r, c = positions[agent_id]
                self.env.agent_positions[agent_id] = torch.tensor([r, c], device=self.device)
                self.env.agent_alive[agent_id] = True
            else:
                # Place inactive agent in a corner
                cr, cc = corner_positions[corner_idx % len(corner_positions)]
                self.env.agent_positions[agent_id] = torch.tensor([cr, cc], device=self.device)
                # Mark as dead so they don't interfere
                self.env.agent_alive[agent_id] = False
                self.env.agent_hp[agent_id] = 0
                corner_idx += 1

        self.env._update_occupancy()

    def run_episode(
        self,
        test_agent_id: int,
        scripted_agents: List[ScriptedAgent],
        initial_positions: Dict[int, Tuple[int, int]],
        social_histories: Dict[Tuple[int, int], Dict[str, float]],
        max_steps: int = 50,
        provocation_step: Optional[int] = None,  # For retaliation test
    ) -> EpisodeMetrics:
        """Run a single test episode and collect metrics."""

        metrics = EpisodeMetrics()

        # Reset and setup
        obs = self.env.reset()
        self.set_positions(initial_positions)
        self.inject_social_history(social_histories)

        # Get fresh observations after position/history setup
        obs = self.env._get_all_observations()

        provoked = False

        for step in range(max_steps):
            # Check if all agents dead
            if not self.env.agent_alive.any():
                break

            # Track provocation for retaliation test
            if provocation_step is not None and step >= provocation_step:
                if not provoked:
                    provoked = True
                    metrics.attacks_before_provocation = metrics.attacks_total
                    metrics.steps_before_provocation = step

            # Get action masks (factored: direction_mask, action_type_mask)
            direction_mask, action_type_mask = self.env.get_action_masks()

            # Get factored actions from policy
            # Returns: directions, action_types, log_probs, entropies, values
            with torch.no_grad():
                directions, action_types, _, _, _ = self.multi_agent.get_actions_and_values(
                    obs, direction_mask=direction_mask, action_type_mask=action_type_mask
                )

            # Override scripted agents' actions
            for scripted in scripted_agents:
                if self.env.agent_alive[scripted.agent_id]:
                    direction, action_type = scripted.get_action(self.env)
                    directions[scripted.agent_id] = direction
                    action_types[scripted.agent_id] = action_type

            # Record test agent's action
            # Action types: 0=MOVE, 1=ATTACK, 2=GIVE, 3=SIGNAL, 4=COOPERATE
            if self.env.agent_alive[test_agent_id]:
                action_type = action_types[test_agent_id].item()
                direction = directions[test_agent_id].item()
                metrics.total_actions += 1

                if action_type == 1 and direction < 4:  # ATTACK with valid direction
                    metrics.attacks_by_direction[direction] += 1
                    metrics.attacks_total += 1
                elif action_type == 2 and direction < 4:  # GIVE with valid direction
                    metrics.gives_by_direction[direction] += 1
                    metrics.gives_total += 1
                elif action_type == 4:  # COOPERATE
                    metrics.coop_attempts += 1

            # Step environment with factored actions
            obs, rewards, dones, infos = self.env.step(directions, action_types)

        # Finalize retaliation metrics
        if provocation_step is not None:
            metrics.attacks_after_provocation = metrics.attacks_total - metrics.attacks_before_provocation
            metrics.steps_after_provocation = max_steps - metrics.steps_before_provocation

        return metrics

    # =========================================================================
    # Test Scenarios
    # =========================================================================

    def test_enemy_recognition(self, num_episodes: int) -> ScenarioResult:
        """Test 1: Does agent attack enemies more than neutrals?"""
        self.load_model(n_agents_for_test=2)

        result = ScenarioResult(
            name="Enemy Recognition",
            episodes=num_episodes,
            metrics=[],
            description="Attack rate toward enemy vs neutral baseline"
        )

        # Run episodes with enemy history
        enemy_metrics = []
        for _ in range(num_episodes):
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[ScriptedAgent(1, ScriptedRole.PASSIVE, target_id=0)],
                initial_positions={0: (7, 7), 1: (7, 8)},  # Adjacent horizontally
                social_histories={(1, 0): make_enemy_history()},  # Agent 1 was enemy to agent 0
            )
            enemy_metrics.append(m)

        # Run episodes with neutral history
        neutral_metrics = []
        for _ in range(num_episodes):
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[ScriptedAgent(1, ScriptedRole.PASSIVE, target_id=0)],
                initial_positions={0: (7, 7), 1: (7, 8)},
                social_histories={(1, 0): make_neutral_history()},
            )
            neutral_metrics.append(m)

        # Calculate attack rates (RIGHT direction = toward agent 1)
        enemy_attack_rates = [m.attacks_by_direction[3] / max(m.total_actions, 1) for m in enemy_metrics]
        neutral_attack_rates = [m.attacks_by_direction[3] / max(m.total_actions, 1) for m in neutral_metrics]

        result.metrics = enemy_metrics
        result.mean_primary = np.mean(enemy_attack_rates)
        result.std_primary = np.std(enemy_attack_rates)
        result.mean_baseline = np.mean(neutral_attack_rates)
        result.std_baseline = np.std(neutral_attack_rates)

        # Statistical test
        if result.mean_baseline > 0:
            result.ratio = result.mean_primary / result.mean_baseline
        else:
            result.ratio = float('inf') if result.mean_primary > 0 else 1.0

        _, result.p_value = stats.ttest_ind(enemy_attack_rates, neutral_attack_rates, alternative='greater')
        result.passed = result.ratio >= 1.5 and result.p_value < 0.05

        return result

    def test_ally_recognition(self, num_episodes: int) -> ScenarioResult:
        """Test 2: Does agent give to allies and avoid attacking them?"""
        self.load_model(n_agents_for_test=2)

        result = ScenarioResult(
            name="Ally Recognition",
            episodes=num_episodes,
            metrics=[],
            description="Give rate toward ally vs attack rate"
        )

        ally_metrics = []
        for _ in range(num_episodes):
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[ScriptedAgent(1, ScriptedRole.PASSIVE, target_id=0)],
                initial_positions={0: (7, 7), 1: (7, 8)},
                social_histories={(1, 0): make_ally_history()},
            )
            ally_metrics.append(m)

        # Calculate rates
        give_rates = [m.gives_by_direction[3] / max(m.total_actions, 1) for m in ally_metrics]
        attack_rates = [m.attacks_by_direction[3] / max(m.total_actions, 1) for m in ally_metrics]

        result.metrics = ally_metrics
        result.mean_primary = np.mean(give_rates)
        result.std_primary = np.std(give_rates)
        result.mean_baseline = np.mean(attack_rates)
        result.std_baseline = np.std(attack_rates)

        # Test: give rate should exceed attack rate
        if result.mean_baseline > 0:
            result.ratio = result.mean_primary / result.mean_baseline
        else:
            result.ratio = float('inf') if result.mean_primary > 0 else 1.0

        _, result.p_value = stats.ttest_rel(give_rates, attack_rates, alternative='greater')
        result.passed = result.ratio >= 1.5 and result.p_value < 0.05

        return result

    def test_discrimination(self, num_episodes: int) -> ScenarioResult:
        """Test 3: Can agent discriminate enemy (RIGHT) from ally (LEFT)?"""
        self.load_model(n_agents_for_test=3)

        result = ScenarioResult(
            name="Enemy vs Ally Discrimination",
            episodes=num_episodes,
            metrics=[],
            description="Attack right (enemy) vs left (ally)"
        )

        metrics_list = []
        for _ in range(num_episodes):
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[
                    ScriptedAgent(1, ScriptedRole.PASSIVE, target_id=0),  # Enemy on RIGHT
                    ScriptedAgent(2, ScriptedRole.PASSIVE, target_id=0),  # Ally on LEFT
                ],
                initial_positions={0: (7, 7), 1: (7, 8), 2: (7, 6)},  # Test in center
                social_histories={
                    (1, 0): make_enemy_history(),  # Agent 1 was enemy
                    (2, 0): make_ally_history(),   # Agent 2 was ally
                },
            )
            metrics_list.append(m)

        # Attack direction: RIGHT=3 (enemy), LEFT=2 (ally)
        attacks_toward_enemy = [m.attacks_by_direction[3] for m in metrics_list]
        attacks_toward_ally = [m.attacks_by_direction[2] for m in metrics_list]

        result.metrics = metrics_list
        result.mean_primary = np.mean(attacks_toward_enemy)
        result.std_primary = np.std(attacks_toward_enemy)
        result.mean_baseline = np.mean(attacks_toward_ally)
        result.std_baseline = np.std(attacks_toward_ally)

        if result.mean_baseline > 0:
            result.ratio = result.mean_primary / result.mean_baseline
        else:
            result.ratio = float('inf') if result.mean_primary > 0 else 1.0

        _, result.p_value = stats.ttest_rel(attacks_toward_enemy, attacks_toward_ally, alternative='greater')
        result.passed = result.ratio >= 1.5 and result.p_value < 0.05

        return result

    def test_cooperation_preference(self, num_episodes: int) -> ScenarioResult:
        """Test 4: Does agent cooperate more with past cooperators?"""
        self.load_model(n_agents_for_test=3)

        result = ScenarioResult(
            name="Cooperation with Cooperators",
            episodes=num_episodes,
            metrics=[],
            description="COOP action rate with cooperator vs stranger nearby"
        )

        # With past cooperator nearby
        coop_metrics = []
        for _ in range(num_episodes):
            # Place rich food and agents near it
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[
                    ScriptedAgent(1, ScriptedRole.PASSIVE, target_id=0),  # Past cooperator
                    ScriptedAgent(2, ScriptedRole.PASSIVE, target_id=0),  # Stranger
                ],
                initial_positions={0: (7, 7), 1: (7, 8), 2: (7, 6)},
                social_histories={
                    (1, 0): make_cooperator_history(),  # Agent 1 cooperated before
                    (2, 0): make_neutral_history(),     # Agent 2 is stranger
                },
            )
            coop_metrics.append(m)

        # With stranger only (no cooperator history)
        stranger_metrics = []
        for _ in range(num_episodes):
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[
                    ScriptedAgent(1, ScriptedRole.PASSIVE, target_id=0),
                    ScriptedAgent(2, ScriptedRole.PASSIVE, target_id=0),
                ],
                initial_positions={0: (7, 7), 1: (7, 8), 2: (7, 6)},
                social_histories={
                    (1, 0): make_neutral_history(),
                    (2, 0): make_neutral_history(),
                },
            )
            stranger_metrics.append(m)

        coop_rates = [m.coop_attempts / max(m.total_actions, 1) for m in coop_metrics]
        stranger_rates = [m.coop_attempts / max(m.total_actions, 1) for m in stranger_metrics]

        result.metrics = coop_metrics
        result.mean_primary = np.mean(coop_rates)
        result.std_primary = np.std(coop_rates)
        result.mean_baseline = np.mean(stranger_rates)
        result.std_baseline = np.std(stranger_rates)

        if result.mean_baseline > 0:
            result.ratio = result.mean_primary / result.mean_baseline
        else:
            result.ratio = float('inf') if result.mean_primary > 0 else 1.0

        _, result.p_value = stats.ttest_ind(coop_rates, stranger_rates, alternative='greater')
        result.passed = result.ratio >= 1.2 and result.p_value < 0.05

        return result

    def test_defense_behavior(self, num_episodes: int) -> ScenarioResult:
        """Test 5: Does agent defend allies from attackers?"""
        self.load_model(n_agents_for_test=3)

        result = ScenarioResult(
            name="Defense Behavior",
            episodes=num_episodes,
            metrics=[],
            description="Attack rate toward attacker when ally is being attacked"
        )

        # Scenario: Attacker (agent 2) attacks ally (agent 1), test agent (0) should attack attacker
        defense_metrics = []
        for _ in range(num_episodes):
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[
                    ScriptedAgent(1, ScriptedRole.PASSIVE, target_id=0),   # Ally being attacked
                    ScriptedAgent(2, ScriptedRole.ATTACKER, target_id=1), # Attacks the ally
                ],
                # Position: test(0) - ally(1) - attacker(2) in a row
                initial_positions={0: (7, 6), 1: (7, 7), 2: (7, 8)},
                social_histories={
                    (1, 0): make_ally_history(),   # Agent 1 is ally
                    (2, 0): make_neutral_history(), # Agent 2 starts neutral
                },
            )
            defense_metrics.append(m)

        # Baseline: no one attacking
        baseline_metrics = []
        for _ in range(num_episodes):
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[
                    ScriptedAgent(1, ScriptedRole.PASSIVE, target_id=0),
                    ScriptedAgent(2, ScriptedRole.PASSIVE, target_id=0),
                ],
                initial_positions={0: (7, 6), 1: (7, 7), 2: (7, 8)},
                social_histories={
                    (1, 0): make_ally_history(),
                    (2, 0): make_neutral_history(),
                },
            )
            baseline_metrics.append(m)

        # Test agent attacks RIGHT to defend (direction 3)
        defense_attacks = [m.attacks_total for m in defense_metrics]
        baseline_attacks = [m.attacks_total for m in baseline_metrics]

        result.metrics = defense_metrics
        result.mean_primary = np.mean(defense_attacks)
        result.std_primary = np.std(defense_attacks)
        result.mean_baseline = np.mean(baseline_attacks)
        result.std_baseline = np.std(baseline_attacks)

        if result.mean_baseline > 0:
            result.ratio = result.mean_primary / result.mean_baseline
        else:
            result.ratio = float('inf') if result.mean_primary > 0 else 1.0

        _, result.p_value = stats.ttest_ind(defense_attacks, baseline_attacks, alternative='greater')
        result.passed = result.ratio >= 1.3 and result.p_value < 0.05

        return result

    def test_retaliation(self, num_episodes: int) -> ScenarioResult:
        """Test 6: Does agent retaliate after being attacked?"""
        self.load_model(n_agents_for_test=2)

        result = ScenarioResult(
            name="Retaliation",
            episodes=num_episodes,
            metrics=[],
            description="Attack rate increase after being attacked"
        )

        retaliation_metrics = []
        for _ in range(num_episodes):
            # Provocation starts at step 10
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[ScriptedAgent(1, ScriptedRole.ATTACKER, target_id=0)],
                initial_positions={0: (7, 7), 1: (7, 8)},
                social_histories={(1, 0): make_neutral_history()},  # Starts neutral
                provocation_step=10,
            )
            retaliation_metrics.append(m)

        # Calculate attack rates before and after provocation
        rates_before = []
        rates_after = []
        for m in retaliation_metrics:
            if m.steps_before_provocation > 0:
                rates_before.append(m.attacks_before_provocation / m.steps_before_provocation)
            else:
                rates_before.append(0)
            if m.steps_after_provocation > 0:
                rates_after.append(m.attacks_after_provocation / m.steps_after_provocation)
            else:
                rates_after.append(0)

        result.metrics = retaliation_metrics
        result.mean_primary = np.mean(rates_after)
        result.std_primary = np.std(rates_after)
        result.mean_baseline = np.mean(rates_before)
        result.std_baseline = np.std(rates_before)

        if result.mean_baseline > 0:
            result.ratio = result.mean_primary / result.mean_baseline
        else:
            result.ratio = float('inf') if result.mean_primary > 0 else 1.0

        # Paired t-test since same episodes
        _, result.p_value = stats.ttest_rel(rates_after, rates_before, alternative='greater')
        result.passed = result.ratio >= 1.3 and result.p_value < 0.05

        return result

    def test_neutral_baseline(self, num_episodes: int) -> ScenarioResult:
        """Test 7: Establish baseline behavior rates with neutral agent."""
        self.load_model(n_agents_for_test=2)

        result = ScenarioResult(
            name="Neutral Baseline",
            episodes=num_episodes,
            metrics=[],
            description="Baseline interaction rates with neutral agents"
        )

        neutral_metrics = []
        for _ in range(num_episodes):
            m = self.run_episode(
                test_agent_id=0,
                scripted_agents=[ScriptedAgent(1, ScriptedRole.PASSIVE, target_id=0)],
                initial_positions={0: (7, 7), 1: (7, 8)},
                social_histories={(1, 0): make_neutral_history()},
            )
            neutral_metrics.append(m)

        attack_rates = [m.attacks_total / max(m.total_actions, 1) for m in neutral_metrics]
        give_rates = [m.gives_total / max(m.total_actions, 1) for m in neutral_metrics]

        result.metrics = neutral_metrics
        result.mean_primary = np.mean(attack_rates)
        result.std_primary = np.std(attack_rates)
        result.mean_baseline = np.mean(give_rates)
        result.std_baseline = np.std(give_rates)

        # This test always passes - it's informational
        result.passed = True
        result.ratio = 1.0
        result.p_value = 1.0

        return result


# =============================================================================
# Feature Distinctness Analysis
# =============================================================================

@dataclass
class FeatureAnalysisResult:
    """Results from feature distinctness analysis."""
    scenario_names: List[str]
    similarity_matrix: np.ndarray
    avg_similarity: float
    min_similarity: float
    max_similarity: float
    features_distinct: bool


class FeatureDistinctnessAnalyzer:
    """Analyzes whether model features are distinct for different social contexts."""

    # Social scenarios to test
    SCENARIOS = {
        'neutral': {},
        'enemy_attacked': {'damage_dealt': 0.8},
        'ally_gave_food': {'food_given': 0.8},
        'ally_defended': {'defense_score': 0.8},
        'cooperator': {'coop_count': 0.8},
        'strong_ally': {'food_given': 0.5, 'defense_score': 0.5, 'coop_count': 0.3},
        'strong_enemy': {'damage_dealt': 1.0},
    }

    def __init__(self, runner: 'SocialTestRunner'):
        self.runner = runner
        self.device = runner.device

    def _fourier_encode(self, dx, dy):
        """Encode position using Fourier features (matching environment)."""
        bands = [1.0, 2.0, 4.0, 8.0]
        features = []
        for freq in bands:
            features.extend([
                torch.sin(torch.tensor(freq * torch.pi * dx, device=self.device)),
                torch.cos(torch.tensor(freq * torch.pi * dx, device=self.device)),
                torch.sin(torch.tensor(freq * torch.pi * dy, device=self.device)),
                torch.cos(torch.tensor(freq * torch.pi * dy, device=self.device)),
            ])
        return torch.stack(features)

    def create_synthetic_observation(self, social_values: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Create a synthetic observation with specified social history.

        Token format (25 features):
            - fourier[16]: 4 freq bands × (sin_dx, cos_dx, sin_dy, cos_dy)
            - velocity[2]: (dv_x, dv_y)
            - type_onehot[2]: [is_food, is_agent]
            - value[1]: HP or quality
            - social[4]: [damage_dealt, food_given, coop_count, defense_score]
        """
        n_agents = self.runner._checkpoint_n_agents
        grid_size = 15
        max_entities = n_agents + 16  # agents + food tokens
        token_dim = 25

        # Create entity tokens [1, max_entities, 25]
        entity_tokens = torch.zeros(1, max_entities, token_dim, device=self.device)

        # Self token at index 0 (relative pos = 0, type = agent, hp = 1.0)
        self_fourier = self._fourier_encode(0.0, 0.0)  # [16]
        entity_tokens[0, 0, 0:16] = self_fourier       # fourier[16]
        entity_tokens[0, 0, 16:18] = 0.0               # velocity[2]
        entity_tokens[0, 0, 18:20] = torch.tensor([0.0, 1.0], device=self.device)  # type: [is_food, is_agent]
        entity_tokens[0, 0, 20] = 1.0                  # HP (full)
        entity_tokens[0, 0, 21:25] = 0.0               # social[4]

        # Other agent at index 1 (to the right, distance 1)
        dx_norm = 0.0 / grid_size
        dy_norm = 1.0 / grid_size  # RIGHT direction, normalized
        other_fourier = self._fourier_encode(dx_norm, dy_norm)  # [16]
        entity_tokens[0, 1, 0:16] = other_fourier      # fourier[16]
        entity_tokens[0, 1, 16:18] = 0.0               # velocity[2]
        entity_tokens[0, 1, 18:20] = torch.tensor([0.0, 1.0], device=self.device)  # type: [is_food, is_agent]
        entity_tokens[0, 1, 20] = 1.0                  # HP (full)

        # Social features (indices 21-24), scaled by 5.0 like in environment
        entity_tokens[0, 1, 21] = social_values.get('damage_dealt', 0) * 5.0
        entity_tokens[0, 1, 22] = social_values.get('food_given', 0) * 5.0
        entity_tokens[0, 1, 23] = social_values.get('coop_count', 0) * 5.0
        entity_tokens[0, 1, 24] = social_values.get('defense_score', 0) * 5.0

        # Entity mask [1, max_entities]
        entity_mask = torch.zeros(1, max_entities, dtype=torch.bool, device=self.device)
        entity_mask[0, 0] = True  # Self
        entity_mask[0, 1] = True  # Other agent

        # Signals [1, n_agents]
        signals = torch.zeros(1, n_agents, device=self.device)

        # Self state
        self_hp = torch.ones(1, 1, device=self.device)
        self_inventory = torch.zeros(1, 1, device=self.device)

        # Agent ID
        agent_id = torch.zeros(1, dtype=torch.long, device=self.device)

        return {
            'entity_tokens': entity_tokens,
            'entity_mask': entity_mask,
            'signals': signals,
            'self_hp': self_hp,
            'self_inventory': self_inventory,
            'agent_id': agent_id,
        }

    def extract_features(self, obs: Dict[str, torch.Tensor], network: ActorCritic) -> torch.Tensor:
        """Extract encoder features from observation."""
        with torch.no_grad():
            features = network.encoder(
                obs['entity_tokens'],
                obs['entity_mask'],
                obs['signals'],
                obs['self_hp'],
                obs['self_inventory'],
                obs['agent_id']
            )
        return features  # [1, 144]

    def analyze(self) -> FeatureAnalysisResult:
        """Run feature distinctness analysis."""
        # Load model if not already loaded
        if self.runner.multi_agent is None:
            self.runner.load_model(n_agents_for_test=2)

        network = self.runner.multi_agent.networks[0]
        network.eval()

        # Collect features for each scenario
        scenario_names = list(self.SCENARIOS.keys())
        all_features = []

        for name in scenario_names:
            social_values = self.SCENARIOS[name]
            obs = self.create_synthetic_observation(social_values)
            features = self.extract_features(obs, network)
            all_features.append(features)

        # Stack features [n_scenarios, 144]
        all_features = torch.cat(all_features, dim=0)

        # Normalize to unit length for cosine similarity
        all_features_norm = F.normalize(all_features, dim=1)

        # Compute cosine similarity matrix
        similarity_matrix = (all_features_norm @ all_features_norm.T).cpu().numpy()

        # Extract upper triangle (excluding diagonal)
        n = len(scenario_names)
        upper_tri_indices = np.triu_indices(n, k=1)
        off_diagonal = similarity_matrix[upper_tri_indices]

        avg_sim = np.mean(off_diagonal)
        min_sim = np.min(off_diagonal)
        max_sim = np.max(off_diagonal)

        # Features are distinct if average similarity < 0.95
        features_distinct = avg_sim < 0.95

        return FeatureAnalysisResult(
            scenario_names=scenario_names,
            similarity_matrix=similarity_matrix,
            avg_similarity=avg_sim,
            min_similarity=min_sim,
            max_similarity=max_sim,
            features_distinct=features_distinct,
        )


def format_feature_analysis(result: FeatureAnalysisResult) -> str:
    """Format feature analysis results as a readable report."""
    lines = []
    lines.append("")
    lines.append("=" * 65)
    lines.append("FEATURE DISTINCTNESS ANALYSIS")
    lines.append("=" * 65)
    lines.append("")

    # Similarity matrix
    lines.append("Cosine Similarity Matrix:")
    lines.append("")

    # Header row
    short_names = [n[:8] for n in result.scenario_names]
    header = "         " + " ".join(f"{n:>8}" for n in short_names)
    lines.append(header)

    # Matrix rows
    for i, name in enumerate(short_names):
        row_values = " ".join(f"{result.similarity_matrix[i, j]:8.3f}" for j in range(len(short_names)))
        lines.append(f"{name:>8} {row_values}")

    lines.append("")

    # Summary stats
    lines.append(f"Average pairwise similarity: {result.avg_similarity:.3f}")
    lines.append(f"Min similarity: {result.min_similarity:.3f}")
    lines.append(f"Max similarity: {result.max_similarity:.3f}")
    lines.append("")

    # Interpretation
    if result.avg_similarity > 0.99:
        interpretation = "POOR - Features nearly identical (network not using social info)"
    elif result.avg_similarity > 0.95:
        interpretation = "WEAK - Minimal feature variation across scenarios"
    elif result.avg_similarity > 0.80:
        interpretation = "MODERATE - Some discrimination between scenarios"
    else:
        interpretation = "GOOD - Features are distinct across social contexts"

    status = "PASS" if result.features_distinct else "FAIL"
    lines.append(f"Interpretation: {interpretation}")
    lines.append(f"Features Distinct: {status} (threshold: avg_sim < 0.95)")
    lines.append("=" * 65)

    return "\n".join(lines)


# =============================================================================
# Report Generation
# =============================================================================

def format_report(results: List[ScenarioResult]) -> str:
    """Format test results as a readable report."""
    lines = []
    lines.append("=" * 65)
    lines.append("SOCIAL ABILITIES TEST REPORT")
    lines.append("=" * 65)
    lines.append("")

    passed_count = 0
    total_count = len(results)

    for i, r in enumerate(results, 1):
        status = "PASS" if r.passed else "FAIL"
        if r.passed:
            passed_count += 1

        lines.append(f"SCENARIO {i}: {r.name}")
        lines.append(f"  {r.description}")
        lines.append(f"  Primary metric:  {r.mean_primary:.3f} +/- {r.std_primary:.3f}")
        lines.append(f"  Baseline metric: {r.mean_baseline:.3f} +/- {r.std_baseline:.3f}")

        if r.name != "Neutral Baseline":
            # Format ratio
            if np.isinf(r.ratio):
                ratio_str = "inf (baseline=0)"
            else:
                ratio_str = f"{r.ratio:.2f}x"

            # Format p-value
            if np.isnan(r.p_value):
                p_str = "N/A (no variance)"
            elif r.p_value < 0.001:
                p_str = "<0.001"
            else:
                p_str = f"{r.p_value:.3f}"

            lines.append(f"  Ratio: {ratio_str} | p-value: {p_str}")

        lines.append(f"  RESULT: {status}")
        lines.append("")

    lines.append("-" * 65)
    lines.append(f"SUMMARY: {passed_count}/{total_count} scenarios passed")
    lines.append("=" * 65)

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test social abilities of trained agents")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per scenario")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer episodes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--features-only", action="store_true", help="Only run feature analysis")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature analysis")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Adjust episodes for quick mode
    num_episodes = 10 if args.quick else args.episodes

    # Setup
    device = get_device()
    config = Config()

    print(f"Loading model from: {args.model}")
    print(f"Device: {device}")
    print()

    # Create runner
    runner = SocialTestRunner(config, device, args.model)

    # Run feature distinctness analysis first
    feature_result = None
    if not args.skip_features:
        print("Running Feature Distinctness Analysis...")
        analyzer = FeatureDistinctnessAnalyzer(runner)
        feature_result = analyzer.analyze()
        print(format_feature_analysis(feature_result))

        if args.features_only:
            return 0 if feature_result.features_distinct else 1

    # Run behavioral scenarios
    print(f"\nRunning {num_episodes} episodes per behavioral scenario...")
    results = []

    print("Running Scenario 1: Enemy Recognition...")
    results.append(runner.test_enemy_recognition(num_episodes))

    print("Running Scenario 2: Ally Recognition...")
    results.append(runner.test_ally_recognition(num_episodes))

    print("Running Scenario 3: Enemy vs Ally Discrimination...")
    results.append(runner.test_discrimination(num_episodes))

    print("Running Scenario 4: Cooperation with Cooperators...")
    results.append(runner.test_cooperation_preference(num_episodes))

    print("Running Scenario 5: Defense Behavior...")
    results.append(runner.test_defense_behavior(num_episodes))

    print("Running Scenario 6: Retaliation...")
    results.append(runner.test_retaliation(num_episodes))

    print("Running Scenario 7: Neutral Baseline...")
    results.append(runner.test_neutral_baseline(num_episodes))

    print()

    # Generate and print report
    report = format_report(results)
    print(report)

    # Return exit code based on pass rate
    pass_rate = sum(1 for r in results if r.passed) / len(results)
    return 0 if pass_rate >= 0.5 else 1


if __name__ == "__main__":
    exit(main())
