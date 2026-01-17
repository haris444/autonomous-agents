"""
Global Ledger - tracks agent-to-agent interaction history.

Structure: [N_agents × N_agents × 4]
- Axis X (Source): The actor who performed the action
- Axis Y (Target): The subject who received the action
- Axis Z (Channels): 4 objective facts about their history

Key principle: Store objective ACTIONS, not subjective relationships.
Agents derive feelings from facts.
"""
import torch


class Ledger:
    """
    Global interaction tensor tracking agent-to-agent history.
    """
    # Channel indices
    DAMAGE_DEALT = 0   # Total HP removed by source from target
    FOOD_GIVEN = 1     # Total food transferred source -> target
    COOP_COUNT = 2     # Times they cooperatively unlocked rich food
    DEFENSE_SCORE = 3  # Damage source dealt defending target

    def __init__(self, n_agents: int, device: torch.device):
        self.n_agents = n_agents
        self.device = device
        self.tensor = torch.zeros((n_agents, n_agents, 4), device=device)

    def reset(self) -> None:
        """Clear all interaction history."""
        self.tensor.zero_()

    def record_damage(self, attacker: int, target: int, amount: float) -> None:
        """Record damage dealt from attacker to target."""
        self.tensor[attacker, target, self.DAMAGE_DEALT] += amount

    def record_food_given(self, giver: int, receiver: int, amount: float) -> None:
        """Record food transfer from giver to receiver."""
        self.tensor[giver, receiver, self.FOOD_GIVEN] += amount

    def record_cooperation(self, agent_a: int, agent_b: int) -> None:
        """Record mutual cooperation on rich food (symmetric update)."""
        self.tensor[agent_a, agent_b, self.COOP_COUNT] += 1
        self.tensor[agent_b, agent_a, self.COOP_COUNT] += 1

    def record_defense(self, defender: int, defended: int, damage_dealt: float) -> None:
        """Record defender protecting defended by dealing damage to attacker."""
        self.tensor[defender, defended, self.DEFENSE_SCORE] += damage_dealt

    def get_tensor(self) -> torch.Tensor:
        """Return the full ledger tensor for observations."""
        return self.tensor.clone()

    def get_normalized_tensor(self, max_damage: float = 100.0, max_food: float = 100.0,
                               max_coop: float = 10.0, max_defense: float = 100.0) -> torch.Tensor:
        """Return normalized ledger tensor (values in [0, 1] range)."""
        normalized = self.tensor.clone()
        normalized[:, :, self.DAMAGE_DEALT] /= max_damage
        normalized[:, :, self.FOOD_GIVEN] /= max_food
        normalized[:, :, self.COOP_COUNT] /= max_coop
        normalized[:, :, self.DEFENSE_SCORE] /= max_defense
        return normalized.clamp(0, 1)
