"""
Neural Network architecture for Multi-Agent RL.

ObservationEncoder: Encodes multi-modal observations into feature vector
ActorCritic: Dual-head policy (move + interact) with value function
"""
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import Config


class ObservationEncoder(nn.Module):
    """
    Encodes multi-modal observations into a flat feature vector.

    Inputs:
        - spatial: [batch, vision_size, vision_size, 5] - local grid view
        - ledger: [batch, n_agents, n_agents, 4] - full interaction history
        - signals: [batch, n_agents] - who is signaling
        - self_hp: [batch, 1] - agent's own HP (normalized)

    Output:
        - features: [batch, output_dim]
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Spatial encoder (flatten NxNx5 and MLP)
        spatial_input_dim = config.vision_size * config.vision_size * config.vision_channels
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Ledger encoder (flatten NxNx4 and MLP)
        ledger_input_dim = config.n_agents * config.n_agents * config.ledger_channels
        self.ledger_encoder = nn.Sequential(
            nn.Linear(ledger_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Signal encoder
        self.signal_encoder = nn.Sequential(
            nn.Linear(config.n_agents, 16),
            nn.ReLU()
        )

        # Self state encoder
        self.self_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU()
        )

        # Total output dimension: 64 + 64 + 16 + 8 = 152
        self.output_dim = 64 + 64 + 16 + 8

    def forward(self, spatial: torch.Tensor, ledger: torch.Tensor,
                signals: torch.Tensor, self_hp: torch.Tensor) -> torch.Tensor:
        """Encode all observation components into unified representation."""
        batch_size = spatial.shape[0]

        # Flatten and encode spatial
        spatial_flat = spatial.reshape(batch_size, -1)
        spatial_features = self.spatial_encoder(spatial_flat)

        # Flatten and encode ledger
        ledger_flat = ledger.reshape(batch_size, -1)
        ledger_features = self.ledger_encoder(ledger_flat)

        # Encode signals
        signal_features = self.signal_encoder(signals)

        # Encode self state
        self_features = self.self_encoder(self_hp)

        # Concatenate all features
        combined = torch.cat([
            spatial_features,
            ledger_features,
            signal_features,
            self_features
        ], dim=1)

        return combined


class ActorCritic(nn.Module):
    """
    Dual-head Actor-Critic network for multi-action PPO.

    Outputs:
        - move_logits: [batch, 5] for [Up, Down, Left, Right, Stay]
        - interact_logits: [batch, 10] for [Attack x4 dirs, Give x4 dirs, Signal, Idle]
        - value: [batch, 1]
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )

        # Actor heads
        self.move_head = nn.Linear(128, config.n_move_actions)      # 5 movement actions
        self.interact_head = nn.Linear(128, config.n_interact_actions)  # 10 interaction actions

        # Critic head
        self.value_head = nn.Linear(128, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

        # Smaller initialization for policy heads
        nn.init.orthogonal_(self.move_head.weight, gain=0.01)
        nn.init.orthogonal_(self.interact_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def _encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations through encoder and shared trunk."""
        features = self.encoder(
            obs['spatial'],
            obs['ledger'],
            obs['signals'],
            obs['self_hp']
        )
        return self.shared(features)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning raw logits and value.

        Returns:
            move_logits: [batch, n_move_actions]
            interact_logits: [batch, n_interact_actions]
            value: [batch, 1]
        """
        hidden = self._encode(obs)
        move_logits = self.move_head(hidden)
        interact_logits = self.interact_head(hidden)
        value = self.value_head(hidden)
        return move_logits, interact_logits, value

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return value estimate only."""
        hidden = self._encode(obs)
        return self.value_head(hidden)

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        move_action: Optional[torch.Tensor] = None,
        interact_action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions, log probabilities, entropy, and value.

        Args:
            obs: Observation dictionary
            move_action: Optional pre-selected move action (for computing log prob)
            interact_action: Optional pre-selected interact action

        Returns:
            move_action, interact_action, log_prob (combined), entropy (combined), value
        """
        hidden = self._encode(obs)

        # Get logits
        move_logits = self.move_head(hidden)
        interact_logits = self.interact_head(hidden)
        value = self.value_head(hidden)

        # Create distributions
        move_dist = Categorical(logits=move_logits)
        interact_dist = Categorical(logits=interact_logits)

        # Sample or use provided actions
        if move_action is None:
            move_action = move_dist.sample()
        if interact_action is None:
            interact_action = interact_dist.sample()

        # Combined log probability (sum of independent log probs)
        log_prob = move_dist.log_prob(move_action) + interact_dist.log_prob(interact_action)

        # Combined entropy
        entropy = move_dist.entropy() + interact_dist.entropy()

        return move_action, interact_action, log_prob, entropy, value.squeeze(-1)
