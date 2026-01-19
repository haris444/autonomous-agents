"""
Neural Network architecture for Multi-Agent RL.

ObservationEncoder: Unified Transformer over all entities (agents + food)
ActorCritic: Factorized action heads (move, interact_type, direction) with value function
"""
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import Config


class ObservationEncoder(nn.Module):
    """
    Encodes observations using a unified Transformer over all entities.

    Architecture:
        - Entity Tokens: Single Transformer over agents + food (explicit dx, dy)
        - Signals: MLP
        - Self HP: MLP

    Inputs:
        - entity_tokens: [batch, max_entities, 8] - (dx, dy, type, value, social[4])
        - entity_mask: [batch, max_entities] - which tokens are valid
        - signals: [batch, n_agents] - who is signaling
        - self_hp: [batch, 1] - agent's own HP (normalized)
        - self_inventory: [batch, 1] - agent's food inventory (normalized)
        - agent_id: [batch] - which agent this observation belongs to

    Output:
        - features: [batch, output_dim]
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 1. ENTITY TOKENS: Unified Transformer over agents + food
        # Each token has 8 features: dx, dy, type, value, social[4]
        self.entity_embed = nn.Linear(config.entity_token_dim, config.attention_embed_dim)
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=config.attention_embed_dim,
            num_heads=config.attention_num_heads,
            batch_first=True
        )
        self.entity_norm = nn.LayerNorm(config.attention_embed_dim)
        entity_output_dim = config.attention_embed_dim  # 64

        # 2. SIGNAL encoder
        self.signal_encoder = nn.Sequential(
            nn.Linear(config.n_agents, 16),
            nn.ReLU()
        )

        # 3. SELF state encoder (HP + inventory = 2 inputs)
        self.self_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )

        # Total output dimension: 64 (entities) + 16 (signals) + 64 (self) = 144
        self.output_dim = entity_output_dim + 16 + 64

    def forward(self, entity_tokens: torch.Tensor, entity_mask: torch.Tensor,
                signals: torch.Tensor, self_hp: torch.Tensor,
                self_inventory: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        """Encode all observation components into unified representation."""
        batch_size = entity_tokens.shape[0]
        device = entity_tokens.device

        # 1. ENTITY TOKENS: Extract this agent's view
        batch_indices = torch.arange(batch_size, device=device)

        # Handle both batched formats
        if entity_tokens.dim() == 4:
            # [B, n_agents, max_entities, 8] - need to select observer's row
            my_tokens = entity_tokens[batch_indices, agent_id.long(), :, :]  # [B, max_entities, 8]
            my_mask = entity_mask[batch_indices, agent_id.long(), :]  # [B, max_entities]
        else:
            # [B, max_entities, 8] - already per-observer
            my_tokens = entity_tokens
            my_mask = entity_mask

        # Embed entity tokens
        tokens = self.entity_embed(my_tokens)  # [B, max_entities, embed_dim]

        # Create attention mask (True = ignore)
        attn_mask = ~my_mask  # [B, max_entities]

        # Self-attention: "Which entities should I pay attention to?"
        attn_out, _ = self.entity_attention(
            tokens, tokens, tokens,
            key_padding_mask=attn_mask
        )
        attn_out = self.entity_norm(attn_out + tokens)  # Residual + LayerNorm

        # Masked mean pooling (exclude self token - it's always at origin, adds no spatial info)
        # Self token is at index 0 (first agent slot)
        pool_mask = my_mask.clone()
        pool_mask[:, 0] = False  # Exclude self from pooling

        mask_expanded = pool_mask.unsqueeze(-1).float()  # [B, max_entities, 1]
        masked_sum = (attn_out * mask_expanded).sum(dim=1)  # [B, embed_dim]
        mask_count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        entity_features = masked_sum / mask_count  # [B, embed_dim]

        # 2. SIGNALS
        signal_features = self.signal_encoder(signals)  # [B, 16]

        # 3. SELF state (HP + inventory)
        self_state = torch.cat([self_hp, self_inventory], dim=-1)  # [B, 2]
        self_features = self.self_encoder(self_state)  # [B, 64]

        # Concatenate all features
        combined = torch.cat([
            entity_features,
            signal_features,
            self_features
        ], dim=1)

        return combined


class ActorCritic(nn.Module):
    """
    Actor-Critic network with unified action head.

    Action space (15 total):
        - 0-4: Movement (UP, DOWN, LEFT, RIGHT, STAY)
        - 5-8: ATTACK (UP, DOWN, LEFT, RIGHT)
        - 9-12: GIVE (UP, DOWN, LEFT, RIGHT)
        - 13: SIGNAL
        - 14: COOPERATE

    Agent chooses ONE action per step (move OR interact, not both).
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)

        # Shared trunk (ReLU to avoid Tanh saturation)
        self.shared = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Unified actor head (15 actions)
        self.action_head = nn.Linear(128, config.n_actions)

        # Critic head
        self.value_head = nn.Linear(128, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        import math
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use sqrt(2) gain for ReLU layers (He initialization equivalent)
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Smaller initialization for policy head (helps with exploration)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        # Value head uses gain=1.0 (outputs should be in reasonable range initially)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def _encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations through encoder and shared trunk."""
        features = self.encoder(
            obs['entity_tokens'],
            obs['entity_mask'],
            obs['signals'],
            obs['self_hp'],
            obs['self_inventory'],
            obs['agent_id']
        )
        return self.shared(features)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning raw logits and value.

        Returns:
            action_logits: [batch, n_actions] (15 actions)
            value: [batch, 1]
        """
        hidden = self._encode(obs)
        action_logits = self.action_head(hidden)
        value = self.value_head(hidden)
        return action_logits, value

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return value estimate only."""
        hidden = self._encode(obs)
        return self.value_head(hidden)

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            obs: Observation dictionary
            action: Optional pre-selected action (for computing log prob during PPO update)
            action_mask: Optional [batch, 15] mask of valid actions

        Returns:
            action: [batch] selected action (0-14)
            log_prob: [batch] log probability of action
            entropy: [batch] policy entropy
            value: [batch] value estimate
        """
        hidden = self._encode(obs)

        # Get logits for unified action head
        action_logits = self.action_head(hidden)
        value = self.value_head(hidden)

        # Apply action mask if provided
        if action_mask is not None:
            LARGE_NEG = -1e8
            action_logits = action_logits.masked_fill(~action_mask, LARGE_NEG)

        # Create distribution
        action_dist = Categorical(logits=action_logits)

        # Sample or use provided action
        if action is None:
            action = action_dist.sample()

        # Compute log probability and entropy
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)
