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

from core.config import Config


class ObservationEncoder(nn.Module):
    """
    Encodes observations using a unified Transformer over all entities.

    Architecture:
        - Entity Tokens: Per-group projection (fourier, velocity, type, value, social
          each → group_embed_dim) then concat → Transformer
        - Signals: MLP
        - Self HP: MLP

    Each feature group is projected to the same dimensionality before concatenation,
    so the number of Fourier bands doesn't affect the relative influence of position
    vs social/type features.

    Output:
        - features: [batch, output_dim]
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.fourier_dim = config.fourier_bands * 4

        # 1. ENTITY TOKENS: Per-group projections to standardize influence
        # Each group gets projected to group_embed_dim, then concatenated and projected to attention_embed_dim
        g = config.group_embed_dim
        self.fourier_embed = nn.Linear(self.fourier_dim, g)
        self.velocity_embed = nn.Linear(2, g)
        self.type_embed = nn.Linear(2, g)
        self.value_embed = nn.Linear(1, g)
        self.social_embed = nn.Linear(4, g)
        self.entity_embed = nn.Linear(5 * g, config.attention_embed_dim)
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
            # [B, n_agents, max_entities, 25] - need to select observer's row
            my_tokens = entity_tokens[batch_indices, agent_id.long(), :, :]  # [B, max_entities, 25]
            my_mask = entity_mask[batch_indices, agent_id.long(), :]  # [B, max_entities]
        else:
            # [B, max_entities, 25] - already per-observer
            my_tokens = entity_tokens
            my_mask = entity_mask

        # Split token into feature groups and project each to equal size
        f = self.fourier_dim
        fourier_proj = self.fourier_embed(my_tokens[..., :f])
        velocity_proj = self.velocity_embed(my_tokens[..., f:f+2])
        type_proj = self.type_embed(my_tokens[..., f+2:f+4])
        value_proj = self.value_embed(my_tokens[..., f+4:f+5])
        social_proj = self.social_embed(my_tokens[..., f+5:f+9])
        grouped = torch.cat([fourier_proj, velocity_proj, type_proj, value_proj, social_proj], dim=-1)
        tokens = self.entity_embed(grouped)  # [B, max_entities, embed_dim]

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
    Actor-Critic network with factored action space.

    Direction head (5 outputs): UP=0, DOWN=1, LEFT=2, RIGHT=3, STAY=4
    Action type head (5 outputs): MOVE=0, ATTACK=1, GIVE=2, SIGNAL=3, COOPERATE=4

    Direction is ignored for SIGNAL and COOPERATE (non-directional actions).
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

        # Factored actor heads
        self.direction_head = nn.Linear(128, config.n_directions)
        self.action_type_head = nn.Linear(128, config.n_action_types)

        # Critic head
        self.value_head = nn.Linear(128, 1)

        # Auxiliary value heads for decomposed reward streams
        self.value_head_survival = nn.Linear(128, 1)
        self.value_head_resource = nn.Linear(128, 1)
        self.value_head_social = nn.Linear(128, 1)

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

        # Smaller initialization for policy heads (helps with exploration)
        nn.init.orthogonal_(self.direction_head.weight, gain=0.01)
        nn.init.orthogonal_(self.action_type_head.weight, gain=0.01)
        # Value head uses gain=1.0 (outputs should be in reasonable range initially)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

        # Auxiliary value heads also use gain=1.0
        nn.init.orthogonal_(self.value_head_survival.weight, gain=1.0)
        nn.init.orthogonal_(self.value_head_resource.weight, gain=1.0)
        nn.init.orthogonal_(self.value_head_social.weight, gain=1.0)
        nn.init.zeros_(self.value_head_survival.bias)
        nn.init.zeros_(self.value_head_resource.bias)
        nn.init.zeros_(self.value_head_social.bias)

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

    def forward(self, obs: Dict[str, torch.Tensor], return_aux: bool = False):
        """
        Forward pass returning raw logits and value.

        Args:
            obs: Observation dictionary
            return_aux: If True, also return auxiliary value head outputs

        Returns:
            direction_logits: [batch, n_directions] (5 directions)
            action_type_logits: [batch, n_action_types] (5 action types)
            value: [batch, 1]
            (if return_aux) value_survival: [batch, 1]
            (if return_aux) value_resource: [batch, 1]
            (if return_aux) value_social: [batch, 1]
        """
        hidden = self._encode(obs)
        direction_logits = self.direction_head(hidden)
        action_type_logits = self.action_type_head(hidden)
        value = self.value_head(hidden)
        if return_aux:
            return (direction_logits, action_type_logits, value,
                    self.value_head_survival(hidden),
                    self.value_head_resource(hidden),
                    self.value_head_social(hidden))
        return direction_logits, action_type_logits, value

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return value estimate only."""
        hidden = self._encode(obs)
        return self.value_head(hidden)

    def get_auxiliary_values(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return auxiliary value estimates for decomposed reward streams."""
        hidden = self._encode(obs)
        return {
            'survival': self.value_head_survival(hidden).squeeze(-1),
            'resource': self.value_head_resource(hidden).squeeze(-1),
            'social': self.value_head_social(hidden).squeeze(-1),
        }

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        direction: Optional[torch.Tensor] = None,
        action_type: Optional[torch.Tensor] = None,
        direction_mask: Optional[torch.Tensor] = None,
        action_type_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get direction, action type, log probability, entropy, and value.

        Samples from two independent Categorical distributions.
        log_prob = log_prob_direction + log_prob_action_type
        entropy = entropy_direction + entropy_action_type

        Args:
            obs: Observation dictionary
            direction: Optional pre-selected direction (for PPO update)
            action_type: Optional pre-selected action type (for PPO update)
            direction_mask: Optional [batch, 5] mask of valid directions
            action_type_mask: Optional [batch, 5] mask of valid action types

        Returns:
            direction: [batch] selected direction (0-4)
            action_type: [batch] selected action type (0-4)
            log_prob: [batch] combined log probability
            entropy: [batch] combined policy entropy
            value: [batch] value estimate
        """
        hidden = self._encode(obs)

        # Get logits for both heads
        direction_logits = self.direction_head(hidden)
        action_type_logits = self.action_type_head(hidden)
        value = self.value_head(hidden)

        LARGE_NEG = -1e8

        # Apply direction mask if provided
        if direction_mask is not None:
            direction_logits = direction_logits.masked_fill(~direction_mask, LARGE_NEG)

        # Apply action type mask if provided
        if action_type_mask is not None:
            action_type_logits = action_type_logits.masked_fill(~action_type_mask, LARGE_NEG)

        # Create distributions
        direction_dist = Categorical(logits=direction_logits)
        action_type_dist = Categorical(logits=action_type_logits)

        # Sample or use provided actions
        if direction is None:
            direction = direction_dist.sample()
        if action_type is None:
            action_type = action_type_dist.sample()

        # Compute combined log probability and entropy
        log_prob_dir = direction_dist.log_prob(direction)
        log_prob_act = action_type_dist.log_prob(action_type)
        log_prob = log_prob_dir + log_prob_act

        entropy_dir = direction_dist.entropy()
        entropy_act = action_type_dist.entropy()
        entropy = entropy_dir + entropy_act

        return direction, action_type, log_prob, entropy, value.squeeze(-1)
