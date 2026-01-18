"""
Neural Network architecture for Multi-Agent RL.

ObservationEncoder: MLP for spatial + Social Transformer for ledger
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
    Encodes multi-modal observations into a flat feature vector.

    Architecture:
        - Spatial: Flatten + MLP (position-aware, no CNN)
        - Ledger: Social Transformer over agents (permutation equivariant)
        - Signals: MLP
        - Self HP: MLP

    Inputs:
        - spatial: [batch, vision_size, vision_size, channels] - local grid view
        - ledger: [batch, n_agents, n_agents, 4] - full interaction history
        - signals: [batch, n_agents] - who is signaling
        - self_hp: [batch, 1] - agent's own HP (normalized)
        - agent_id: [batch] - which agent this observation belongs to

    Output:
        - features: [batch, output_dim]
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 1. SPATIAL: Flatten + MLP (not CNN - respects ego-centric positioning)
        spatial_input_dim = config.vision_size * config.vision_size * config.vision_channels
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_input_dim, 128),
            nn.ReLU()
        )
        spatial_output_dim = 128

        # 2. SOCIAL: Transformer over agents
        # Each agent is a token with 4 features (damage_dealt, food_given, coop_count, defense_score)
        self.agent_embed = nn.Linear(config.ledger_channels, config.attention_embed_dim)
        self.social_attention = nn.MultiheadAttention(
            embed_dim=config.attention_embed_dim,
            num_heads=config.attention_num_heads,
            batch_first=True
        )
        self.social_norm = nn.LayerNorm(config.attention_embed_dim)
        social_output_dim = config.attention_embed_dim  # 64

        # 3. SIGNAL encoder
        self.signal_encoder = nn.Sequential(
            nn.Linear(config.n_agents, 16),
            nn.ReLU()
        )

        # 4. SELF state encoder
        self.self_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU()
        )

        # Total output dimension: 128 (spatial) + 64 (social) + 16 (signals) + 8 (self) = 216
        self.output_dim = spatial_output_dim + social_output_dim + 16 + 8

    def forward(self, spatial: torch.Tensor, ledger: torch.Tensor,
                signals: torch.Tensor, self_hp: torch.Tensor,
                agent_id: torch.Tensor) -> torch.Tensor:
        """Encode all observation components into unified representation."""
        batch_size = spatial.shape[0]
        device = spatial.device

        # 1. SPATIAL: Flatten + MLP
        spatial_flat = spatial.reshape(batch_size, -1)
        spatial_features = self.spatial_encoder(spatial_flat)  # [B, 128]

        # 2. SOCIAL: Extract this agent's row from ledger, then transformer
        # ledger: [B, n_agents, n_agents, 4]
        # agent_id: [B] - which agent each observation belongs to
        # We want ledger[b, agent_id[b], :, :] for each b -> [B, n_agents, 4]
        batch_indices = torch.arange(batch_size, device=device)
        my_ledger_row = ledger[batch_indices, agent_id.long(), :, :]  # [B, n_agents, 4]

        # Embed each agent's interaction history as a token
        tokens = self.agent_embed(my_ledger_row)  # [B, n_agents, embed_dim]

        # Self-attention over agents: "Which agent should I pay attention to?"
        attn_out, _ = self.social_attention(tokens, tokens, tokens)
        attn_out = self.social_norm(attn_out + tokens)  # Residual + LayerNorm

        # Pool over agents (mean pooling)
        social_features = attn_out.mean(dim=1)  # [B, embed_dim]

        # 3. SIGNALS
        signal_features = self.signal_encoder(signals)  # [B, 16]

        # 4. SELF HP
        self_features = self.self_encoder(self_hp)  # [B, 8]

        # Concatenate all features
        combined = torch.cat([
            spatial_features,
            social_features,
            signal_features,
            self_features
        ], dim=1)

        return combined


class ActorCritic(nn.Module):
    """
    Actor-Critic network with factorized action heads.

    Outputs:
        - move_action: [batch] from 9 options (8 dirs + stay)
        - interact_action: [batch] converted to env format (0-18)
        - log_prob: combined log probability
        - entropy: combined entropy
        - value: [batch]

    Factorized action heads:
        - Move: 9 actions (8 directions + stay)
        - Interact Type: 5 actions (attack, give, signal, cooperate, idle)
        - Direction: 8 actions (for attack/give only)
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

        # Factorized actor heads
        self.move_head = nn.Linear(128, config.n_move_actions)  # 9 movement actions
        self.interact_type_head = nn.Linear(128, config.n_interact_types)  # 5 types
        self.direction_head = nn.Linear(128, config.n_directions)  # 8 directions

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

        # Smaller initialization for policy heads (helps with exploration)
        nn.init.orthogonal_(self.move_head.weight, gain=0.01)
        nn.init.orthogonal_(self.interact_type_head.weight, gain=0.01)
        nn.init.orthogonal_(self.direction_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def _encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations through encoder and shared trunk."""
        features = self.encoder(
            obs['spatial'],
            obs['ledger'],
            obs['signals'],
            obs['self_hp'],
            obs['agent_id']
        )
        return self.shared(features)

    def _factorized_to_env_action(self, interact_type: torch.Tensor,
                                   direction: torch.Tensor) -> torch.Tensor:
        """
        Convert factorized (type, direction) to environment's single action index.

        Mapping:
            type=0 (attack): action = direction (0-7)
            type=1 (give): action = 8 + direction (8-15)
            type=2 (signal): action = 16
            type=3 (cooperate): action = 17
            type=4 (idle): action = 18
        """
        return torch.where(interact_type == 0, direction,
               torch.where(interact_type == 1, 8 + direction,
               torch.where(interact_type == 2, torch.tensor(16, device=interact_type.device),
               torch.where(interact_type == 3, torch.tensor(17, device=interact_type.device),
                           torch.tensor(18, device=interact_type.device)))))

    def _env_action_to_factorized(self, interact_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert environment's single action index to factorized (type, direction).

        Inverse mapping:
            0-7: type=0 (attack), direction=action
            8-15: type=1 (give), direction=action-8
            16: type=2 (signal), direction=0 (unused)
            17: type=3 (cooperate), direction=0 (unused)
            18: type=4 (idle), direction=0 (unused)
        """
        interact_type = torch.where(interact_action <= 7, torch.tensor(0, device=interact_action.device),
                        torch.where(interact_action <= 15, torch.tensor(1, device=interact_action.device),
                        torch.where(interact_action == 16, torch.tensor(2, device=interact_action.device),
                        torch.where(interact_action == 17, torch.tensor(3, device=interact_action.device),
                                    torch.tensor(4, device=interact_action.device)))))

        direction = torch.where(interact_action <= 7, interact_action,
                    torch.where(interact_action <= 15, interact_action - 8,
                                torch.zeros_like(interact_action)))

        return interact_type, direction

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning raw logits and value.

        Returns:
            move_logits: [batch, n_move_actions]
            interact_type_logits: [batch, n_interact_types]
            direction_logits: [batch, n_directions]
            value: [batch, 1]
        """
        hidden = self._encode(obs)
        move_logits = self.move_head(hidden)
        interact_type_logits = self.interact_type_head(hidden)
        direction_logits = self.direction_head(hidden)
        value = self.value_head(hidden)
        return move_logits, interact_type_logits, direction_logits, value

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return value estimate only."""
        hidden = self._encode(obs)
        return self.value_head(hidden)

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        move_action: Optional[torch.Tensor] = None,
        interact_action: Optional[torch.Tensor] = None,
        action_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions, log probabilities, entropy, and value.

        Args:
            obs: Observation dictionary
            move_action: Optional pre-selected move action (for computing log prob)
            interact_action: Optional pre-selected interact action (in env format 0-18)
            action_masks: Optional dict with 'move_mask', 'interact_type_mask', 'direction_mask'

        Returns:
            move_action, interact_action (env format), log_prob (combined), entropy (combined), value
        """
        hidden = self._encode(obs)

        # Get logits for all heads
        move_logits = self.move_head(hidden)
        interact_type_logits = self.interact_type_head(hidden)
        direction_logits = self.direction_head(hidden)
        value = self.value_head(hidden)

        # Apply action masks if provided
        if action_masks is not None:
            LARGE_NEG = -1e8

            # Move mask: [batch, 9]
            move_mask = action_masks['move_mask']
            move_logits = move_logits.masked_fill(~move_mask, LARGE_NEG)

            # Direction mask: [batch, 8]
            dir_mask = action_masks['direction_mask']

            # Check if ANY direction has a valid target
            has_any_target = dir_mask.any(dim=1)  # [batch]

            # Interact type mask: [batch, 5]
            # Mask ATTACK (0) and GIVE (1) when no valid direction targets exist
            type_mask = action_masks['interact_type_mask'].clone()
            type_mask[:, 0] = type_mask[:, 0] & has_any_target  # ATTACK only if targets exist
            type_mask[:, 1] = type_mask[:, 1] & has_any_target  # GIVE only if targets exist
            interact_type_logits = interact_type_logits.masked_fill(~type_mask, LARGE_NEG)

            # Direction mask - apply directly (no fallback needed now)
            # If agent can't select ATTACK/GIVE, direction doesn't matter
            direction_logits = direction_logits.masked_fill(~dir_mask, LARGE_NEG)

        # Create distributions
        move_dist = Categorical(logits=move_logits)
        type_dist = Categorical(logits=interact_type_logits)
        dir_dist = Categorical(logits=direction_logits)

        # Sample or use provided actions
        if move_action is None:
            move_action = move_dist.sample()

        if interact_action is None:
            # Sample factorized actions
            interact_type = type_dist.sample()
            direction = dir_dist.sample()
            # Convert to env format
            interact_action = self._factorized_to_env_action(interact_type, direction)
        else:
            # Convert provided env action to factorized form for log prob computation
            interact_type, direction = self._env_action_to_factorized(interact_action)

        # Compute log probabilities
        move_log_prob = move_dist.log_prob(move_action)
        type_log_prob = type_dist.log_prob(interact_type)

        # Direction log prob only contributes for attack (type=0) and give (type=1)
        needs_direction = (interact_type <= 1)
        dir_log_prob = torch.where(
            needs_direction,
            dir_dist.log_prob(direction),
            torch.zeros_like(move_log_prob)
        )

        # Combined log probability
        log_prob = move_log_prob + type_log_prob + dir_log_prob

        # Compute entropy
        move_entropy = move_dist.entropy()
        type_entropy = type_dist.entropy()
        # Direction entropy only contributes when direction matters
        dir_entropy = torch.where(
            needs_direction,
            dir_dist.entropy(),
            torch.zeros_like(move_entropy)
        )

        entropy = move_entropy + type_entropy + dir_entropy

        return move_action, interact_action, log_prob, entropy, value.squeeze(-1)
