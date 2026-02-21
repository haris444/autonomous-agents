"""
Neural Network architecture for Multi-Agent RL.

ObservationEncoder: Unified Transformer over all entities (agents + food)
SharedTrunkActorCritic: Shared encoder+trunk with per-agent heads
"""
import math
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


class SharedTrunkActorCritic(nn.Module):
    """
    Shared encoder+trunk with per-agent action/value heads.

    Architecture:
        ObservationEncoder (144-dim) -> Shared Trunk (144->256->128)
        -> Per-Agent Heads (N sets of 6 linear layers, stored as stacked parameters)

    Per-agent heads stored as stacked nn.Parameter tensors enabling
    batched matmul (einsum/bmm) for parallel forward across all agents.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_agents = config.n_agents

        # Shared encoder + trunk
        self.encoder = ObservationEncoder(config)
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.trunk_dim = 128

        N = config.n_agents
        D = self.trunk_dim
        n_dir = config.n_directions    # 5
        n_act = config.n_action_types  # 5

        # Per-agent head parameters: [N, out_dim, D] weights + [N, out_dim] biases
        # Policy heads (small init for exploration)
        self.head_dir_w = nn.Parameter(torch.empty(N, n_dir, D))
        self.head_dir_b = nn.Parameter(torch.empty(N, n_dir))
        self.head_act_w = nn.Parameter(torch.empty(N, n_act, D))
        self.head_act_b = nn.Parameter(torch.empty(N, n_act))
        # Value head
        self.head_val_w = nn.Parameter(torch.empty(N, 1, D))
        self.head_val_b = nn.Parameter(torch.empty(N, 1))
        # Auxiliary value heads
        self.head_val_surv_w = nn.Parameter(torch.empty(N, 1, D))
        self.head_val_surv_b = nn.Parameter(torch.empty(N, 1))
        self.head_val_res_w = nn.Parameter(torch.empty(N, 1, D))
        self.head_val_res_b = nn.Parameter(torch.empty(N, 1))
        self.head_val_soc_w = nn.Parameter(torch.empty(N, 1, D))
        self.head_val_soc_b = nn.Parameter(torch.empty(N, 1))

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights using orthogonal initialization."""
        # Shared encoder + trunk: sqrt(2) gain for ReLU
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Per-agent heads: init template, clone to all N agents
        self._init_head(self.head_dir_w, self.head_dir_b, gain=0.01)
        self._init_head(self.head_act_w, self.head_act_b, gain=0.01)
        self._init_head(self.head_val_w, self.head_val_b, gain=1.0)
        self._init_head(self.head_val_surv_w, self.head_val_surv_b, gain=1.0)
        self._init_head(self.head_val_res_w, self.head_val_res_b, gain=1.0)
        self._init_head(self.head_val_soc_w, self.head_val_soc_b, gain=1.0)

    def _init_head(self, weight, bias, gain):
        """Init one template with orthogonal init, clone to all N agents."""
        template_w = torch.empty(weight.shape[1], weight.shape[2])
        nn.init.orthogonal_(template_w, gain=gain)
        template_b = torch.zeros(bias.shape[1])
        with torch.no_grad():
            for i in range(self.n_agents):
                weight[i].copy_(template_w)
                bias[i].copy_(template_b)

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Shared encoder+trunk. Returns [B, 128]."""
        features = self.encoder(
            obs['entity_tokens'], obs['entity_mask'],
            obs['signals'], obs['self_hp'],
            obs['self_inventory'], obs['agent_id']
        )
        return self.trunk(features)

    def apply_heads(self, hidden, agent_indices, return_aux=False):
        """
        Apply per-agent heads to trunk output.

        Args:
            hidden: [B, 128] trunk output
            agent_indices: int (single agent) or [B] long tensor (mixed agents)
            return_aux: whether to return auxiliary value heads

        Returns:
            dir_logits [B, 5], act_logits [B, 5], value [B, 1]
            (+ v_surv, v_res, v_soc each [B, 1] if return_aux)
        """
        if isinstance(agent_indices, int):
            # Single agent index — F.linear (efficient for per-agent training)
            i = agent_indices
            dir_logits = F.linear(hidden, self.head_dir_w[i], self.head_dir_b[i])
            act_logits = F.linear(hidden, self.head_act_w[i], self.head_act_b[i])
            value = F.linear(hidden, self.head_val_w[i], self.head_val_b[i])
            if return_aux:
                v_surv = F.linear(hidden, self.head_val_surv_w[i], self.head_val_surv_b[i])
                v_res = F.linear(hidden, self.head_val_res_w[i], self.head_val_res_b[i])
                v_soc = F.linear(hidden, self.head_val_soc_w[i], self.head_val_soc_b[i])
                return dir_logits, act_logits, value, v_surv, v_res, v_soc
            return dir_logits, act_logits, value
        else:
            # Mixed agent indices — gather + bmm
            h = hidden.unsqueeze(-1)  # [B, 128, 1]
            dir_logits = torch.bmm(self.head_dir_w[agent_indices], h).squeeze(-1) + self.head_dir_b[agent_indices]
            act_logits = torch.bmm(self.head_act_w[agent_indices], h).squeeze(-1) + self.head_act_b[agent_indices]
            value = torch.bmm(self.head_val_w[agent_indices], h).squeeze(-1) + self.head_val_b[agent_indices]
            if return_aux:
                v_surv = torch.bmm(self.head_val_surv_w[agent_indices], h).squeeze(-1) + self.head_val_surv_b[agent_indices]
                v_res = torch.bmm(self.head_val_res_w[agent_indices], h).squeeze(-1) + self.head_val_res_b[agent_indices]
                v_soc = torch.bmm(self.head_val_soc_w[agent_indices], h).squeeze(-1) + self.head_val_soc_b[agent_indices]
                return dir_logits, act_logits, value, v_surv, v_res, v_soc
            return dir_logits, act_logits, value

    def apply_heads_parallel(self, hidden, n_active, return_aux=False):
        """
        Apply all agent heads in parallel via einsum.

        Args:
            hidden: [N, B, 128] per-agent hidden states
            n_active: number of active agents to process

        Returns shapes: [N, B, out_dim]
        """
        dir_logits = torch.einsum('noh,nbh->nbo', self.head_dir_w[:n_active], hidden) + self.head_dir_b[:n_active].unsqueeze(1)
        act_logits = torch.einsum('noh,nbh->nbo', self.head_act_w[:n_active], hidden) + self.head_act_b[:n_active].unsqueeze(1)
        value = torch.einsum('noh,nbh->nbo', self.head_val_w[:n_active], hidden) + self.head_val_b[:n_active].unsqueeze(1)

        if return_aux:
            v_surv = torch.einsum('noh,nbh->nbo', self.head_val_surv_w[:n_active], hidden) + self.head_val_surv_b[:n_active].unsqueeze(1)
            v_res = torch.einsum('noh,nbh->nbo', self.head_val_res_w[:n_active], hidden) + self.head_val_res_b[:n_active].unsqueeze(1)
            v_soc = torch.einsum('noh,nbh->nbo', self.head_val_soc_w[:n_active], hidden) + self.head_val_soc_b[:n_active].unsqueeze(1)
            return dir_logits, act_logits, value, v_surv, v_res, v_soc
        return dir_logits, act_logits, value

    def forward(self, obs, agent_indices, return_aux=False):
        """Full forward: encode + apply_heads."""
        hidden = self.encode(obs)
        return self.apply_heads(hidden, agent_indices, return_aux=return_aux)

    def get_action_and_value(self, obs, agent_indices,
                             direction=None, action_type=None,
                             direction_mask=None, action_type_mask=None):
        """Sample actions, compute log_prob, entropy, value. For rollouts."""
        hidden = self.encode(obs)
        dir_logits, act_logits, value = self.apply_heads(hidden, agent_indices)

        LARGE_NEG = -1e8
        if direction_mask is not None:
            dir_logits = dir_logits.masked_fill(~direction_mask, LARGE_NEG)
        if action_type_mask is not None:
            act_logits = act_logits.masked_fill(~action_type_mask, LARGE_NEG)

        dir_dist = Categorical(logits=dir_logits)
        act_dist = Categorical(logits=act_logits)

        if direction is None:
            direction = dir_dist.sample()
        if action_type is None:
            action_type = act_dist.sample()

        log_prob = dir_dist.log_prob(direction) + act_dist.log_prob(action_type)
        entropy = dir_dist.entropy() + act_dist.entropy()

        return direction, action_type, log_prob, entropy, value.squeeze(-1)
