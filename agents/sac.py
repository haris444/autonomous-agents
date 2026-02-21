"""
Discrete Soft Actor-Critic (SAC) for Multi-Agent RL.

Unified SAC with shared trunk + per-agent heads architecture.
Actor: SharedTrunkActorCritic (policy heads only)
Critic: SharedTrunkCritic (twin Q-heads per agent)

References:
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018)
- Christodoulou, "Soft Actor-Critic for Discrete Action Settings" (2019)
"""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core.config import Config
from agents.network import ObservationEncoder, SharedTrunkActorCritic


class SharedTrunkCritic(nn.Module):
    """
    Shared encoder+trunk with per-agent twin Q-heads.

    Architecture:
        ObservationEncoder (144-dim) -> Trunk (144->256->128)
        -> Per-Agent Twin Q-heads (N sets of 4: q1_dir, q1_act, q2_dir, q2_act)

    Q-heads stored as stacked nn.Parameter tensors for batched matmul.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_agents = config.n_agents

        # Shared encoder + trunk (separate from actor)
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
        n_dir = config.n_directions
        n_act = config.n_action_types

        # Per-agent twin Q-head parameters: [N, out_dim, D] weights + [N, out_dim] biases
        self.q1_dir_w = nn.Parameter(torch.empty(N, n_dir, D))
        self.q1_dir_b = nn.Parameter(torch.empty(N, n_dir))
        self.q1_act_w = nn.Parameter(torch.empty(N, n_act, D))
        self.q1_act_b = nn.Parameter(torch.empty(N, n_act))
        self.q2_dir_w = nn.Parameter(torch.empty(N, n_dir, D))
        self.q2_dir_b = nn.Parameter(torch.empty(N, n_dir))
        self.q2_act_w = nn.Parameter(torch.empty(N, n_act, D))
        self.q2_act_b = nn.Parameter(torch.empty(N, n_act))

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights using orthogonal initialization."""
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Q-heads: gain=1.0, template cloned to all N agents
        for w, b in [(self.q1_dir_w, self.q1_dir_b), (self.q1_act_w, self.q1_act_b),
                      (self.q2_dir_w, self.q2_dir_b), (self.q2_act_w, self.q2_act_b)]:
            self._init_head(w, b, gain=1.0)

    def _init_head(self, weight, bias, gain):
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

    def apply_q_heads(self, hidden, agent_indices):
        """
        Apply per-agent twin Q-heads.

        Args:
            hidden: [B, 128] trunk output
            agent_indices: [B] long tensor (mixed agent ids from replay buffer)

        Returns:
            q1_dir [B, 5], q1_act [B, 5], q2_dir [B, 5], q2_act [B, 5]
        """
        h = hidden.unsqueeze(-1)  # [B, 128, 1]
        q1_dir = torch.bmm(self.q1_dir_w[agent_indices], h).squeeze(-1) + self.q1_dir_b[agent_indices]
        q1_act = torch.bmm(self.q1_act_w[agent_indices], h).squeeze(-1) + self.q1_act_b[agent_indices]
        q2_dir = torch.bmm(self.q2_dir_w[agent_indices], h).squeeze(-1) + self.q2_dir_b[agent_indices]
        q2_act = torch.bmm(self.q2_act_w[agent_indices], h).squeeze(-1) + self.q2_act_b[agent_indices]
        return q1_dir, q1_act, q2_dir, q2_act

    def forward(self, obs, agent_indices):
        """Full forward: encode + apply_q_heads."""
        hidden = self.encode(obs)
        return self.apply_q_heads(hidden, agent_indices)


class SAC:
    """
    Unified SAC using SharedTrunkActorCritic (actor) and SharedTrunkCritic (critic).

    One actor network, one critic network, one target critic.
    Per-agent learned entropy coefficients. Three optimizers.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.n_agents = config.n_agents

        # Actor: shared trunk + per-agent policy heads
        self.actor = SharedTrunkActorCritic(config).to(device)

        # Critic: shared trunk + per-agent twin Q-heads
        self.critic = SharedTrunkCritic(config).to(device)
        self.critic_target = SharedTrunkCritic(config).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        # Per-agent learned entropy coefficients
        self.log_alpha_dirs = nn.Parameter(
            torch.full((config.n_agents,), math.log(config.sac_alpha_init), device=device)
        )
        self.log_alpha_acts = nn.Parameter(
            torch.full((config.n_agents,), math.log(config.sac_alpha_init), device=device)
        )

        # Target entropies
        self.target_entropy_dir = config.sac_target_entropy_scale * math.log(config.n_directions)
        self.target_entropy_act = config.sac_target_entropy_scale * math.log(config.n_action_types)

        # Optimizers
        lr = config.sac_learning_rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha_dirs, self.log_alpha_acts], lr=lr)

    @torch.no_grad()
    def get_actions(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """
        Get actions for all agents from vectorized env observations.

        Args:
            obs: Dict with [n_envs, n_agents, ...] shaped tensors
            deterministic: If True, use argmax instead of sampling

        Returns:
            directions: [n_envs, n_agents]
            action_types: [n_envs, n_agents]
        """
        n_envs = obs['entity_tokens'].shape[0]
        N = self.n_agents

        # Flatten: [n_envs, N, ...] -> [n_envs*N, ...]
        obs_flat = {k: v.reshape(n_envs * N, *v.shape[2:]) for k, v in obs.items()}
        hidden = self.actor.encode(obs_flat)  # [n_envs*N, 128]

        # Reshape to [N, n_envs, 128] for parallel heads
        hidden = hidden.view(n_envs, N, -1).transpose(0, 1)  # [N, n_envs, 128]

        # Apply policy heads via einsum (all agents in parallel)
        dir_logits, act_logits, _ = self.actor.apply_heads_parallel(hidden, N)
        # [N, n_envs, 5]

        if deterministic:
            directions = dir_logits.argmax(dim=-1)
            action_types = act_logits.argmax(dim=-1)
        else:
            directions = Categorical(logits=dir_logits).sample()
            action_types = Categorical(logits=act_logits).sample()

        # Transpose: [N, n_envs] -> [n_envs, N]
        return directions.transpose(0, 1), action_types.transpose(0, 1)

    def update(self, replay_buffer, batch_size: int) -> Dict[str, float]:
        """
        Run one SAC update step using mixed-agent batch from replay buffer.

        Samples once, routes to per-agent heads via agent_id indexing + bmm.
        """
        cfg = self.config
        tau = cfg.sac_tau
        gamma = cfg.sac_gamma

        batch = replay_buffer.sample(batch_size)
        obs = batch['obs']
        next_obs = batch['next_obs']
        dirs = batch['directions']
        acts = batch['action_types']
        rewards = batch['rewards']
        dones = batch['dones']
        agent_ids = obs['agent_id']  # [B] — mixed agent indices

        # Per-sample alpha values (indexed by agent_id)
        alpha_dir = self.log_alpha_dirs[agent_ids].exp().detach()  # [B]
        alpha_act = self.log_alpha_acts[agent_ids].exp().detach()  # [B]

        # ---- CRITIC UPDATE ----
        q1_dir, q1_act, q2_dir, q2_act = self.critic(obs, agent_ids)
        # Q-value for taken actions
        q1 = q1_dir.gather(1, dirs.unsqueeze(1)).squeeze(1) + q1_act.gather(1, acts.unsqueeze(1)).squeeze(1)
        q2 = q2_dir.gather(1, dirs.unsqueeze(1)).squeeze(1) + q2_act.gather(1, acts.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Next-state policy (through actor)
            next_hidden = self.actor.encode(next_obs)
            next_dir_logits, next_act_logits, _ = self.actor.apply_heads(next_hidden, agent_ids)
            next_dir_probs = F.softmax(next_dir_logits, dim=-1)
            next_act_probs = F.softmax(next_act_logits, dim=-1)

            # Next-state Q-values from target critic
            nq1_dir, nq1_act, nq2_dir, nq2_act = self.critic_target(next_obs, agent_ids)
            next_q_dir = torch.min(nq1_dir, nq2_dir)
            next_q_act = torch.min(nq1_act, nq2_act)

            # Soft state value: V(s') = E_a[Q(s',a) - alpha * log pi(a|s')]
            next_log_dir = torch.log(next_dir_probs + 1e-8)
            next_log_act = torch.log(next_act_probs + 1e-8)
            next_v_dir = (next_dir_probs * (next_q_dir - alpha_dir.unsqueeze(-1) * next_log_dir)).sum(dim=-1)
            next_v_act = (next_act_probs * (next_q_act - alpha_act.unsqueeze(-1) * next_log_act)).sum(dim=-1)

            target_q = rewards + gamma * (1.0 - dones) * (next_v_dir + next_v_act)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()

        # ---- ACTOR UPDATE ----
        actor_hidden = self.actor.encode(obs)
        dir_logits, act_logits, _ = self.actor.apply_heads(actor_hidden, agent_ids)
        dir_probs = F.softmax(dir_logits, dim=-1)
        act_probs = F.softmax(act_logits, dim=-1)
        log_dir = torch.log(dir_probs + 1e-8)
        log_act = torch.log(act_probs + 1e-8)

        with torch.no_grad():
            q1d, q1a, _, _ = self.critic(obs, agent_ids)

        actor_loss_dir = (dir_probs * (alpha_dir.unsqueeze(-1) * log_dir - q1d)).sum(dim=-1).mean()
        actor_loss_act = (act_probs * (alpha_act.unsqueeze(-1) * log_act - q1a)).sum(dim=-1).mean()
        actor_loss = actor_loss_dir + actor_loss_act

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()

        # ---- ALPHA UPDATE ----
        if cfg.sac_auto_alpha:
            # Per-sample entropy
            entropy_dir = -(dir_probs.detach() * log_dir.detach()).sum(dim=-1)  # [B]
            entropy_act = -(act_probs.detach() * log_act.detach()).sum(dim=-1)  # [B]

            # Per-agent alpha loss: aggregate by agent_id
            alpha_loss = (
                -(self.log_alpha_dirs[agent_ids] * (self.target_entropy_dir - entropy_dir).detach()).mean()
                + -(self.log_alpha_acts[agent_ids] * (self.target_entropy_act - entropy_act).detach()).mean()
            )

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ---- TARGET UPDATE (Polyak) ----
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

        # ---- METRICS ----
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_dir': self.log_alpha_dirs.exp().mean().item(),
            'alpha_act': self.log_alpha_acts.exp().mean().item(),
            'q1_mean': q1.mean().item(),
            'entropy_dir': -(dir_probs * log_dir).sum(dim=-1).mean().item(),
            'entropy_act': -(act_probs * log_act).sum(dim=-1).mean().item(),
        }

    def state_dict(self):
        """Return full state for checkpointing."""
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'log_alpha_dirs': self.log_alpha_dirs.detach().cpu(),
            'log_alpha_acts': self.log_alpha_acts.detach().cpu(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state': self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        """Load full state from checkpoint."""
        self.actor.load_state_dict(state['actor_state_dict'])
        self.critic.load_state_dict(state['critic_state_dict'])
        self.critic_target.load_state_dict(state['critic_target_state_dict'])

        with torch.no_grad():
            self.log_alpha_dirs.copy_(state['log_alpha_dirs'].to(self.device))
            self.log_alpha_acts.copy_(state['log_alpha_acts'].to(self.device))

        if 'actor_optimizer_state' in state:
            self.actor_optimizer.load_state_dict(state['actor_optimizer_state'])
            self.critic_optimizer.load_state_dict(state['critic_optimizer_state'])
            # Rebuild alpha optimizer with current params
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha_dirs, self.log_alpha_acts],
                lr=self.config.sac_learning_rate
            )
            self.alpha_optimizer.load_state_dict(state['alpha_optimizer_state'])
