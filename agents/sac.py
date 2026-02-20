"""
Discrete Soft Actor-Critic (SAC) for Multi-Agent RL.

Implements:
- SACCritic: Twin Q-networks with per-head (direction + action_type) Q-values
- IndependentSAC: N independent actor-critic pairs (one per agent)

References:
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018)
- Christodoulou, "Soft Actor-Critic for Discrete Action Settings" (2019)
"""
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core.config import Config
from agents.network import ObservationEncoder, ActorCritic


class SACCritic(nn.Module):
    """Twin Q-networks for discrete SAC with factored action heads.

    Each Q-network outputs Q-values for all actions in each head:
    Q1_dir[B, 5], Q1_act[B, 5], Q2_dir[B, 5], Q2_act[B, 5].
    """

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = ObservationEncoder(config)
        edim = self.encoder.output_dim

        # Shared trunk (same architecture as ActorCritic)
        self.shared = nn.Sequential(
            nn.Linear(edim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Twin Q-networks — per-head Q-values
        self.q1_dir = nn.Linear(128, config.n_directions)
        self.q1_act = nn.Linear(128, config.n_action_types)
        self.q2_dir = nn.Linear(128, config.n_directions)
        self.q2_act = nn.Linear(128, config.n_action_types)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Q-heads: gain=1.0
        for head in [self.q1_dir, self.q1_act, self.q2_dir, self.q2_act]:
            nn.init.orthogonal_(head.weight, gain=1.0)

    def _encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.encoder(
            obs['entity_tokens'], obs['entity_mask'],
            obs['signals'], obs['self_hp'],
            obs['self_inventory'], obs['agent_id']
        )
        return self.shared(features)

    def forward(self, obs: Dict[str, torch.Tensor]):
        """Returns (q1_dir[B,5], q1_act[B,5], q2_dir[B,5], q2_act[B,5])."""
        h = self._encode(obs)
        return self.q1_dir(h), self.q1_act(h), self.q2_dir(h), self.q2_act(h)


class IndependentSAC:
    """N independent SAC actor-critic pairs (one per agent).

    Each agent has:
    - Actor: ActorCritic (reused from network.py, only policy heads used)
    - Critic: SACCritic (twin Q-networks)
    - Critic target: SACCritic (Polyak averaged)
    - Learned alpha (entropy coefficient) per action head
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.n_agents = config.n_agents

        # Per-agent networks
        self.actors = nn.ModuleList([ActorCritic(config).to(device) for _ in range(config.n_agents)])
        self.critics = nn.ModuleList([SACCritic(config).to(device) for _ in range(config.n_agents)])
        self.critic_targets = nn.ModuleList([SACCritic(config).to(device) for _ in range(config.n_agents)])

        # Init target networks as copies
        for i in range(config.n_agents):
            self.critic_targets[i].load_state_dict(self.critics[i].state_dict())
            self.critic_targets[i].requires_grad_(False)

        # Learned entropy coefficients (one alpha per action head, per agent)
        self.log_alpha_dirs = [
            torch.tensor(math.log(config.sac_alpha_init), device=device, requires_grad=True)
            for _ in range(config.n_agents)
        ]
        self.log_alpha_acts = [
            torch.tensor(math.log(config.sac_alpha_init), device=device, requires_grad=True)
            for _ in range(config.n_agents)
        ]

        # Target entropies: -scale * log(1/|A|) = scale * log(|A|)
        self.target_entropy_dir = config.sac_target_entropy_scale * math.log(config.n_directions)
        self.target_entropy_act = config.sac_target_entropy_scale * math.log(config.n_action_types)

        # Optimizers (per agent)
        lr = config.sac_learning_rate
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(), lr=lr) for i in range(config.n_agents)]
        self.critic_optimizers = [torch.optim.Adam(self.critics[i].parameters(), lr=lr) for i in range(config.n_agents)]
        self.alpha_optimizers = [
            torch.optim.Adam([self.log_alpha_dirs[i], self.log_alpha_acts[i]], lr=lr)
            for i in range(config.n_agents)
        ]

    @torch.no_grad()
    def get_actions(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """Get actions for all agents from vectorized env observations.

        Args:
            obs: Dict with [n_envs, n_agents, ...] shaped tensors
            deterministic: If True, use argmax instead of sampling

        Returns:
            directions: [n_envs, n_agents]
            action_types: [n_envs, n_agents]
        """
        ne = obs['entity_tokens'].shape[0]
        n = self.n_agents
        directions = torch.zeros(ne, n, device=self.device, dtype=torch.long)
        action_types = torch.zeros(ne, n, device=self.device, dtype=torch.long)

        for i in range(n):
            # Extract per-agent obs: [n_envs, ...]
            obs_i = {k: v[:, i] for k, v in obs.items()}
            dir_logits, act_logits, _ = self.actors[i](obs_i)

            if deterministic:
                directions[:, i] = dir_logits.argmax(dim=-1)
                action_types[:, i] = act_logits.argmax(dim=-1)
            else:
                directions[:, i] = Categorical(logits=dir_logits).sample()
                action_types[:, i] = Categorical(logits=act_logits).sample()

        return directions, action_types

    def update(self, replay_buffer, batch_size: int) -> Dict[str, float]:
        """Run one SAC update step for all agents.

        Each agent independently samples from the shared replay buffer.

        Returns: Dict of averaged metrics across agents.
        """
        cfg = self.config
        tau = cfg.sac_tau
        gamma = cfg.sac_gamma

        metrics_sum = {}
        for i in range(self.n_agents):
            batch = replay_buffer.sample(batch_size)
            obs = batch['obs']
            next_obs = batch['next_obs']
            dirs = batch['directions']
            acts = batch['action_types']
            rewards = batch['rewards']
            dones = batch['dones']

            alpha_dir = self.log_alpha_dirs[i].exp().detach()
            alpha_act = self.log_alpha_acts[i].exp().detach()

            # ---- CRITIC UPDATE ----
            q1_dir, q1_act, q2_dir, q2_act = self.critics[i](obs)
            # Q-value for taken actions
            q1 = q1_dir.gather(1, dirs.unsqueeze(1)).squeeze(1) + q1_act.gather(1, acts.unsqueeze(1)).squeeze(1)
            q2 = q2_dir.gather(1, dirs.unsqueeze(1)).squeeze(1) + q2_act.gather(1, acts.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # Next-state policy probabilities
                next_dir_logits, next_act_logits, _ = self.actors[i](next_obs)
                next_dir_probs = F.softmax(next_dir_logits, dim=-1)
                next_act_probs = F.softmax(next_act_logits, dim=-1)

                # Next-state Q-values from target
                nq1_dir, nq1_act, nq2_dir, nq2_act = self.critic_targets[i](next_obs)
                next_q_dir = torch.min(nq1_dir, nq2_dir)
                next_q_act = torch.min(nq1_act, nq2_act)

                # Soft state value: V(s') = E_a[Q(s',a) - alpha * log pi(a|s')]
                next_log_dir = torch.log(next_dir_probs + 1e-8)
                next_log_act = torch.log(next_act_probs + 1e-8)
                next_v_dir = (next_dir_probs * (next_q_dir - alpha_dir * next_log_dir)).sum(dim=-1)
                next_v_act = (next_act_probs * (next_q_act - alpha_act * next_log_act)).sum(dim=-1)

                target_q = rewards + gamma * (1.0 - dones) * (next_v_dir + next_v_act)

            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), cfg.max_grad_norm)
            self.critic_optimizers[i].step()

            # ---- ACTOR UPDATE ----
            dir_logits, act_logits, _ = self.actors[i](obs)
            dir_probs = F.softmax(dir_logits, dim=-1)
            act_probs = F.softmax(act_logits, dim=-1)
            log_dir = torch.log(dir_probs + 1e-8)
            log_act = torch.log(act_probs + 1e-8)

            # Use Q1 for actor (no need for min here)
            with torch.no_grad():
                q1d, q1a, _, _ = self.critics[i](obs)

            actor_loss_dir = (dir_probs * (alpha_dir * log_dir - q1d)).sum(dim=-1).mean()
            actor_loss_act = (act_probs * (alpha_act * log_act - q1a)).sum(dim=-1).mean()
            actor_loss = actor_loss_dir + actor_loss_act

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), cfg.max_grad_norm)
            self.actor_optimizers[i].step()

            # ---- ALPHA UPDATE ----
            if cfg.sac_auto_alpha:
                entropy_dir = -(dir_probs.detach() * log_dir.detach()).sum(dim=-1).mean()
                entropy_act = -(act_probs.detach() * log_act.detach()).sum(dim=-1).mean()

                alpha_loss = (
                    -self.log_alpha_dirs[i] * (self.target_entropy_dir - entropy_dir).detach()
                    + -self.log_alpha_acts[i] * (self.target_entropy_act - entropy_act).detach()
                )

                self.alpha_optimizers[i].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizers[i].step()

            # ---- TARGET UPDATE (Polyak) ----
            with torch.no_grad():
                for tp, p in zip(self.critic_targets[i].parameters(), self.critics[i].parameters()):
                    tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

            # ---- METRICS ----
            m = {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha_dir': alpha_dir.item(),
                'alpha_act': alpha_act.item(),
                'q1_mean': q1.mean().item(),
                'entropy_dir': -(dir_probs * log_dir).sum(dim=-1).mean().item(),
                'entropy_act': -(act_probs * log_act).sum(dim=-1).mean().item(),
            }
            for k, v in m.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

        # Average across agents
        return {k: v / self.n_agents for k, v in metrics_sum.items()}

    def state_dict(self):
        """Return full state for checkpointing."""
        return {
            'actor_state_dicts': [a.state_dict() for a in self.actors],
            'critic_state_dicts': [c.state_dict() for c in self.critics],
            'critic_target_state_dicts': [ct.state_dict() for ct in self.critic_targets],
            'log_alpha_dirs': [la.item() for la in self.log_alpha_dirs],
            'log_alpha_acts': [la.item() for la in self.log_alpha_acts],
            'actor_optimizer_states': [o.state_dict() for o in self.actor_optimizers],
            'critic_optimizer_states': [o.state_dict() for o in self.critic_optimizers],
            'alpha_optimizer_states': [o.state_dict() for o in self.alpha_optimizers],
        }

    def load_state_dict(self, state):
        """Load full state from checkpoint."""
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(state['actor_state_dicts'][i])
            self.critics[i].load_state_dict(state['critic_state_dicts'][i])
            self.critic_targets[i].load_state_dict(state['critic_target_state_dicts'][i])
            self.log_alpha_dirs[i] = torch.tensor(
                state['log_alpha_dirs'][i], device=self.device, requires_grad=True
            )
            self.log_alpha_acts[i] = torch.tensor(
                state['log_alpha_acts'][i], device=self.device, requires_grad=True
            )
            if 'actor_optimizer_states' in state:
                self.actor_optimizers[i].load_state_dict(state['actor_optimizer_states'][i])
                self.critic_optimizers[i].load_state_dict(state['critic_optimizer_states'][i])
                # Rebuild alpha optimizers with new tensors
                self.alpha_optimizers[i] = torch.optim.Adam(
                    [self.log_alpha_dirs[i], self.log_alpha_acts[i]],
                    lr=self.config.sac_learning_rate
                )
                self.alpha_optimizers[i].load_state_dict(state['alpha_optimizer_states'][i])
