"""
Proximal Policy Optimization (PPO) algorithm.

Unified PPO with shared trunk + per-agent heads architecture.
Single network, single optimizer. Supports single-env and vec-env training.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core.config import Config
from agents.network import SharedTrunkActorCritic
from agents.buffer import SingleAgentBuffer, VecBuffer


class PPO:
    """
    Unified PPO using SharedTrunkActorCritic.

    One network with shared encoder+trunk and per-agent heads.
    One Adam optimizer. Supports single-env and vec-env training.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.n_agents = config.n_agents
        self.n_active = config.pretrain_spawn_agents if config.pretrain_mode else config.n_agents
        self.clone_mode = False

        self.network = SharedTrunkActorCritic(config).to(device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )

    def set_curriculum_phase(self, phase: int):
        """Update n_active and clone_mode based on curriculum phase."""
        if phase >= 6:
            self.n_active = 2
            self.clone_mode = True
        else:
            self.n_active = self.config.pretrain_spawn_agents if self.config.pretrain_mode else self.config.n_agents
            self.clone_mode = False

    def set_n_active(self, n: int):
        self.n_active = min(n, self.n_agents)

    def set_clone_mode(self, enabled: bool):
        self.clone_mode = enabled

    def _agent_ids(self, n: int) -> torch.Tensor:
        """Get agent indices. In clone mode, all map to head 0."""
        if self.clone_mode:
            return torch.zeros(n, device=self.device, dtype=torch.long)
        return torch.arange(n, device=self.device)

    def _pad_inactive(self, *tensors, pad_value_map=None):
        """Pad tensors for inactive agents. Default pad value is 0."""
        if self.n_active >= self.n_agents:
            return tensors if len(tensors) > 1 else tensors[0]
        pad_size = self.n_agents - self.n_active
        results = []
        for i, t in enumerate(tensors):
            if pad_value_map and i in pad_value_map:
                pv = pad_value_map[i]
                p = torch.full((pad_size,), pv, device=self.device, dtype=t.dtype)
            else:
                p = torch.zeros(pad_size, device=self.device, dtype=t.dtype)
            results.append(torch.cat([t, p]))
        return tuple(results) if len(results) > 1 else results[0]

    # =========================================================================
    # SINGLE-ENV INFERENCE
    # =========================================================================

    def get_actions_and_values(
        self,
        obs: Dict[str, torch.Tensor],
        direction_mask: torch.Tensor = None,
        action_type_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-env: obs [n_agents, ...] -> [n_agents] tensors."""
        N = self.n_active
        agent_ids = self._agent_ids(N)
        obs_active = {k: v[:N] for k, v in obs.items()}
        hidden = self.network.encode(obs_active)  # [N, 128]

        dir_logits, act_logits, value = self.network.apply_heads(hidden, agent_ids)

        LARGE_NEG = -1e8
        if direction_mask is not None:
            dir_logits = dir_logits.masked_fill(~direction_mask[:N], LARGE_NEG)
        if action_type_mask is not None:
            act_logits = act_logits.masked_fill(~action_type_mask[:N], LARGE_NEG)

        dir_dist = Categorical(logits=dir_logits)
        act_dist = Categorical(logits=act_logits)
        directions = dir_dist.sample()
        action_types = act_dist.sample()
        log_probs = dir_dist.log_prob(directions) + act_dist.log_prob(action_types)
        entropies = dir_dist.entropy() + act_dist.entropy()
        values = value.squeeze(-1)

        # Pad inactive agents
        if N < self.n_agents:
            pad = self.n_agents - N
            directions = torch.cat([directions, torch.full((pad,), 4, device=self.device, dtype=torch.long)])
            action_types = torch.cat([action_types, torch.zeros(pad, device=self.device, dtype=torch.long)])
            log_probs = torch.cat([log_probs, torch.zeros(pad, device=self.device)])
            entropies = torch.cat([entropies, torch.zeros(pad, device=self.device)])
            values = torch.cat([values, torch.zeros(pad, device=self.device)])

        return directions, action_types, log_probs, entropies, values

    def get_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single-env: get value estimates [n_agents]."""
        N = self.n_active
        agent_ids = self._agent_ids(N)
        obs_active = {k: v[:N] for k, v in obs.items()}
        hidden = self.network.encode(obs_active)
        _, _, value = self.network.apply_heads(hidden, agent_ids)
        values = value.squeeze(-1)

        if N < self.n_agents:
            values = torch.cat([values, torch.zeros(self.n_agents - N, device=self.device)])
        return values

    def get_auxiliary_values(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single-env: get auxiliary value estimates."""
        N = self.n_active
        agent_ids = self._agent_ids(N)
        obs_active = {k: v[:N] for k, v in obs.items()}
        hidden = self.network.encode(obs_active)
        _, _, _, v_surv, v_res, v_soc = self.network.apply_heads(hidden, agent_ids, return_aux=True)

        s_vals = v_surv.squeeze(-1)
        r_vals = v_res.squeeze(-1)
        soc_vals = v_soc.squeeze(-1)

        if N < self.n_agents:
            pad = torch.zeros(self.n_agents - N, device=self.device)
            s_vals = torch.cat([s_vals, pad])
            r_vals = torch.cat([r_vals, pad])
            soc_vals = torch.cat([soc_vals, pad])

        return {'survival': s_vals, 'resource': r_vals, 'social': soc_vals}

    def get_action_probs(
        self,
        obs: Dict[str, torch.Tensor],
        direction_mask: torch.Tensor = None,
        action_type_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action probabilities for all active agents (for visualization)."""
        N = self.n_active
        agent_ids = self._agent_ids(N)
        obs_active = {k: v[:N] for k, v in obs.items()}

        with torch.no_grad():
            hidden = self.network.encode(obs_active)
            dir_logits, act_logits, _ = self.network.apply_heads(hidden, agent_ids)

            LARGE_NEG = -1e8
            if direction_mask is not None:
                dir_logits = dir_logits.masked_fill(~direction_mask[:N], LARGE_NEG)
            if action_type_mask is not None:
                act_logits = act_logits.masked_fill(~action_type_mask[:N], LARGE_NEG)

            dir_probs = F.softmax(dir_logits, dim=-1)
            act_probs = F.softmax(act_logits, dim=-1)

        if N < self.n_agents:
            pad = self.n_agents - N
            dir_probs = torch.cat([dir_probs, torch.ones(pad, 5, device=self.device) / 5])
            act_probs = torch.cat([act_probs, torch.ones(pad, 5, device=self.device) / 5])

        return dir_probs, act_probs

    # =========================================================================
    # VEC-ENV INFERENCE
    # =========================================================================

    def vec_get_actions_and_values(
        self,
        obs: Dict[str, torch.Tensor],
        direction_mask: torch.Tensor = None,
        action_type_mask: torch.Tensor = None,
        n_envs: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vec-env inference: obs [n_envs, n_agents, ...] -> [n_envs, n_agents] tensors.
        Returns: directions, action_types, log_probs, entropies, values, aux_values
        """
        if n_envs is None:
            n_envs = obs['entity_tokens'].shape[0]
        N = self.n_active
        n_agents = self.n_agents

        # Flatten active agents: [n_envs, N, ...] -> [n_envs*N, ...]
        obs_flat = {k: v[:, :N].reshape(n_envs * N, *v.shape[2:]) for k, v in obs.items()}
        hidden = self.network.encode(obs_flat)  # [n_envs*N, 128]

        # Reshape to [N, n_envs, 128] for parallel heads
        hidden = hidden.view(n_envs, N, -1).transpose(0, 1)  # [N, n_envs, 128]

        # Apply heads
        if self.clone_mode:
            # All agents use head 0
            h_flat = hidden.reshape(N * n_envs, -1)
            result = self.network.apply_heads(h_flat, 0, return_aux=True)
            dir_logits = result[0].view(N, n_envs, -1)
            act_logits = result[1].view(N, n_envs, -1)
            values_out = result[2].view(N, n_envs, -1)
            v_surv = result[3].view(N, n_envs, -1)
            v_res = result[4].view(N, n_envs, -1)
            v_soc = result[5].view(N, n_envs, -1)
        else:
            dir_logits, act_logits, values_out, v_surv, v_res, v_soc = \
                self.network.apply_heads_parallel(hidden, N, return_aux=True)

        # Apply masks: need [N, n_envs, 5]
        LARGE_NEG = -1e8
        if direction_mask is not None:
            mask = direction_mask[:, :N].transpose(0, 1)
            dir_logits = dir_logits.masked_fill(~mask, LARGE_NEG)
        if action_type_mask is not None:
            mask = action_type_mask[:, :N].transpose(0, 1)
            act_logits = act_logits.masked_fill(~mask, LARGE_NEG)

        # Sample actions
        dir_dist = Categorical(logits=dir_logits)
        act_dist = Categorical(logits=act_logits)
        directions = dir_dist.sample()
        action_types = act_dist.sample()
        log_probs = dir_dist.log_prob(directions) + act_dist.log_prob(action_types)
        entropies = dir_dist.entropy() + act_dist.entropy()

        # Transpose: [N, n_envs] -> [n_envs, N]
        directions = directions.transpose(0, 1)
        action_types = action_types.transpose(0, 1)
        log_probs = log_probs.transpose(0, 1)
        entropies = entropies.transpose(0, 1)
        values = values_out.squeeze(-1).transpose(0, 1)
        s_vals = v_surv.squeeze(-1).transpose(0, 1)
        r_vals = v_res.squeeze(-1).transpose(0, 1)
        soc_vals = v_soc.squeeze(-1).transpose(0, 1)

        # Pad inactive agents
        if N < n_agents:
            pad_size = n_agents - N
            pad_dir = torch.full((n_envs, pad_size), 4, device=self.device, dtype=torch.long)
            pad_act = torch.zeros((n_envs, pad_size), device=self.device, dtype=torch.long)
            pad_float = torch.zeros((n_envs, pad_size), device=self.device)

            directions = torch.cat([directions, pad_dir], dim=1)
            action_types = torch.cat([action_types, pad_act], dim=1)
            log_probs = torch.cat([log_probs, pad_float], dim=1)
            entropies = torch.cat([entropies, pad_float], dim=1)
            values = torch.cat([values, pad_float], dim=1)
            s_vals = torch.cat([s_vals, pad_float], dim=1)
            r_vals = torch.cat([r_vals, pad_float], dim=1)
            soc_vals = torch.cat([soc_vals, pad_float], dim=1)

        aux_values = {'survival': s_vals, 'resource': r_vals, 'social': soc_vals}
        return directions, action_types, log_probs, entropies, values, aux_values

    def vec_get_values(self, obs: Dict[str, torch.Tensor], n_envs: int = None) -> torch.Tensor:
        """Vec-env: get value estimates [n_envs, n_agents]."""
        if n_envs is None:
            n_envs = obs['entity_tokens'].shape[0]
        N = self.n_active

        obs_flat = {k: v[:, :N].reshape(n_envs * N, *v.shape[2:]) for k, v in obs.items()}
        hidden = self.network.encode(obs_flat)
        hidden = hidden.view(n_envs, N, -1).transpose(0, 1)

        if self.clone_mode:
            h_flat = hidden.reshape(N * n_envs, -1)
            _, _, value = self.network.apply_heads(h_flat, 0)
            values = value.squeeze(-1).view(N, n_envs).transpose(0, 1)
        else:
            _, _, value = self.network.apply_heads_parallel(hidden, N)
            values = value.squeeze(-1).transpose(0, 1)

        if N < self.n_agents:
            pad = torch.zeros((n_envs, self.n_agents - N), device=self.device)
            values = torch.cat([values, pad], dim=1)
        return values

    def vec_get_auxiliary_values(self, obs: Dict[str, torch.Tensor], n_envs: int = None) -> Dict[str, torch.Tensor]:
        """Vec-env: get auxiliary value estimates."""
        if n_envs is None:
            n_envs = obs['entity_tokens'].shape[0]
        N = self.n_active

        obs_flat = {k: v[:, :N].reshape(n_envs * N, *v.shape[2:]) for k, v in obs.items()}
        hidden = self.network.encode(obs_flat)
        hidden = hidden.view(n_envs, N, -1).transpose(0, 1)

        if self.clone_mode:
            h_flat = hidden.reshape(N * n_envs, -1)
            _, _, _, v_surv, v_res, v_soc = self.network.apply_heads(h_flat, 0, return_aux=True)
            s_vals = v_surv.squeeze(-1).view(N, n_envs).transpose(0, 1)
            r_vals = v_res.squeeze(-1).view(N, n_envs).transpose(0, 1)
            soc_vals = v_soc.squeeze(-1).view(N, n_envs).transpose(0, 1)
        else:
            _, _, _, v_surv, v_res, v_soc = self.network.apply_heads_parallel(hidden, N, return_aux=True)
            s_vals = v_surv.squeeze(-1).transpose(0, 1)
            r_vals = v_res.squeeze(-1).transpose(0, 1)
            soc_vals = v_soc.squeeze(-1).transpose(0, 1)

        if N < self.n_agents:
            pad = torch.zeros((n_envs, self.n_agents - N), device=self.device)
            s_vals = torch.cat([s_vals, pad], dim=1)
            r_vals = torch.cat([r_vals, pad], dim=1)
            soc_vals = torch.cat([soc_vals, pad], dim=1)

        return {'survival': s_vals, 'resource': r_vals, 'social': soc_vals}

    # =========================================================================
    # PPO UPDATES
    # =========================================================================

    def update(self, buffers: List[SingleAgentBuffer]) -> Dict[str, float]:
        """Update from per-agent SingleAgentBuffers (single-env training)."""
        metric_sums = {
            'policy_loss': torch.tensor(0.0, device=self.device),
            'value_loss': torch.tensor(0.0, device=self.device),
            'aux_value_loss': torch.tensor(0.0, device=self.device),
            'entropy': torch.tensor(0.0, device=self.device),
            'total_loss': torch.tensor(0.0, device=self.device),
            'approx_kl': torch.tensor(0.0, device=self.device),
            'clip_fraction': torch.tensor(0.0, device=self.device)
        }
        n_batches = 0

        for epoch in range(self.config.update_epochs):
            for agent_idx in range(self.n_active):
                head_idx = 0 if self.clone_mode else agent_idx
                for batch in buffers[agent_idx].get_batches():
                    loss, metrics = self._compute_loss(batch, head_idx)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                    for k in metric_sums:
                        metric_sums[k] += metrics[k]
                    n_batches += 1

        return {k: (v / max(1, n_batches)).item() for k, v in metric_sums.items()}

    def update_from_vec_buffer(self, buffer: VecBuffer) -> Dict[str, float]:
        """Update from VecBuffer (vectorized environment training)."""
        metric_keys = ['policy_loss', 'value_loss', 'aux_value_loss',
                        'entropy', 'total_loss', 'approx_kl', 'clip_fraction']
        metric_sums = {k: torch.tensor(0.0, device=self.device) for k in metric_keys}
        n_batches = 0

        has_aux = buffer.returns_survival.abs().sum() > 0

        if self.clone_mode:
            # Clone mode: flat batches, all through head 0
            for epoch in range(self.config.update_epochs):
                for batch in buffer.get_batches(n_active=self.n_active):
                    loss, metrics = self._compute_loss(batch, agent_idx=0)
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                    for k in metric_keys:
                        metric_sums[k] += metrics[k]
                    n_batches += 1
        else:
            # Independent mode: aligned batches [n_active, mb, ...]
            for epoch in range(self.config.update_epochs):
                for batch in buffer.get_aligned_batches(n_active=self.n_active):
                    loss, metrics = self._compute_loss_aligned(batch, has_aux)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                    for k in metric_keys:
                        metric_sums[k] += metrics[k]
                    n_batches += 1

        if n_batches > 0:
            return {k: (v / n_batches).item() for k, v in metric_sums.items()}
        return {k: 0.0 for k in metric_keys}

    def _compute_loss(self, batch, agent_idx):
        """Compute PPO loss for a flat minibatch using a single agent head."""
        obs = batch['obs']
        has_aux = 'returns_survival' in batch and batch['returns_survival'].abs().sum() > 0

        hidden = self.network.encode(obs)

        if has_aux:
            dir_logits, act_logits, value, v_surv, v_res, v_soc = \
                self.network.apply_heads(hidden, agent_idx, return_aux=True)
        else:
            dir_logits, act_logits, value = self.network.apply_heads(hidden, agent_idx)

        # Apply action masks
        LARGE_NEG = -1e8
        direction_mask = batch.get('direction_mask')
        action_type_mask = batch.get('action_type_mask')
        if direction_mask is not None:
            dir_logits = dir_logits.masked_fill(~direction_mask, LARGE_NEG)
        if action_type_mask is not None:
            act_logits = act_logits.masked_fill(~action_type_mask, LARGE_NEG)

        dir_dist = Categorical(logits=dir_logits)
        act_dist = Categorical(logits=act_logits)
        new_log_prob = dir_dist.log_prob(batch['directions']) + act_dist.log_prob(batch['action_types'])
        entropy = dir_dist.entropy() + act_dist.entropy()
        new_value = value.squeeze(-1)

        # PPO objectives
        log_ratio = new_log_prob - batch['log_probs']
        ratio = log_ratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()

        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((new_value - batch['returns']) ** 2).mean()

        aux_v_loss = torch.tensor(0.0, device=self.device)
        if has_aux:
            aux_v_loss = (
                0.5 * ((v_surv.squeeze(-1) - batch['returns_survival']) ** 2).mean() +
                0.5 * ((v_res.squeeze(-1) - batch['returns_resource']) ** 2).mean() +
                0.5 * ((v_soc.squeeze(-1) - batch['returns_social']) ** 2).mean()
            )

        entropy_loss = entropy.mean()
        ent_coef = self.config.ent_coef_coop if self.clone_mode else self.config.ent_coef
        loss = pg_loss + self.config.vf_coef * v_loss + self.config.aux_vf_coef * aux_v_loss - ent_coef * entropy_loss

        metrics = {
            'policy_loss': pg_loss.detach(),
            'value_loss': v_loss.detach(),
            'aux_value_loss': aux_v_loss.detach(),
            'entropy': entropy_loss.detach(),
            'total_loss': loss.detach(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction
        }

        return loss, metrics

    def _compute_loss_aligned(self, batch, has_aux):
        """
        Compute PPO loss for aligned minibatch [n_active, mb_size, ...].
        All agents processed in parallel via einsum.
        """
        N = self.n_active
        obs = batch['obs']
        mb = batch['directions'].shape[1]

        # Flatten obs for shared trunk: [N, mb, ...] -> [N*mb, ...]
        obs_flat = {k: v.reshape(N * mb, *v.shape[2:]) for k, v in obs.items()}
        hidden = self.network.encode(obs_flat)  # [N*mb, 128]
        hidden = hidden.view(N, mb, -1)  # [N, mb, 128]

        # Apply all heads via einsum
        if has_aux:
            dir_logits, act_logits, value, v_surv, v_res, v_soc = \
                self.network.apply_heads_parallel(hidden, N, return_aux=True)
        else:
            dir_logits, act_logits, value = \
                self.network.apply_heads_parallel(hidden, N)

        # Apply masks: [N, mb, 5]
        LARGE_NEG = -1e8
        direction_mask = batch.get('direction_mask')
        action_type_mask = batch.get('action_type_mask')
        if direction_mask is not None:
            dir_logits = dir_logits.masked_fill(~direction_mask, LARGE_NEG)
        if action_type_mask is not None:
            act_logits = act_logits.masked_fill(~action_type_mask, LARGE_NEG)

        # Log probs and entropy: [N, mb]
        dir_log_sm = F.log_softmax(dir_logits, dim=-1)
        act_log_sm = F.log_softmax(act_logits, dim=-1)
        dir_log_prob = dir_log_sm.gather(2, batch['directions'].unsqueeze(-1)).squeeze(-1)
        act_log_prob = act_log_sm.gather(2, batch['action_types'].unsqueeze(-1)).squeeze(-1)
        new_log_prob = dir_log_prob + act_log_prob

        dir_probs = F.softmax(dir_logits, dim=-1)
        act_probs = F.softmax(act_logits, dim=-1)
        entropy = -(dir_probs * dir_log_sm).sum(-1) - (act_probs * act_log_sm).sum(-1)

        new_value = value.squeeze(-1)  # [N, mb]

        # PPO objectives: all [N, mb]
        log_ratio = new_log_prob - batch['log_probs']
        ratio = log_ratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()

        # Normalize advantages per agent
        advantages = batch['advantages']
        adv_mean = advantages.mean(dim=1, keepdim=True)
        adv_std = advantages.std(dim=1, keepdim=True) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((new_value - batch['returns']) ** 2).mean()

        aux_v_loss = torch.tensor(0.0, device=self.device)
        if has_aux:
            aux_v_loss = (
                0.5 * ((v_surv.squeeze(-1) - batch['returns_survival']) ** 2).mean() +
                0.5 * ((v_res.squeeze(-1) - batch['returns_resource']) ** 2).mean() +
                0.5 * ((v_soc.squeeze(-1) - batch['returns_social']) ** 2).mean()
            )

        entropy_loss = entropy.mean()
        ent_coef = self.config.ent_coef_coop if self.clone_mode else self.config.ent_coef
        loss = pg_loss + self.config.vf_coef * v_loss + self.config.aux_vf_coef * aux_v_loss - ent_coef * entropy_loss

        metrics = {
            'policy_loss': pg_loss.detach(),
            'value_loss': v_loss.detach(),
            'aux_value_loss': aux_v_loss.detach(),
            'entropy': entropy_loss.detach(),
            'total_loss': loss.detach(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction
        }

        return loss, metrics
