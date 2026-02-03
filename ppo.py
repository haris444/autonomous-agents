"""
Proximal Policy Optimization (PPO) algorithm.

Handles dual-action policy with shared value function.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.func as func

from config import Config
from network import ActorCritic
from buffer import RolloutBuffer, SingleAgentBuffer


class PPO:
    """
    PPO algorithm implementation.

    Uses clipped surrogate objective with entropy bonus.
    """

    def __init__(self, config: Config, network: ActorCritic, device: torch.device):
        self.config = config
        self.network = network
        self.device = device

        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout.

        Returns:
            Dictionary of training metrics
        """
        # Accumulate metrics on GPU to reduce CPU-GPU syncs
        metric_sums = {
            'policy_loss': torch.tensor(0.0, device=self.device),
            'value_loss': torch.tensor(0.0, device=self.device),
            'entropy': torch.tensor(0.0, device=self.device),
            'total_loss': torch.tensor(0.0, device=self.device),
            'approx_kl': torch.tensor(0.0, device=self.device),
            'clip_fraction': torch.tensor(0.0, device=self.device)
        }

        n_batches = 0

        for epoch in range(self.config.update_epochs):
            for batch in buffer.get_batches():
                loss, batch_metrics = self._compute_loss(batch)

                # Gradient update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics on GPU (no .item() calls)
                for key in metric_sums:
                    metric_sums[key] += batch_metrics[key]
                n_batches += 1

        # Average and sync to CPU once at the end
        metrics = {key: (val / n_batches).item() for key, val in metric_sums.items()}

        return metrics

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Compute combined PPO loss for one minibatch."""
        # Get current policy outputs (with action masks for correct log prob computation)
        _, _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
            batch['obs'],
            direction=batch['directions'],
            action_type=batch['action_types'],
            direction_mask=batch.get('direction_mask'),
            action_type_mask=batch.get('action_type_mask')
        )

        # Policy ratio
        log_ratio = new_log_prob - batch['log_probs']
        ratio = log_ratio.exp()

        # Approximate KL divergence (for monitoring) - keep on GPU
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()

        # Normalize advantages
        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (clipped surrogate objective)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1 - self.config.clip_coef,
            1 + self.config.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((new_value - batch['returns']) ** 2).mean()

        # Entropy bonus (negative because we maximize entropy)
        entropy_loss = entropy.mean()

        # Total loss
        loss = pg_loss + self.config.vf_coef * v_loss - self.config.ent_coef * entropy_loss

        # Return tensors (no .item() calls) - synced once in update()
        batch_metrics = {
            'policy_loss': pg_loss.detach(),
            'value_loss': v_loss.detach(),
            'entropy': entropy_loss.detach(),
            'total_loss': loss.detach(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction
        }

        return loss, batch_metrics


class IndependentPPO:
    """
    Manages N independent ActorCritic networks and optimizers.

    Each agent has its own network trained only on its own experiences.
    This enables heterogeneous policy learning.

    Supports "clone mode" for cooperation curriculum where agent 1 uses agent 0's network.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.n_agents = config.n_agents

        # Number of agents to actually train (all agents, or fewer in pretrain mode)
        self.n_active = config.pretrain_spawn_agents if config.pretrain_mode else config.n_agents

        # Clone mode: agent 1 uses agent 0's network (for cooperation curriculum)
        self.clone_mode = False

        # Create N separate networks and optimizers
        self.networks = nn.ModuleList([
            ActorCritic(config).to(device)
            for _ in range(config.n_agents)
        ])

        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=config.learning_rate, eps=1e-5)
            for net in self.networks
        ]

    def set_curriculum_phase(self, phase: int):
        """Update n_active and clone_mode based on curriculum phase (legacy method)."""
        if phase >= 6:
            # Cooperation phases: 2 active agents, both use network[0]
            self.n_active = 2
            self.clone_mode = True
        else:
            # Solo phases: 1 active agent
            self.n_active = self.config.pretrain_spawn_agents if self.config.pretrain_mode else self.config.n_agents
            self.clone_mode = False

    def set_n_active(self, n: int):
        """Set the number of active agents directly."""
        self.n_active = min(n, self.n_agents)

    def set_clone_mode(self, enabled: bool):
        """Enable or disable clone mode (all agents share network[0])."""
        self.clone_mode = enabled

    def get_actions_and_values(
        self,
        obs: Dict[str, torch.Tensor],
        direction_mask: torch.Tensor = None,
        action_type_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions and values from each agent's network.

        Args:
            obs: Batched observations [n_agents, ...]
            direction_mask: Optional batched direction mask [n_agents, 5]
            action_type_mask: Optional batched action type mask [n_agents, 5]

        Returns:
            directions: [n_agents]
            action_types: [n_agents]
            log_probs: [n_agents]
            entropies: [n_agents]
            values: [n_agents]
        """
        directions = []
        action_types = []
        log_probs = []
        entropies = []
        values = []

        # Only process active agents (all in normal mode, fewer in pretrain)
        for i in range(self.n_active):
            # In clone mode, all agents use network[0]
            net = self.networks[0] if self.clone_mode else self.networks[i]
            # Extract single agent observation (add batch dim)
            obs_i = {k: v[i:i+1] for k, v in obs.items()}

            # Extract single agent action masks if provided
            dir_mask_i = None
            act_mask_i = None
            if direction_mask is not None:
                dir_mask_i = direction_mask[i:i+1]
            if action_type_mask is not None:
                act_mask_i = action_type_mask[i:i+1]

            # Get actions and value from this agent's network
            dir_i, act_i, log_p, ent, val = net.get_action_and_value(
                obs_i, direction_mask=dir_mask_i, action_type_mask=act_mask_i
            )

            directions.append(dir_i)
            action_types.append(act_i)
            log_probs.append(log_p)
            entropies.append(ent)
            values.append(val)

        # For inactive agents, return dummy values (they're dead anyway)
        for i in range(self.n_active, self.n_agents):
            directions.append(torch.tensor([4], device=self.device))  # STAY
            action_types.append(torch.tensor([0], device=self.device))  # MOVE
            log_probs.append(torch.tensor([0.0], device=self.device))
            entropies.append(torch.tensor([0.0], device=self.device))
            values.append(torch.tensor([0.0], device=self.device))

        # Stack results back to [n_agents] tensors
        return (
            torch.cat(directions),
            torch.cat(action_types),
            torch.cat(log_probs),
            torch.cat(entropies),
            torch.cat(values)
        )

    def get_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get value estimates from each agent's network.

        Args:
            obs: Batched observations [n_agents, ...]

        Returns:
            values: [n_agents]
        """
        values = []
        # Only process active agents
        for i in range(self.n_active):
            # In clone mode, all agents use network[0]
            net = self.networks[0] if self.clone_mode else self.networks[i]
            obs_i = {k: v[i:i+1] for k, v in obs.items()}
            val = net.get_value(obs_i).squeeze(-1)
            values.append(val)

        # Dummy values for inactive agents
        for i in range(self.n_active, self.n_agents):
            values.append(torch.tensor([0.0], device=self.device))

        return torch.cat(values)

    def update(self, buffers: List[SingleAgentBuffer]) -> Dict[str, float]:
        """
        Update each agent's network using its own buffer.

        Args:
            buffers: List of SingleAgentBuffer, one per agent

        Returns:
            Averaged metrics across all agents
        """
        # Accumulate metrics across all agents
        all_metrics = {
            'policy_loss': torch.tensor(0.0, device=self.device),
            'value_loss': torch.tensor(0.0, device=self.device),
            'aux_value_loss': torch.tensor(0.0, device=self.device),
            'entropy': torch.tensor(0.0, device=self.device),
            'total_loss': torch.tensor(0.0, device=self.device),
            'approx_kl': torch.tensor(0.0, device=self.device),
            'clip_fraction': torch.tensor(0.0, device=self.device)
        }

        total_batches = 0

        # In clone mode, combine all active agents' buffers and train network[0]
        if self.clone_mode:
            # Combine buffers from all active agents
            combined_metrics, n_batches = self._update_with_combined_buffers(
                self.networks[0],
                self.optimizers[0],
                [buffers[i] for i in range(self.n_active)]
            )
            for key in all_metrics:
                all_metrics[key] += combined_metrics[key]
            total_batches += n_batches
        else:
            # Normal mode: each agent updates its own network
            for i in range(self.n_active):
                agent_metrics, n_batches = self._update_single_agent(
                    self.networks[i],
                    self.optimizers[i],
                    buffers[i]
                )

                # Accumulate metrics
                for key in all_metrics:
                    all_metrics[key] += agent_metrics[key]
                total_batches += n_batches

        # Average metrics and sync to CPU
        metrics = {key: (val / total_batches).item() for key, val in all_metrics.items()}

        return metrics

    def _update_single_agent(
        self,
        network: ActorCritic,
        optimizer: torch.optim.Optimizer,
        buffer: SingleAgentBuffer
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Perform PPO update for a single agent.

        Returns:
            Tuple of (accumulated metrics dict, number of batches)
        """
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
            for batch in buffer.get_batches():
                loss, batch_metrics = self._compute_loss(network, batch)

                # Gradient update
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), self.config.max_grad_norm)
                optimizer.step()

                # Accumulate metrics
                for key in metric_sums:
                    metric_sums[key] += batch_metrics[key]
                n_batches += 1

        return metric_sums, n_batches

    def _update_with_combined_buffers(
        self,
        network: ActorCritic,
        optimizer: torch.optim.Optimizer,
        buffers: List[SingleAgentBuffer]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Perform PPO update using combined data from multiple agents' buffers.
        Used in clone mode where all agents share a single network.

        Returns:
            Tuple of (accumulated metrics dict, number of batches)
        """
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
            # Get batches from all buffers and interleave them
            all_batch_iters = [buf.get_batches() for buf in buffers]
            for batch_tuple in zip(*all_batch_iters):
                # Process each agent's batch through the shared network
                for batch in batch_tuple:
                    loss, batch_metrics = self._compute_loss(network, batch)

                    # Gradient update
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                    # Accumulate metrics
                    for key in metric_sums:
                        metric_sums[key] += batch_metrics[key]
                    n_batches += 1

        return metric_sums, n_batches

    def _compute_loss(
        self,
        network: ActorCritic,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined PPO loss for one minibatch."""
        # Get current policy outputs
        _, _, new_log_prob, entropy, new_value = network.get_action_and_value(
            batch['obs'],
            direction=batch['directions'],
            action_type=batch['action_types'],
            direction_mask=batch.get('direction_mask'),
            action_type_mask=batch.get('action_type_mask')
        )

        # Policy ratio
        log_ratio = new_log_prob - batch['log_probs']
        ratio = log_ratio.exp()

        # Approximate KL divergence (for monitoring)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()

        # Normalize advantages
        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (clipped surrogate objective)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1 - self.config.clip_coef,
            1 + self.config.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((new_value - batch['returns']) ** 2).mean()

        # Auxiliary value losses for decomposed reward streams
        aux_v_loss = torch.tensor(0.0, device=self.device)
        if 'returns_survival' in batch and batch['returns_survival'].abs().sum() > 0:
            aux_values = network.get_auxiliary_values(batch['obs'])
            v_loss_survival = 0.5 * ((aux_values['survival'] - batch['returns_survival']) ** 2).mean()
            v_loss_resource = 0.5 * ((aux_values['resource'] - batch['returns_resource']) ** 2).mean()
            v_loss_social = 0.5 * ((aux_values['social'] - batch['returns_social']) ** 2).mean()
            aux_v_loss = v_loss_survival + v_loss_resource + v_loss_social

        # Entropy bonus
        entropy_loss = entropy.mean()

        # Total loss - use higher entropy in coop phases to encourage exploration of COOP action
        ent_coef = self.config.ent_coef_coop if self.clone_mode else self.config.ent_coef
        loss = pg_loss + self.config.vf_coef * v_loss + self.config.aux_vf_coef * aux_v_loss - ent_coef * entropy_loss

        batch_metrics = {
            'policy_loss': pg_loss.detach(),
            'value_loss': v_loss.detach(),
            'aux_value_loss': aux_v_loss.detach(),
            'entropy': entropy_loss.detach(),
            'total_loss': loss.detach(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction
        }

        return loss, batch_metrics


class VmapPPO:
    """
    Parallelized version of IndependentPPO using torch.func.vmap.

    Each agent has its own network, but forward passes are batched
    across all networks using vmap for GPU parallelism.

    Supports "clone mode" for cooperation curriculum where agent 1 uses agent 0's network.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.n_agents = config.n_agents

        # Number of agents to actually train
        self.n_active = config.pretrain_spawn_agents if config.pretrain_mode else config.n_agents

        # Clone mode: agent 1 uses agent 0's network (for cooperation curriculum)
        self.clone_mode = False

        # Keep separate networks (for heterogeneous policies)
        self.networks = nn.ModuleList([
            ActorCritic(config).to(device)
            for _ in range(config.n_agents)
        ])

        # Separate optimizers
        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=config.learning_rate, eps=1e-5)
            for net in self.networks
        ]

        # Base network for functional calls (architecture template)
        self.base_network = ActorCritic(config).to(device)

    def set_curriculum_phase(self, phase: int):
        """Update n_active and clone_mode based on curriculum phase (legacy method)."""
        if phase >= 6:
            # Cooperation phases: 2 active agents, both use network[0]
            self.n_active = 2
            self.clone_mode = True
        else:
            # Solo phases: 1 active agent
            self.n_active = self.config.pretrain_spawn_agents if self.config.pretrain_mode else self.config.n_agents
            self.clone_mode = False

    def set_n_active(self, n: int):
        """Set the number of active agents directly."""
        self.n_active = min(n, self.n_agents)

    def set_clone_mode(self, enabled: bool):
        """Enable or disable clone mode (all agents share network[0])."""
        self.clone_mode = enabled

    def _get_stacked_params(self):
        """Stack parameters from all active networks for vmap."""
        return func.stack_module_state(self.networks[:self.n_active])

    def get_actions_and_values(
        self,
        obs: Dict[str, torch.Tensor],
        direction_mask: torch.Tensor = None,
        action_type_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions and values from all agents in parallel using vmap.

        Args:
            obs: Batched observations [n_agents, ...]
            direction_mask: Optional batched direction mask [n_agents, 5]
            action_type_mask: Optional batched action type mask [n_agents, 5]

        Returns:
            directions, action_types, log_probs, entropies, values: all [n_agents]
        """
        from torch.distributions import Categorical

        # In clone mode, fall back to sequential processing with shared network
        if self.clone_mode:
            return self._get_actions_and_values_sequential(obs, direction_mask, action_type_mask)

        # Stack parameters from all active networks
        params, buffers = self._get_stacked_params()

        # Add batch dimension for vmap: [n_agents, ...] -> [n_agents, 1, ...]
        stacked_obs = {k: v[:self.n_active].unsqueeze(1) for k, v in obs.items()}

        # Stateless forward function - returns direction_logits, action_type_logits, value
        def forward_single(params, buffers, obs_i):
            # functional_call calls forward() which returns (dir_logits, act_logits, value)
            return func.functional_call(
                self.base_network,
                (params, buffers),
                args=(obs_i,)
            )

        # Vectorize: run all networks in parallel
        batched_forward = func.vmap(forward_single, in_dims=(0, 0, 0))

        # Execute batched forward pass - get logits [n_active, 1, ...]
        direction_logits, action_type_logits, values = batched_forward(params, buffers, stacked_obs)

        # Squeeze batch dim: [n_active, 1, ...] -> [n_active, ...]
        direction_logits = direction_logits.squeeze(1)  # [n_active, 5]
        action_type_logits = action_type_logits.squeeze(1)  # [n_active, 5]
        values = values.squeeze(1).squeeze(-1)    # [n_active]

        LARGE_NEG = -1e8

        # Apply direction mask if provided (batched)
        if direction_mask is not None:
            mask = direction_mask[:self.n_active]
            direction_logits = direction_logits.masked_fill(~mask, LARGE_NEG)

        # Apply action type mask if provided (batched)
        if action_type_mask is not None:
            mask = action_type_mask[:self.n_active]
            action_type_logits = action_type_logits.masked_fill(~mask, LARGE_NEG)

        # Create distributions and sample (batched)
        direction_dist = Categorical(logits=direction_logits)
        action_type_dist = Categorical(logits=action_type_logits)
        directions = direction_dist.sample()  # [n_active]
        action_types = action_type_dist.sample()  # [n_active]

        # Compute combined log probs and entropy
        log_probs = direction_dist.log_prob(directions) + action_type_dist.log_prob(action_types)
        entropies = direction_dist.entropy() + action_type_dist.entropy()

        # Handle inactive agents (pad with dummy values)
        if self.n_active < self.n_agents:
            pad_size = self.n_agents - self.n_active
            directions = torch.cat([directions, torch.full((pad_size,), 4, device=self.device)])  # STAY
            action_types = torch.cat([action_types, torch.zeros(pad_size, dtype=torch.long, device=self.device)])  # MOVE
            log_probs = torch.cat([log_probs, torch.zeros(pad_size, device=self.device)])
            entropies = torch.cat([entropies, torch.zeros(pad_size, device=self.device)])
            values = torch.cat([values, torch.zeros(pad_size, device=self.device)])

        return directions, action_types, log_probs, entropies, values

    def _get_actions_and_values_sequential(
        self,
        obs: Dict[str, torch.Tensor],
        direction_mask: torch.Tensor = None,
        action_type_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequential fallback for clone mode - all agents use network[0]."""
        directions = []
        action_types = []
        log_probs = []
        entropies = []
        values = []

        net = self.networks[0]  # Shared network in clone mode

        for i in range(self.n_active):
            obs_i = {k: v[i:i+1] for k, v in obs.items()}
            dir_mask_i = None
            act_mask_i = None
            if direction_mask is not None:
                dir_mask_i = direction_mask[i:i+1]
            if action_type_mask is not None:
                act_mask_i = action_type_mask[i:i+1]

            dir_i, act_i, log_p, ent, val = net.get_action_and_value(
                obs_i, direction_mask=dir_mask_i, action_type_mask=act_mask_i
            )

            directions.append(dir_i)
            action_types.append(act_i)
            log_probs.append(log_p)
            entropies.append(ent)
            values.append(val)

        # Pad for inactive agents
        for i in range(self.n_active, self.n_agents):
            directions.append(torch.tensor([4], device=self.device))  # STAY
            action_types.append(torch.tensor([0], device=self.device))  # MOVE
            log_probs.append(torch.tensor([0.0], device=self.device))
            entropies.append(torch.tensor([0.0], device=self.device))
            values.append(torch.tensor([0.0], device=self.device))

        return (
            torch.cat(directions),
            torch.cat(action_types),
            torch.cat(log_probs),
            torch.cat(entropies),
            torch.cat(values)
        )

    def get_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get value estimates from all agents in parallel.

        Args:
            obs: Batched observations [n_agents, ...]

        Returns:
            values: [n_agents]
        """
        # In clone mode, fall back to sequential processing
        if self.clone_mode:
            return self._get_values_sequential(obs)

        params, buffers = self._get_stacked_params()
        stacked_obs = {k: v[:self.n_active].unsqueeze(1) for k, v in obs.items()}

        # Use forward() and extract just the value (3rd output now)
        def get_value_via_forward(params, buffers, obs_i):
            _, _, val = func.functional_call(
                self.base_network,
                (params, buffers),
                args=(obs_i,)
            )
            return val

        batched_value = func.vmap(get_value_via_forward, in_dims=(0, 0, 0))
        values = batched_value(params, buffers, stacked_obs).squeeze(1).squeeze(-1)

        # Pad for inactive agents
        if self.n_active < self.n_agents:
            pad_size = self.n_agents - self.n_active
            values = torch.cat([values, torch.zeros(pad_size, device=self.device)])

        return values

    def get_auxiliary_values(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get auxiliary value estimates from all agents.

        Args:
            obs: Batched observations [n_agents, ...]

        Returns:
            Dict with 'survival', 'resource', 'social' keys, each [n_agents] tensor
        """
        # Sequential processing for simplicity (auxiliary values are cheap)
        survival_values = []
        resource_values = []
        social_values = []

        for i in range(self.n_active):
            net_idx = 0 if self.clone_mode else i
            net = self.networks[net_idx]
            obs_i = {k: v[i:i+1] for k, v in obs.items()}
            aux_vals = net.get_auxiliary_values(obs_i)
            survival_values.append(aux_vals['survival'])
            resource_values.append(aux_vals['resource'])
            social_values.append(aux_vals['social'])

        # Pad for inactive agents
        for i in range(self.n_active, self.n_agents):
            survival_values.append(torch.tensor([0.0], device=self.device))
            resource_values.append(torch.tensor([0.0], device=self.device))
            social_values.append(torch.tensor([0.0], device=self.device))

        return {
            'survival': torch.cat(survival_values),
            'resource': torch.cat(resource_values),
            'social': torch.cat(social_values),
        }

    def _get_values_sequential(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sequential fallback for clone mode - all agents use network[0]."""
        values = []
        net = self.networks[0]

        for i in range(self.n_active):
            obs_i = {k: v[i:i+1] for k, v in obs.items()}
            val = net.get_value(obs_i).squeeze(-1)
            values.append(val)

        # Pad for inactive agents
        for i in range(self.n_active, self.n_agents):
            values.append(torch.tensor([0.0], device=self.device))

        return torch.cat(values)

    def get_action_probs(
        self,
        obs: Dict[str, torch.Tensor],
        direction_mask: torch.Tensor = None,
        action_type_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action probabilities for all active agents (for visualization).

        Args:
            obs: Batched observations [n_agents, ...]
            direction_mask: Optional batched direction mask [n_agents, 5]
            action_type_mask: Optional batched action type mask [n_agents, 5]

        Returns:
            direction_probs: [n_agents, 5] - probabilities for directions
            action_type_probs: [n_agents, 5] - probabilities for action types
        """
        import torch.nn.functional as F

        direction_probs_list = []
        action_type_probs_list = []

        for i in range(self.n_active):
            # Get the network for this agent (shared in clone mode)
            net_idx = 0 if self.clone_mode else i
            net = self.networks[net_idx]

            # Extract single agent observation
            obs_i = {k: v[i:i+1] for k, v in obs.items()}

            # Extract single agent action masks if provided
            dir_mask_i = None
            act_mask_i = None
            if direction_mask is not None:
                dir_mask_i = direction_mask[i:i+1]
            if action_type_mask is not None:
                act_mask_i = action_type_mask[i:i+1]

            with torch.no_grad():
                # Forward pass to get logits
                direction_logits, action_type_logits, _ = net.forward(obs_i)

                # Apply masks
                LARGE_NEG = -1e8
                if dir_mask_i is not None:
                    direction_logits = direction_logits.masked_fill(~dir_mask_i, LARGE_NEG)
                if act_mask_i is not None:
                    action_type_logits = action_type_logits.masked_fill(~act_mask_i, LARGE_NEG)

                # Convert to probabilities
                direction_probs = F.softmax(direction_logits, dim=-1)
                action_type_probs = F.softmax(action_type_logits, dim=-1)
                direction_probs_list.append(direction_probs)
                action_type_probs_list.append(action_type_probs)

        # Pad for inactive agents with uniform distributions
        for i in range(self.n_active, self.n_agents):
            direction_probs_list.append(torch.ones(1, 5, device=self.device) / 5)
            action_type_probs_list.append(torch.ones(1, 5, device=self.device) / 5)

        return torch.cat(direction_probs_list), torch.cat(action_type_probs_list)

    def update(self, buffers: List[SingleAgentBuffer]) -> Dict[str, float]:
        """
        Update each agent's network using its own buffer.
        (Sequential updates - parallelizing this is optional future work)

        Args:
            buffers: List of SingleAgentBuffer, one per agent

        Returns:
            Averaged metrics across all agents
        """
        all_metrics = {
            'policy_loss': torch.tensor(0.0, device=self.device),
            'value_loss': torch.tensor(0.0, device=self.device),
            'aux_value_loss': torch.tensor(0.0, device=self.device),
            'entropy': torch.tensor(0.0, device=self.device),
            'total_loss': torch.tensor(0.0, device=self.device),
            'approx_kl': torch.tensor(0.0, device=self.device),
            'clip_fraction': torch.tensor(0.0, device=self.device)
        }

        total_batches = 0

        # In clone mode, combine all active agents' buffers and train network[0]
        if self.clone_mode:
            combined_metrics, n_batches = self._update_with_combined_buffers(
                self.networks[0],
                self.optimizers[0],
                [buffers[i] for i in range(self.n_active)]
            )
            for key in all_metrics:
                all_metrics[key] += combined_metrics[key]
            total_batches += n_batches
        else:
            # Normal mode: Sequential updates (each agent has separate optimizer)
            for i in range(self.n_active):
                agent_metrics, n_batches = self._update_single_agent(
                    self.networks[i],
                    self.optimizers[i],
                    buffers[i]
                )

                for key in all_metrics:
                    all_metrics[key] += agent_metrics[key]
                total_batches += n_batches

        metrics = {key: (val / total_batches).item() for key, val in all_metrics.items()}
        return metrics

    def _update_single_agent(
        self,
        network: ActorCritic,
        optimizer: torch.optim.Optimizer,
        buffer: SingleAgentBuffer
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Perform PPO update for a single agent."""
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
            for batch in buffer.get_batches():
                loss, batch_metrics = self._compute_loss(network, batch)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), self.config.max_grad_norm)
                optimizer.step()

                for key in metric_sums:
                    metric_sums[key] += batch_metrics[key]
                n_batches += 1

        return metric_sums, n_batches

    def _update_with_combined_buffers(
        self,
        network: ActorCritic,
        optimizer: torch.optim.Optimizer,
        buffers: List[SingleAgentBuffer]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Perform PPO update using combined data from multiple agents' buffers.
        Used in clone mode where all agents share a single network.
        """
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
            all_batch_iters = [buf.get_batches() for buf in buffers]
            for batch_tuple in zip(*all_batch_iters):
                for batch in batch_tuple:
                    loss, batch_metrics = self._compute_loss(network, batch)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                    for key in metric_sums:
                        metric_sums[key] += batch_metrics[key]
                    n_batches += 1

        return metric_sums, n_batches

    def _compute_loss(
        self,
        network: ActorCritic,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined PPO loss for one minibatch."""
        _, _, new_log_prob, entropy, new_value = network.get_action_and_value(
            batch['obs'],
            direction=batch['directions'],
            action_type=batch['action_types'],
            direction_mask=batch.get('direction_mask'),
            action_type_mask=batch.get('action_type_mask')
        )

        log_ratio = new_log_prob - batch['log_probs']
        ratio = log_ratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()

        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1 - self.config.clip_coef,
            1 + self.config.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((new_value - batch['returns']) ** 2).mean()

        # Auxiliary value losses for decomposed reward streams
        aux_v_loss = torch.tensor(0.0, device=self.device)
        if 'returns_survival' in batch and batch['returns_survival'].abs().sum() > 0:
            aux_values = network.get_auxiliary_values(batch['obs'])
            v_loss_survival = 0.5 * ((aux_values['survival'] - batch['returns_survival']) ** 2).mean()
            v_loss_resource = 0.5 * ((aux_values['resource'] - batch['returns_resource']) ** 2).mean()
            v_loss_social = 0.5 * ((aux_values['social'] - batch['returns_social']) ** 2).mean()
            aux_v_loss = v_loss_survival + v_loss_resource + v_loss_social

        entropy_loss = entropy.mean()

        # Use higher entropy in coop phases to encourage exploration of COOP action
        ent_coef = self.config.ent_coef_coop if self.clone_mode else self.config.ent_coef
        loss = pg_loss + self.config.vf_coef * v_loss + self.config.aux_vf_coef * aux_v_loss - ent_coef * entropy_loss

        batch_metrics = {
            'policy_loss': pg_loss.detach(),
            'value_loss': v_loss.detach(),
            'aux_value_loss': aux_v_loss.detach(),
            'entropy': entropy_loss.detach(),
            'total_loss': loss.detach(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction
        }

        return loss, batch_metrics
