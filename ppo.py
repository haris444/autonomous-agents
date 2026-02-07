"""
Proximal Policy Optimization (PPO) algorithm.

Handles dual-action policy with shared value function.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.func as func

from config import Config
from network import ActorCritic
from buffer import RolloutBuffer, SingleAgentBuffer


def load_state_dict_flexible(net, state_dict):
    """Load state dict, resizing mismatched layers (e.g. signal_encoder) via slice/pad.
    Returns True if any layers were resized (caller should skip optimizer state)."""
    try:
        net.load_state_dict(state_dict, strict=True)
        return False
    except RuntimeError as e:
        if "size mismatch" not in str(e):
            raise
        model_dict = net.state_dict()
        resized = []
        for k, v in list(state_dict.items()):
            if k in model_dict and v.shape != model_dict[k].shape:
                target_shape = model_dict[k].shape
                new_v = torch.zeros(target_shape, dtype=v.dtype, device=v.device)
                slices = tuple(slice(0, min(s, t)) for s, t in zip(v.shape, target_shape))
                new_v[slices] = v[slices]
                state_dict[k] = new_v
                resized.append(f"{k}: {list(v.shape)} -> {list(target_shape)}")
        net.load_state_dict(state_dict, strict=True)
        for r in resized:
            print(f"  [Resized] {r}")
        return True


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
        """Compute combined PPO loss for one minibatch.

        Uses forward(return_aux=True) to get all heads in a single encode pass.
        """
        obs = batch['obs']
        has_aux = 'returns_survival' in batch and batch['returns_survival'].abs().sum() > 0

        # Single forward pass for all heads
        result = network.forward(obs, return_aux=has_aux)
        if has_aux:
            dir_logits, act_logits, value, v_surv, v_res, v_soc = result
        else:
            dir_logits, act_logits, value = result

        # Apply action masks
        LARGE_NEG = -1e8
        direction_mask = batch.get('direction_mask')
        action_type_mask = batch.get('action_type_mask')
        if direction_mask is not None:
            dir_logits = dir_logits.masked_fill(~direction_mask, LARGE_NEG)
        if action_type_mask is not None:
            act_logits = act_logits.masked_fill(~action_type_mask, LARGE_NEG)

        # Compute log probs and entropy from logits
        from torch.distributions import Categorical
        dir_dist = Categorical(logits=dir_logits)
        act_dist = Categorical(logits=act_logits)
        new_log_prob = dir_dist.log_prob(batch['directions']) + act_dist.log_prob(batch['action_types'])
        entropy = dir_dist.entropy() + act_dist.entropy()
        new_value = value.squeeze(-1)

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

        # Auxiliary value losses (already computed from same forward pass)
        aux_v_loss = torch.tensor(0.0, device=self.device)
        if has_aux:
            v_loss_survival = 0.5 * ((v_surv.squeeze(-1) - batch['returns_survival']) ** 2).mean()
            v_loss_resource = 0.5 * ((v_res.squeeze(-1) - batch['returns_resource']) ** 2).mean()
            v_loss_social = 0.5 * ((v_soc.squeeze(-1) - batch['returns_social']) ** 2).mean()
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

    def _get_vmapped_forward(self):
        """Get cached vmapped forward function (return_aux=True), creating on first use."""
        cached = getattr(self, '_cached_vmapped_forward', None)
        if cached is not None:
            return cached
        base_network = self.base_network

        def forward_batch(params, buffers, obs_batch):
            return func.functional_call(
                base_network,
                (params, buffers),
                args=(obs_batch,),
                kwargs={'return_aux': True}
            )

        self._cached_vmapped_forward = func.vmap(forward_batch, in_dims=(0, 0, 0))
        return self._cached_vmapped_forward

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
        """Compute combined PPO loss for one minibatch.

        Uses forward(return_aux=True) to get all heads in a single encode pass,
        avoiding the double-encoding that previously happened when
        get_action_and_value() and get_auxiliary_values() each called _encode().
        """
        obs = batch['obs']
        has_aux = 'returns_survival' in batch and batch['returns_survival'].abs().sum() > 0

        # Single forward pass for all heads
        result = network.forward(obs, return_aux=has_aux)
        if has_aux:
            dir_logits, act_logits, value, v_surv, v_res, v_soc = result
        else:
            dir_logits, act_logits, value = result

        # Apply action masks
        LARGE_NEG = -1e8
        direction_mask = batch.get('direction_mask')
        action_type_mask = batch.get('action_type_mask')
        if direction_mask is not None:
            dir_logits = dir_logits.masked_fill(~direction_mask, LARGE_NEG)
        if action_type_mask is not None:
            act_logits = act_logits.masked_fill(~action_type_mask, LARGE_NEG)

        # Compute log probs and entropy from logits
        from torch.distributions import Categorical
        dir_dist = Categorical(logits=dir_logits)
        act_dist = Categorical(logits=act_logits)
        new_log_prob = dir_dist.log_prob(batch['directions']) + act_dist.log_prob(batch['action_types'])
        entropy = dir_dist.entropy() + act_dist.entropy()
        new_value = value.squeeze(-1)

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

        # Auxiliary value losses (already computed from same forward pass)
        aux_v_loss = torch.tensor(0.0, device=self.device)
        if has_aux:
            v_loss_survival = 0.5 * ((v_surv.squeeze(-1) - batch['returns_survival']) ** 2).mean()
            v_loss_resource = 0.5 * ((v_res.squeeze(-1) - batch['returns_resource']) ** 2).mean()
            v_loss_social = 0.5 * ((v_soc.squeeze(-1) - batch['returns_social']) ** 2).mean()
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

    # =========================================================================
    # VECTORIZED ENVIRONMENT METHODS
    # Process [n_envs, n_agents, ...] shaped inputs for parallel training
    # =========================================================================

    def vec_get_actions_and_values(
        self,
        obs: Dict[str, torch.Tensor],
        direction_mask: torch.Tensor = None,
        action_type_mask: torch.Tensor = None,
        n_envs: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get actions, values, and auxiliary values in a single forward pass.

        Args:
            obs: Batched observations [n_envs, n_agents, ...]
            direction_mask: [n_envs, n_agents, 5]
            action_type_mask: [n_envs, n_agents, 5]
            n_envs: Number of environments (inferred from obs if not provided)

        Returns:
            directions, action_types, log_probs, entropies, values: all [n_envs, n_agents]
            aux_values: dict with 'survival', 'resource', 'social' keys, each [n_envs, n_agents]
        """
        # Infer n_envs from observations
        if n_envs is None:
            n_envs = obs['entity_tokens'].shape[0]
        n_agents = self.n_agents

        from torch.distributions import Categorical

        # Prepare inputs: [n_active, n_envs, ...]
        obs_transposed = {
            k: v[:, :self.n_active].transpose(0, 1) for k, v in obs.items()
        }

        dir_mask_transposed = None
        if direction_mask is not None:
             dir_mask_transposed = direction_mask[:, :self.n_active].transpose(0, 1)

        act_mask_transposed = None
        if action_type_mask is not None:
             act_mask_transposed = action_type_mask[:, :self.n_active].transpose(0, 1)

        if self.clone_mode:
            obs_flat = {
                k: v.reshape(-1, *v.shape[2:]) for k, v in obs_transposed.items()
            }
            dir_mask_flat = dir_mask_transposed.reshape(-1, 5) if dir_mask_transposed is not None else None
            act_mask_flat = act_mask_transposed.reshape(-1, 5) if act_mask_transposed is not None else None

            # Single forward pass with aux values
            net = self.networks[0]
            dir_logits, act_logits, val, v_surv, v_res, v_soc = net.forward(
                obs_flat, return_aux=True
            )

            # Apply masks and sample actions
            LARGE_NEG = -1e8
            if dir_mask_flat is not None:
                dir_logits = dir_logits.masked_fill(~dir_mask_flat, LARGE_NEG)
            if act_mask_flat is not None:
                act_logits = act_logits.masked_fill(~act_mask_flat, LARGE_NEG)

            dir_dist = Categorical(logits=dir_logits)
            act_dist = Categorical(logits=act_logits)
            dirs = dir_dist.sample()
            acts = act_dist.sample()
            lps = dir_dist.log_prob(dirs) + act_dist.log_prob(acts)
            ents = dir_dist.entropy() + act_dist.entropy()

            # Reshape: [n_active * n_envs] -> [n_active, n_envs] -> [n_envs, n_active]
            directions = dirs.view(self.n_active, n_envs).transpose(0, 1)
            action_types = acts.view(self.n_active, n_envs).transpose(0, 1)
            log_probs = lps.view(self.n_active, n_envs).transpose(0, 1)
            entropies = ents.view(self.n_active, n_envs).transpose(0, 1)
            values = val.squeeze(-1).view(self.n_active, n_envs).transpose(0, 1)
            s_vals = v_surv.squeeze(-1).view(self.n_active, n_envs).transpose(0, 1)
            r_vals = v_res.squeeze(-1).view(self.n_active, n_envs).transpose(0, 1)
            soc_vals = v_soc.squeeze(-1).view(self.n_active, n_envs).transpose(0, 1)

        else:
            # Independent mode: vmap over n_active agents
            params, buffers = self._get_stacked_params()
            batched_forward = self._get_vmapped_forward()

            # Returns [n_active, n_envs, ...] for each of 6 outputs
            direction_logits, action_type_logits, values_out, v_surv, v_res, v_soc = batched_forward(
                params, buffers, obs_transposed
            )

            values_out = values_out.squeeze(-1)

            # Masking
            LARGE_NEG = -1e8
            if dir_mask_transposed is not None:
                direction_logits = direction_logits.masked_fill(~dir_mask_transposed, LARGE_NEG)
            if act_mask_transposed is not None:
                action_type_logits = action_type_logits.masked_fill(~act_mask_transposed, LARGE_NEG)

            # Sampling
            direction_dist = Categorical(logits=direction_logits)
            action_type_dist = Categorical(logits=action_type_logits)

            directions_out = direction_dist.sample()
            action_types_out = action_type_dist.sample()

            log_probs_out = direction_dist.log_prob(directions_out) + action_type_dist.log_prob(action_types_out)
            entropies_out = direction_dist.entropy() + action_type_dist.entropy()

            # Transpose results back to [n_envs, n_active]
            directions = directions_out.transpose(0, 1)
            action_types = action_types_out.transpose(0, 1)
            log_probs = log_probs_out.transpose(0, 1)
            entropies = entropies_out.transpose(0, 1)
            values = values_out.transpose(0, 1)
            s_vals = v_surv.squeeze(-1).transpose(0, 1)
            r_vals = v_res.squeeze(-1).transpose(0, 1)
            soc_vals = v_soc.squeeze(-1).transpose(0, 1)

        # Handle inactive agents (padding)
        if self.n_active < n_agents:
            pad_size = n_agents - self.n_active
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

        aux_values = {
            'survival': s_vals,
            'resource': r_vals,
            'social': soc_vals
        }
        return directions, action_types, log_probs, entropies, values, aux_values

    def vec_get_values(
        self,
        obs: Dict[str, torch.Tensor],
        n_envs: int = None
    ) -> torch.Tensor:
        """
        Get values for vectorized environments.

        Args:
            obs: Batched observations [n_envs, n_agents, ...]
            n_envs: Number of environments (inferred from obs if not provided)

        Returns:
            values: [n_envs, n_agents]
        """
        if n_envs is None:
            n_envs = obs['entity_tokens'].shape[0]
        n_agents = self.n_agents

        # Prepare inputs: [n_active, n_envs, ...]
        obs_transposed = {k: v[:, :self.n_active].transpose(0, 1) for k, v in obs.items()}

        if self.clone_mode:
            # Flatten [n_active, n_envs] -> [n_active * n_envs, ...]
            obs_flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs_transposed.items()}

            # Batch all active agents*envs through shared network
            net = self.networks[0]
            vals = net.get_value(obs_flat).squeeze(-1)

            # Reshape [n_active * n_envs] -> [n_active, n_envs] -> [n_envs, n_active]
            values = vals.view(self.n_active, n_envs).transpose(0, 1)
        
        else:
            params, buffers = self._get_stacked_params()
            batched_forward = self._get_vmapped_forward()

            # Reuse cached forward (returns 6 outputs, we only need value)
            _, _, values_out, _, _, _ = batched_forward(params, buffers, obs_transposed)

            # [n_active, n_envs] -> [n_envs, n_active]
            values = values_out.squeeze(-1).transpose(0, 1)

        # Pad for inactive agents
        if self.n_active < n_agents:
            pad_size = n_agents - self.n_active
            pad_val = torch.zeros((n_envs, pad_size), device=self.device)
            values = torch.cat([values, pad_val], dim=1)

        return values

    def vec_get_auxiliary_values(
        self,
        obs: Dict[str, torch.Tensor],
        n_envs: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get auxiliary values for vectorized environments using vmap.

        Args:
            obs: Batched observations [n_envs, n_agents, ...]
            n_envs: Number of environments (inferred from obs if not provided)

        Returns:
            Dict with 'survival', 'resource', 'social' keys, each [n_envs, n_agents]
        """
        if n_envs is None:
            n_envs = obs['entity_tokens'].shape[0]
        n_agents = self.n_agents

        # Prepare inputs: [n_active, n_envs, ...]
        obs_transposed = {k: v[:, :self.n_active].transpose(0, 1) for k, v in obs.items()}

        if self.clone_mode:
            # Flatten [n_active, n_envs] -> [n_active * n_envs, ...]
            obs_flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs_transposed.items()}
            net = self.networks[0]
            # forward with return_aux=True: single encode for all heads
            _, _, _, v_surv, v_res, v_soc = net.forward(obs_flat, return_aux=True)
            # Reshape [n_active*n_envs, 1] -> [n_active, n_envs] -> [n_envs, n_active]
            s_vals = v_surv.squeeze(-1).view(self.n_active, n_envs).transpose(0, 1)
            r_vals = v_res.squeeze(-1).view(self.n_active, n_envs).transpose(0, 1)
            soc_vals = v_soc.squeeze(-1).view(self.n_active, n_envs).transpose(0, 1)
        else:
            # Independent mode: vmap over agents
            params, buffers = self._get_stacked_params()
            batched_forward = self._get_vmapped_forward()

            # Returns [n_active, n_envs, ...] for each output
            _, _, _, v_surv, v_res, v_soc = batched_forward(params, buffers, obs_transposed)

            # [n_active, n_envs, 1] -> [n_active, n_envs] -> [n_envs, n_active]
            s_vals = v_surv.squeeze(-1).transpose(0, 1)
            r_vals = v_res.squeeze(-1).transpose(0, 1)
            soc_vals = v_soc.squeeze(-1).transpose(0, 1)

        # Pad for inactive agents
        if self.n_active < n_agents:
            pad_size = n_agents - self.n_active
            pad = torch.zeros((n_envs, pad_size), device=self.device)
            s_vals = torch.cat([s_vals, pad], dim=1)
            r_vals = torch.cat([r_vals, pad], dim=1)
            soc_vals = torch.cat([soc_vals, pad], dim=1)

        return {
            'survival': s_vals,
            'resource': r_vals,
            'social': soc_vals
        }

    def update_from_vec_buffer(self, buffer: 'VecBuffer') -> Dict[str, float]:
        """
        Update networks from VecBuffer (vectorized environment buffer).

        In clone mode, all agents share weights so we update once with all data.
        In independent mode, uses vmapped parallel update across all agents.
        """
        if self.clone_mode:
            # Clone mode: all data → network[0]
            epoch_metrics = defaultdict(list)
            for epoch in range(self.config.update_epochs):
                for batch in buffer.get_batches(n_active=self.n_active):
                    loss, metrics = self._compute_loss(self.networks[0], batch)
                    self.optimizers[0].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.networks[0].parameters(), self.config.max_grad_norm)
                    self.optimizers[0].step()

                    for k, v in metrics.items():
                        epoch_metrics[k].append(v)
            return {k: torch.stack(v).mean().item() for k, v in epoch_metrics.items()}
        else:
            # Independent mode: vmapped parallel update
            return self._vmapped_update_from_vec_buffer(buffer)

    def _make_loss_fn(self, has_aux: bool):
        """Create a pure-functional PPO loss function (closed over config + has_aux)."""
        clip_coef = self.config.clip_coef
        vf_coef = self.config.vf_coef
        aux_vf_coef = self.config.aux_vf_coef
        ent_coef = self.config.ent_coef
        base_network = self.base_network
        device = self.device

        def loss_fn(params, buffs, obs, directions, action_types,
                   old_log_probs, advantages, returns,
                   direction_mask, action_type_mask,
                   returns_survival, returns_resource, returns_social):
            """Pure-functional PPO loss. All args are tensors for vmap."""
            if has_aux:
                dir_logits, act_logits, value, v_surv, v_res, v_soc = func.functional_call(
                    base_network, (params, buffs),
                    args=(obs,), kwargs={'return_aux': True}
                )
            else:
                dir_logits, act_logits, value = func.functional_call(
                    base_network, (params, buffs), args=(obs,)
                )

            LARGE_NEG = -1e8
            dir_logits = dir_logits.masked_fill(~direction_mask, LARGE_NEG)
            act_logits = act_logits.masked_fill(~action_type_mask, LARGE_NEG)

            dir_log_sm = torch.nn.functional.log_softmax(dir_logits, dim=-1)
            act_log_sm = torch.nn.functional.log_softmax(act_logits, dim=-1)
            dir_log_prob = dir_log_sm.gather(1, directions.unsqueeze(-1)).squeeze(-1)
            act_log_prob = act_log_sm.gather(1, action_types.unsqueeze(-1)).squeeze(-1)
            new_log_prob = dir_log_prob + act_log_prob

            dir_probs = torch.nn.functional.softmax(dir_logits, dim=-1)
            act_probs = torch.nn.functional.softmax(act_logits, dim=-1)
            entropy = -(dir_probs * dir_log_sm).sum(-1) - (act_probs * act_log_sm).sum(-1)

            new_value = value.squeeze(-1)

            log_ratio = new_log_prob - old_log_probs
            ratio = log_ratio.exp()

            adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((new_value - returns) ** 2).mean()

            aux_v_loss = torch.tensor(0.0, device=device)
            if has_aux:
                aux_v_loss = (
                    0.5 * ((v_surv.squeeze(-1) - returns_survival) ** 2).mean() +
                    0.5 * ((v_res.squeeze(-1) - returns_resource) ** 2).mean() +
                    0.5 * ((v_soc.squeeze(-1) - returns_social) ** 2).mean()
                )

            entropy_loss = entropy.mean()
            loss = pg_loss + vf_coef * v_loss + aux_vf_coef * aux_v_loss - ent_coef * entropy_loss

            approx_kl = ((ratio - 1) - log_ratio).mean().detach()
            clip_frac = ((ratio - 1.0).abs() > clip_coef).float().mean().detach()

            metrics = torch.stack([
                pg_loss.detach(), v_loss.detach(), aux_v_loss.detach(),
                entropy_loss.detach(), loss.detach(), approx_kl, clip_frac
            ])
            return loss, metrics

        return loss_fn

    def _get_vmapped_grad_fn(self, has_aux: bool):
        """Get cached vmapped grad function, creating on first use."""
        attr = '_vmapped_grad_fn_aux' if has_aux else '_vmapped_grad_fn_noaux'
        cached = getattr(self, attr, None)
        if cached is not None:
            return cached
        loss_fn = self._make_loss_fn(has_aux)
        vmapped_grad_fn = func.vmap(
            func.grad(loss_fn, has_aux=True),
            in_dims=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
        setattr(self, attr, vmapped_grad_fn)
        return vmapped_grad_fn

    def _vmapped_update_from_vec_buffer(self, buffer: 'VecBuffer') -> Dict[str, float]:
        """
        Vmapped parallel PPO update for independent agents.

        Uses func.vmap(func.grad(...)) to compute gradients for all agents
        in parallel, then applies them to individual networks sequentially
        (optimizer step is cheap relative to forward+backward).
        """
        metric_keys = ['policy_loss', 'value_loss', 'aux_value_loss',
                        'entropy', 'total_loss', 'approx_kl', 'clip_fraction']
        all_metrics = torch.zeros(len(metric_keys), device=self.device)
        n_batches = 0

        # Determine has_aux once upfront (consistent for entire update)
        has_aux = buffer.returns_survival.abs().sum() > 0

        # Get cached vmapped grad function (no retracing per minibatch)
        vmapped_grad_fn = self._get_vmapped_grad_fn(has_aux)

        for epoch in range(self.config.update_epochs):
            for batch in buffer.get_aligned_batches(n_active=self.n_active):
                # Stack current params from all active networks
                params, buffs = self._get_stacked_params()

                # Execute: get gradients and metrics for all agents in parallel
                grads, metrics_stacked = vmapped_grad_fn(
                    params, buffs,
                    batch['obs'], batch['directions'], batch['action_types'],
                    batch['log_probs'], batch['advantages'], batch['returns'],
                    batch['direction_mask'], batch['action_type_mask'],
                    batch['returns_survival'], batch['returns_resource'], batch['returns_social']
                )

                # Accumulate metrics (mean across agents)
                all_metrics += metrics_stacked.mean(dim=0)
                n_batches += 1

                # Apply gradients to individual networks
                for agent_idx in range(self.n_active):
                    net = self.networks[agent_idx]
                    opt = self.optimizers[agent_idx]

                    opt.zero_grad()

                    # Copy vmapped grads into network .grad fields
                    for name, param in net.named_parameters():
                        if param.requires_grad:
                            param.grad = grads[name][agent_idx]

                    nn.utils.clip_grad_norm_(net.parameters(), self.config.max_grad_norm)
                    opt.step()

        # Average metrics
        if n_batches > 0:
            all_metrics /= n_batches

        return {k: all_metrics[i].item() for i, k in enumerate(metric_keys)}
