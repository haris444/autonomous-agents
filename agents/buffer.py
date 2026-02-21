"""
Rollout Buffers for PPO training.

Stores trajectories from environment rollouts and computes
Generalized Advantage Estimation (GAE).
"""
from typing import Dict, Generator

import torch

from core.config import Config


class SingleAgentBuffer:
    """
    Rollout buffer for a single agent.

    Stores trajectories shaped [num_steps, ...] instead of [num_steps, n_agents, ...].
    Used with single-env training where each agent has its own buffer.
    """

    def __init__(self, config: Config, device: torch.device, agent_id: int):
        self.config = config
        self.device = device
        self.agent_id = agent_id
        self.num_steps = config.num_steps

        # Pre-allocate storage tensors - shape: [num_steps, ...]
        max_ent = config.max_entities
        token_dim = config.entity_token_dim
        n = config.n_agents

        # Observations - unified entity tokens (agents + food)
        self.entity_tokens_obs = torch.zeros((self.num_steps, max_ent, token_dim), device=device)
        self.entity_mask_obs = torch.zeros((self.num_steps, max_ent), device=device, dtype=torch.bool)
        self.signal_obs = torch.zeros((self.num_steps, n), device=device)
        self.self_hp_obs = torch.zeros((self.num_steps, 1), device=device)
        self.self_inventory_obs = torch.zeros((self.num_steps, 1), device=device)
        self.agent_ids = torch.zeros((self.num_steps,), device=device, dtype=torch.long)

        # Actions (factored action space)
        self.directions = torch.zeros((self.num_steps,), device=device, dtype=torch.long)
        self.action_types = torch.zeros((self.num_steps,), device=device, dtype=torch.long)

        # Other data
        self.log_probs = torch.zeros((self.num_steps,), device=device)
        self.rewards = torch.zeros((self.num_steps,), device=device)
        self.dones = torch.zeros((self.num_steps,), device=device)
        self.values = torch.zeros((self.num_steps,), device=device)

        # Computed after rollout
        self.advantages = torch.zeros((self.num_steps,), device=device)
        self.returns = torch.zeros((self.num_steps,), device=device)

        # Action masks (factored)
        self.direction_masks = torch.zeros((self.num_steps, config.n_directions), device=device, dtype=torch.bool)
        self.action_type_masks = torch.zeros((self.num_steps, config.n_action_types), device=device, dtype=torch.bool)

        # Decomposed rewards for auxiliary value heads
        self.rewards_survival = torch.zeros((self.num_steps,), device=device)
        self.rewards_resource = torch.zeros((self.num_steps,), device=device)
        self.rewards_social = torch.zeros((self.num_steps,), device=device)

        # Decomposed returns (computed in compute_gae)
        self.returns_survival = torch.zeros((self.num_steps,), device=device)
        self.returns_resource = torch.zeros((self.num_steps,), device=device)
        self.returns_social = torch.zeros((self.num_steps,), device=device)

        # Auxiliary values (stored during rollout)
        self.values_survival = torch.zeros((self.num_steps,), device=device)
        self.values_resource = torch.zeros((self.num_steps,), device=device)
        self.values_social = torch.zeros((self.num_steps,), device=device)

        self.step_idx = 0

    def reset(self) -> None:
        """Reset buffer for new rollout."""
        self.step_idx = 0

    def store(
        self,
        obs: Dict[str, torch.Tensor],
        direction: torch.Tensor,
        action_type: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        direction_mask: torch.Tensor = None,
        action_type_mask: torch.Tensor = None,
        reward_survival: torch.Tensor = None,
        reward_resource: torch.Tensor = None,
        reward_social: torch.Tensor = None,
        value_survival: torch.Tensor = None,
        value_resource: torch.Tensor = None,
        value_social: torch.Tensor = None
    ) -> None:
        """Store one step of experience for this single agent."""
        t = self.step_idx

        # Store observations (already single-agent tensors)
        self.entity_tokens_obs[t] = obs['entity_tokens']
        self.entity_mask_obs[t] = obs['entity_mask']
        self.signal_obs[t] = obs['signals']
        self.self_hp_obs[t] = obs['self_hp']
        self.self_inventory_obs[t] = obs['self_inventory']
        self.agent_ids[t] = obs['agent_id']

        # Store actions and values
        self.directions[t] = direction
        self.action_types[t] = action_type
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done.float() if isinstance(done, torch.Tensor) else float(done)
        self.values[t] = value

        # Store action masks if provided
        if direction_mask is not None:
            self.direction_masks[t] = direction_mask
        if action_type_mask is not None:
            self.action_type_masks[t] = action_type_mask

        # Store decomposed rewards if provided
        if reward_survival is not None:
            self.rewards_survival[t] = reward_survival
        if reward_resource is not None:
            self.rewards_resource[t] = reward_resource
        if reward_social is not None:
            self.rewards_social[t] = reward_social

        # Store auxiliary values if provided
        if value_survival is not None:
            self.values_survival[t] = value_survival
        if value_resource is not None:
            self.values_resource[t] = value_resource
        if value_social is not None:
            self.values_social[t] = value_social

        self.step_idx += 1

    def compute_gae(
        self,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        next_value_survival: torch.Tensor = None,
        next_value_resource: torch.Tensor = None,
        next_value_social: torch.Tensor = None
    ) -> None:
        """
        Compute Generalized Advantage Estimation for this single agent.

        Args:
            next_value: Value estimate for state after last step (scalar tensor)
            next_done: Whether episode ended after last step (scalar tensor)
            next_value_survival: Auxiliary value estimate for survival rewards
            next_value_resource: Auxiliary value estimate for resource rewards
            next_value_social: Auxiliary value estimate for social rewards
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        last_gae = torch.tensor(0.0, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done.float()
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]  # Fix: block bootstrap if CURRENT step is done
                next_values = self.values[t + 1]

            # TD error
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]

            # GAE
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Returns = advantages + values
        self.returns = self.advantages + self.values

        # Compute auxiliary returns (simple discounted returns, not GAE)
        # These are used for auxiliary value head supervision
        if next_value_survival is not None:
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_non_terminal = 1.0 - next_done.float()
                    next_ret_surv = next_value_survival
                    next_ret_res = next_value_resource
                    next_ret_soc = next_value_social
                else:
                    next_non_terminal = 1.0 - self.dones[t]  # Fix: block bootstrap if CURRENT step is done
                    next_ret_surv = self.returns_survival[t + 1]
                    next_ret_res = self.returns_resource[t + 1]
                    next_ret_soc = self.returns_social[t + 1]

                self.returns_survival[t] = self.rewards_survival[t] + gamma * next_ret_surv * next_non_terminal
                self.returns_resource[t] = self.rewards_resource[t] + gamma * next_ret_res * next_non_terminal
                self.returns_social[t] = self.rewards_social[t] + gamma * next_ret_soc * next_non_terminal

    def get_batches(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield minibatches for PPO update.

        For single agent: batch_size = num_steps
        """
        batch_size = self.num_steps
        # Minibatch size for single agent (divide by num_minibatches from config)
        minibatch_size = max(1, batch_size // self.config.num_minibatches)

        indices = torch.randperm(batch_size, device=self.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield {
                'obs': {
                    'entity_tokens': self.entity_tokens_obs[mb_indices],
                    'entity_mask': self.entity_mask_obs[mb_indices],
                    'signals': self.signal_obs[mb_indices],
                    'self_hp': self.self_hp_obs[mb_indices],
                    'self_inventory': self.self_inventory_obs[mb_indices],
                    'agent_id': self.agent_ids[mb_indices]
                },
                'directions': self.directions[mb_indices],
                'action_types': self.action_types[mb_indices],
                'log_probs': self.log_probs[mb_indices],
                'advantages': self.advantages[mb_indices],
                'returns': self.returns[mb_indices],
                'direction_mask': self.direction_masks[mb_indices],
                'action_type_mask': self.action_type_masks[mb_indices],
                # Auxiliary returns for decomposed value heads
                'returns_survival': self.returns_survival[mb_indices],
                'returns_resource': self.returns_resource[mb_indices],
                'returns_social': self.returns_social[mb_indices],
            }


class VecBuffer:
    """
    Rollout buffer for vectorized environments.

    Stores transitions from N parallel environments.
    Shape: [num_steps, n_envs, n_agents, ...] for main storage
    Flattens to [batch_size, ...] for PPO updates where batch_size = num_steps * n_envs * n_agents
    """

    def __init__(self, config: Config, device: torch.device, n_envs: int):
        self.config = config
        self.device = device
        self.n_envs = n_envs
        self.n_agents = config.n_agents
        self.num_steps = config.num_steps

        # Pre-allocate storage tensors
        # Shape: [num_steps, n_envs, n_agents, ...]
        max_ent = config.max_entities
        token_dim = config.entity_token_dim
        n = config.n_agents

        # Observations
        self.entity_tokens_obs = torch.zeros(
            (self.num_steps, n_envs, n, max_ent, token_dim), device=device
        )
        self.entity_mask_obs = torch.zeros(
            (self.num_steps, n_envs, n, max_ent), device=device, dtype=torch.bool
        )
        self.signal_obs = torch.zeros(
            (self.num_steps, n_envs, n, n), device=device
        )
        self.self_hp_obs = torch.zeros(
            (self.num_steps, n_envs, n, 1), device=device
        )
        self.self_inventory_obs = torch.zeros(
            (self.num_steps, n_envs, n, 1), device=device
        )
        self.agent_ids = torch.zeros(
            (self.num_steps, n_envs, n), device=device, dtype=torch.long
        )

        # Actions (factored action space)
        self.directions = torch.zeros(
            (self.num_steps, n_envs, n), device=device, dtype=torch.long
        )
        self.action_types = torch.zeros(
            (self.num_steps, n_envs, n), device=device, dtype=torch.long
        )

        # Other data
        self.log_probs = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.rewards = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.dones = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.values = torch.zeros((self.num_steps, n_envs, n), device=device)

        # Computed after rollout
        self.advantages = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.returns = torch.zeros((self.num_steps, n_envs, n), device=device)

        # Action masks (factored)
        self.direction_masks = torch.zeros(
            (self.num_steps, n_envs, n, config.n_directions), device=device, dtype=torch.bool
        )
        self.action_type_masks = torch.zeros(
            (self.num_steps, n_envs, n, config.n_action_types), device=device, dtype=torch.bool
        )

        # Decomposed rewards for auxiliary value heads
        self.rewards_survival = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.rewards_resource = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.rewards_social = torch.zeros((self.num_steps, n_envs, n), device=device)

        # Decomposed returns (computed in compute_gae)
        self.returns_survival = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.returns_resource = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.returns_social = torch.zeros((self.num_steps, n_envs, n), device=device)

        # Auxiliary values (stored during rollout)
        self.values_survival = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.values_resource = torch.zeros((self.num_steps, n_envs, n), device=device)
        self.values_social = torch.zeros((self.num_steps, n_envs, n), device=device)

        self.step_idx = 0

    def reset(self):
        """Reset buffer for new rollout."""
        self.step_idx = 0

    def store(
        self,
        obs: Dict[str, torch.Tensor],  # [n_envs, n_agents, ...]
        directions: torch.Tensor,       # [n_envs, n_agents]
        action_types: torch.Tensor,     # [n_envs, n_agents]
        log_probs: torch.Tensor,        # [n_envs, n_agents]
        rewards: torch.Tensor,          # [n_envs, n_agents]
        dones: torch.Tensor,            # [n_envs, n_agents]
        values: torch.Tensor,           # [n_envs, n_agents]
        direction_mask: torch.Tensor,   # [n_envs, n_agents, 5]
        action_type_mask: torch.Tensor, # [n_envs, n_agents, 5]
        reward_survival: torch.Tensor,  # [n_envs, n_agents]
        reward_resource: torch.Tensor,  # [n_envs, n_agents]
        reward_social: torch.Tensor,    # [n_envs, n_agents]
        value_survival: torch.Tensor,   # [n_envs, n_agents]
        value_resource: torch.Tensor,   # [n_envs, n_agents]
        value_social: torch.Tensor,     # [n_envs, n_agents]
    ) -> None:
        """Store one step of experience from all envs."""
        t = self.step_idx

        # Store observations
        self.entity_tokens_obs[t] = obs['entity_tokens']
        self.entity_mask_obs[t] = obs['entity_mask']
        self.signal_obs[t] = obs['signals']
        self.self_hp_obs[t] = obs['self_hp']
        self.self_inventory_obs[t] = obs['self_inventory']
        self.agent_ids[t] = obs['agent_id']

        # Store actions
        self.directions[t] = directions
        self.action_types[t] = action_types
        self.log_probs[t] = log_probs

        # Store other data
        self.rewards[t] = rewards
        self.dones[t] = dones.float()
        self.values[t] = values

        # Store masks
        self.direction_masks[t] = direction_mask
        self.action_type_masks[t] = action_type_mask

        # Store decomposed rewards
        self.rewards_survival[t] = reward_survival
        self.rewards_resource[t] = reward_resource
        self.rewards_social[t] = reward_social

        # Store auxiliary values
        self.values_survival[t] = value_survival
        self.values_resource[t] = value_resource
        self.values_social[t] = value_social

        self.step_idx += 1

    def compute_gae(
        self,
        next_value: torch.Tensor,  # [n_envs, n_agents]
        next_done: torch.Tensor,   # [n_envs, n_agents]
        next_value_survival: torch.Tensor = None,
        next_value_resource: torch.Tensor = None,
        next_value_social: torch.Tensor = None,
    ) -> None:
        """Compute GAE for all envs/agents (fully vectorized)."""
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        # Main GAE
        last_gae = torch.zeros(self.n_envs, self.n_agents, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done.float()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]  # Fix: block bootstrap if CURRENT step is done
                next_val = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

        # Auxiliary GAE (simplified - returns only, no separate advantages)
        if next_value_survival is not None:
            self._compute_auxiliary_returns(
                self.rewards_survival, self.values_survival,
                next_value_survival, next_done, self.returns_survival
            )
        if next_value_resource is not None:
            self._compute_auxiliary_returns(
                self.rewards_resource, self.values_resource,
                next_value_resource, next_done, self.returns_resource
            )
        if next_value_social is not None:
            self._compute_auxiliary_returns(
                self.rewards_social, self.values_social,
                next_value_social, next_done, self.returns_social
            )

    def _compute_auxiliary_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        out_returns: torch.Tensor
    ) -> None:
        """Compute discounted returns for auxiliary value heads."""
        gamma = self.config.gamma

        last_return = next_value * (1.0 - next_done.float())
        for t in reversed(range(self.num_steps)):
            next_non_terminal = 1.0 - (self.dones[t] if t < self.num_steps - 1 else next_done.float())
            last_return = rewards[t] + gamma * last_return * next_non_terminal
            out_returns[t] = last_return

    def get_batches(self, n_active: int = None) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield minibatches for PPO update.

        Flattens [num_steps, n_envs, n_agents] to [batch_size].
        Only includes first n_active agents from each env (for curriculum phases with fewer agents).
        """
        if n_active is None:
            n_active = self.n_agents

        # Compute effective batch size
        batch_size = self.num_steps * self.n_envs * n_active
        minibatch_size = batch_size // self.config.num_minibatches

        # Create mask for active agents [num_steps, n_envs, n_agents]
        active_mask = torch.zeros(
            self.num_steps, self.n_envs, self.n_agents,
            device=self.device, dtype=torch.bool
        )
        active_mask[:, :, :n_active] = True

        # Get linear indices of active entries
        active_indices = active_mask.flatten().nonzero(as_tuple=True)[0]

        # Shuffle the active indices
        perm = torch.randperm(len(active_indices), device=self.device)
        shuffled_indices = active_indices[perm]

        # Flatten all tensors
        def flatten_and_index(tensor, indices):
            """Flatten first 3 dims and index."""
            flat = tensor.reshape(-1, *tensor.shape[3:]) if tensor.dim() > 3 else tensor.flatten()
            return flat[indices]

        for start in range(0, batch_size, minibatch_size):
            mb_indices = shuffled_indices[start:start + minibatch_size]

            yield {
                'obs': {
                    'entity_tokens': flatten_and_index(self.entity_tokens_obs, mb_indices),
                    'entity_mask': flatten_and_index(self.entity_mask_obs, mb_indices),
                    'signals': flatten_and_index(self.signal_obs, mb_indices),
                    'self_hp': flatten_and_index(self.self_hp_obs, mb_indices),
                    'self_inventory': flatten_and_index(self.self_inventory_obs, mb_indices),
                    'agent_id': flatten_and_index(self.agent_ids, mb_indices),
                },
                'directions': flatten_and_index(self.directions, mb_indices),
                'action_types': flatten_and_index(self.action_types, mb_indices),
                'log_probs': flatten_and_index(self.log_probs, mb_indices),
                'advantages': flatten_and_index(self.advantages, mb_indices),
                'returns': flatten_and_index(self.returns, mb_indices),
                'direction_mask': flatten_and_index(self.direction_masks, mb_indices),
                'action_type_mask': flatten_and_index(self.action_type_masks, mb_indices),
                'returns_survival': flatten_and_index(self.returns_survival, mb_indices),
                'returns_resource': flatten_and_index(self.returns_resource, mb_indices),
                'returns_social': flatten_and_index(self.returns_social, mb_indices),
            }

    def get_agent_batches(self, agent_idx: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield minibatches for a specific agent only.

        Uses direct slicing [:, :, agent_idx] instead of mask+nonzero for speed.

        Args:
            agent_idx: Which agent's data to return (0-indexed)
        """
        # Direct slice for this agent: [num_steps, n_envs, ...] -> flatten to [num_steps*n_envs, ...]
        agent_batch_size = self.num_steps * self.n_envs
        minibatch_size = max(1, agent_batch_size // self.config.num_minibatches)

        # Shuffle indices
        perm = torch.randperm(agent_batch_size, device=self.device)

        # Pre-slice and flatten all tensors for this agent
        def agent_flat(tensor):
            """Slice agent dim and flatten [num_steps, n_envs, ...] -> [num_steps*n_envs, ...]."""
            sliced = tensor[:, :, agent_idx]  # [num_steps, n_envs, ...]
            return sliced.reshape(agent_batch_size, *sliced.shape[2:])

        et = agent_flat(self.entity_tokens_obs)
        em = agent_flat(self.entity_mask_obs)
        sig = agent_flat(self.signal_obs)
        shp = agent_flat(self.self_hp_obs)
        sinv = agent_flat(self.self_inventory_obs)
        aid = agent_flat(self.agent_ids)
        dirs = agent_flat(self.directions)
        acts = agent_flat(self.action_types)
        lps = agent_flat(self.log_probs)
        advs = agent_flat(self.advantages)
        rets = agent_flat(self.returns)
        dm = agent_flat(self.direction_masks)
        am = agent_flat(self.action_type_masks)
        rs = agent_flat(self.returns_survival)
        rr = agent_flat(self.returns_resource)
        rso = agent_flat(self.returns_social)

        for start in range(0, agent_batch_size, minibatch_size):
            idx = perm[start:start + minibatch_size]

            yield {
                'obs': {
                    'entity_tokens': et[idx],
                    'entity_mask': em[idx],
                    'signals': sig[idx],
                    'self_hp': shp[idx],
                    'self_inventory': sinv[idx],
                    'agent_id': aid[idx],
                },
                'directions': dirs[idx],
                'action_types': acts[idx],
                'log_probs': lps[idx],
                'advantages': advs[idx],
                'returns': rets[idx],
                'direction_mask': dm[idx],
                'action_type_mask': am[idx],
                'returns_survival': rs[idx],
                'returns_resource': rr[idx],
                'returns_social': rso[idx],
            }

    def get_aligned_batches(self, n_active: int = None) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield minibatches with all agents stacked in dim 0, using the SAME
        shuffled permutation across agents. Shape: [n_active, minibatch_size, ...].

        This enables vmapped PPO updates where all agents are processed in parallel.

        Args:
            n_active: Number of active agents (defaults to self.n_agents)
        """
        if n_active is None:
            n_active = self.n_agents

        # Per-agent batch size
        agent_batch_size = self.num_steps * self.n_envs
        minibatch_size = max(1, agent_batch_size // self.config.num_minibatches)

        # Shared permutation across all agents
        perm = torch.randperm(agent_batch_size, device=self.device)

        # Pre-slice all agents and flatten: [n_active, num_steps*n_envs, ...]
        def agents_flat(tensor):
            """Slice first n_active agents, reshape to [n_active, num_steps*n_envs, ...]."""
            sliced = tensor[:, :, :n_active]  # [num_steps, n_envs, n_active, ...]
            # Permute agent dim to front: [n_active, num_steps, n_envs, ...]
            perm_dims = [2, 0, 1] + list(range(3, sliced.dim()))
            permuted = sliced.permute(*perm_dims)
            return permuted.reshape(n_active, agent_batch_size, *permuted.shape[3:])

        et = agents_flat(self.entity_tokens_obs)
        em = agents_flat(self.entity_mask_obs)
        sig = agents_flat(self.signal_obs)
        shp = agents_flat(self.self_hp_obs)
        sinv = agents_flat(self.self_inventory_obs)
        aid = agents_flat(self.agent_ids)
        dirs = agents_flat(self.directions)
        acts = agents_flat(self.action_types)
        lps = agents_flat(self.log_probs)
        advs = agents_flat(self.advantages)
        rets = agents_flat(self.returns)
        dm = agents_flat(self.direction_masks)
        am = agents_flat(self.action_type_masks)
        rs = agents_flat(self.returns_survival)
        rr = agents_flat(self.returns_resource)
        rso = agents_flat(self.returns_social)

        for start in range(0, agent_batch_size, minibatch_size):
            idx = perm[start:start + minibatch_size]

            yield {
                'obs': {
                    'entity_tokens': et[:, idx],  # [n_active, mb, ...]
                    'entity_mask': em[:, idx],
                    'signals': sig[:, idx],
                    'self_hp': shp[:, idx],
                    'self_inventory': sinv[:, idx],
                    'agent_id': aid[:, idx],
                },
                'directions': dirs[:, idx],
                'action_types': acts[:, idx],
                'log_probs': lps[:, idx],
                'advantages': advs[:, idx],
                'returns': rets[:, idx],
                'direction_mask': dm[:, idx],
                'action_type_mask': am[:, idx],
                'returns_survival': rs[:, idx],
                'returns_resource': rr[:, idx],
                'returns_social': rso[:, idx],
            }
