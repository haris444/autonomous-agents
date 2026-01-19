"""
Rollout Buffer for PPO training.

Stores trajectories from environment rollouts and computes
Generalized Advantage Estimation (GAE).
"""
from typing import Dict, Generator

import torch

from config import Config


class RolloutBuffer:
    """
    Stores trajectories for PPO training.

    Handles multi-agent rollouts where each agent has:
    - Entity tokens (unified agents + food representation)
    - Dual actions (move + interact)
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.num_steps = config.num_steps
        self.n_agents = config.n_agents

        # Pre-allocate storage tensors
        n = config.n_agents
        max_ent = config.max_entities
        token_dim = config.entity_token_dim

        # Observations - shape: [num_steps, n_agents, ...]
        self.entity_tokens_obs = torch.zeros((self.num_steps, n, max_ent, token_dim), device=device)
        self.entity_mask_obs = torch.zeros((self.num_steps, n, max_ent), device=device, dtype=torch.bool)
        self.signal_obs = torch.zeros((self.num_steps, n, n), device=device)
        self.self_obs = torch.zeros((self.num_steps, n, 1), device=device)
        self.agent_ids = torch.zeros((self.num_steps, n), device=device, dtype=torch.long)

        # Actions - shape: [num_steps, n_agents] (unified action space)
        self.actions = torch.zeros((self.num_steps, n), device=device, dtype=torch.long)

        # Other data
        self.log_probs = torch.zeros((self.num_steps, n), device=device)
        self.rewards = torch.zeros((self.num_steps, n), device=device)
        self.dones = torch.zeros((self.num_steps, n), device=device)
        self.values = torch.zeros((self.num_steps, n), device=device)

        # Computed after rollout
        self.advantages = torch.zeros((self.num_steps, n), device=device)
        self.returns = torch.zeros((self.num_steps, n), device=device)

        # Action mask - shape: [num_steps, n_agents, 15] (unified)
        self.action_masks = torch.zeros((self.num_steps, n, config.n_actions), device=device, dtype=torch.bool)

        self.step_idx = 0

    def reset(self) -> None:
        """Reset buffer for new rollout."""
        self.step_idx = 0

    def store(
        self,
        obs: Dict[int, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: Dict[int, float],
        dones: Dict[int, bool],
        values: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> None:
        """Store one step of experience for all agents."""
        t = self.step_idx

        for agent_id in range(self.n_agents):
            self.entity_tokens_obs[t, agent_id] = obs[agent_id]['entity_tokens']
            self.entity_mask_obs[t, agent_id] = obs[agent_id]['entity_mask']
            self.signal_obs[t, agent_id] = obs[agent_id]['signals']
            self.self_obs[t, agent_id] = obs[agent_id]['self_hp']
            self.agent_ids[t, agent_id] = obs[agent_id]['agent_id']

            self.rewards[t, agent_id] = rewards[agent_id]
            self.dones[t, agent_id] = float(dones[agent_id])

        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values

        if action_mask is not None:
            self.action_masks[t] = action_mask

        self.step_idx += 1

    def store_batched(
        self,
        obs: Dict[str, torch.Tensor],  # Already batched [n_agents, ...]
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,  # [n_agents] tensor, not dict
        dones: torch.Tensor,    # [n_agents] tensor, not dict
        values: torch.Tensor,
        action_mask: torch.Tensor = None  # Optional [n_agents, 15] mask
    ) -> None:
        """Store one step - fully batched, no loops or .item() calls."""
        t = self.step_idx

        # Direct tensor assignment - no loops!
        self.entity_tokens_obs[t] = obs['entity_tokens']
        self.entity_mask_obs[t] = obs['entity_mask']
        self.signal_obs[t] = obs['signals']
        self.self_obs[t] = obs['self_hp']
        self.agent_ids[t] = obs['agent_id']

        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones.float()
        self.values[t] = values

        # Store action mask if provided
        if action_mask is not None:
            self.action_masks[t] = action_mask

        self.step_idx += 1

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor) -> None:
        """
        Compute Generalized Advantage Estimation.

        Args:
            next_value: Value estimate for state after last step [n_agents]
            next_done: Whether episode ended after last step [n_agents]
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        last_gae = torch.zeros(self.n_agents, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done.float()
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            # TD error
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]

            # GAE
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Returns = advantages + values
        self.returns = self.advantages + self.values

    def get_batches(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield minibatches for PPO update.

        Flattens [num_steps, n_agents] to [batch_size] and shuffles.
        """
        batch_size = self.num_steps * self.n_agents
        indices = torch.randperm(batch_size, device=self.device)

        # Flatten all tensors - unified entity tokens format
        entity_tokens_flat = self.entity_tokens_obs.reshape(batch_size, *self.entity_tokens_obs.shape[2:])
        entity_mask_flat = self.entity_mask_obs.reshape(batch_size, -1)
        signal_flat = self.signal_obs.reshape(batch_size, -1)
        self_flat = self.self_obs.reshape(batch_size, -1)
        agent_id_flat = self.agent_ids.reshape(batch_size)

        actions_flat = self.actions.reshape(batch_size)
        log_probs_flat = self.log_probs.reshape(batch_size)
        advantages_flat = self.advantages.reshape(batch_size)
        returns_flat = self.returns.reshape(batch_size)

        # Flatten action mask
        action_masks_flat = self.action_masks.reshape(batch_size, -1)

        # Yield minibatches
        minibatch_size = self.config.minibatch_size
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield {
                'obs': {
                    'entity_tokens': entity_tokens_flat[mb_indices],
                    'entity_mask': entity_mask_flat[mb_indices],
                    'signals': signal_flat[mb_indices],
                    'self_hp': self_flat[mb_indices],
                    'agent_id': agent_id_flat[mb_indices]
                },
                'actions': actions_flat[mb_indices],
                'log_probs': log_probs_flat[mb_indices],
                'advantages': advantages_flat[mb_indices],
                'returns': returns_flat[mb_indices],
                'action_mask': action_masks_flat[mb_indices]
            }


class SingleAgentBuffer:
    """
    Rollout buffer for a single agent.

    Stores trajectories shaped [num_steps, ...] instead of [num_steps, n_agents, ...].
    Used with IndependentPPO where each agent has its own network and buffer.
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

        # Actions (unified action space)
        self.actions = torch.zeros((self.num_steps,), device=device, dtype=torch.long)

        # Other data
        self.log_probs = torch.zeros((self.num_steps,), device=device)
        self.rewards = torch.zeros((self.num_steps,), device=device)
        self.dones = torch.zeros((self.num_steps,), device=device)
        self.values = torch.zeros((self.num_steps,), device=device)

        # Computed after rollout
        self.advantages = torch.zeros((self.num_steps,), device=device)
        self.returns = torch.zeros((self.num_steps,), device=device)

        # Action mask (unified)
        self.action_masks = torch.zeros((self.num_steps, config.n_actions), device=device, dtype=torch.bool)

        self.step_idx = 0

    def reset(self) -> None:
        """Reset buffer for new rollout."""
        self.step_idx = 0

    def store(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        action_mask: torch.Tensor = None
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

        # Store action and values
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done.float() if isinstance(done, torch.Tensor) else float(done)
        self.values[t] = value

        # Store action mask if provided
        if action_mask is not None:
            self.action_masks[t] = action_mask

        self.step_idx += 1

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor) -> None:
        """
        Compute Generalized Advantage Estimation for this single agent.

        Args:
            next_value: Value estimate for state after last step (scalar tensor)
            next_done: Whether episode ended after last step (scalar tensor)
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        last_gae = torch.tensor(0.0, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done.float()
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            # TD error
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]

            # GAE
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Returns = advantages + values
        self.returns = self.advantages + self.values

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
                'actions': self.actions[mb_indices],
                'log_probs': self.log_probs[mb_indices],
                'advantages': self.advantages[mb_indices],
                'returns': self.returns[mb_indices],
                'action_mask': self.action_masks[mb_indices]
            }
