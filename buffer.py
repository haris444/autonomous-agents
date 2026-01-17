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
    - Multi-modal observations (spatial, ledger, signals, self)
    - Dual actions (move + interact)
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.num_steps = config.num_steps
        self.n_agents = config.n_agents

        # Pre-allocate storage tensors
        vs = config.vision_size
        vc = config.vision_channels
        n = config.n_agents
        lc = config.ledger_channels

        # Observations - shape: [num_steps, n_agents, ...]
        self.spatial_obs = torch.zeros((self.num_steps, n, vs, vs, vc), device=device)
        self.ledger_obs = torch.zeros((self.num_steps, n, n, n, lc), device=device)
        self.signal_obs = torch.zeros((self.num_steps, n, n), device=device)
        self.self_obs = torch.zeros((self.num_steps, n, 1), device=device)

        # Actions - shape: [num_steps, n_agents]
        self.move_actions = torch.zeros((self.num_steps, n), device=device, dtype=torch.long)
        self.interact_actions = torch.zeros((self.num_steps, n), device=device, dtype=torch.long)

        # Other data
        self.log_probs = torch.zeros((self.num_steps, n), device=device)
        self.rewards = torch.zeros((self.num_steps, n), device=device)
        self.dones = torch.zeros((self.num_steps, n), device=device)
        self.values = torch.zeros((self.num_steps, n), device=device)

        # Computed after rollout
        self.advantages = torch.zeros((self.num_steps, n), device=device)
        self.returns = torch.zeros((self.num_steps, n), device=device)

        self.step_idx = 0

    def reset(self) -> None:
        """Reset buffer for new rollout."""
        self.step_idx = 0

    def store(
        self,
        obs: Dict[int, Dict[str, torch.Tensor]],
        move_actions: torch.Tensor,
        interact_actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: Dict[int, float],
        dones: Dict[int, bool],
        values: torch.Tensor
    ) -> None:
        """Store one step of experience for all agents."""
        t = self.step_idx

        for agent_id in range(self.n_agents):
            self.spatial_obs[t, agent_id] = obs[agent_id]['spatial']
            self.ledger_obs[t, agent_id] = obs[agent_id]['ledger']
            self.signal_obs[t, agent_id] = obs[agent_id]['signals']
            self.self_obs[t, agent_id] = obs[agent_id]['self_hp']

            self.rewards[t, agent_id] = rewards[agent_id]
            self.dones[t, agent_id] = float(dones[agent_id])

        self.move_actions[t] = move_actions
        self.interact_actions[t] = interact_actions
        self.log_probs[t] = log_probs
        self.values[t] = values

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

        # Flatten all tensors
        spatial_flat = self.spatial_obs.reshape(batch_size, *self.spatial_obs.shape[2:])
        ledger_flat = self.ledger_obs.reshape(batch_size, *self.ledger_obs.shape[2:])
        signal_flat = self.signal_obs.reshape(batch_size, -1)
        self_flat = self.self_obs.reshape(batch_size, -1)

        move_flat = self.move_actions.reshape(batch_size)
        interact_flat = self.interact_actions.reshape(batch_size)
        log_probs_flat = self.log_probs.reshape(batch_size)
        advantages_flat = self.advantages.reshape(batch_size)
        returns_flat = self.returns.reshape(batch_size)

        # Yield minibatches
        minibatch_size = self.config.minibatch_size
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield {
                'obs': {
                    'spatial': spatial_flat[mb_indices],
                    'ledger': ledger_flat[mb_indices],
                    'signals': signal_flat[mb_indices],
                    'self_hp': self_flat[mb_indices]
                },
                'move_actions': move_flat[mb_indices],
                'interact_actions': interact_flat[mb_indices],
                'log_probs': log_probs_flat[mb_indices],
                'advantages': advantages_flat[mb_indices],
                'returns': returns_flat[mb_indices]
            }
