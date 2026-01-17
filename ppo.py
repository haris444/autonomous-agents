"""
Proximal Policy Optimization (PPO) algorithm.

Handles dual-action policy with shared value function.
"""
from typing import Dict

import torch
import torch.nn as nn

from config import Config
from network import ActorCritic
from buffer import RolloutBuffer


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
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0
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

                # Accumulate metrics
                for key in metrics:
                    metrics[key] += batch_metrics[key]
                n_batches += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= n_batches

        return metrics

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Compute combined PPO loss for one minibatch."""
        # Get current policy outputs
        _, _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
            batch['obs'],
            batch['move_actions'],
            batch['interact_actions']
        )

        # Policy ratio
        log_ratio = new_log_prob - batch['log_probs']
        ratio = log_ratio.exp()

        # Approximate KL divergence (for monitoring)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()

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

        batch_metrics = {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy': entropy_loss.item(),
            'total_loss': loss.item(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction
        }

        return loss, batch_metrics
