"""
Vectorized Environment Wrapper for Parallel Training.

Manages N independent GridWorld instances and provides batched interface.
All outputs are stacked with shape [n_envs, n_agents, ...].
"""
from typing import Dict, List, Tuple, Optional
import torch

from config import Config
from environment import GridWorld


class VecEnv:
    """
    Vectorized environment wrapper that runs N GridWorld instances in parallel.

    Key features:
    - All outputs batched with [n_envs, ...] dimension
    - Automatic reset of individual envs when they terminate
    - Compatible with existing scenarios and curriculum

    Usage:
        vec_env = VecEnv(config, device, n_envs=16)
        obs = vec_env.reset()  # [n_envs, n_agents, ...]
        obs, rewards, dones, infos = vec_env.step(directions, action_types)

    Note:
        All tensor operations on outputs are batched for GPU efficiency.
    """

    def __init__(self, config: Config, device: torch.device, n_envs: int = 8):
        self.config = config
        self.device = device
        self.n_envs = n_envs
        self.n_agents = config.n_agents

        # Create N independent environments
        self.envs: List[GridWorld] = [
            GridWorld(config, device) for _ in range(n_envs)
        ]

        # Track which envs need reset (for auto-reset on done)
        self.needs_reset = torch.zeros(n_envs, dtype=torch.bool, device=device)

        # Current scenario (applied to all envs)
        self._current_scenario = None

    @property
    def grid_size(self) -> int:
        return self.config.grid_size

    def get_curriculum_phase(self) -> int:
        """Get curriculum phase from first env (they should all be synced)."""
        return self.envs[0].get_curriculum_phase()

    def set_curriculum_phase(self, phase: int) -> None:
        """Set curriculum phase for all environments."""
        for env in self.envs:
            env.curriculum_phase = phase
            env.episode_returns = []

    def reset(self) -> Dict[str, torch.Tensor]:
        """
        Reset all environments and return stacked observations.

        Returns:
            Dict with observation tensors shaped [n_envs, n_agents, ...]
        """
        all_obs = [env.reset() for env in self.envs]
        self.needs_reset.zero_()
        return self._stack_observations(all_obs)

    def apply_scenario(self, scenario) -> None:
        """Apply scenario to all environments."""
        self._current_scenario = scenario
        for env in self.envs:
            env.apply_scenario(scenario)

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get current observations from all environments (stacked)."""
        all_obs = [env._get_all_observations() for env in self.envs]
        return self._stack_observations(all_obs)

    def get_action_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action masks from all environments.

        Returns:
            direction_mask: [n_envs, n_agents, 5]
            action_type_mask: [n_envs, n_agents, 5]
        """
        dir_masks = []
        act_masks = []
        for env in self.envs:
            d, a = env.get_action_masks()
            dir_masks.append(d)
            act_masks.append(a)

        return (
            torch.stack(dir_masks, dim=0),  # [n_envs, n_agents, 5]
            torch.stack(act_masks, dim=0),  # [n_envs, n_agents, 5]
        )

    def step(
        self,
        directions: torch.Tensor,
        action_types: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """
        Step all environments with batched actions.

        Args:
            directions: [n_envs, n_agents] direction indices
            action_types: [n_envs, n_agents] action type indices

        Returns:
            obs: Dict with [n_envs, n_agents, ...] tensors
            rewards: [n_envs, n_agents]
            dones: [n_envs, n_agents]
            infos: Dict with [n_envs, n_agents] tensors
        """
        all_obs = []
        all_rewards = []
        all_dones = []
        all_survival = []
        all_resource = []
        all_social = []

        for i, env in enumerate(self.envs):
            # Step this environment
            obs, rewards, dones, infos = env.step(
                directions[i], action_types[i]
            )

            # Auto-reset if done
            if dones.all():
                # Store terminal info if I needed it (not currently used by buffer)
                # Reset instantly so 'obs' becomes the NEW start state
                obs = env.reset()
                if self._current_scenario is not None:
                    env.apply_scenario(self._current_scenario)
                    obs = env._get_all_observations()
            
            all_obs.append(obs)
            all_rewards.append(rewards)
            all_dones.append(dones)
            all_survival.append(infos['reward_survival'])
            all_resource.append(infos['reward_resource'])
            all_social.append(infos['reward_social'])

        # Stack outputs
        stacked_obs = self._stack_observations(all_obs)
        stacked_rewards = torch.stack(all_rewards, dim=0)
        stacked_dones = torch.stack(all_dones, dim=0)

        stacked_infos = {
            'reward_survival': torch.stack(all_survival, dim=0),
            'reward_resource': torch.stack(all_resource, dim=0),
            'reward_social': torch.stack(all_social, dim=0),
        }

        return stacked_obs, stacked_rewards, stacked_dones, stacked_infos

    def _auto_reset(self) -> None:
        """Reset environments that have terminated."""
        for i in range(self.n_envs):
            if self.needs_reset[i]:
                self.envs[i].reset()
                if self._current_scenario is not None:
                    self.envs[i].apply_scenario(self._current_scenario)

        self.needs_reset.zero_()

    def _stack_observations(
        self, all_obs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Stack observations from multiple envs into batched tensors."""
        return {
            'entity_tokens': torch.stack([o['entity_tokens'] for o in all_obs], dim=0),
            'entity_mask': torch.stack([o['entity_mask'] for o in all_obs], dim=0),
            'signals': torch.stack([o['signals'] for o in all_obs], dim=0),
            'self_hp': torch.stack([o['self_hp'] for o in all_obs], dim=0),
            'self_inventory': torch.stack([o['self_inventory'] for o in all_obs], dim=0),
            'agent_id': torch.stack([o['agent_id'] for o in all_obs], dim=0),
        }

    # === Expose environment properties ===

    @property
    def ledger(self):
        """Return ledger from first env (for logging only)."""
        return self.envs[0].ledger

    @property
    def partner_relationship(self):
        """Return partner_relationship from first env."""
        return getattr(self.envs[0], 'partner_relationship', None)

    @property
    def last_coop_success_count(self):
        """Sum of coop successes across all envs."""
        return sum(getattr(env, 'last_coop_success_count', 0) for env in self.envs)

    def get_first_env(self) -> GridWorld:
        """Get first environment (for visualization, recording, etc)."""
        return self.envs[0]
