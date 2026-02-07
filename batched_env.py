"""
Batched GridWorld Environment — processes N environments simultaneously.

Drop-in replacement for VecEnv. All state tensors have leading [n_envs, ...] dimension.
Uses batched tensor operations instead of looping over independent GridWorld instances.
"""
import time
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from config import Config
from environment import GridWorld
from ledger import Ledger

# Factored action space constants (mirrored from environment.py)
DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT, DIR_STAY = 0, 1, 2, 3, 4
ACT_MOVE, ACT_ATTACK, ACT_GIVE, ACT_SIGNAL, ACT_COOPERATE = 0, 1, 2, 3, 4


class BatchedGridWorld:
    """
    Fully batched multi-environment GridWorld.
    Same public API as VecEnv for drop-in replacement.
    """

    def __init__(self, config: Config, device: torch.device, n_envs: int = 8):
        self.config = config
        self.device = device
        self.n_envs = n_envs
        self.n_agents = config.n_agents
        self.grid_size = config.grid_size
        gs = config.grid_size
        n = config.n_agents

        # Direction deltas [5, 2]: UP, DOWN, LEFT, RIGHT, STAY
        self.direction_deltas = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]],
            device=device, dtype=torch.long
        )
        # Cardinal-only deltas [4, 2] for attacks/give
        self.dir_deltas_4 = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]],
            device=device, dtype=torch.long
        )

        # Precomputed coordinate grid [gs, gs, 2] for nonzero-free distance computation
        rows_grid = torch.arange(gs, device=device).unsqueeze(1).expand(gs, gs)
        cols_grid = torch.arange(gs, device=device).unsqueeze(0).expand(gs, gs)
        self.coord_grid = torch.stack([rows_grid, cols_grid], dim=-1).float()

        # Precomputed edge cells [num_edge, 2] for predator respawn
        edge_list = []
        for c in range(gs):
            edge_list.extend([(0, c), (gs - 1, c)])
        for r in range(1, gs - 1):
            edge_list.extend([(r, 0), (r, gs - 1)])
        self._edge_cells = torch.tensor(edge_list, device=device, dtype=torch.long)

        # Ledger normalization scale [4]
        self._ledger_norm_scale = torch.tensor(
            [1.0 / 100.0, 1.0 / 100.0, 1.0 / 10.0, 1.0 / 100.0],
            device=device
        )

        # Identity matrix for defense diagonal zeroing [n, n]
        self._eye = torch.eye(n, device=device)

        # Curriculum tracking (synced across all envs)
        self.curriculum_phase = config.curriculum_phase if config.curriculum_enabled else 4
        self.episode_returns = []
        self._current_scenario = None

        # Shadow env for scenario handling and get_first_env()
        self._shadow_env = GridWorld(config, device)
        self.partner_relationships = [None] * n_envs  # Per-slot relationship tracking

        # Per-env scalars
        self.step_count = torch.zeros(n_envs, device=device, dtype=torch.long)
        self.episode_count = torch.zeros(n_envs, device=device, dtype=torch.long)
        self.current_episode_reward = torch.zeros(n_envs, device=device)
        self.warmup_episodes = 0

        # ---- Batched state tensors (allocated in reset) ----
        self.agent_positions = torch.zeros((n_envs, n, 2), device=device, dtype=torch.long)
        self.prev_agent_positions = torch.zeros((n_envs, n, 2), device=device, dtype=torch.long)
        self.agent_hp = torch.zeros((n_envs, n), device=device)
        self.agent_alive = torch.zeros((n_envs, n), device=device, dtype=torch.bool)
        self.agent_inventory = torch.zeros((n_envs, n), device=device)
        self.food_eaten_total = torch.zeros((n_envs, n), device=device)
        self.occupancy = torch.full((n_envs, gs, gs), -1, device=device, dtype=torch.long)
        self.poor_food = torch.zeros((n_envs, gs, gs), device=device, dtype=torch.bool)
        self.rich_food = torch.zeros((n_envs, gs, gs), device=device, dtype=torch.bool)
        self.signals = torch.zeros((n_envs, n), device=device, dtype=torch.bool)
        self.ledger_tensor = torch.zeros((n_envs, n, n, 4), device=device)
        self.recent_attacks = torch.zeros((n_envs, 3, n, n), device=device)

        # Predator state (environmental enemy)
        self.n_predators = config.n_predators
        n_pred = config.n_predators
        if n_pred > 0:
            self.predator_positions = torch.zeros((n_envs, n_pred, 2), device=device, dtype=torch.long)
            self.predator_prev_positions = torch.zeros((n_envs, n_pred, 2), device=device, dtype=torch.long)
            self.predator_hp = torch.zeros((n_envs, n_pred), device=device)
            self.predator_alive = torch.zeros((n_envs, n_pred), device=device, dtype=torch.bool)
            self.predator_respawn_timer = torch.zeros((n_envs, n_pred), device=device, dtype=torch.long)
            self.predator_ledger = torch.zeros((n_envs, n, n_pred, 2), device=device)

        # Cooperation tracking
        self.last_coop_success_count = 0

        # Step sub-timers (cumulative)
        self.step_timers = {
            'interactions': 0.0, 'movement': 0.0, 'predator': 0.0,
            'hp_decay': 0.0, 'food': 0.0, 'hierarchy': 0.0,
            'spawn_food': 0.0, 'deaths': 0.0, 'rewards': 0.0,
            'observations': 0.0, 'reset': 0.0, 'nearest_food': 0.0,
        }
        self.step_timer_calls = 0

    # =====================================================================
    # PUBLIC API (matches VecEnv)
    # =====================================================================

    def get_step_timing_str(self) -> str:
        """Return a formatted string of step sub-timers and reset them."""
        T = self.step_timers
        total = sum(T.values())
        if total <= 0:
            return ""
        parts = []
        for k, v in sorted(T.items(), key=lambda x: -x[1]):
            pct = 100 * v / total
            if pct >= 1.0:
                parts.append(f"{k}: {v:.2f}s ({pct:.0f}%)")
        # Reset
        for k in T:
            T[k] = 0.0
        self.step_timer_calls = 0
        return " | ".join(parts)

    @property
    def grid_size_prop(self) -> int:
        return self.config.grid_size

    def get_curriculum_phase(self) -> int:
        return self.curriculum_phase

    def set_curriculum_phase(self, phase: int) -> None:
        self.curriculum_phase = phase
        self.episode_returns = []

    @property
    def ledger(self):
        """Return a Ledger-like object from env 0 for logging."""
        self._shadow_env.ledger.tensor = self.ledger_tensor[0].clone()
        return self._shadow_env.ledger

    @property
    def partner_relationship(self):
        return self.partner_relationships[0]

    def get_first_env(self) -> GridWorld:
        """Sync shadow env from batch slot 0 and return it."""
        s = self._shadow_env
        s.agent_positions = self.agent_positions[0].clone()
        s.prev_agent_positions = self.prev_agent_positions[0].clone()
        s.agent_hp = self.agent_hp[0].clone()
        s.agent_alive = self.agent_alive[0].clone()
        s.agent_inventory = self.agent_inventory[0].clone()
        s.occupancy = self.occupancy[0].clone()
        s.poor_food = self.poor_food[0].clone()
        s.rich_food = self.rich_food[0].clone()
        s.signals = self.signals[0].clone()
        s.ledger.tensor = self.ledger_tensor[0].clone()
        s.step_count = self.step_count[0].item()
        if self.n_predators > 0:
            s.predator_positions = self.predator_positions[0].clone()
            s.predator_prev_positions = self.predator_prev_positions[0].clone()
            s.predator_hp = self.predator_hp[0].clone()
            s.predator_alive = self.predator_alive[0].clone()
            s.predator_respawn_timer = self.predator_respawn_timer[0].clone()
            if hasattr(self, 'predator_ledger'):
                s.predator_ledger = self.predator_ledger[0].clone()
        return s

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset all environments."""
        all_mask = torch.ones(self.n_envs, device=self.device, dtype=torch.bool)
        self._masked_reset(all_mask, is_initial=True)
        return self._get_all_observations()

    def apply_scenario(self, scenario) -> None:
        """Apply scenario to all environments via shadow env."""
        self._current_scenario = scenario
        for i in range(self.n_envs):
            self._apply_scenario_to_slot(i)

    def get_observations(self) -> Dict[str, torch.Tensor]:
        return self._get_all_observations()

    def get_action_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._get_action_masks_batched()

    def step(
        self,
        directions: torch.Tensor,
        action_types: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """
        Step all environments with batched actions.

        Args:
            directions: [n_envs, n_agents]
            action_types: [n_envs, n_agents]
        Returns:
            obs, rewards, dones, infos — all batched [n_envs, ...]
        """
        self.step_count += 1
        self.step_timer_calls += 1
        ne, n = self.n_envs, self.n_agents
        T = self.step_timers

        self.prev_agent_positions = self.agent_positions.clone()

        # Decompose actions
        is_move = (action_types == ACT_MOVE)
        is_attack = (action_types == ACT_ATTACK)
        is_give = (action_types == ACT_GIVE)
        is_signal = (action_types == ACT_SIGNAL)

        hp_before = self.agent_hp.clone()
        _t = time.time()
        dist_before, _ = self._compute_nearest_food_info()
        T['nearest_food'] += time.time() - _t

        # Clear signals
        self.signals.zero_()

        # 1. Interactions (from current position, includes agent-vs-predator)
        _t = time.time()
        attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, give_rewards, predator_kill_rewards = (
            self._resolve_interactions(directions, action_types, is_attack, is_give, is_signal)
        )
        T['interactions'] += time.time() - _t

        # 2. Movement (predators block cells)
        _t = time.time()
        self._resolve_movement(directions, is_move)
        self._update_occupancy()
        T['movement'] += time.time() - _t

        # 3. Predator movement + attack
        _t = time.time()
        predator_damage = self._predator_step()
        T['predator'] += time.time() - _t

        # 4. HP decay (agents only)
        _t = time.time()
        self._apply_hp_decay()
        T['hp_decay'] += time.time() - _t

        # 5. Food eating
        _t = time.time()
        food_rewards = self._process_food_eating(action_types)
        intrinsic_coop_rewards = self._compute_intrinsic_coop_rewards(action_types)
        self._consume_inventory()
        T['food'] += time.time() - _t

        # 5c. Hierarchy rewards (competitive ranking bonus)
        _t = time.time()
        hierarchy_rewards = self._compute_hierarchy_rewards()
        T['hierarchy'] += time.time() - _t

        # 6. Spawn food
        _t = time.time()
        self._spawn_food()
        T['spawn_food'] += time.time() - _t

        # 7. Deaths (agents + predators)
        _t = time.time()
        death_rewards = self._check_deaths()
        self._update_occupancy()
        T['deaths'] += time.time() - _t

        # Reward computation
        _t = time.time()
        total_damage_taken = damage_taken + predator_damage
        hp_after = self.agent_hp.clone()
        hp_ratio = hp_before / self.config.max_hp
        pain_multiplier = 1.0 / (hp_ratio + 0.1)
        damage_pain = total_damage_taken * pain_multiplier * self.config.r_damage_taken

        hp_ratio_after = hp_after / self.config.max_hp
        low_hp_penalty = (1.0 - hp_ratio_after) * self.config.r_low_hp * self.agent_alive.float()

        _t2 = time.time()
        dist_after, _ = self._compute_nearest_food_info()
        T['nearest_food'] += time.time() - _t2

        approach_delta = dist_before - dist_after
        approach_delta = torch.where(
            torch.isinf(dist_before) | torch.isinf(dist_after),
            torch.zeros_like(approach_delta),
            approach_delta
        )
        approach_reward = approach_delta * self.config.r_approach_food * self.agent_alive.float()

        # Survival bonus (constant positive reward for staying alive)
        survival_bonus = self.config.r_survival * self.agent_alive.float()

        rewards = (
            death_rewards + food_rewards + intrinsic_coop_rewards
            + attack_rewards + defense_rewards + revenge_rewards + betrayal_rewards
            + give_rewards + predator_kill_rewards + hierarchy_rewards
            + damage_pain + low_hp_penalty + approach_reward + survival_bonus
        )
        T['rewards'] += time.time() - _t

        _t = time.time()
        observations = self._get_all_observations()
        T['observations'] += time.time() - _t

        dones = ~self.agent_alive  # [n_envs, n_agents]

        # Episode termination
        all_dead = ~self.agent_alive.any(dim=1)  # [n_envs]
        max_steps = self.step_count >= self.config.max_steps_per_episode
        episode_done = all_dead | max_steps  # [n_envs]
        # In pretrain, end episode when agent 0 dies (scripted partners can outlive it)
        if self.config.pretrain_mode:
            agent0_dead = ~self.agent_alive[:, 0]  # [n_envs]
            episode_done = episode_done | agent0_dead

        # Set all agents done for finished envs
        dones = dones | episode_done.unsqueeze(1)

        # Curriculum tracking (env 0 only)
        self.current_episode_reward += rewards[:, 0]
        if episode_done.any():
            done_indices = torch.where(episode_done)[0]
            if episode_done[0].item() and self.config.pretrain_mode and self.config.curriculum_enabled:
                self._check_curriculum_advance(self.current_episode_reward[0].item())
            self.current_episode_reward[done_indices] = 0.0

        # Save terminal state before auto-reset wipes it
        terminal_ledger = self.ledger_tensor.clone() if episode_done.any() else None
        terminal_relationships = list(self.partner_relationships) if episode_done.any() else None

        # Auto-reset finished envs (obs already captured above)
        _t = time.time()
        if episode_done.any():
            self._masked_reset(episode_done, is_initial=False)
            # Overwrite observations for reset envs with fresh state
            new_obs = self._get_all_observations()
            observations = {
                key: torch.where(
                    episode_done.view(ne, *([1] * (observations[key].dim() - 1))).expand_as(observations[key]),
                    new_obs[key],
                    observations[key]
                )
                for key in observations
            }
        T['reset'] += time.time() - _t

        # Decomposed rewards
        survival_rewards = damage_pain + low_hp_penalty + death_rewards
        resource_rewards = food_rewards + approach_reward
        social_rewards = (attack_rewards + defense_rewards + revenge_rewards
                          + betrayal_rewards + intrinsic_coop_rewards + give_rewards
                          + predator_kill_rewards + hierarchy_rewards)

        infos = {
            'reward_survival': survival_rewards,
            'reward_resource': resource_rewards,
            'reward_social': social_rewards,
            'terminal_ledger': terminal_ledger,
            'terminal_relationships': terminal_relationships,
        }

        return observations, rewards, dones, infos

    # =====================================================================
    # RESET
    # =====================================================================

    def _masked_reset(self, mask: torch.Tensor, is_initial: bool = False):
        """Reset envs where mask is True."""
        n_reset = mask.sum().item()
        if n_reset == 0:
            return

        gs = self.grid_size
        n = self.n_agents

        self.step_count[mask] = 0
        self.episode_count[mask] += 1
        self.current_episode_reward[mask] = 0.0
        self.ledger_tensor[mask] = 0
        self.recent_attacks[mask] = 0
        self.signals[mask] = False

        # Random non-overlapping positions via argsort
        rand = torch.rand(n_reset, gs * gs, device=self.device)
        all_positions = rand.argsort(dim=1)[:, :n]  # [n_reset, n_agents]
        rows = all_positions // gs
        cols = all_positions % gs
        self.agent_positions[mask] = torch.stack([rows, cols], dim=-1)

        self.agent_hp[mask] = self.config.max_hp
        self.agent_alive[mask] = True
        self.agent_inventory[mask] = 0
        self.food_eaten_total[mask] = 0

        self.poor_food[mask] = False
        self.rich_food[mask] = False

        self.prev_agent_positions[mask] = self.agent_positions[mask].clone()

        # Reset predator state for reset envs
        if self.n_predators > 0:
            self.predator_hp[mask] = self.config.max_hp * self.config.predator_hp_mult
            self.predator_alive[mask] = True
            self.predator_respawn_timer[mask] = 0
            # Spawn predators at random edge cells
            reset_indices = torch.where(mask)[0]
            for idx in reset_indices:
                i = idx.item()
                for p in range(self.n_predators):
                    self._respawn_predator_in_slot(i, p)
            self.predator_prev_positions[mask] = self.predator_positions[mask].clone()
            self.predator_ledger[mask] = 0

        # Update occupancy for all envs (cheap)
        self._update_occupancy()

        # Spawn initial food for reset envs
        self._spawn_initial_food(mask)

        # Apply scenario if active
        if self._current_scenario is not None:
            reset_indices = torch.where(mask)[0]
            for idx in reset_indices:
                self._apply_scenario_to_slot(idx.item())
            self._update_occupancy()

    def _apply_scenario_to_slot(self, slot: int):
        """Apply current scenario to a single batch slot via shadow env."""
        s = self._shadow_env
        s.curriculum_phase = self.curriculum_phase
        s.reset()
        s.apply_scenario(self._current_scenario)

        self.agent_positions[slot] = s.agent_positions
        self.agent_hp[slot] = s.agent_hp
        self.agent_alive[slot] = s.agent_alive
        self.agent_inventory[slot] = s.agent_inventory
        self.food_eaten_total[slot] = getattr(s, 'food_eaten_total', torch.zeros(self.n_agents, device=self.device))
        self.poor_food[slot] = s.poor_food
        self.rich_food[slot] = s.rich_food
        self.ledger_tensor[slot] = s.ledger.tensor
        self.signals[slot] = s.signals
        self.prev_agent_positions[slot] = s.agent_positions.clone()
        self.partner_relationships[slot] = getattr(s, 'partner_relationship', None)
        if self.n_predators > 0 and s.predator_positions is not None:
            self.predator_positions[slot] = s.predator_positions
            self.predator_prev_positions[slot] = s.predator_prev_positions
            self.predator_hp[slot] = s.predator_hp
            self.predator_alive[slot] = s.predator_alive
            self.predator_respawn_timer[slot] = s.predator_respawn_timer
            if hasattr(s, 'predator_ledger') and s.predator_ledger is not None:
                self.predator_ledger[slot] = s.predator_ledger

    def _spawn_initial_food(self, mask: torch.Tensor):
        """Spawn food at full capacity for reset envs."""
        gs = self.grid_size
        total_cells = gs * gs
        cap_cells = int(total_cells * self.config.food_coverage_cap)

        # For each reset env, pick random empty cells for food
        reset_indices = torch.where(mask)[0]
        for idx in reset_indices:
            i = idx.item()
            empty = (self.occupancy[i] == -1)
            empty_flat = empty.flatten().nonzero(as_tuple=True)[0]
            if len(empty_flat) == 0:
                continue
            perm = torch.randperm(len(empty_flat), device=self.device)
            shuffled = empty_flat[perm]

            # Poor food
            poor_count = min(cap_cells, len(shuffled))
            if poor_count > 0:
                poor_indices = shuffled[:poor_count]
                poor_rows = poor_indices // gs
                poor_cols = poor_indices % gs
                self.poor_food[i, poor_rows, poor_cols] = True

            # Rich food (skip in pretrain mode)
            if not self.config.pretrain_mode:
                rich_start = poor_count
                rich_count = min(cap_cells, len(shuffled) - rich_start)
                if rich_count > 0:
                    rich_indices = shuffled[rich_start:rich_start + rich_count]
                    rich_rows = rich_indices // gs
                    rich_cols = rich_indices % gs
                    self.rich_food[i, rich_rows, rich_cols] = True

    # =====================================================================
    # OCCUPANCY
    # =====================================================================

    def _update_occupancy(self):
        """Rebuild occupancy grid for all envs — fully batched, including predators."""
        ne, n = self.n_envs, self.n_agents
        self.occupancy.fill_(-1)

        env_idx = torch.arange(ne, device=self.device).unsqueeze(1).expand(ne, n)
        agent_idx = torch.arange(n, device=self.device).unsqueeze(0).expand(ne, n)
        rows = self.agent_positions[:, :, 0]  # [ne, n]
        cols = self.agent_positions[:, :, 1]  # [ne, n]
        alive = self.agent_alive  # [ne, n]

        self.occupancy[env_idx[alive], rows[alive], cols[alive]] = agent_idx[alive]

        # Predators occupy cells as n_agents + pred_idx
        if self.n_predators > 0:
            n_pred = self.n_predators
            env_idx_p = torch.arange(ne, device=self.device).unsqueeze(1).expand(ne, n_pred)
            pred_idx = torch.arange(n_pred, device=self.device).unsqueeze(0).expand(ne, n_pred) + n
            pred_rows = self.predator_positions[:, :, 0]
            pred_cols = self.predator_positions[:, :, 1]
            pred_alive = self.predator_alive
            self.occupancy[env_idx_p[pred_alive], pred_rows[pred_alive], pred_cols[pred_alive]] = pred_idx[pred_alive]

    # =====================================================================
    # TRIVIAL METHODS
    # =====================================================================

    def _apply_hp_decay(self):
        decay = self.config.max_hp * self.config.hp_decay_rate
        self.agent_hp = self.agent_hp - decay * self.agent_alive.float()

    def _consume_inventory(self):
        hp_missing = self.config.max_hp - self.agent_hp
        cap = torch.full_like(self.agent_inventory, self.config.heal_per_tick)
        heal = torch.minimum(torch.minimum(hp_missing, self.agent_inventory), cap)
        heal = heal * self.agent_alive.float() * (hp_missing > 0).float()
        self.agent_hp = self.agent_hp + heal
        self.agent_inventory = self.agent_inventory - heal

    def _check_deaths(self) -> torch.Tensor:
        was_alive = self.agent_alive.clone()
        self.agent_alive = self.agent_hp > 0
        newly_dead = was_alive & ~self.agent_alive
        return newly_dead.float() * self.config.r_death

    # =====================================================================
    # MOVEMENT (Heavyweight Rule)
    # =====================================================================

    def _resolve_movement(self, move_actions: torch.Tensor, is_move: torch.Tensor):
        """Batched heavyweight movement resolution."""
        ne, n, gs = self.n_envs, self.n_agents, self.grid_size

        alive_mask = self.agent_alive & is_move  # [ne, n]

        # Intended destinations
        intended = self.agent_positions + self.direction_deltas[move_actions]  # [ne, n, 2]
        intended = intended.clamp(0, gs - 1)
        intended = torch.where(alive_mask.unsqueeze(2), intended, self.agent_positions)

        # Flat indices per env
        flat_intended = intended[:, :, 0] * gs + intended[:, :, 1]  # [ne, n]
        flat_current = self.agent_positions[:, :, 0] * gs + self.agent_positions[:, :, 1]

        is_staying = (flat_intended == flat_current)

        # Occupant at intended destination
        dest_rows = intended[:, :, 0]  # [ne, n]
        dest_cols = intended[:, :, 1]  # [ne, n]
        env_idx = torch.arange(ne, device=self.device).unsqueeze(1).expand(ne, n)
        occupant_at_dest = self.occupancy[env_idx, dest_rows, dest_cols]  # [ne, n]

        has_occupant = occupant_at_dest >= 0
        is_predator_occ = occupant_at_dest >= n  # Predators stored as n_agents + pred_idx
        safe_occ = occupant_at_dest.clamp(min=0, max=n - 1)  # Clamp to agent range for gather

        # Check if occupant is staying — predators always block
        agent_occupant_staying = torch.gather(is_staying, 1, safe_occ) & has_occupant & ~is_predator_occ
        occupant_is_staying = agent_occupant_staying | (is_predator_occ & has_occupant)

        agent_ids = torch.arange(n, device=self.device).unsqueeze(0).expand(ne, n)
        is_self_occupant = (occupant_at_dest == agent_ids)
        blocked = has_occupant & occupant_is_staying & ~is_self_occupant

        valid_contender = alive_mask & ~blocked

        tiebreaker = torch.rand(ne, n, device=self.device) * 0.001
        hp_with_tie = self.agent_hp + tiebreaker
        hp_contest = torch.where(
            valid_contender, hp_with_tie,
            torch.full_like(hp_with_tie, float('-inf'))
        )

        # Scatter reduce: find max HP per destination (global flat index)
        env_offset = torch.arange(ne, device=self.device).unsqueeze(1) * (gs * gs)
        flat_global = flat_intended + env_offset  # [ne, n]
        flat_global_1d = flat_global.reshape(-1)
        hp_1d = hp_contest.reshape(-1)

        max_hp_buf = torch.full((ne * gs * gs,), float('-inf'), device=self.device)
        max_hp_buf.scatter_reduce_(0, flat_global_1d, hp_1d, reduce='amax', include_self=True)

        max_hp_at_dest = max_hp_buf[flat_global_1d].reshape(ne, n)
        is_winner = valid_contender & (hp_contest == max_hp_at_dest)

        self.agent_positions = torch.where(is_winner.unsqueeze(2), intended, self.agent_positions)

    # =====================================================================
    # INTERACTIONS (attacks, give, signal)
    # =====================================================================

    def _resolve_interactions(self, directions, action_types, is_attack, is_give, is_signal):
        signal_mask = is_signal & self.agent_alive
        self.signals = signal_mask

        attack_mask = is_attack & self.agent_alive
        attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, predator_kill_rewards = (
            self._resolve_attacks(directions, attack_mask)
        )

        give_mask = is_give & self.agent_alive
        give_rewards = self._resolve_give_food(directions, give_mask)

        return attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, give_rewards, predator_kill_rewards

    def _resolve_attacks(self, directions, attack_mask):
        """Fully batched attack resolution including defense/revenge/betrayal and agent-vs-predator."""
        ne, n, gs = self.n_envs, self.n_agents, self.grid_size

        attack_rewards = torch.zeros(ne, n, device=self.device)
        damage_taken = torch.zeros(ne, n, device=self.device)
        defense_rewards = torch.zeros(ne, n, device=self.device)
        revenge_rewards = torch.zeros(ne, n, device=self.device)
        betrayal_rewards = torch.zeros(ne, n, device=self.device)
        predator_kill_rewards = torch.zeros(ne, n, device=self.device)

        if not attack_mask.any():
            return attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, predator_kill_rewards

        # Attack cost
        attack_cost = self.config.max_hp * 0.05
        self.agent_hp = self.agent_hp - attack_cost * attack_mask.float()

        # Target positions
        attack_dirs = directions.clamp(0, 3)
        deltas = self.dir_deltas_4[attack_dirs]  # [ne, n, 2]
        target_pos = self.agent_positions + deltas
        target_rows = target_pos[:, :, 0]
        target_cols = target_pos[:, :, 1]

        in_bounds = (
            (target_rows >= 0) & (target_rows < gs) &
            (target_cols >= 0) & (target_cols < gs)
        )
        valid_attack = attack_mask & in_bounds

        safe_rows = target_rows.clamp(0, gs - 1)
        safe_cols = target_cols.clamp(0, gs - 1)

        env_idx = torch.arange(ne, device=self.device).unsqueeze(1).expand(ne, n)
        target_ids = self.occupancy[env_idx, safe_rows, safe_cols]  # [ne, n]

        # Separate agent targets from predator targets
        has_target = target_ids >= 0
        is_predator_target = target_ids >= n
        is_agent_target = has_target & ~is_predator_target

        # === Agent-vs-Agent attacks ===
        safe_target_ids = target_ids.clamp(min=0, max=n - 1)
        target_alive = torch.gather(self.agent_alive, 1, safe_target_ids) & is_agent_target
        valid_hit = valid_attack & is_agent_target & target_alive

        damage = (self.agent_hp * self.config.attack_damage_fraction) * valid_hit.float()  # [ne, n]

        # Build damage matrix [ne, n, n]: damage_matrix[e, attacker, victim]
        damage_matrix = torch.zeros(ne, n, n, device=self.device)
        attacker_idx = torch.arange(n, device=self.device).unsqueeze(0).expand(ne, n)  # [ne, n]
        hit_mask = valid_hit  # [ne, n]
        if hit_mask.any():
            e_flat = env_idx[hit_mask]
            a_flat = attacker_idx[hit_mask]
            t_flat = safe_target_ids[hit_mask]
            d_flat = damage[hit_mask]
            damage_matrix[e_flat, a_flat, t_flat] = d_flat

        # Apply damage to targets
        damage_per_target = damage_matrix.sum(dim=1)  # [ne, n] sum over attackers
        self.agent_hp = self.agent_hp - damage_per_target

        # === Agent-vs-Predator attacks ===
        if self.n_predators > 0:
            valid_pred_hit = valid_attack & is_predator_target
            if valid_pred_hit.any():
                pred_hit_e = env_idx[valid_pred_hit]
                pred_hit_a = attacker_idx[valid_pred_hit]
                pred_target_p = (target_ids[valid_pred_hit] - n)  # Predator index
                pred_dmg = self.agent_hp[pred_hit_e, pred_hit_a] * self.config.attack_damage_fraction

                # Apply damage to predators (loop over hits since scatter doesn't handle well here)
                for i in range(pred_hit_e.numel()):
                    e_i = pred_hit_e[i].item()
                    a_i = pred_hit_a[i].item()
                    p_i = pred_target_p[i].item()
                    d_i = pred_dmg[i].item()
                    if self.predator_alive[e_i, p_i]:
                        self.predator_hp[e_i, p_i] -= d_i
                        # Record in predator ledger: agent dealt damage to predator
                        self.predator_ledger[e_i, a_i, p_i, 0] += d_i
                        attack_rewards[e_i, a_i] += d_i * self.config.r_attack_mult
                        if self.predator_hp[e_i, p_i] <= 0:
                            self.predator_alive[e_i, p_i] = False
                            self.predator_respawn_timer[e_i, p_i] = self.config.predator_respawn_steps
                            predator_kill_rewards[e_i, a_i] += self.config.predator_kill_reward

        # Update ledger
        self.ledger_tensor[:, :, :, Ledger.DAMAGE_DEALT] += damage_matrix

        # Roll attack history
        self.recent_attacks = torch.roll(self.recent_attacks, 1, dims=1)
        self.recent_attacks[:, 0] = damage_matrix

        # === VECTORIZED DEFENSE/REVENGE/BETRAYAL ===
        recent_victims = (self.recent_attacks > 0).any(dim=1)  # [ne, n, n]
        valid_hits = (damage_matrix > 0)  # [ne, n_c, n_a] — C hit A

        # REVENGE
        revenge_mask = valid_hits & recent_victims.transpose(1, 2)
        revenge_rewards = (damage_matrix * revenge_mask.float() * self.config.r_revenge).sum(dim=2)

        # BETRAYAL
        help_matrix = (self.ledger_tensor[:, :, :, Ledger.FOOD_GIVEN]
                       + self.ledger_tensor[:, :, :, Ledger.COOP_COUNT])
        help_from = help_matrix.transpose(1, 2)
        has_helped = (help_from > 0) & valid_hits
        betrayal_rewards = (help_from * has_helped.float() * self.config.r_betrayal).sum(dim=2)

        # DEFENSE
        defense_credit = torch.bmm(damage_matrix, recent_victims.float())  # [ne, n, n]
        defense_credit = defense_credit * (1.0 - self._eye.unsqueeze(0))  # zero diagonal
        defense_rewards = defense_credit.sum(dim=2) * self.config.r_defense
        self.ledger_tensor[:, :, :, Ledger.DEFENSE_SCORE] += defense_credit

        # Attack rewards for agent-vs-agent (predator rewards already added above)
        total_damage_dealt = damage_matrix.sum(dim=2)  # [ne, n]
        attack_rewards += total_damage_dealt * self.config.r_attack_mult
        damage_taken = damage_per_target

        return attack_rewards, damage_taken, defense_rewards, revenge_rewards, betrayal_rewards, predator_kill_rewards

    def _resolve_give_food(self, directions, give_mask):
        """Batched food giving from inventory."""
        ne, n, gs = self.n_envs, self.n_agents, self.grid_size
        food_value = self.config.poor_food_value
        give_rewards = torch.zeros(ne, n, device=self.device)

        if not give_mask.any():
            return give_rewards

        give_amount = torch.minimum(
            torch.full((ne, n), food_value, device=self.device),
            self.agent_inventory
        ) * give_mask.float()

        valid_giver = give_mask & (give_amount > 0)

        give_dirs = directions.clamp(0, 3)
        deltas = self.dir_deltas_4[give_dirs]
        target_pos = self.agent_positions + deltas
        target_rows = target_pos[:, :, 0]
        target_cols = target_pos[:, :, 1]

        in_bounds = (
            (target_rows >= 0) & (target_rows < gs) &
            (target_cols >= 0) & (target_cols < gs)
        )

        safe_rows = target_rows.clamp(0, gs - 1)
        safe_cols = target_cols.clamp(0, gs - 1)

        env_idx = torch.arange(ne, device=self.device).unsqueeze(1).expand(ne, n)
        target_ids = self.occupancy[env_idx, safe_rows, safe_cols]

        has_target = target_ids >= 0
        is_agent_target = has_target & (target_ids < n)
        safe_tid = target_ids.clamp(min=0, max=n - 1)
        target_alive = torch.gather(self.agent_alive, 1, safe_tid) & is_agent_target
        valid_transfer = valid_giver & in_bounds & is_agent_target & target_alive

        if not valid_transfer.any():
            return give_rewards

        transfer_mask = valid_transfer
        e_flat = env_idx[transfer_mask]
        giver_flat = torch.arange(n, device=self.device).unsqueeze(0).expand(ne, n)[transfer_mask]
        target_flat = safe_tid[transfer_mask]
        amount_flat = give_amount[transfer_mask]

        # Deduct from givers
        self.agent_inventory[e_flat, giver_flat] -= amount_flat

        # Add to targets (scatter_add for collisions)
        inv_gain = torch.zeros(ne, n, device=self.device)
        # Flatten env+agent into single dim for scatter
        flat_target = e_flat * n + target_flat
        inv_gain_flat = inv_gain.reshape(-1)
        inv_gain_flat.scatter_add_(0, flat_target, amount_flat)
        self.agent_inventory = self.agent_inventory + inv_gain.reshape(ne, n)

        # Reward givers
        give_rewards[e_flat, giver_flat] = amount_flat * self.config.r_food_share

        # Update ledger
        self.ledger_tensor[e_flat, giver_flat, target_flat, Ledger.FOOD_GIVEN] += amount_flat

        return give_rewards

    # =====================================================================
    # PREDATOR METHODS
    # =====================================================================

    def _respawn_predator_in_slot(self, env_i: int, pred_i: int):
        """Respawn a single predator in a specific env slot at a random edge cell."""
        gs = self.grid_size
        self.predator_hp[env_i, pred_i] = self.config.max_hp * self.config.predator_hp_mult
        self.predator_alive[env_i, pred_i] = True
        self.predator_respawn_timer[env_i, pred_i] = 0

        # Collect edge cells
        edge_cells = []
        for c in range(gs):
            edge_cells.extend([(0, c), (gs - 1, c)])
        for r in range(1, gs - 1):
            edge_cells.extend([(r, 0), (r, gs - 1)])

        perm = torch.randperm(len(edge_cells))
        for idx in perm:
            r, c = edge_cells[idx.item()]
            if self.occupancy[env_i, r, c] == -1:
                self.predator_positions[env_i, pred_i] = torch.tensor([r, c], device=self.device)
                return

        # Fallback
        self.predator_positions[env_i, pred_i] = torch.tensor([0, 0], device=self.device)

    def _predator_step(self) -> torch.Tensor:
        """Move predators toward closest agent, attack adjacent agents, handle respawn.

        Fully vectorized over [n_envs, n_predators] — no Python loops in hot path.

        Returns:
            predator_damage_to_agents: [n_envs, n_agents] damage dealt by predators
        """
        ne, n = self.n_envs, self.n_agents
        n_pred = self.n_predators
        predator_damage = torch.zeros(ne, n, device=self.device)
        if n_pred == 0:
            return predator_damage

        # === Phase 1: Respawn dead predators ===
        dead_mask = ~self.predator_alive  # [ne, n_pred]
        if dead_mask.any():
            self.predator_respawn_timer[dead_mask] -= 1
            respawn_mask = dead_mask & (self.predator_respawn_timer <= 0)  # [ne, n_pred]
            if respawn_mask.any():
                self._update_occupancy()
                re, rp = torch.where(respawn_mask)
                for i in range(re.numel()):
                    self._respawn_predator_in_slot(re[i].item(), rp[i].item())
                self._update_occupancy()

        # === Phase 2: Movement (fully vectorized) ===
        alive = self.predator_alive  # [ne, n_pred]
        if not alive.any():
            return predator_damage

        self.predator_prev_positions = self.predator_positions.clone()

        # Distances from each predator to each agent: [ne, n_pred, n_agents]
        pred_pos_f = self.predator_positions.float()  # [ne, n_pred, 2]
        agent_pos_f = self.agent_positions.float()     # [ne, n, 2]
        dists = (pred_pos_f[:, :, None, :] - agent_pos_f[:, None, :, :]).abs().sum(dim=-1)
        # Mask dead agents
        dists = dists.masked_fill(~self.agent_alive[:, None, :], float('inf'))

        # Closest agent per predator: [ne, n_pred]
        closest_idx = dists.argmin(dim=2)

        # Gather closest agent position: [ne, n_pred, 2]
        closest_pos = torch.gather(
            agent_pos_f,  # [ne, n, 2]
            1,
            closest_idx.unsqueeze(-1).expand(ne, n_pred, 2)
        )  # [ne, n_pred, 2]

        # Direction to move
        diff = closest_pos - pred_pos_f  # [ne, n_pred, 2]
        abs_diff = diff.abs()

        # Determine step direction: move along larger axis (row-priority on tie)
        move_row = (abs_diff[:, :, 0] >= abs_diff[:, :, 1]) & (abs_diff[:, :, 0] > 0)
        move_col = ~move_row & (abs_diff[:, :, 1] > 0)

        # Compute step: +1 or -1 in chosen axis
        step_vec = torch.zeros(ne, n_pred, 2, device=self.device)
        step_vec[:, :, 0] = move_row.float() * diff[:, :, 0].sign()
        step_vec[:, :, 1] = move_col.float() * diff[:, :, 1].sign()

        candidates = self.predator_positions + step_vec.long()
        candidates = candidates.clamp(0, self.grid_size - 1)

        # Check occupancy at candidate positions (batched lookup)
        cand_r = candidates[:, :, 0]  # [ne, n_pred]
        cand_c = candidates[:, :, 1]  # [ne, n_pred]
        env_idx = torch.arange(ne, device=self.device).unsqueeze(1).expand(ne, n_pred)
        occ_at_cand = self.occupancy[env_idx, cand_r, cand_c]  # [ne, n_pred]
        can_move = (occ_at_cand == -1) & alive  # [ne, n_pred]

        # Clear old occupancy for moving predators
        old_r = self.predator_positions[:, :, 0]  # [ne, n_pred]
        old_c = self.predator_positions[:, :, 1]
        pred_occ_id = torch.arange(n_pred, device=self.device).unsqueeze(0).expand(ne, n_pred) + n
        old_is_self = (self.occupancy[env_idx, old_r, old_c] == pred_occ_id)
        clear_mask = can_move & old_is_self
        self.occupancy[env_idx[clear_mask], old_r[clear_mask], old_c[clear_mask]] = -1

        # Update positions and set new occupancy
        self.predator_positions[can_move] = candidates[can_move]
        new_r = self.predator_positions[:, :, 0]
        new_c = self.predator_positions[:, :, 1]
        self.occupancy[env_idx[alive], new_r[alive], new_c[alive]] = pred_occ_id[alive]

        # === Phase 3: Attack adjacent agents (fully vectorized) ===
        pred_pos_f = self.predator_positions.float()
        dists_atk = (pred_pos_f[:, :, None, :] - agent_pos_f[:, None, :, :]).abs().sum(dim=-1)  # [ne, n_pred, n]
        adjacent = (dists_atk <= 1.0) & self.agent_alive[:, None, :] & alive[:, :, None]

        # Pick one target per predator (closest adjacent)
        atk_dist = dists_atk.masked_fill(~adjacent, float('inf'))
        target_idx = atk_dist.argmin(dim=2)  # [ne, n_pred]
        has_target = adjacent.any(dim=2)      # [ne, n_pred]

        # Apply damage via scatter_add
        dmg = self.config.predator_damage
        dmg_per_pred = torch.zeros(ne, n_pred, device=self.device)
        dmg_per_pred[has_target] = dmg

        # Scatter damage to agent dimension: [ne, n_pred] -> [ne, n]
        predator_damage.scatter_add_(1, target_idx, dmg_per_pred)
        self.agent_hp -= predator_damage

        # Update predator ledger: channel 1 = damage pred dealt to agent
        # Expand for scatter: [ne, n_pred] -> [ne, n_pred, 1]
        hit_envs, hit_preds = torch.where(has_target)
        if hit_envs.numel() > 0:
            hit_agents = target_idx[hit_envs, hit_preds]
            self.predator_ledger[hit_envs, hit_agents, hit_preds, 1] += dmg

        return predator_damage

    # =====================================================================
    # FOOD METHODS
    # =====================================================================

    def _compute_nearest_food_info(self):
        """Batched distance to nearest food using coord_grid. Returns [ne, n] distances and values."""
        ne, n, gs = self.n_envs, self.n_agents, self.grid_size

        food_mask = self.poor_food | self.rich_food  # [ne, gs, gs]
        has_any_food = food_mask.any(dim=2).any(dim=1)  # [ne]

        # Distance from each agent to each cell: [ne, n, gs, gs]
        agent_pos_f = self.agent_positions.float()  # [ne, n, 2]
        diff = agent_pos_f[:, :, None, None, :] - self.coord_grid[None, None, :, :, :]
        dist = diff.abs().sum(dim=-1)  # [ne, n, gs, gs]

        # Mask out non-food cells
        dist_masked = dist.masked_fill(~food_mask.unsqueeze(1), float('inf'))

        # Min distance
        dist_flat = dist_masked.reshape(ne, n, gs * gs)
        min_dist, min_idx = dist_flat.min(dim=2)  # [ne, n]

        # Get food value at nearest location
        # Determine if nearest is rich or poor
        min_row = min_idx // gs
        min_col = min_idx % gs
        env_idx = torch.arange(ne, device=self.device).unsqueeze(1).expand(ne, n)
        is_rich = self.rich_food[env_idx, min_row, min_col]  # [ne, n]
        nearest_value = torch.where(is_rich, self.config.rich_food_value, self.config.poor_food_value)

        # Set inf for envs with no food
        min_dist = torch.where(
            has_any_food.unsqueeze(1).expand(ne, n),
            min_dist,
            torch.full_like(min_dist, float('inf'))
        )

        return min_dist, nearest_value

    def _process_food_eating(self, action_types):
        """Process agents eating food they're standing on."""
        ne, n, gs = self.n_envs, self.n_agents, self.grid_size

        food_rewards = torch.zeros(ne, n, device=self.device)

        env_idx = torch.arange(ne, device=self.device).unsqueeze(1).expand(ne, n)
        rows = self.agent_positions[:, :, 0]
        cols = self.agent_positions[:, :, 1]

        # Poor food
        on_poor = self.poor_food[env_idx, rows, cols] & self.agent_alive
        food_rewards += on_poor.float() * self.config.r_small
        self.agent_inventory = self.agent_inventory + on_poor.float() * self.config.poor_food_value
        self.food_eaten_total = self.food_eaten_total + on_poor.float()

        # Remove eaten poor food
        self.poor_food[env_idx[on_poor], rows[on_poor], cols[on_poor]] = False

        # Rich food
        coop_mask = (action_types == ACT_COOPERATE) & self.agent_alive
        rich_rewards = self._process_rich_food(coop_mask)
        food_rewards += rich_rewards

        return food_rewards

    def _process_rich_food(self, coop_mask):
        """Batched rich food cooperation using grid-based distances."""
        ne, n, gs = self.n_envs, self.n_agents, self.grid_size

        has_rich = self.rich_food.any(dim=2).any(dim=1)  # [ne]
        has_coops = coop_mask.sum(dim=1) >= 2  # [ne]

        if not (has_rich & has_coops).any():
            self.last_coop_success_count = 0
            return torch.zeros(ne, n, device=self.device)

        # Distance from each agent to each cell
        agent_pos_f = self.agent_positions.float()
        diff = agent_pos_f[:, :, None, None, :] - self.coord_grid[None, None, :, :, :]
        dist = diff.abs().sum(dim=-1)  # [ne, n, gs, gs]

        # Eligible: within range AND chose cooperate AND alive
        within_range = (dist <= 1) & coop_mask[:, :, None, None]  # [ne, n, gs, gs]
        # Only consider rich food cells
        eligible = within_range & self.rich_food.unsqueeze(1)  # [ne, n, gs, gs]

        # Cooperators per food cell
        cooperators_per_cell = eligible.sum(dim=1)  # [ne, gs, gs]
        consumed = cooperators_per_cell >= 2  # [ne, gs, gs]

        if not consumed.any():
            self.last_coop_success_count = 0
            return torch.zeros(ne, n, device=self.device)

        self.last_coop_success_count = consumed.sum().item()

        # Inventory: count consumed foods each agent participated in
        # eligible[e, agent, r, c] & consumed[e, r, c]
        participated = eligible & consumed.unsqueeze(1)  # [ne, n, gs, gs]
        foods_per_agent = participated.float().sum(dim=(2, 3))  # [ne, n]
        self.agent_inventory = self.agent_inventory + foods_per_agent * self.config.rich_food_value

        # Cooperation ledger update
        # For each consumed food cell, agents who participated cooperated with each other
        # participated: [ne, n, gs, gs] — flatten spatial dims
        part_flat = participated.float().reshape(ne, n, gs * gs)  # [ne, n, F]
        # consumed foods only
        consumed_flat = consumed.reshape(ne, gs * gs)  # [ne, F]
        part_consumed = part_flat * consumed_flat.unsqueeze(1)  # [ne, n, F]
        coop_pairs = torch.bmm(part_consumed, part_consumed.transpose(1, 2))  # [ne, n, n]
        coop_pairs = coop_pairs * (1.0 - self._eye.unsqueeze(0))  # zero diagonal

        # Reciprocity bonus
        prior_help = (self.ledger_tensor[:, :, :, Ledger.FOOD_GIVEN]
                      + self.ledger_tensor[:, :, :, Ledger.COOP_COUNT])
        reciprocity_bonus = (coop_pairs * prior_help).sum(dim=2) * self.config.r_reciprocity

        self.ledger_tensor[:, :, :, Ledger.COOP_COUNT] += coop_pairs

        # Remove consumed food
        self.rich_food = self.rich_food & ~consumed

        # Reward
        did_participate = (foods_per_agent > 0).float()
        self.food_eaten_total = self.food_eaten_total + did_participate
        food_reward = did_participate * self.config.r_large
        return food_reward + reciprocity_bonus

    def _compute_intrinsic_coop_rewards(self, action_types):
        """Intrinsic reward for COOP attempt near rich food + ally."""
        ne, n, gs = self.n_envs, self.n_agents, self.grid_size

        coop_mask = (action_types == ACT_COOPERATE) & self.agent_alive
        if not coop_mask.any() or not self.rich_food.any():
            return torch.zeros(ne, n, device=self.device)

        # Distance from agents to cells
        agent_pos_f = self.agent_positions.float()
        diff = agent_pos_f[:, :, None, None, :] - self.coord_grid[None, None, :, :, :]
        dist = diff.abs().sum(dim=-1)  # [ne, n, gs, gs]

        within_range = (dist <= 1) & self.agent_alive[:, :, None, None]
        within_range_rich = within_range & self.rich_food.unsqueeze(1)

        agents_per_food = within_range_rich.sum(dim=1)  # [ne, gs, gs]
        coop_opportunity = agents_per_food >= 2

        near_coop_opp = (within_range_rich & coop_opportunity.unsqueeze(1)).any(dim=(2, 3))
        reward_mask = coop_mask & near_coop_opp
        return reward_mask.float() * self.config.r_coop_attempt

    def _compute_hierarchy_rewards(self) -> torch.Tensor:
        """Compute per-step hierarchy bonus based on agent ranking (batched).

        Returns [n_envs, n_agents] tensor of hierarchy rewards.
        """
        ne, n = self.n_envs, self.n_agents
        cfg = self.config

        if cfg.r_hierarchy == 0.0:
            return torch.zeros(ne, n, device=self.device)

        alive = self.agent_alive.float()  # [ne, n]
        n_active = alive.sum(dim=1)  # [ne]
        valid_env = n_active >= 2  # [ne]

        if not valid_env.any():
            return torch.zeros(ne, n, device=self.device)

        eps = 1e-8

        # Component scores (normalize per-env by max across agents)
        food = self.food_eaten_total * alive  # [ne, n]
        food_max = food.max(dim=1, keepdim=True).values.clamp(min=eps)
        food_norm = food / food_max

        # Total damage dealt (sum over all targets from ledger)
        damage = self.ledger_tensor[:, :, :, Ledger.DAMAGE_DEALT].sum(dim=2) * alive  # [ne, n]
        damage_max = damage.max(dim=1, keepdim=True).values.clamp(min=eps)
        damage_norm = damage / damage_max

        hp_ratio = (self.agent_hp / cfg.max_hp) * alive  # [ne, n]
        hp_max = hp_ratio.max(dim=1, keepdim=True).values.clamp(min=eps)
        hp_norm = hp_ratio / hp_max

        # Weighted hierarchy score
        score = (cfg.hierarchy_food_weight * food_norm
                 + cfg.hierarchy_damage_weight * damage_norm
                 + cfg.hierarchy_hp_weight * hp_norm) * alive

        # Rank via argsort of argsort (rank 0 = lowest score)
        ranks = score.argsort(dim=1).argsort(dim=1).float()

        # Reward: linear from 0 (bottom) to r_hierarchy (top)
        denom = (n_active - 1).clamp(min=1.0).unsqueeze(1)
        rewards = (ranks / denom) * cfg.r_hierarchy * alive

        # Zero out invalid envs (< 2 active agents)
        rewards = rewards * valid_env.unsqueeze(1).float()

        return rewards

    def _spawn_food(self):
        """Spawn new food — batched for normal mode, shadow env for curriculum."""
        if self.config.pretrain_mode and self.config.curriculum_enabled:
            self._spawn_food_curriculum()
            return

        ne, gs = self.n_envs, self.grid_size
        total_cells = gs * gs
        cap_cells = int(total_cells * self.config.food_coverage_cap)

        empty = (self.occupancy == -1) & ~self.poor_food & ~self.rich_food

        # Poor food
        poor_count = self.poor_food.sum(dim=(1, 2))  # [ne]
        below_cap_poor = poor_count < cap_cells
        if below_cap_poor.any():
            poor_rate = min(1.0, self.config.poor_food_spawn_rate * 2.0)
            rand = torch.rand(ne, gs, gs, device=self.device)
            spawn_poor = empty & (rand < poor_rate) & below_cap_poor[:, None, None]
            self.poor_food = self.poor_food | spawn_poor

        # Rich food (skip in pretrain)
        if not self.config.pretrain_mode:
            rich_count = self.rich_food.sum(dim=(1, 2))
            below_cap_rich = rich_count < cap_cells
            if below_cap_rich.any():
                empty_after = (self.occupancy == -1) & ~self.poor_food & ~self.rich_food
                rich_rate = min(1.0, self.config.rich_food_spawn_rate * 2.0)
                rand2 = torch.rand(ne, gs, gs, device=self.device)
                spawn_rich = empty_after & (rand2 < rich_rate) & below_cap_rich[:, None, None]
                self.rich_food = self.rich_food | spawn_rich

    def _spawn_food_curriculum(self):
        """Curriculum food spawning via shadow env (called every step during pretrain)."""
        # Check which envs need food respawned
        for i in range(self.n_envs):
            if self._current_scenario is not None:
                # Load state into shadow, respawn, copy back
                self._load_shadow_from_slot(i)
                self._current_scenario.respawn_food(self._shadow_env)
                self.poor_food[i] = self._shadow_env.poor_food
                self.rich_food[i] = self._shadow_env.rich_food
            else:
                # Legacy curriculum spawning
                self._load_shadow_from_slot(i)
                self._shadow_env._spawn_food_curriculum_continuous()
                self.poor_food[i] = self._shadow_env.poor_food
                self.rich_food[i] = self._shadow_env.rich_food

    def _load_shadow_from_slot(self, slot: int):
        """Load shadow env state from a batch slot."""
        s = self._shadow_env
        s.agent_positions = self.agent_positions[slot].clone()
        s.agent_hp = self.agent_hp[slot].clone()
        s.agent_alive = self.agent_alive[slot].clone()
        s.agent_inventory = self.agent_inventory[slot].clone()
        s.occupancy = self.occupancy[slot].clone()
        s.poor_food = self.poor_food[slot].clone()
        s.rich_food = self.rich_food[slot].clone()
        s.signals = self.signals[slot].clone()
        s.ledger.tensor = self.ledger_tensor[slot].clone()
        s.step_count = self.step_count[slot].item()
        s.curriculum_phase = self.curriculum_phase
        s._current_scenario = self._current_scenario
        if self.n_predators > 0:
            s.predator_positions = self.predator_positions[slot].clone()
            s.predator_prev_positions = self.predator_prev_positions[slot].clone()
            s.predator_hp = self.predator_hp[slot].clone()
            s.predator_alive = self.predator_alive[slot].clone()
            s.predator_respawn_timer = self.predator_respawn_timer[slot].clone()
            if hasattr(self, 'predator_ledger'):
                s.predator_ledger = self.predator_ledger[slot].clone()

    # =====================================================================
    # OBSERVATIONS
    # =====================================================================

    def _fourier_encode(self, rel_pos):
        """Fourier encode relative positions. rel_pos: [..., 2] -> [..., fourier_bands*4]."""
        bands = [2.0 ** i for i in range(self.config.fourier_bands)]
        dx = rel_pos[..., 0:1]
        dy = rel_pos[..., 1:2]
        features = []
        for freq in bands:
            features.append(torch.sin(freq * torch.pi * dx))
            features.append(torch.cos(freq * torch.pi * dx))
            features.append(torch.sin(freq * torch.pi * dy))
            features.append(torch.cos(freq * torch.pi * dy))
        return torch.cat(features, dim=-1)

    def _get_all_observations(self) -> Dict[str, torch.Tensor]:
        """Build batched observations [n_envs, n_agents, ...]."""
        entity_tokens, entity_mask = self._get_entity_tokens()
        ne, n = self.n_envs, self.n_agents
        return {
            'entity_tokens': entity_tokens,
            'entity_mask': entity_mask,
            'signals': self.signals.float().unsqueeze(1).expand(ne, n, n),
            'self_hp': (self.agent_hp / self.config.max_hp).unsqueeze(2),
            'self_inventory': (self.agent_inventory / self.config.max_hp).unsqueeze(2),
            'agent_id': torch.arange(n, device=self.device).unsqueeze(0).expand(ne, n),
        }

    def _get_entity_tokens(self):
        """Build entity tokens for all envs/agents. Returns [ne, n, max_ent, 25], [ne, n, max_ent]."""
        ne, n, gs = self.n_envs, self.n_agents, self.grid_size
        max_food = self.config.max_food_tokens
        max_ent = self.config.max_entities
        token_dim = self.config.entity_token_dim

        tokens = torch.zeros(ne, n, max_ent, token_dim, device=self.device)
        mask = torch.zeros(ne, n, max_ent, device=self.device, dtype=torch.bool)

        # === AGENT TOKENS (first n slots) ===
        # Relative positions [ne, n_observer, n_target, 2]
        pos_f = self.agent_positions.float()
        rel_pos = pos_f.unsqueeze(1) - pos_f.unsqueeze(2)  # [ne, n, n, 2]
        rel_pos_norm = rel_pos / gs

        prev_f = self.prev_agent_positions.float()
        rel_prev = prev_f.unsqueeze(1) - prev_f.unsqueeze(2)
        rel_prev_norm = rel_prev / gs

        fourier_agents = self._fourier_encode(rel_pos_norm)  # [ne, n, n, 16]
        velocity_agents = (rel_pos_norm - rel_prev_norm) * 10.0  # [ne, n, n, 2]

        type_onehot = torch.zeros(ne, n, n, 2, device=self.device)
        type_onehot[:, :, :, 1] = 1  # is_agent

        agent_hp_norm = (self.agent_hp / self.config.max_hp).view(ne, 1, n, 1).expand(ne, n, n, 1)

        # Social features from ledger [ne, n, n, 4]
        if self.config.ablate_ledger:
            social = torch.zeros(ne, n, n, 4, device=self.device)
        else:
            social = (self.ledger_tensor * self._ledger_norm_scale).clamp(0, 1) * 5.0

        agent_tokens = torch.cat([fourier_agents, velocity_agents, type_onehot, agent_hp_norm, social], dim=-1)
        tokens[:, :, :n, :] = agent_tokens
        mask[:, :, :n] = self.agent_alive.unsqueeze(1).expand(ne, n, n)

        # === FOOD TOKENS (next max_food slots) ===
        # Grid-based approach: compute distances, topk
        agent_pos_f = self.agent_positions.float()  # [ne, n, 2]
        # diff to all cells: [ne, n, gs, gs, 2]
        diff_food = agent_pos_f[:, :, None, None, :] - self.coord_grid[None, None, :, :, :]
        dist_food = diff_food.abs().sum(dim=-1)  # [ne, n, gs, gs]

        # Food presence and quality
        poor_f = self.poor_food.unsqueeze(1).expand(ne, n, gs, gs)
        rich_f = self.rich_food.unsqueeze(1).expand(ne, n, gs, gs)
        any_food = poor_f | rich_f  # [ne, n, gs, gs]
        quality = torch.where(rich_f, torch.ones_like(dist_food), torch.full_like(dist_food, 0.5))
        quality = quality * any_food.float()

        # Mask non-food cells
        dist_food_masked = dist_food.masked_fill(~any_food, float('inf'))
        dist_flat = dist_food_masked.reshape(ne, n, gs * gs)

        # Number of food tokens to use
        n_food_total = any_food.reshape(ne, n, gs * gs).sum(dim=2)  # [ne, n]
        k = min(max_food, gs * gs)

        # Topk nearest
        _, nearest_idx = dist_flat.topk(k, dim=2, largest=False)  # [ne, n, k]

        # Gather coordinates and quality
        nearest_row = nearest_idx // gs
        nearest_col = nearest_idx % gs

        # Relative positions for food tokens
        food_rel_row = nearest_row.float() - agent_pos_f[:, :, 0:1]
        food_rel_col = nearest_col.float() - agent_pos_f[:, :, 1:2]
        food_rel = torch.stack([food_rel_row, food_rel_col], dim=-1) / gs  # [ne, n, k, 2]

        fourier_food = self._fourier_encode(food_rel)  # [ne, n, k, 16]
        velocity_food = torch.zeros(ne, n, k, 2, device=self.device)

        type_food = torch.zeros(ne, n, k, 2, device=self.device)
        type_food[:, :, :, 0] = 1  # is_food

        # Gather quality from flat grid
        quality_flat = quality.reshape(ne, n, gs * gs)
        food_val = torch.gather(quality_flat, 2, nearest_idx).unsqueeze(-1)  # [ne, n, k, 1]

        food_social = torch.zeros(ne, n, k, 4, device=self.device)

        food_tokens = torch.cat([fourier_food, velocity_food, type_food, food_val, food_social], dim=-1)

        # Mask: only valid if there was actually food there
        food_dist_gathered = torch.gather(dist_flat, 2, nearest_idx)
        food_valid = ~torch.isinf(food_dist_gathered)

        actual_k = min(k, max_food)
        tokens[:, :, n:n + actual_k, :] = food_tokens[:, :, :actual_k, :]
        mask[:, :, n:n + actual_k] = food_valid[:, :, :actual_k]

        # === PREDATOR TOKENS (after food tokens) ===
        if self.n_predators > 0:
            pred_start = n + max_food
            for p in range(self.n_predators):
                slot = pred_start + p
                if slot >= max_ent:
                    break

                # Relative position: pred_pos - observer_pos, normalized [ne, n, 2]
                pred_pos_f = self.predator_positions[:, p:p+1, :].float()  # [ne, 1, 2]
                observer_pos_f = self.agent_positions.float()  # [ne, n, 2]
                rel_pos_pred = (pred_pos_f - observer_pos_f) / gs  # [ne, n, 2]

                fourier_pred = self._fourier_encode(rel_pos_pred)  # [ne, n, 16]

                # Velocity
                pred_prev_f = self.predator_prev_positions[:, p:p+1, :].float()
                observer_prev_f = self.prev_agent_positions.float()
                rel_prev_pred = (pred_prev_f - observer_prev_f) / gs
                velocity_pred = (rel_pos_pred - rel_prev_pred) * 10.0  # [ne, n, 2]

                # Type: agent-like [0, 1]
                type_pred = torch.zeros(ne, n, 2, device=self.device)
                type_pred[:, :, 1] = 1  # is_agent

                # HP normalized by max_hp (>1.0 distinguishes from agents)
                hp_val = (self.predator_hp[:, p] / self.config.max_hp).view(ne, 1, 1).expand(ne, n, 1)

                # Social: from predator ledger (damage dealt/received)
                social_pred = torch.zeros(ne, n, 4, device=self.device)
                if hasattr(self, 'predator_ledger'):
                    social_pred[:, :, 0] = (self.predator_ledger[:, :, p, 0] / 100.0).clamp(0, 1) * 5.0
                    social_pred[:, :, 1] = (self.predator_ledger[:, :, p, 1] / 100.0).clamp(0, 1) * 5.0

                pred_token = torch.cat([fourier_pred, velocity_pred, type_pred, hp_val, social_pred], dim=-1)  # [ne, n, 25]
                tokens[:, :, slot, :] = pred_token

                # Mask: alive predators
                pred_alive_expanded = self.predator_alive[:, p].unsqueeze(1).expand(ne, n)  # [ne, n]
                mask[:, :, slot] = pred_alive_expanded

        return tokens, mask

    def _get_action_masks_batched(self):
        """Compute action masks for all envs. Returns [ne, n, 5], [ne, n, 5]."""
        ne, n, gs = self.n_envs, self.n_agents, self.grid_size

        # Direction mask [ne, n, 5]
        intended = self.agent_positions.unsqueeze(2) + self.direction_deltas.unsqueeze(0).unsqueeze(0)
        in_bounds = (
            (intended[..., 0] >= 0) & (intended[..., 0] < gs) &
            (intended[..., 1] >= 0) & (intended[..., 1] < gs)
        )
        direction_mask = in_bounds & self.agent_alive.unsqueeze(2)

        # Check adjacent agents (for ATTACK/GIVE)
        target_pos = self.agent_positions.unsqueeze(2) + self.dir_deltas_4.unsqueeze(0).unsqueeze(0)
        target_in_bounds = (
            (target_pos[..., 0] >= 0) & (target_pos[..., 0] < gs) &
            (target_pos[..., 1] >= 0) & (target_pos[..., 1] < gs)
        )
        safe_rows = target_pos[..., 0].clamp(0, gs - 1)
        safe_cols = target_pos[..., 1].clamp(0, gs - 1)
        env_idx = torch.arange(ne, device=self.device).view(ne, 1, 1).expand(ne, n, 4)
        occupants = self.occupancy[env_idx, safe_rows, safe_cols]  # [ne, n, 4]

        has_target = occupants >= 0
        is_agent_occ = has_target & (occupants < n)
        safe_occ = occupants.clamp(min=0, max=n - 1)
        # Gather alive status for agent occupants
        env_idx_flat = env_idx.reshape(-1)
        occ_flat = safe_occ.reshape(-1)
        target_alive_flat = self.agent_alive[env_idx_flat, occ_flat]
        target_alive = target_alive_flat.reshape(ne, n, 4) & is_agent_occ
        direction_has_agent = target_in_bounds & target_alive

        # Check for adjacent predators too
        direction_has_predator = target_in_bounds & has_target & (occupants >= n)
        any_adjacent_agent = direction_has_agent.any(dim=2)
        any_adjacent_target = (direction_has_agent | direction_has_predator).any(dim=2)

        # Action type mask [ne, n, 5]
        action_type_mask = torch.zeros(ne, n, 5, dtype=torch.bool, device=self.device)
        action_type_mask[:, :, ACT_MOVE] = self.agent_alive
        action_type_mask[:, :, ACT_ATTACK] = any_adjacent_target & self.agent_alive
        action_type_mask[:, :, ACT_GIVE] = any_adjacent_agent & self.agent_alive
        action_type_mask[:, :, ACT_SIGNAL] = self.agent_alive
        # COOPERATE: rich food within distance 1
        if self.rich_food.any():
            agent_pos_f = self.agent_positions.float()
            diff = agent_pos_f[:, :, None, None, :] - self.coord_grid[None, None, :, :, :]
            dist = diff.abs().sum(dim=-1)  # [ne, n, gs, gs]
            rich_nearby = ((dist <= 1) & self.rich_food.unsqueeze(1)).any(dim=(2, 3))
        else:
            rich_nearby = torch.zeros(ne, n, dtype=torch.bool, device=self.device)
        action_type_mask[:, :, ACT_COOPERATE] = rich_nearby & self.agent_alive

        return direction_mask, action_type_mask

    # =====================================================================
    # CURRICULUM
    # =====================================================================

    def _check_curriculum_advance(self, episode_return: float):
        from scenarios import THRESHOLDS
        max_phase = max(THRESHOLDS.keys())
        if self.curriculum_phase >= max_phase:
            return
        threshold = THRESHOLDS.get(self.curriculum_phase, 999999)
        if episode_return >= threshold and self.curriculum_phase < max_phase:
            self.curriculum_phase += 1
            print(f"[Curriculum] Return {episode_return:.1f} >= {threshold} -> Advanced to phase {self.curriculum_phase}")
