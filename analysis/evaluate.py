"""
Evaluation & Personality Profiling System.

Loads a trained checkpoint (PPO or SAC), profiles each agent's emergent
personality based on behavioral tendencies, then evaluates them across varied
scenarios to see which personality types perform best.

Supports:
- PPO checkpoints (SharedTrunkActorCritic via PPO)
- SAC checkpoints (SharedTrunkActorCritic via SAC)
- Predator behavior metrics (kills, damage dealt/taken)
- Social hierarchy ranking

Usage:
    python evaluate.py --config eval_config.yaml
"""
import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import yaml

from core.config import Config
from env.environment import GridWorld
from core.ledger import Ledger
from agents.ppo import PPO
from agents.sac import SAC
from training.scenarios import SocialScenario, MultiTeamScenario, CoopFoodScenario

# Action / direction names matching the factored action space
ACT_NAMES = ['MOVE', 'ATTACK', 'GIVE', 'SIGNAL', 'COOP']
DIR_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


@dataclass
class AgentProfile:
    agent_id: int
    personality: str
    action_pcts: Dict[str, float] = field(default_factory=dict)
    direction_pcts: Dict[str, float] = field(default_factory=dict)
    avg_return: float = 0.0
    avg_survival: float = 0.0
    avg_damage_dealt: float = 0.0
    avg_food_given: float = 0.0
    avg_coop_count: float = 0.0
    avg_predator_kills: float = 0.0
    avg_predator_damage_dealt: float = 0.0
    avg_predator_damage_taken: float = 0.0
    avg_hierarchy_rank: float = 0.0
    avg_hierarchy_score: float = 0.0


@dataclass
class EpisodeResult:
    agent_metrics: List[dict] = field(default_factory=list)


class EvaluationRunner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.yaml_cfg = yaml.safe_load(f)

        # Device
        dev = self.yaml_cfg.get('device', 'auto')
        if dev == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(dev)
        print(f"Device: {self.device}")

        self.greedy = self.yaml_cfg['evaluation'].get('greedy', False)
        self.episodes_per_scenario = self.yaml_cfg['evaluation']['episodes_per_scenario']
        self.profiling_episodes = self.yaml_cfg['profiling']['episodes']
        self.thresholds = self.yaml_cfg.get('personality', {})

        self.ppo = None   # PPO or None
        self.sac = None   # SAC or None
        self.algo = 'ppo'  # 'ppo' or 'sac'
        self.config: Optional[Config] = None

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def load_checkpoint(self):
        path = self.yaml_cfg['checkpoint']
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Reconstruct Config
        raw_cfg = ckpt['config']
        if isinstance(raw_cfg, dict):
            self.config = Config.from_dict(raw_cfg)
        else:
            self.config = raw_cfg

        # --- SAC checkpoint detection ---
        if 'sac_state' in ckpt:
            self.algo = 'sac'
            self.sac = SAC(self.config, self.device)
            self.sac.load_state_dict(ckpt['sac_state'])
            self.sac.actor.eval()
            mode = "SAC (shared trunk)"
        # --- PPO checkpoint detection ---
        elif 'network_state_dict' in ckpt:
            self.algo = 'ppo'
            self.ppo = PPO(self.config, self.device)
            self.ppo.network.load_state_dict(ckpt['network_state_dict'])
            self.ppo.network.eval()
            mode = "PPO (shared trunk)"
        else:
            raise ValueError("Checkpoint has no recognized format "
                             "(expected 'sac_state' or 'network_state_dict')")

        print(f"Loaded {self.config.n_agents} agent networks [{mode}] "
              f"(grid {self.config.grid_size}x{self.config.grid_size})")

    # ------------------------------------------------------------------
    # Scenario factory
    # ------------------------------------------------------------------
    def create_scenario(self, entry: dict):
        t = entry['type']
        p = entry.get('params', {})
        if t == 'social':
            return SocialScenario(**p)
        elif t == 'team':
            return MultiTeamScenario(**p)
        elif t == 'coop':
            return CoopFoodScenario(**p)
        else:
            raise ValueError(f"Unknown scenario type: {t}")

    # ------------------------------------------------------------------
    # Hierarchy scoring (mirrors environment._compute_hierarchy_rewards)
    # ------------------------------------------------------------------
    def _compute_hierarchy(self, env: GridWorld, n_active: int):
        """Compute hierarchy score and rank for each agent.

        Returns (scores[n_active], ranks[n_active]) where rank 0 = worst.
        """
        cfg = self.config
        alive = env.agent_alive[:n_active].float()
        n_alive = alive.sum()
        eps = 1e-8

        food = env.food_eaten_total[:n_active] * alive
        food_norm = food / food.max().clamp(min=eps)

        damage = env.ledger.tensor[:n_active, :, Ledger.DAMAGE_DEALT].sum(dim=1) * alive
        damage_norm = damage / damage.max().clamp(min=eps)

        hp_ratio = (env.agent_hp[:n_active] / cfg.max_hp) * alive
        hp_norm = hp_ratio / hp_ratio.max().clamp(min=eps)

        score = (getattr(cfg, 'hierarchy_food_weight', 1.0) * food_norm
                 + getattr(cfg, 'hierarchy_damage_weight', 1.0) * damage_norm
                 + getattr(cfg, 'hierarchy_hp_weight', 1.0) * hp_norm) * alive

        # Include predator kills if weight > 0
        kill_w = getattr(cfg, 'hierarchy_kill_weight', 0.0)
        if kill_w > 0 and hasattr(env, 'predator_kills_total'):
            kills = env.predator_kills_total[:n_active] * alive
            kills_norm = kills / kills.max().clamp(min=eps)
            score = score + kill_w * kills_norm * alive

        # Rank via double argsort (rank 0 = lowest score)
        ranks = score.argsort().argsort().float()

        return score, ranks

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------
    def run_episode(self, env: GridWorld, scenario) -> List[dict]:
        obs = env.reset()
        env.apply_scenario(scenario)

        sc = scenario.get_config()
        n_active = sc.n_active_agents
        n = self.config.n_agents

        # Configure PPO for this scenario's n_active and clone_mode
        if self.algo == 'ppo':
            self.ppo.set_n_active(n_active)
            self.ppo.set_clone_mode(sc.clone_mode)

        trackers = [{
            'steps_alive': 0,
            'total_return': 0.0,
            'action_counts': [0] * 5,
            'direction_counts': [0] * 5,
            'reward_survival': 0.0,
            'reward_resource': 0.0,
            'reward_social': 0.0,
        } for _ in range(n)]

        for step in range(self.config.max_steps_per_episode):
            dir_mask, act_mask = env.get_action_masks()

            with torch.no_grad():
                if self.algo == 'ppo':
                    dirs, acts, _, _, _ = self.ppo.get_actions_and_values(
                        obs, dir_mask, act_mask
                    )
                else:
                    # SAC path: add env batch dim [1, n_agents, ...]
                    obs_batched = {k: v.unsqueeze(0) for k, v in obs.items()}
                    dirs, acts = self.sac.get_actions(
                        obs_batched, deterministic=self.greedy
                    )
                    dirs = dirs.squeeze(0)
                    acts = acts.squeeze(0)

            obs, rewards, dones, infos = env.step(dirs, acts)

            for i in range(n_active):
                if env.agent_alive[i] or step == 0:
                    trackers[i]['steps_alive'] += 1
                    trackers[i]['total_return'] += rewards[i].item()
                    trackers[i]['action_counts'][acts[i].item()] += 1
                    trackers[i]['direction_counts'][dirs[i].item()] += 1
                    trackers[i]['reward_survival'] += infos['reward_survival'][i].item()
                    trackers[i]['reward_resource'] += infos['reward_resource'][i].item()
                    trackers[i]['reward_social'] += infos['reward_social'][i].item()

            if dones.all():
                break

        # Extract final ledger stats
        for i in range(n_active):
            trackers[i]['damage_dealt'] = env.ledger.tensor[i, :, Ledger.DAMAGE_DEALT].sum().item()
            trackers[i]['food_given'] = env.ledger.tensor[i, :, Ledger.FOOD_GIVEN].sum().item()
            trackers[i]['coop_count'] = env.ledger.tensor[i, :, Ledger.COOP_COUNT].sum().item()
            trackers[i]['defense_score'] = env.ledger.tensor[i, :, Ledger.DEFENSE_SCORE].sum().item()

        # Predator metrics
        has_predators = (getattr(self.config, 'n_predators', 0) > 0
                         and hasattr(env, 'predator_kills_total')
                         and env.predator_kills_total is not None)
        for i in range(n_active):
            if has_predators:
                trackers[i]['predator_kills'] = env.predator_kills_total[i].item()
                trackers[i]['predator_damage_dealt'] = env.predator_ledger[i, :, 0].sum().item() if env.predator_ledger is not None else 0.0
                trackers[i]['predator_damage_taken'] = env.predator_ledger[i, :, 1].sum().item() if env.predator_ledger is not None else 0.0
            else:
                trackers[i]['predator_kills'] = 0.0
                trackers[i]['predator_damage_dealt'] = 0.0
                trackers[i]['predator_damage_taken'] = 0.0

        # Hierarchy ranking
        scores, ranks = self._compute_hierarchy(env, n_active)
        for i in range(n_active):
            trackers[i]['hierarchy_score'] = scores[i].item()
            trackers[i]['hierarchy_rank'] = int(ranks[i].item())

        return trackers[:n_active]

    # ------------------------------------------------------------------
    # Personality classification
    # ------------------------------------------------------------------
    def classify_personality(self, action_counts: List[int], direction_counts: List[int]) -> str:
        total_acts = sum(action_counts) or 1
        total_dirs = sum(direction_counts) or 1

        act_pcts = [c / total_acts for c in action_counts]
        dir_pcts = [c / total_dirs for c in direction_counts]

        agg_thresh = self.thresholds.get('aggressive_threshold', 0.15)
        coop_thresh = self.thresholds.get('cooperative_threshold', 0.20)
        forage_thresh = self.thresholds.get('forager_threshold', 0.60)
        passive_thresh = self.thresholds.get('passive_stay_threshold', 0.40)

        # Priority-based classification
        # 1. Aggressive -- attack % > threshold
        if act_pcts[1] > agg_thresh:
            return 'Aggressive'
        # 2. Cooperative -- (coop + give) % > threshold
        if (act_pcts[4] + act_pcts[2]) > coop_thresh:
            return 'Cooperative'
        # 3. Forager -- move % > threshold
        if act_pcts[0] > forage_thresh:
            return 'Forager'
        # 4. Passive -- STAY direction % > threshold
        if dir_pcts[4] > passive_thresh:
            return 'Passive'
        # 5. Balanced
        return 'Balanced'

    # ------------------------------------------------------------------
    # Profiling phase
    # ------------------------------------------------------------------
    def profile_agents(self) -> List[AgentProfile]:
        print("\n" + "=" * 64)
        print("               PROFILING AGENTS")
        print("=" * 64)

        # Use a neutral free-for-all for profiling
        scenario = SocialScenario(n_agents=self.config.n_agents, inject_histories=False, n_rich_food=5)
        env = GridWorld(self.config, self.device)

        # Accumulators per agent
        n = self.config.n_agents
        accum = [{
            'action_counts': [0] * 5,
            'direction_counts': [0] * 5,
            'total_return': 0.0,
            'steps_alive': 0,
            'damage_dealt': 0.0,
            'food_given': 0.0,
            'coop_count': 0.0,
            'predator_kills': 0.0,
            'predator_damage_dealt': 0.0,
            'predator_damage_taken': 0.0,
            'hierarchy_rank': 0.0,
            'hierarchy_score': 0.0,
        } for _ in range(n)]

        for ep in range(self.profiling_episodes):
            trackers = self.run_episode(env, scenario)
            for i, t in enumerate(trackers):
                for a in range(5):
                    accum[i]['action_counts'][a] += t['action_counts'][a]
                    accum[i]['direction_counts'][a] += t['direction_counts'][a]
                accum[i]['total_return'] += t['total_return']
                accum[i]['steps_alive'] += t['steps_alive']
                accum[i]['damage_dealt'] += t.get('damage_dealt', 0.0)
                accum[i]['food_given'] += t.get('food_given', 0.0)
                accum[i]['coop_count'] += t.get('coop_count', 0.0)
                accum[i]['predator_kills'] += t.get('predator_kills', 0.0)
                accum[i]['predator_damage_dealt'] += t.get('predator_damage_dealt', 0.0)
                accum[i]['predator_damage_taken'] += t.get('predator_damage_taken', 0.0)
                accum[i]['hierarchy_rank'] += t.get('hierarchy_rank', 0.0)
                accum[i]['hierarchy_score'] += t.get('hierarchy_score', 0.0)
            print(f"  Profiling episode {ep + 1}/{self.profiling_episodes} done")

        # Build profiles
        profiles = []
        eps = self.profiling_episodes
        for i in range(n):
            a = accum[i]
            total_acts = sum(a['action_counts']) or 1
            total_dirs = sum(a['direction_counts']) or 1

            act_pcts = {ACT_NAMES[j]: a['action_counts'][j] / total_acts for j in range(5)}
            dir_pcts = {DIR_NAMES[j]: a['direction_counts'][j] / total_dirs for j in range(5)}
            personality = self.classify_personality(a['action_counts'], a['direction_counts'])

            profiles.append(AgentProfile(
                agent_id=i,
                personality=personality,
                action_pcts=act_pcts,
                direction_pcts=dir_pcts,
                avg_return=a['total_return'] / eps,
                avg_survival=a['steps_alive'] / eps,
                avg_damage_dealt=a['damage_dealt'] / eps,
                avg_food_given=a['food_given'] / eps,
                avg_coop_count=a['coop_count'] / eps,
                avg_predator_kills=a['predator_kills'] / eps,
                avg_predator_damage_dealt=a['predator_damage_dealt'] / eps,
                avg_predator_damage_taken=a['predator_damage_taken'] / eps,
                avg_hierarchy_rank=a['hierarchy_rank'] / eps,
                avg_hierarchy_score=a['hierarchy_score'] / eps,
            ))

        return profiles

    # ------------------------------------------------------------------
    # Evaluation phase
    # ------------------------------------------------------------------
    def evaluate_scenarios(self, profiles: List[AgentProfile]) -> Dict[str, List[EpisodeResult]]:
        scenarios_cfg = self.yaml_cfg.get('scenarios', [])
        results: Dict[str, List[EpisodeResult]] = {}

        for sc_entry in scenarios_cfg:
            name = sc_entry['name']
            print(f"\n{'=' * 64}")
            print(f"  Scenario: {name}")
            print(f"{'=' * 64}")

            scenario = self.create_scenario(sc_entry)
            env = GridWorld(self.config, self.device)
            sc_cfg = scenario.get_config()

            ep_results = []
            for ep in range(self.episodes_per_scenario):
                trackers = self.run_episode(env, scenario)
                ep_results.append(EpisodeResult(agent_metrics=trackers))
                print(f"  Episode {ep + 1}/{self.episodes_per_scenario} done")

            results[name] = ep_results

        return results

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def print_report(self, profiles: List[AgentProfile], results: Dict[str, List[EpisodeResult]]):
        has_predators = getattr(self.config, 'n_predators', 0) > 0

        print("\n" + "=" * 64)
        print("                AGENT PERSONALITY PROFILES")
        print("=" * 64)

        for p in profiles:
            act_str = " | ".join(f"{k} {v*100:.0f}%" for k, v in p.action_pcts.items())
            print(f"\nAgent {p.agent_id}: {p.personality}")
            print(f"  Actions: {act_str}")
            print(f"  Avg Return: {p.avg_return:.1f}  |  Survival: {p.avg_survival:.0f}/{self.config.max_steps_per_episode} steps")
            print(f"  Damage Dealt: {p.avg_damage_dealt:.1f}  |  Food Given: {p.avg_food_given:.1f}  |  Coops: {p.avg_coop_count:.1f}")
            if has_predators:
                print(f"  Pred Kills: {p.avg_predator_kills:.1f}  |  Pred Dmg Dealt: {p.avg_predator_damage_dealt:.1f}  |  Pred Dmg Taken: {p.avg_predator_damage_taken:.1f}")
            print(f"  Hierarchy Rank: {p.avg_hierarchy_rank:.1f}  |  Hierarchy Score: {p.avg_hierarchy_score:.2f}")

        # Build personality map: agent_id -> personality
        personality_map = {p.agent_id: p.personality for p in profiles}

        print("\n" + "=" * 64)
        print("              SCENARIO EVALUATION RESULTS")
        print("=" * 64)

        # For global rankings
        personality_global: Dict[str, List[float]] = defaultdict(list)
        personality_combat: Dict[str, List[float]] = defaultdict(list)
        personality_survival: Dict[str, List[float]] = defaultdict(list)

        for scenario_name, ep_results in results.items():
            print(f"\n{scenario_name} ({len(ep_results)} episodes):")

            # Per-agent breakdown — build header dynamically
            hdr = f"  {'Agent':<8}{'Type':<14}| {'Return':>8} | {'Alive':>6} | {'Dmg':>7} | {'Food':>6} | {'Coop':>5}"
            if has_predators:
                hdr += f" | {'PKills':>6} | {'PDmg':>6}"
            hdr += f" | {'Rank':>4} | {'R_surv':>7} | {'R_res':>7} | {'R_soc':>7}"
            print(hdr)

            sep = f"  {'-'*8}{'-'*14}|{'-'*10}|{'-'*8}|{'-'*9}|{'-'*8}|{'-'*7}"
            if has_predators:
                sep += f"|{'-'*8}|{'-'*8}"
            sep += f"|{'-'*6}|{'-'*9}|{'-'*9}|{'-'*9}"
            print(sep)

            for er in ep_results:
                for idx, am in enumerate(er.agent_metrics):
                    aid = idx
                    pers = personality_map.get(aid, '?')
                    row = f"  {aid:<8}{pers:<14}| {am['total_return']:>8.1f} | {am['steps_alive']:>6} | {am.get('damage_dealt', 0):>7.1f} | {am.get('food_given', 0):>6.1f} | {am.get('coop_count', 0):>5.1f}"
                    if has_predators:
                        row += f" | {am.get('predator_kills', 0):>6.0f} | {am.get('predator_damage_dealt', 0):>6.1f}"
                    row += f" | {am.get('hierarchy_rank', 0):>4} | {am['reward_survival']:>7.1f} | {am['reward_resource']:>7.1f} | {am['reward_social']:>7.1f}"
                    print(row)

            # Aggregate by personality
            by_personality: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for er in ep_results:
                for am in er.agent_metrics:
                    aid = am.get('agent_id', er.agent_metrics.index(am))
                    if aid < len(profiles):
                        pers = personality_map[aid]
                    else:
                        pers = 'Unknown'
                    by_personality[pers]['return'].append(am['total_return'])
                    by_personality[pers]['survival'].append(am['steps_alive'])
                    by_personality[pers]['food'].append(am.get('reward_resource', 0.0))
                    by_personality[pers]['damage'].append(am.get('damage_dealt', 0.0))
                    if has_predators:
                        by_personality[pers]['pred_kills'].append(am.get('predator_kills', 0.0))
                    by_personality[pers]['hierarchy_rank'].append(am.get('hierarchy_rank', 0))

            is_combat = 'battle' in scenario_name.lower() or 'team' in scenario_name.lower()

            print(f"\n  By personality:")
            pers_hdr = f"  {'Personality':<16}| {'Avg Return':>10} | {'Avg Survival':>12} | {'Avg Damage':>10}"
            if has_predators:
                pers_hdr += f" | {'Avg PKills':>10}"
            pers_hdr += f" | {'Avg Rank':>8}"
            print(pers_hdr)

            pers_sep = f"  {'-'*16}|{'-'*12}|{'-'*14}|{'-'*12}"
            if has_predators:
                pers_sep += f"|{'-'*12}"
            pers_sep += f"|{'-'*10}"
            print(pers_sep)

            for pers, metrics in sorted(by_personality.items()):
                n_samples = len(metrics['return']) or 1
                avg_ret = sum(metrics['return']) / n_samples
                avg_surv = sum(metrics['survival']) / n_samples
                avg_dmg = sum(metrics['damage']) / n_samples
                row = f"  {pers:<16}| {avg_ret:>10.1f} | {avg_surv:>12.1f} | {avg_dmg:>10.1f}"
                if has_predators:
                    avg_pk = sum(metrics.get('pred_kills', [0])) / max(len(metrics.get('pred_kills', [1])), 1)
                    row += f" | {avg_pk:>10.1f}"
                avg_rank = sum(metrics.get('hierarchy_rank', [0])) / max(len(metrics.get('hierarchy_rank', [1])), 1)
                row += f" | {avg_rank:>8.1f}"
                print(row)

                personality_global[pers].append(avg_ret)
                personality_survival[pers].append(avg_surv)
                if is_combat:
                    personality_combat[pers].append(avg_ret)

        # Rankings
        print("\n" + "=" * 64)
        print("                   PERSONALITY RANKINGS")
        print("=" * 64)

        def best_of(data: Dict[str, List[float]], metric_name: str):
            if not data:
                return
            avg = {k: sum(v) / len(v) for k, v in data.items() if v}
            if avg:
                best = max(avg, key=avg.get)
                print(f"  {metric_name}: {best} ({avg[best]:.1f})")

        best_of(personality_global, "Best overall (avg return)")
        best_of(personality_combat, "Best in combat (avg return in team battles)")
        best_of(personality_survival, "Best at survival (avg steps alive)")
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    def save_results(self, profiles: List[AgentProfile], results: Dict[str, List[EpisodeResult]]):
        out_dir = self.yaml_cfg.get('output', {}).get('directory', './eval_results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'results.yaml')

        has_predators = getattr(self.config, 'n_predators', 0) > 0

        # Build profiles section
        profiles_data = []
        for p in profiles:
            pd = {
                'agent_id': p.agent_id,
                'personality': p.personality,
                'action_pcts': {k: round(v, 4) for k, v in p.action_pcts.items()},
                'direction_pcts': {k: round(v, 4) for k, v in p.direction_pcts.items()},
                'avg_return': round(p.avg_return, 2),
                'avg_survival': round(p.avg_survival, 1),
                'avg_damage_dealt': round(p.avg_damage_dealt, 2),
                'avg_food_given': round(p.avg_food_given, 2),
                'avg_coop_count': round(p.avg_coop_count, 2),
                'avg_hierarchy_rank': round(p.avg_hierarchy_rank, 1),
                'avg_hierarchy_score': round(p.avg_hierarchy_score, 3),
            }
            if has_predators:
                pd['avg_predator_kills'] = round(p.avg_predator_kills, 2)
                pd['avg_predator_damage_dealt'] = round(p.avg_predator_damage_dealt, 2)
                pd['avg_predator_damage_taken'] = round(p.avg_predator_damage_taken, 2)
            profiles_data.append(pd)

        # Build scenario section
        personality_map = {p.agent_id: p.personality for p in profiles}
        scenarios_data = {}
        for scenario_name, ep_results in results.items():
            episodes_data = []
            for ep_idx, er in enumerate(ep_results):
                agents_data = []
                for idx, am in enumerate(er.agent_metrics):
                    ad = {
                        'agent_id': idx,
                        'personality': personality_map.get(idx, 'Unknown'),
                        'return': round(am['total_return'], 2),
                        'survival': am['steps_alive'],
                        'damage_dealt': round(am.get('damage_dealt', 0.0), 2),
                        'food_given': round(am.get('food_given', 0.0), 2),
                        'coop_count': round(am.get('coop_count', 0.0), 2),
                        'hierarchy_rank': am.get('hierarchy_rank', 0),
                        'hierarchy_score': round(am.get('hierarchy_score', 0.0), 3),
                        'reward_survival': round(am['reward_survival'], 2),
                        'reward_resource': round(am['reward_resource'], 2),
                        'reward_social': round(am['reward_social'], 2),
                    }
                    if has_predators:
                        ad['predator_kills'] = am.get('predator_kills', 0)
                        ad['predator_damage_dealt'] = round(am.get('predator_damage_dealt', 0.0), 2)
                        ad['predator_damage_taken'] = round(am.get('predator_damage_taken', 0.0), 2)
                    agents_data.append(ad)
                episodes_data.append({'episode': ep_idx, 'agents': agents_data})

            # Personality summary for this scenario
            by_pers: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for er in ep_results:
                for idx, am in enumerate(er.agent_metrics):
                    pers = personality_map.get(idx, 'Unknown')
                    by_pers[pers]['avg_return'].append(am['total_return'])
                    by_pers[pers]['avg_survival'].append(am['steps_alive'])
                    by_pers[pers]['avg_damage'].append(am.get('damage_dealt', 0.0))
                    by_pers[pers]['avg_hierarchy_rank'].append(am.get('hierarchy_rank', 0))
                    if has_predators:
                        by_pers[pers]['avg_pred_kills'].append(am.get('predator_kills', 0.0))

            summary = {}
            for pers, metrics in by_pers.items():
                n = len(metrics['avg_return']) or 1
                s = {
                    'avg_return': round(sum(metrics['avg_return']) / n, 2),
                    'avg_survival': round(sum(metrics['avg_survival']) / n, 1),
                    'avg_damage': round(sum(metrics['avg_damage']) / n, 2),
                    'avg_hierarchy_rank': round(sum(metrics['avg_hierarchy_rank']) / n, 1),
                }
                if has_predators:
                    s['avg_pred_kills'] = round(sum(metrics.get('avg_pred_kills', [0])) / n, 2)
                summary[pers] = s

            scenarios_data[scenario_name] = {
                'episodes': episodes_data,
                'personality_summary': summary,
            }

        output = {
            'algo': self.algo,
            'profiles': profiles_data,
            'scenarios': scenarios_data,
        }

        with open(out_path, 'w') as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

        print(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained agents')
    parser.add_argument('--config', type=str, default='eval_config.yaml',
                        help='Path to evaluation config YAML')
    args = parser.parse_args()

    runner = EvaluationRunner(args.config)
    runner.load_checkpoint()

    profiles = runner.profile_agents()
    results = runner.evaluate_scenarios(profiles)

    runner.print_report(profiles, results)
    runner.save_results(profiles, results)


if __name__ == '__main__':
    main()
