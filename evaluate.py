"""
Evaluation & Personality Profiling System.

Loads a trained checkpoint, profiles each agent's emergent personality based on
behavioral tendencies, then evaluates them across varied scenarios to see which
personality types perform best.

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

from config import Config
from environment import GridWorld
from ppo import IndependentPPO, VmapPPO, load_state_dict_flexible
from scenarios import SocialScenario, MultiTeamScenario, CoopFoodScenario

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

        self.multi_agent = None  # IndependentPPO or VmapPPO
        self.config: Optional[Config] = None
        self.shared_weights = False  # True when all agents share one network

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

        # Handle both formats: list of state_dicts vs single state_dict
        if 'network_state_dicts' in ckpt:
            # Independent networks → use VmapPPO for parallel inference
            self.multi_agent = VmapPPO(self.config, self.device)
            for i, sd in enumerate(ckpt['network_state_dicts']):
                load_state_dict_flexible(self.multi_agent.networks[i], sd)
            for net in self.multi_agent.networks:
                net.eval()
            self.multi_agent.base_network.eval()
            mode = "independent (vmap)"
        elif 'model_state_dict' in ckpt:
            # train_vec.py format: single network — use batched path
            self.multi_agent = IndependentPPO(self.config, self.device)
            load_state_dict_flexible(self.multi_agent.networks[0], ckpt['model_state_dict'])
            self.multi_agent.networks[0].eval()
            self.shared_weights = True
            mode = "shared (batched)"
        else:
            raise ValueError("Checkpoint has neither 'network_state_dicts' nor 'model_state_dict'")

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
    # Episode runner
    # ------------------------------------------------------------------
    def run_episode(self, env: GridWorld, scenario) -> List[dict]:
        obs = env.reset()
        env.apply_scenario(scenario)

        sc = scenario.get_config()
        n_active = sc.n_active_agents
        n = self.config.n_agents

        # Choose inference path
        if self.shared_weights:
            get_actions = self._batched_actions
        else:
            self.multi_agent.set_n_active(n_active)
            self.multi_agent.set_clone_mode(sc.clone_mode)
            get_actions = self._vmap_actions

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
                dirs, acts = get_actions(obs, dir_mask, act_mask, n_active)

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
            trackers[i]['damage_dealt'] = env.ledger.tensor[i, :, 0].sum().item()
            trackers[i]['food_given'] = env.ledger.tensor[i, :, 1].sum().item()
            trackers[i]['coop_count'] = env.ledger.tensor[i, :, 2].sum().item()
            trackers[i]['defense_score'] = env.ledger.tensor[i, :, 3].sum().item()

        return trackers[:n_active]

    def _batched_actions(self, obs, dir_mask, act_mask, n_active):
        """Fast path: one forward pass for all agents through shared network."""
        from torch.distributions import Categorical
        LARGE_NEG = -1e8
        net = self.multi_agent.networks[0]

        # obs is already [n_agents, ...] — just slice to n_active and forward
        obs_batch = {k: v[:n_active] for k, v in obs.items()}

        if self.greedy:
            dir_logits, act_logits, _ = net.forward(obs_batch)
            if dir_mask is not None:
                dir_logits = dir_logits.masked_fill(~dir_mask[:n_active], LARGE_NEG)
            if act_mask is not None:
                act_logits = act_logits.masked_fill(~act_mask[:n_active], LARGE_NEG)
            dirs = dir_logits.argmax(dim=-1)
            acts = act_logits.argmax(dim=-1)
        else:
            dirs, acts, _, _, _ = net.get_action_and_value(
                obs_batch,
                direction_mask=dir_mask[:n_active] if dir_mask is not None else None,
                action_type_mask=act_mask[:n_active] if act_mask is not None else None,
            )

        # Pad inactive agents
        if n_active < self.config.n_agents:
            pad = self.config.n_agents - n_active
            dirs = torch.cat([dirs, torch.full((pad,), 4, device=self.device, dtype=dirs.dtype)])
            acts = torch.cat([acts, torch.zeros(pad, device=self.device, dtype=acts.dtype)])

        return dirs, acts

    def _vmap_actions(self, obs, dir_mask, act_mask, n_active):
        """Fast path: vmap-parallel forward pass for independent networks."""
        if self.greedy:
            return self._vmap_greedy_actions(obs, dir_mask, act_mask, n_active)
        dirs, acts, _, _, _ = self.multi_agent.get_actions_and_values(obs, dir_mask, act_mask)
        return dirs, acts

    def _vmap_greedy_actions(self, obs, dir_mask, act_mask, n_active):
        """Deterministic action selection via vmap forward + masked argmax."""
        import torch.func as func

        LARGE_NEG = -1e8
        ma = self.multi_agent  # VmapPPO instance

        # In clone mode, fall back to single-network batched forward
        if ma.clone_mode:
            net = ma.networks[0]
            obs_batch = {k: v[:n_active] for k, v in obs.items()}
            dir_logits, act_logits, _ = net.forward(obs_batch)
        else:
            # Stack params and vmap forward
            params, buffers = ma._get_stacked_params()
            stacked_obs = {k: v[:n_active].unsqueeze(1) for k, v in obs.items()}

            def forward_single(params, buffers, obs_i):
                return func.functional_call(ma.base_network, (params, buffers), args=(obs_i,))

            batched_forward = func.vmap(forward_single, in_dims=(0, 0, 0))
            dir_logits, act_logits, _ = batched_forward(params, buffers, stacked_obs)
            dir_logits = dir_logits.squeeze(1)
            act_logits = act_logits.squeeze(1)

        if dir_mask is not None:
            dir_logits = dir_logits.masked_fill(~dir_mask[:n_active], LARGE_NEG)
        if act_mask is not None:
            act_logits = act_logits.masked_fill(~act_mask[:n_active], LARGE_NEG)

        dirs = dir_logits.argmax(dim=-1)
        acts = act_logits.argmax(dim=-1)

        # Pad inactive agents
        if n_active < self.config.n_agents:
            pad = self.config.n_agents - n_active
            dirs = torch.cat([dirs, torch.full((pad,), 4, device=self.device, dtype=dirs.dtype)])
            acts = torch.cat([acts, torch.zeros(pad, device=self.device, dtype=acts.dtype)])

        return dirs, acts

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
        # 1. Aggressive — attack % > threshold
        if act_pcts[1] > agg_thresh:
            return 'Aggressive'
        # 2. Cooperative — (coop + give) % > threshold
        if (act_pcts[4] + act_pcts[2]) > coop_thresh:
            return 'Cooperative'
        # 3. Forager — move % > threshold
        if act_pcts[0] > forage_thresh:
            return 'Forager'
        # 4. Passive — STAY direction % > threshold
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
        print("\n" + "=" * 64)
        print("                AGENT PERSONALITY PROFILES")
        print("=" * 64)

        for p in profiles:
            act_str = " | ".join(f"{k} {v*100:.0f}%" for k, v in p.action_pcts.items())
            print(f"\nAgent {p.agent_id}: {p.personality}")
            print(f"  Actions: {act_str}")
            print(f"  Avg Return: {p.avg_return:.1f}  |  Survival: {p.avg_survival:.0f}/{self.config.max_steps_per_episode} steps")
            print(f"  Damage Dealt: {p.avg_damage_dealt:.1f}  |  Food Given: {p.avg_food_given:.1f}  |  Coops: {p.avg_coop_count:.1f}")

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

            # Per-agent breakdown
            print(f"  {'Agent':<8}{'Type':<14}| {'Return':>8} | {'Alive':>6} | {'Dmg':>7} | {'Food':>6} | {'Coop':>5} | {'R_surv':>7} | {'R_res':>7} | {'R_soc':>7}")
            print(f"  {'-'*8}{'-'*14}|{'-'*10}|{'-'*8}|{'-'*9}|{'-'*8}|{'-'*7}|{'-'*9}|{'-'*9}|{'-'*9}")
            for er in ep_results:
                for idx, am in enumerate(er.agent_metrics):
                    aid = idx
                    pers = personality_map.get(aid, '?')
                    print(f"  {aid:<8}{pers:<14}| {am['total_return']:>8.1f} | {am['steps_alive']:>6} | {am.get('damage_dealt', 0):>7.1f} | {am.get('food_given', 0):>6.1f} | {am.get('coop_count', 0):>5.1f} | {am['reward_survival']:>7.1f} | {am['reward_resource']:>7.1f} | {am['reward_social']:>7.1f}")

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

            is_combat = 'battle' in scenario_name.lower() or 'team' in scenario_name.lower()

            print(f"\n  By personality:")
            print(f"  {'Personality':<16}| {'Avg Return':>10} | {'Avg Survival':>12} | {'Avg Damage':>10}")
            print(f"  {'-'*16}|{'-'*12}|{'-'*14}|{'-'*12}")
            for pers, metrics in sorted(by_personality.items()):
                n_samples = len(metrics['return']) or 1
                avg_ret = sum(metrics['return']) / n_samples
                avg_surv = sum(metrics['survival']) / n_samples
                avg_dmg = sum(metrics['damage']) / n_samples
                print(f"  {pers:<16}| {avg_ret:>10.1f} | {avg_surv:>12.1f} | {avg_dmg:>10.1f}")

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

        # Build profiles section
        profiles_data = []
        for p in profiles:
            profiles_data.append({
                'agent_id': p.agent_id,
                'personality': p.personality,
                'action_pcts': {k: round(v, 4) for k, v in p.action_pcts.items()},
                'direction_pcts': {k: round(v, 4) for k, v in p.direction_pcts.items()},
                'avg_return': round(p.avg_return, 2),
                'avg_survival': round(p.avg_survival, 1),
                'avg_damage_dealt': round(p.avg_damage_dealt, 2),
                'avg_food_given': round(p.avg_food_given, 2),
                'avg_coop_count': round(p.avg_coop_count, 2),
            })

        # Build scenario section
        personality_map = {p.agent_id: p.personality for p in profiles}
        scenarios_data = {}
        for scenario_name, ep_results in results.items():
            episodes_data = []
            for ep_idx, er in enumerate(ep_results):
                agents_data = []
                for idx, am in enumerate(er.agent_metrics):
                    agents_data.append({
                        'agent_id': idx,
                        'personality': personality_map.get(idx, 'Unknown'),
                        'return': round(am['total_return'], 2),
                        'survival': am['steps_alive'],
                        'damage_dealt': round(am.get('damage_dealt', 0.0), 2),
                        'food_given': round(am.get('food_given', 0.0), 2),
                        'coop_count': round(am.get('coop_count', 0.0), 2),
                        'reward_survival': round(am['reward_survival'], 2),
                        'reward_resource': round(am['reward_resource'], 2),
                        'reward_social': round(am['reward_social'], 2),
                    })
                episodes_data.append({'episode': ep_idx, 'agents': agents_data})

            # Personality summary for this scenario
            by_pers: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for er in ep_results:
                for idx, am in enumerate(er.agent_metrics):
                    pers = personality_map.get(idx, 'Unknown')
                    by_pers[pers]['avg_return'].append(am['total_return'])
                    by_pers[pers]['avg_survival'].append(am['steps_alive'])
                    by_pers[pers]['avg_damage'].append(am.get('damage_dealt', 0.0))

            summary = {}
            for pers, metrics in by_pers.items():
                n = len(metrics['avg_return']) or 1
                summary[pers] = {
                    'avg_return': round(sum(metrics['avg_return']) / n, 2),
                    'avg_survival': round(sum(metrics['avg_survival']) / n, 1),
                    'avg_damage': round(sum(metrics['avg_damage']) / n, 2),
                }

            scenarios_data[scenario_name] = {
                'episodes': episodes_data,
                'personality_summary': summary,
            }

        output = {
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
