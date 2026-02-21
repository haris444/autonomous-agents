"""
Cross-environment generalization test.

Takes checkpoints trained in one environment and evaluates them in different
environments to test robustness and generalization of learned behaviors.
"""
import argparse
import json
import os
import torch
import numpy as np
from collections import Counter
from core.config import Config
from env.environment import GridWorld
from analysis.utils import load_sac, load_auto, get_device


# Test environments: each overrides specific config values
TEST_ENVS = {
    'standard': {},
    'no_predator': {'n_predators': 0},
    '3_predators': {'n_predators': 3, 'predator_damage': 5.0},
    'no_rich_food': {'rich_food_spawn_rate': 0.0, 'r_large': 0.0},
    'scarce_food': {'poor_food_spawn_rate': 0.000168, 'rich_food_spawn_rate': 0.0042},
    'crowded_6x6': {'grid_size': 6},
}


def make_config(base_config, env_overrides):
    """Create a new config with environment overrides applied."""
    config = Config.from_dict(base_config.to_dict())
    for k, v in env_overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.__post_init__()
    return config


def eval_in_env(sac, config, env_overrides, n_episodes=50, device=None):
    """Run evaluation episodes in a modified environment."""
    if device is None:
        device = get_device()

    test_config = make_config(config, env_overrides)
    n = test_config.n_agents

    agent_stats = {i: {
        'returns': [], 'survived': [], 'action_counts': [],
    } for i in range(n)}
    episode_stats = []

    for ep in range(n_episodes):
        env = GridWorld(test_config, device=device)
        obs = env.reset()

        ac = {i: Counter() for i in range(n)}
        tr = {i: 0.0 for i in range(n)}

        for step in range(test_config.max_steps_per_episode):
            obs_batched = {k: v.unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                directions, action_types = sac.get_actions(obs_batched, deterministic=True)
            dirs = directions.squeeze(0)
            acts = action_types.squeeze(0)
            for i in range(n):
                ac[i][acts[i].item()] += 1
            obs, rewards, dones, info = env.step(dirs, acts)
            for i in range(n):
                tr[i] += rewards[i].item()
            if dones.all():
                break

        alive = env.agent_alive.cpu().numpy()
        ledger = env.ledger.tensor.cpu().numpy()

        for i in range(n):
            agent_stats[i]['returns'].append(tr[i])
            agent_stats[i]['survived'].append(bool(alive[i]))
            agent_stats[i]['action_counts'].append(dict(ac[i]))

        ep_return = np.mean([tr[i] for i in range(n)])
        ep_deaths = sum(1 for i in range(n) if not alive[i])
        ep_coops = ledger[:, :, 2].sum()
        episode_stats.append({
            'return': ep_return, 'deaths': ep_deaths,
            'coops': float(ep_coops), 'steps': step + 1,
        })

    # Aggregate
    results = {
        'avg_return': float(np.mean([s['return'] for s in episode_stats])),
        'std_return': float(np.std([s['return'] for s in episode_stats])),
        'avg_deaths': float(np.mean([s['deaths'] for s in episode_stats])),
        'avg_coops': float(np.mean([s['coops'] for s in episode_stats])),
        'avg_survival': float(np.mean([
            np.mean(agent_stats[i]['survived']) for i in range(n)
        ]) * 100),
    }

    # Per-agent action rates
    avg_atk, avg_coop, avg_give = [], [], []
    for i in range(n):
        atks, coops, gives = [], [], []
        for ac_dict in agent_stats[i]['action_counts']:
            t = max(sum(ac_dict.values()), 1)
            atks.append(ac_dict.get(1, 0) / t * 100)
            coops.append(ac_dict.get(4, 0) / t * 100)
            gives.append(ac_dict.get(2, 0) / t * 100)
        avg_atk.append(np.mean(atks))
        avg_coop.append(np.mean(coops))
        avg_give.append(np.mean(gives))

    results['avg_attack_pct'] = float(np.mean(avg_atk))
    results['avg_coop_pct'] = float(np.mean(avg_coop))
    results['avg_give_pct'] = float(np.mean(avg_give))

    return results


def main():
    parser = argparse.ArgumentParser(description='Cross-environment generalization test')
    parser.add_argument('checkpoints', nargs='+', help='Checkpoint paths to test')
    parser.add_argument('--episodes', '-n', type=int, default=50)
    parser.add_argument('--envs', nargs='*', default=None,
                        help='Test env names (default: all)')
    parser.add_argument('--output', '-o', default='results/cross_env_test',
                        help='Output directory')
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.output, exist_ok=True)

    env_names = args.envs if args.envs else list(TEST_ENVS.keys())
    all_results = {}

    for ckpt_path in args.checkpoints:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        # Include parent dir for uniqueness
        parent = os.path.basename(os.path.dirname(ckpt_path))
        ckpt_label = f'{parent}/{ckpt_name}'

        print(f'\n{"="*80}')
        print(f'CHECKPOINT: {ckpt_path}')
        print(f'{"="*80}')

        try:
            sac, config, ckpt = load_sac(ckpt_path, device)
        except Exception:
            agent, config, ckpt, agent_type = load_auto(ckpt_path, device)
            if agent_type == 'sac':
                sac = agent
            else:
                print(f'  Skipping {ckpt_path} — not SAC')
                continue

        ckpt_results = {}
        for env_name in env_names:
            overrides = TEST_ENVS[env_name]
            print(f'  Testing in: {env_name} ({args.episodes} episodes)...', end=' ', flush=True)

            try:
                results = eval_in_env(sac, config, overrides, args.episodes, device)
                ckpt_results[env_name] = results
                print(f'Return={results["avg_return"]:+.1f} Surv={results["avg_survival"]:.0f}% '
                      f'Coop={results["avg_coop_pct"]:.1f}% Atk={results["avg_attack_pct"]:.1f}%')
            except Exception as e:
                print(f'FAILED: {e}')
                ckpt_results[env_name] = {'error': str(e)}

        all_results[ckpt_label] = ckpt_results

    # Save raw JSON
    json_path = os.path.join(args.output, 'cross_env_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nRaw results saved to: {json_path}')

    # Print comparison table
    print(f'\n{"="*120}')
    print('CROSS-ENVIRONMENT RESULTS')
    print(f'{"="*120}')

    # Header
    header = f'{"Checkpoint":<35}'
    for env_name in env_names:
        header += f' | {env_name:>14}'
    print(header)
    print('-' * len(header))

    # Return row per checkpoint
    for ckpt_label, ckpt_results in all_results.items():
        row = f'{ckpt_label:<35}'
        for env_name in env_names:
            if env_name in ckpt_results and 'error' not in ckpt_results[env_name]:
                r = ckpt_results[env_name]['avg_return']
                row += f' | {r:>+13.1f}'
            else:
                row += f' | {"ERROR":>14}'
        print(row)

    # Survival row
    print()
    print('SURVIVAL %:')
    for ckpt_label, ckpt_results in all_results.items():
        row = f'{ckpt_label:<35}'
        for env_name in env_names:
            if env_name in ckpt_results and 'error' not in ckpt_results[env_name]:
                s = ckpt_results[env_name]['avg_survival']
                row += f' | {s:>13.0f}%'
            else:
                row += f' | {"ERROR":>14}'
        print(row)

    # Coop row
    print()
    print('COOPERATION %:')
    for ckpt_label, ckpt_results in all_results.items():
        row = f'{ckpt_label:<35}'
        for env_name in env_names:
            if env_name in ckpt_results and 'error' not in ckpt_results[env_name]:
                c = ckpt_results[env_name]['avg_coop_pct']
                row += f' | {c:>13.1f}%'
            else:
                row += f' | {"ERROR":>14}'
        print(row)

    print(f'\n{args.episodes} episodes per cell. All with greedy policy.')


if __name__ == '__main__':
    main()
