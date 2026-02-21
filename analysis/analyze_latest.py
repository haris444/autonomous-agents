"""Run 10 eval episodes with the latest SAC checkpoint and analyze agent personalities."""
import argparse
import torch
import numpy as np
from collections import Counter
from env.environment import GridWorld
from analysis.utils import load_sac

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate SAC checkpoint and analyze agent personalities.')
parser.add_argument('--checkpoint', default='results/runs/sac_phase11_lifesteal/checkpoint_ep16200.pt',
                    help='Path to SAC checkpoint (default: %(default)s)')
args = parser.parse_args()

# Load checkpoint using utility
from analysis.utils import get_device
device = get_device()
sac, config, ckpt = load_sac(args.checkpoint, device)
print(f'Loaded: {args.checkpoint}')
print(f'Config: grid={config.grid_size} agents={config.n_agents} predators={config.n_predators} lifesteal={config.lifesteal_fraction:.2f}')

n_episodes = 10
n_agents = config.n_agents
all_ep_stats = []

for ep in range(n_episodes):
    env = GridWorld(config, device=device)
    obs_dict = env.reset()

    ac = {i: Counter() for i in range(n_agents)}
    tr = {i: 0.0 for i in range(n_agents)}

    for step_i in range(config.max_steps_per_episode):
        # Add env batch dim: [n_agents, ...] -> [1, n_agents, ...]
        obs_batched = {k: v.unsqueeze(0) for k, v in obs_dict.items()}

        with torch.no_grad():
            directions, action_types = sac.get_actions(obs_batched, deterministic=True)

        # Remove env batch dim: [1, n_agents] -> [n_agents]
        dirs = directions.squeeze(0)
        acts = action_types.squeeze(0)

        for i in range(n_agents):
            ac[i][acts[i].item()] += 1

        obs_dict, rewards, dones, info = env.step(dirs, acts)

        for i in range(n_agents):
            tr[i] += rewards[i].item()

        if dones.all():
            break

    fa = env.agent_alive.cpu().numpy()
    fh = env.agent_hp.cpu().numpy()
    fl = env.ledger.tensor.cpu().numpy()
    n_steps = step_i + 1

    avg_atk = np.mean([ac[i][1] / max(sum(ac[i].values()), 1) for i in range(n_agents)]) * 100
    avg_coop = np.mean([ac[i][4] / max(sum(ac[i].values()), 1) for i in range(n_agents)]) * 100
    avg_ret = np.mean([tr[i] for i in range(n_agents)])
    n_dead = sum(1 for i in range(n_agents) if not fa[i])
    total_dmg = fl[:, :, 0].sum()
    total_coops = fl[:, :, 2].sum()

    all_ep_stats.append({
        'ac': ac, 'tr': tr, 'fa': fa, 'fh': fh, 'fl': fl,
        'avg_atk': avg_atk, 'avg_coop': avg_coop, 'avg_ret': avg_ret,
        'n_dead': n_dead, 'total_dmg': total_dmg, 'total_coops': total_coops,
        'n_steps': n_steps
    })

# Print per-episode summaries
print()
print('EVALUATION: 10 episodes with checkpoint ep16200 (greedy policy)')
print('=' * 80)
print('%4s | %7s | %5s %5s | %4s | %5s %5s | %5s' % (
    'Ep', 'Return', 'Atk%', 'Coop%', 'Dead', 'Dmg', 'Coops', 'Steps'))
print('-' * 80)
for i, s in enumerate(all_ep_stats):
    print('  %2d | %+7.1f | %4.1f%% %4.1f%% | %4d | %5.0f %5.0f | %5d' % (
        i, s['avg_ret'], s['avg_atk'], s['avg_coop'], s['n_dead'],
        s['total_dmg'], s['total_coops'], s['n_steps']))

avg_ret = np.mean([s['avg_ret'] for s in all_ep_stats])
avg_atk = np.mean([s['avg_atk'] for s in all_ep_stats])
avg_coop = np.mean([s['avg_coop'] for s in all_ep_stats])
avg_dead = np.mean([s['n_dead'] for s in all_ep_stats])
avg_dmg = np.mean([s['total_dmg'] for s in all_ep_stats])
avg_coops = np.mean([s['total_coops'] for s in all_ep_stats])
avg_steps = np.mean([s['n_steps'] for s in all_ep_stats])
print('-' * 80)
print('  AVG | %+7.1f | %4.1f%% %4.1f%% | %4.1f | %5.0f %5.0f | %5.0f' % (
    avg_ret, avg_atk, avg_coop, avg_dead, avg_dmg, avg_coops, avg_steps))

# Detailed per-agent profile
print()
print('PER-AGENT PROFILES (averaged across 10 episodes)')
print('=' * 80)
print('%6s | %5s %5s %5s %5s %5s | %5s %7s | Type' % (
    'Agent', 'Move%', 'Atk%', 'Give%', 'Sig%', 'Coop%', 'Surv%', 'AvgRet'))
print('-' * 80)

for aid in range(n_agents):
    move_pcts, atk_pcts, give_pcts, sig_pcts, coop_pcts = [], [], [], [], []
    rets = []
    surv = 0
    for s in all_ep_stats:
        c = s['ac'][aid]
        t = max(sum(c.values()), 1)
        move_pcts.append(c[0] / t * 100)
        atk_pcts.append(c[1] / t * 100)
        give_pcts.append(c[2] / t * 100)
        sig_pcts.append(c[3] / t * 100)
        coop_pcts.append(c[4] / t * 100)
        rets.append(s['tr'][aid])
        if s['fa'][aid]:
            surv += 1

    m = np.mean(move_pcts)
    a = np.mean(atk_pcts)
    g = np.mean(give_pcts)
    sg = np.mean(sig_pcts)
    co = np.mean(coop_pcts)
    sr = surv / len(all_ep_stats) * 100
    ar = np.mean(rets)

    if a > 30:
        ptype = 'AGGRESSIVE'
    elif co > 30:
        ptype = 'COOPERATIVE'
    elif m > 50:
        ptype = 'FORAGER'
    elif g > 15:
        ptype = 'ALTRUIST'
    elif a > 15:
        ptype = 'Fighter'
    elif co > 15:
        ptype = 'Social'
    else:
        ptype = 'Balanced'

    print('  A%3d | %4.1f%% %4.1f%% %4.1f%% %4.1f%% %4.1f%% | %4.0f%% %+7.1f | %s' % (
        aid, m, a, g, sg, co, sr, ar, ptype))

print()
print('COMPARISON TO EARLIER PHASES:')
print('  Phase 1 (ep259-348):   Return=%+.1f | Atk=%.1f%% | Coop=%.1f%% | Deaths=%.1f' % (66.1, 13.2, 33.6, 4.4))
print('  Phase 4 (ep1106-1709): Return=%+.1f | Atk=%.1f%% | Coop=%.1f%% | Deaths=%.1f' % (-18.0, 17.8, 23.8, 6.5))
print('  NOW (ep16200):         Return=%+.1f | Atk=%.1f%% | Coop=%.1f%% | Deaths=%.1f' % (
    avg_ret, avg_atk, avg_coop, avg_dead))
