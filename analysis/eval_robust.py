"""
Robust per-agent behavioral evaluation over many episodes.

Runs N eval episodes with greedy policy and computes detailed per-agent stats
with confidence intervals.
"""
import argparse
import torch
import numpy as np
from collections import Counter
from env.environment import GridWorld
from analysis.utils import load_sac, load_auto, get_device


def run_eval(checkpoint_path, n_episodes=50):
    device = get_device()

    # Try loading as SAC first, then auto-detect
    try:
        sac, config, ckpt = load_sac(checkpoint_path, device)
        agent_type = 'sac'
    except Exception:
        agent, config, ckpt, agent_type = load_auto(checkpoint_path, device)
        if agent_type == 'sac':
            sac = agent
        else:
            raise ValueError("Only SAC supported for now")

    n = config.n_agents
    print(f'Loaded: {checkpoint_path} ({agent_type}, {n} agents)')
    print(f'Running {n_episodes} eval episodes (greedy policy)...')

    # Per-agent accumulators
    agent_stats = {i: {
        'returns': [],
        'survived': [],
        'final_hp': [],
        'action_counts': [],       # list of Counters
        'damage_dealt': [],        # total damage dealt per episode
        'damage_taken': [],        # total damage received per episode
        'food_given': [],          # total food given per episode
        'coop_count': [],          # total cooperations per episode
        'kills': [],               # agents killed per episode
    } for i in range(n)}

    # Episode-level stats
    episode_stats = []

    for ep in range(n_episodes):
        env = GridWorld(config, device=device)
        obs = env.reset()

        ac = {i: Counter() for i in range(n)}
        tr = {i: 0.0 for i in range(n)}

        for step in range(config.max_steps_per_episode):
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

        # Collect final state
        alive = env.agent_alive.cpu().numpy()
        hp = env.agent_hp.cpu().numpy()
        ledger = env.ledger.tensor.cpu().numpy()  # [n, n, 4]

        for i in range(n):
            total_steps = max(sum(ac[i].values()), 1)
            agent_stats[i]['returns'].append(tr[i])
            agent_stats[i]['survived'].append(bool(alive[i]))
            agent_stats[i]['final_hp'].append(float(hp[i]))
            agent_stats[i]['action_counts'].append(dict(ac[i]))

            # Ledger: [i, j, ch] — ch0=damage, ch1=food_given, ch2=coop, ch3=defense
            agent_stats[i]['damage_dealt'].append(float(ledger[i, :, 0].sum()))
            agent_stats[i]['damage_taken'].append(float(ledger[:, i, 0].sum()))
            agent_stats[i]['food_given'].append(float(ledger[i, :, 1].sum()))
            agent_stats[i]['coop_count'].append(float(ledger[i, :, 2].sum()))

        # Episode-level
        ep_return = np.mean([tr[i] for i in range(n)])
        ep_deaths = sum(1 for i in range(n) if not alive[i])
        ep_dmg = ledger[:, :, 0].sum()
        ep_coops = ledger[:, :, 2].sum()
        episode_stats.append({
            'return': ep_return, 'deaths': ep_deaths,
            'damage': ep_dmg, 'coops': ep_coops,
            'steps': step + 1,
        })

        if (ep + 1) % 10 == 0:
            print(f'  Episode {ep+1}/{n_episodes} done (avg return so far: {np.mean([s["return"] for s in episode_stats]):+.1f})')

    # ================================================================
    # RESULTS
    # ================================================================
    print()
    print('=' * 110)
    print(f'ROBUST EVALUATION: {n_episodes} episodes, greedy policy')
    print(f'Checkpoint: {checkpoint_path}')
    print('=' * 110)

    # Episode-level summary
    returns = [s['return'] for s in episode_stats]
    deaths = [s['deaths'] for s in episode_stats]
    dmgs = [s['damage'] for s in episode_stats]
    coops = [s['coops'] for s in episode_stats]

    print(f'\nEPISODE AVERAGES (+/- std):')
    print(f'  Return:      {np.mean(returns):+8.1f} +/- {np.std(returns):5.1f}')
    print(f'  Deaths/ep:   {np.mean(deaths):8.2f} +/- {np.std(deaths):5.2f}')
    print(f'  Damage/ep:   {np.mean(dmgs):8.1f} +/- {np.std(dmgs):5.1f}')
    print(f'  Coops/ep:    {np.mean(coops):8.1f} +/- {np.std(coops):5.1f}')

    # Per-agent table
    print()
    print('=' * 110)
    print('PER-AGENT PROFILES (averaged over %d episodes, +/- std)' % n_episodes)
    print('=' * 110)
    print()
    print(f'{"Agent":<6} | {"Return":>14} | {"Surv%":>5} | {"Move%":>8} {"Atk%":>8} {"Give%":>8} {"Sig%":>8} {"Coop%":>8}'
          f' | {"DmgOut":>8} {"DmgIn":>8} {"FoodGvn":>8}')
    print('-' * 110)

    agent_summaries = []

    for i in range(n):
        s = agent_stats[i]
        ret_mean = np.mean(s['returns'])
        ret_std = np.std(s['returns'])
        surv = np.mean(s['survived']) * 100

        # Action percentages
        move_pcts, atk_pcts, give_pcts, sig_pcts, coop_pcts = [], [], [], [], []
        for ac_dict in s['action_counts']:
            t = max(sum(ac_dict.values()), 1)
            move_pcts.append(ac_dict.get(0, 0) / t * 100)
            atk_pcts.append(ac_dict.get(1, 0) / t * 100)
            give_pcts.append(ac_dict.get(2, 0) / t * 100)
            sig_pcts.append(ac_dict.get(3, 0) / t * 100)
            coop_pcts.append(ac_dict.get(4, 0) / t * 100)

        m, a, g, sg, co = np.mean(move_pcts), np.mean(atk_pcts), np.mean(give_pcts), np.mean(sig_pcts), np.mean(coop_pcts)
        m_s, a_s, g_s, sg_s, co_s = np.std(move_pcts), np.std(atk_pcts), np.std(give_pcts), np.std(sig_pcts), np.std(coop_pcts)

        dmg_out = np.mean(s['damage_dealt'])
        dmg_in = np.mean(s['damage_taken'])
        food_gvn = np.mean(s['food_given'])

        print(f'  A{i:<3d} | {ret_mean:+7.1f}+/-{ret_std:4.1f} | {surv:4.0f}%'
              f' | {m:5.1f}+/-{m_s:3.1f} {a:5.1f}+/-{a_s:3.1f} {g:5.1f}+/-{g_s:3.1f}'
              f' {sg:5.1f}+/-{sg_s:3.1f} {co:5.1f}+/-{co_s:3.1f}'
              f' | {dmg_out:7.1f}  {dmg_in:7.1f}  {food_gvn:7.1f}')

        agent_summaries.append({
            'id': i, 'return': ret_mean, 'return_std': ret_std,
            'survival': surv, 'move': m, 'attack': a, 'give': g,
            'signal': sg, 'coop': co, 'dmg_out': dmg_out, 'dmg_in': dmg_in,
            'food_given': food_gvn,
        })

    # Personality classification
    print()
    print('=' * 110)
    print('PERSONALITY CLASSIFICATION')
    print('=' * 110)
    print()

    for s in agent_summaries:
        traits = []
        if s['attack'] > 10: traits.append('AGGRESSIVE')
        if s['coop'] > 30: traits.append('COOPERATIVE')
        if s['give'] > 5: traits.append('ALTRUIST')
        if s['survival'] == 100: traits.append('SURVIVOR')
        if s['survival'] < 80: traits.append('FRAGILE')
        if s['move'] > 70: traits.append('EXPLORER')
        if s['return'] > np.mean([x['return'] for x in agent_summaries]) + np.std([x['return'] for x in agent_summaries]):
            traits.append('TOP-PERFORMER')
        if s['return'] < np.mean([x['return'] for x in agent_summaries]) - np.std([x['return'] for x in agent_summaries]):
            traits.append('UNDERPERFORMER')

        if not traits:
            traits.append('BALANCED')

        print(f'  A{s["id"]}: {", ".join(traits)}')
        print(f'       Return={s["return"]:+.1f} Surv={s["survival"]:.0f}% Atk={s["attack"]:.1f}% Coop={s["coop"]:.1f}% Give={s["give"]:.1f}%')

    # Inter-agent variance
    print()
    print('=' * 110)
    print('BEHAVIORAL DIVERSITY (variance across agents)')
    print('=' * 110)
    print()

    for metric in ['return', 'attack', 'coop', 'give', 'survival']:
        vals = [s[metric] for s in agent_summaries]
        print(f'  {metric:<10}: mean={np.mean(vals):.2f}  std={np.std(vals):.2f}  min={np.min(vals):.2f}  max={np.max(vals):.2f}  range={np.max(vals)-np.min(vals):.2f}')

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--episodes', '-n', type=int, default=50)
    args = parser.parse_args()
    run_eval(args.checkpoint, args.episodes)
