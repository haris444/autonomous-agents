"""Analyze replay .pt files to study agent personality evolution across training."""
import torch, numpy as np, glob, re
from collections import Counter

replay_files = sorted(glob.glob('results/runs/sac_phase11_lifesteal/replay_ep*.pt'),
                       key=lambda x: int(re.search(r'ep(\d+)', x).group(1)))

all_results = []
for path in replay_files:
    ep = int(re.search(r'ep(\d+)', path).group(1))
    data = torch.load(path, weights_only=False)
    steps = data['steps']
    n_agents = 8

    ac = {i: Counter() for i in range(n_agents)}
    tr = {i: 0.0 for i in range(n_agents)}
    for step in steps:
        for aid, (d, at) in step.actions.items():
            ac[aid][at] += 1
        if step.rewards:
            for aid, r in step.rewards.items():
                tr[aid] += r

    fa = steps[-1].alive
    fh = steps[-1].hp
    fl = data['ledger_snapshots'][-1]

    agent_stats = []
    for i in range(n_agents):
        t = max(sum(ac[i].values()), 1)
        agent_stats.append({
            'move_pct': ac[i][0]/t*100, 'atk_pct': ac[i][1]/t*100,
            'give_pct': ac[i][2]/t*100, 'sig_pct': ac[i][3]/t*100,
            'coop_pct': ac[i][4]/t*100, 'hp': fh[i], 'alive': bool(fa[i]),
            'dmg_dealt': fl[i,:,0].sum(), 'dmg_taken': fl[:,i,0].sum(),
            'food_given': fl[i,:,1].sum(), 'coops': fl[i,:,2].sum(),
            'return': tr[i]
        })

    avg_atk = np.mean([s['atk_pct'] for s in agent_stats])
    avg_coop = np.mean([s['coop_pct'] for s in agent_stats])
    avg_ret = np.mean([s['return'] for s in agent_stats])
    n_dead = sum(1 for s in agent_stats if not s['alive'])
    total_dmg = fl[:,:,0].sum()
    total_coops = fl[:,:,2].sum()
    total_food = fl[:,:,1].sum()

    surv_coop = [s['coop_pct'] for s in agent_stats if s['alive']]
    dead_coop = [s['coop_pct'] for s in agent_stats if not s['alive']]
    surv_atk = [s['atk_pct'] for s in agent_stats if s['alive']]
    dead_atk = [s['atk_pct'] for s in agent_stats if not s['alive']]

    all_results.append({
        'ep': ep, 'avg_ret': avg_ret, 'avg_atk': avg_atk, 'avg_coop': avg_coop,
        'n_dead': n_dead, 'total_dmg': total_dmg, 'total_coops': total_coops,
        'total_food': total_food, 'n_steps': len(steps), 'agent_stats': agent_stats,
        'surv_coop': np.mean(surv_coop) if surv_coop else 0,
        'dead_coop': np.mean(dead_coop) if dead_coop else 0,
        'surv_atk': np.mean(surv_atk) if surv_atk else 0,
        'dead_atk': np.mean(dead_atk) if dead_atk else 0,
    })

early = [r for r in all_results if r['ep'] <= 350]
mid = [r for r in all_results if 350 < r['ep'] <= 600]
late = [r for r in all_results if 600 < r['ep'] <= 1100]
final = [r for r in all_results if r['ep'] > 1100]

def phase_summary(name, phase):
    if not phase:
        return
    avg_ret = np.mean([r['avg_ret'] for r in phase])
    avg_atk = np.mean([r['avg_atk'] for r in phase])
    avg_coop = np.mean([r['avg_coop'] for r in phase])
    avg_dead = np.mean([r['n_dead'] for r in phase])
    avg_dmg = np.mean([r['total_dmg'] for r in phase])
    avg_coops = np.mean([r['total_coops'] for r in phase])
    avg_steps = np.mean([r['n_steps'] for r in phase])
    best_ret = max(r['avg_ret'] for r in phase)
    worst_ret = min(r['avg_ret'] for r in phase)
    surv_coop_vals = [r['surv_coop'] for r in phase if r['surv_coop'] > 0]
    dead_coop_vals = [r['dead_coop'] for r in phase if r['dead_coop'] > 0]
    surv_atk_vals = [r['surv_atk'] for r in phase if r['surv_atk'] > 0]
    dead_atk_vals = [r['dead_atk'] for r in phase if r['dead_atk'] > 0]
    surv_coop = np.mean(surv_coop_vals) if surv_coop_vals else 0
    dead_coop = np.mean(dead_coop_vals) if dead_coop_vals else 0
    surv_atk = np.mean(surv_atk_vals) if surv_atk_vals else 0
    dead_atk = np.mean(dead_atk_vals) if dead_atk_vals else 0
    print(name)
    print("  Episodes: %d-%d (%d replays)" % (phase[0]['ep'], phase[-1]['ep'], len(phase)))
    print("  Avg Return: %+.1f  (best: %+.1f, worst: %+.1f)" % (avg_ret, best_ret, worst_ret))
    print("  Avg Atk%%: %.1f%%  |  Avg Coop%%: %.1f%%" % (avg_atk, avg_coop))
    print("  Avg Deaths: %.1f/8  |  Avg Steps: %.0f/128" % (avg_dead, avg_steps))
    print("  Avg Damage: %.0f  |  Avg Coops: %.0f" % (avg_dmg, avg_coops))
    print("  Survivors -> Avg Coop%%=%.1f%% Atk%%=%.1f%%" % (surv_coop, surv_atk))
    print("  Dead agents -> Avg Coop%%=%.1f%% Atk%%=%.1f%%" % (dead_coop, dead_atk))
    print()

print("PHASE SUMMARIES")
print("=" * 60)
phase_summary("PHASE 1: Golden Age", early)
phase_summary("PHASE 2: Transition / Arms Race", mid)
phase_summary("PHASE 3: Dark Age", late)
phase_summary("PHASE 4: Noisy Equilibrium", final)

# Survivor vs dead
print("=" * 60)
print("SURVIVOR vs DEAD ANALYSIS (all %d episodes)" % len(all_results))
print("=" * 60)
all_surv_coop = [r['surv_coop'] for r in all_results if r['surv_coop'] > 0]
all_dead_coop = [r['dead_coop'] for r in all_results if r['dead_coop'] > 0]
all_surv_atk = [r['surv_atk'] for r in all_results if r['surv_atk'] > 0]
all_dead_atk = [r['dead_atk'] for r in all_results if r['dead_atk'] > 0]
print("  Survivors:   Avg Coop%%=%.1f%%  Avg Atk%%=%.1f%%" % (np.mean(all_surv_coop), np.mean(all_surv_atk)))
print("  Dead agents: Avg Coop%%=%.1f%%  Avg Atk%%=%.1f%%" % (np.mean(all_dead_coop), np.mean(all_dead_atk)))
print()

# Correlations
print("RETURN vs BEHAVIOR CORRELATION (per-agent, all episodes)")
print("=" * 60)
all_agent_rets, all_agent_atks, all_agent_coops, all_agent_moves = [], [], [], []
for r in all_results:
    for s in r['agent_stats']:
        all_agent_rets.append(s['return'])
        all_agent_atks.append(s['atk_pct'])
        all_agent_coops.append(s['coop_pct'])
        all_agent_moves.append(s['move_pct'])

arr = lambda x: np.array(x)
corr_atk = np.corrcoef(arr(all_agent_atks), arr(all_agent_rets))[0,1]
corr_coop = np.corrcoef(arr(all_agent_coops), arr(all_agent_rets))[0,1]
corr_move = np.corrcoef(arr(all_agent_moves), arr(all_agent_rets))[0,1]
print("  Correlation(Attack%%, Return)  = %.3f" % corr_atk)
print("  Correlation(Coop%%, Return)    = %.3f" % corr_coop)
print("  Correlation(Move%%, Return)    = %.3f" % corr_move)
print()

# Top/bottom performers
print("TOP 10 INDIVIDUAL AGENT PERFORMANCES")
print("=" * 60)
all_perf = []
for r in all_results:
    for i, s in enumerate(r['agent_stats']):
        all_perf.append((r['ep'], i, s))

top10 = sorted(all_perf, key=lambda x: x[2]['return'], reverse=True)[:10]
bot10 = sorted(all_perf, key=lambda x: x[2]['return'])[:10]

print("  Best:")
for ep, aid, s in top10:
    print("    ep%d A%d: Ret=%+.0f | Atk=%.0f%% Coop=%.0f%% Move=%.0f%% | HP=%.0f %s | Dmg=%.0f Coops=%.0f" % (
        ep, aid, s['return'], s['atk_pct'], s['coop_pct'], s['move_pct'], s['hp'],
        "ALIVE" if s['alive'] else "DEAD", s['dmg_dealt'], s['coops']))
print("  Worst:")
for ep, aid, s in bot10:
    print("    ep%d A%d: Ret=%+.0f | Atk=%.0f%% Coop=%.0f%% Move=%.0f%% | HP=%.0f %s | Dmg=%.0f Coops=%.0f" % (
        ep, aid, s['return'], s['atk_pct'], s['coop_pct'], s['move_pct'], s['hp'],
        "ALIVE" if s['alive'] else "DEAD", s['dmg_dealt'], s['coops']))
print()

# Per-agent identity
print("PER-AGENT IDENTITY CONSISTENCY")
print("=" * 60)
for aid in range(8):
    atks = [r['agent_stats'][aid]['atk_pct'] for r in all_results]
    coops = [r['agent_stats'][aid]['coop_pct'] for r in all_results]
    rets = [r['agent_stats'][aid]['return'] for r in all_results]
    surv_rate = sum(1 for r in all_results if r['agent_stats'][aid]['alive']) / len(all_results) * 100
    print("  Agent %d: Atk=%.1f%%+/-%.1f | Coop=%.1f%%+/-%.1f | Ret=%+.0f+/-%.0f | Survived=%.0f%%" % (
        aid, np.mean(atks), np.std(atks), np.mean(coops), np.std(coops),
        np.mean(rets), np.std(rets), surv_rate))
print()

# Personality distribution
print("PERSONALITY DISTRIBUTION OVER TIME")
print("=" * 60)
for name, phase in [("PHASE 1", early), ("PHASE 2", mid), ("PHASE 3", late), ("PHASE 4", final)]:
    types = Counter()
    for r in phase:
        for s in r['agent_stats']:
            if s['atk_pct'] > 30: types['AGGRESSIVE'] += 1
            elif s['coop_pct'] > 30: types['COOPERATIVE'] += 1
            elif s['move_pct'] > 50: types['FORAGER'] += 1
            elif s['give_pct'] > 15: types['ALTRUIST'] += 1
            elif s['atk_pct'] > 15: types['Fighter'] += 1
            elif s['coop_pct'] > 15: types['Social'] += 1
            else: types['Balanced'] += 1
    total = sum(types.values())
    print("  %s (%d agents across %d eps):" % (name, total, len(phase)))
    for t in ['COOPERATIVE', 'AGGRESSIVE', 'FORAGER', 'ALTRUIST', 'Fighter', 'Social', 'Balanced']:
        if types[t] > 0:
            bar = "#" * int(types[t] / total * 40)
            print("    %-12s: %3d (%5.1f%%) %s" % (t, types[t], types[t]/total*100, bar))
    print()
