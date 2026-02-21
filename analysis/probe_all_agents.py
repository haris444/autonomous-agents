"""
Per-agent personality probe: test all 8 agent heads against social scenarios.

Runs diagnose_social + diagnose_friend_foe for every agent head, not just agent 0.
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from core.config import Config
from agents.network import SharedTrunkActorCritic
from analysis.utils import load_network, get_device


def fourier_encode(dx, dy, device):
    bands = [1.0, 2.0, 4.0, 8.0]
    features = []
    for freq in bands:
        features.extend([
            torch.sin(torch.tensor(freq * torch.pi * dx, device=device)),
            torch.cos(torch.tensor(freq * torch.pi * dx, device=device)),
            torch.sin(torch.tensor(freq * torch.pi * dy, device=device)),
            torch.cos(torch.tensor(freq * torch.pi * dy, device=device)),
        ])
    return torch.stack(features)


def create_obs(config, social_profile, device):
    """Observation: self at origin, other agent 1 cell to the right with given social history."""
    entity_tokens = torch.zeros(config.max_entities, config.entity_token_dim, device=device)
    entity_mask = torch.zeros(config.max_entities, device=device, dtype=torch.bool)

    # Self at origin
    entity_tokens[0, 0:16] = fourier_encode(0.0, 0.0, device)
    entity_tokens[0, 18:20] = torch.tensor([0.0, 1.0], device=device)
    entity_tokens[0, 20] = 1.0
    entity_mask[0] = True

    # Other agent 1 cell right
    entity_tokens[1, 0:16] = fourier_encode(0.0, 1.0 / config.grid_size, device)
    entity_tokens[1, 18:20] = torch.tensor([0.0, 1.0], device=device)
    entity_tokens[1, 20] = 1.0
    entity_tokens[1, 21] = social_profile.get('damage_dealt', 0.0) * 5.0
    entity_tokens[1, 22] = social_profile.get('food_given', 0.0) * 5.0
    entity_tokens[1, 23] = social_profile.get('coop_count', 0.0) * 5.0
    entity_tokens[1, 24] = social_profile.get('defense_score', 0.0) * 5.0
    entity_mask[1] = True

    return {
        'entity_tokens': entity_tokens.unsqueeze(0),
        'entity_mask': entity_mask.unsqueeze(0),
        'signals': torch.zeros(1, config.n_agents, device=device),
        'self_hp': torch.ones(1, 1, device=device),
        'self_inventory': torch.zeros(1, 1, device=device),
        'agent_id': torch.tensor([0], device=device),
    }


SCENARIOS = {
    'neutral':      {'damage_dealt': 0.0, 'food_given': 0.0, 'coop_count': 0.0, 'defense_score': 0.0},
    'enemy':        {'damage_dealt': 0.8, 'food_given': 0.0, 'coop_count': 0.0, 'defense_score': 0.0},
    'ally_defend':  {'damage_dealt': 0.0, 'food_given': 0.0, 'coop_count': 0.0, 'defense_score': 0.8},
    'ally_fed':     {'damage_dealt': 0.0, 'food_given': 0.8, 'coop_count': 0.0, 'defense_score': 0.0},
    'ally_coop':    {'damage_dealt': 0.0, 'food_given': 0.0, 'coop_count': 0.8, 'defense_score': 0.0},
    'strong_ally':  {'damage_dealt': 0.0, 'food_given': 0.5, 'coop_count': 0.5, 'defense_score': 0.5},
    'strong_enemy': {'damage_dealt': 1.0, 'food_given': 0.0, 'coop_count': 0.0, 'defense_score': 0.0},
}

ALLY_PROFILE = {'damage_dealt': 0.0, 'food_given': 0.5, 'coop_count': 0.5, 'defense_score': 0.3}
ENEMY_PROFILE = {'damage_dealt': 0.55, 'food_given': 0.0, 'coop_count': 0.05, 'defense_score': 0.0}

ACT_NAMES = ['MOVE/ATK', 'GIVE', 'SIGNAL', 'COOP', 'IDLE']


def probe_agent(model, config, device, agent_idx):
    """Run all scenarios through one agent head. Returns dict of results."""
    results = {}
    with torch.no_grad():
        for name, social in SCENARIOS.items():
            obs = create_obs(config, social, device)
            hidden = model.encode(obs)
            dir_logits, act_logits, value = model.apply_heads(hidden, agent_idx)
            dir_probs = F.softmax(dir_logits, dim=-1).squeeze().cpu().numpy()
            act_probs = F.softmax(act_logits, dim=-1).squeeze().cpu().numpy()
            results[name] = {
                'act_probs': act_probs,
                'dir_probs': dir_probs,
                'value': value.item(),
                'features': hidden.squeeze().cpu(),
            }

        # Ally vs enemy comparison
        for label, profile in [('ally', ALLY_PROFILE), ('foe', ENEMY_PROFILE)]:
            obs = create_obs(config, profile, device)
            hidden = model.encode(obs)
            dir_logits, act_logits, value = model.apply_heads(hidden, agent_idx)
            act_probs = F.softmax(act_logits, dim=-1).squeeze().cpu().numpy()
            results[f'_ff_{label}'] = {
                'act_probs': act_probs,
                'value': value.item(),
                'features': hidden.squeeze().cpu(),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='Probe all agent heads against social scenarios')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    args = parser.parse_args()

    device = get_device()
    model, config, ckpt = load_network(args.checkpoint, device)
    n = config.n_agents
    print(f'Loaded: {args.checkpoint} ({n} agents)')

    # Probe all agents
    all_results = {}
    for aid in range(n):
        all_results[aid] = probe_agent(model, config, device, aid)

    # ============================================================
    # TABLE 1: Action type per scenario per agent
    # ============================================================
    print()
    print('=' * 100)
    print('TABLE 1: DOMINANT ACTION TYPE PER SCENARIO (all 8 agents)')
    print('=' * 100)
    print()
    print(f'{"Scenario":<14}', end='')
    for aid in range(n):
        print(f' | A{aid:d} action  (%)  ', end='')
    print()
    print('-' * (14 + n * 20))

    for sname in SCENARIOS:
        print(f'{sname:<14}', end='')
        for aid in range(n):
            ap = all_results[aid][sname]['act_probs']
            best = int(np.argmax(ap))
            print(f' | {ACT_NAMES[best]:<7} {ap[best]:5.1%}', end='')
        print()

    # ============================================================
    # TABLE 2: Attack probability per scenario per agent
    # ============================================================
    print()
    print('=' * 100)
    print('TABLE 2: ATTACK PROBABILITY PER SCENARIO')
    print('=' * 100)
    print()
    print(f'{"Scenario":<14}', end='')
    for aid in range(n):
        print(f' |   A{aid:d}  ', end='')
    print()
    print('-' * (14 + n * 9))

    for sname in SCENARIOS:
        print(f'{sname:<14}', end='')
        for aid in range(n):
            ap = all_results[aid][sname]['act_probs'][0]  # ATTACK = index 0
            print(f' | {ap:5.1%}', end='')
        print()

    # ============================================================
    # TABLE 3: Cooperation probability per scenario per agent
    # ============================================================
    print()
    print('=' * 100)
    print('TABLE 3: COOPERATION PROBABILITY PER SCENARIO')
    print('=' * 100)
    print()
    print(f'{"Scenario":<14}', end='')
    for aid in range(n):
        print(f' |   A{aid:d}  ', end='')
    print()
    print('-' * (14 + n * 9))

    for sname in SCENARIOS:
        print(f'{sname:<14}', end='')
        for aid in range(n):
            cp = all_results[aid][sname]['act_probs'][3]  # COOP = index 3
            print(f' | {cp:5.1%}', end='')
        print()

    # ============================================================
    # TABLE 4: Value estimates per scenario per agent
    # ============================================================
    print()
    print('=' * 100)
    print('TABLE 4: VALUE ESTIMATES PER SCENARIO')
    print('=' * 100)
    print()
    print(f'{"Scenario":<14}', end='')
    for aid in range(n):
        print(f' |    A{aid:d}  ', end='')
    print()
    print('-' * (14 + n * 10))

    for sname in SCENARIOS:
        print(f'{sname:<14}', end='')
        for aid in range(n):
            v = all_results[aid][sname]['value']
            print(f' | {v:+6.2f}', end='')
        print()

    # ============================================================
    # TABLE 5: Friend/Foe discrimination per agent
    # ============================================================
    print()
    print('=' * 100)
    print('TABLE 5: FRIEND vs FOE DISCRIMINATION (per agent)')
    print('=' * 100)
    print()
    print(f'{"Agent":<6} | {"V(ally)":>8} {"V(foe)":>8} {"V_diff":>8}'
          f' | {"Atk_ally":>8} {"Atk_foe":>8} {"Atk_diff":>9}'
          f' | {"Coop_ally":>9} {"Coop_foe":>9} {"Coop_diff":>10}'
          f' | {"CosSim":>7}')
    print('-' * 120)

    for aid in range(n):
        ra = all_results[aid]['_ff_ally']
        re = all_results[aid]['_ff_foe']

        va, ve = ra['value'], re['value']
        atk_a, atk_e = ra['act_probs'][0], re['act_probs'][0]
        coop_a, coop_e = ra['act_probs'][3], re['act_probs'][3]

        fa = F.normalize(ra['features'].unsqueeze(0), dim=1)
        fe = F.normalize(re['features'].unsqueeze(0), dim=1)
        cos = (fa @ fe.T).item()

        verdict = ''
        if atk_e > atk_a + 0.05 or coop_a > coop_e + 0.05:
            verdict = ' << DISCRIMINATES'
        elif va > ve + 0.5:
            verdict = ' << values differ'

        print(f'  A{aid:<3d} | {va:+8.2f} {ve:+8.2f} {va-ve:+8.2f}'
              f' | {atk_a:8.1%} {atk_e:8.1%} {atk_e-atk_a:+9.1%}'
              f' | {coop_a:9.1%} {coop_e:9.1%} {coop_a-coop_e:+10.1%}'
              f' | {cos:7.4f}{verdict}')

    # ============================================================
    # TABLE 6: Per-agent personality summary
    # ============================================================
    print()
    print('=' * 100)
    print('TABLE 6: AGENT PERSONALITY SUMMARY')
    print('=' * 100)
    print()
    print(f'{"Agent":<6} | {"Neutral":>9} | {"vs Enemy":>9} | {"vs Ally":>9}'
          f' | {"V(neutral)":>10} | {"V_range":>8} | Personality')
    print('-' * 90)

    for aid in range(n):
        rn = all_results[aid]['neutral']
        re = all_results[aid]['strong_enemy']
        ra = all_results[aid]['strong_ally']

        # Dominant action in neutral
        neutral_best = ACT_NAMES[int(np.argmax(rn['act_probs']))]
        enemy_best = ACT_NAMES[int(np.argmax(re['act_probs']))]
        ally_best = ACT_NAMES[int(np.argmax(ra['act_probs']))]

        v_neutral = rn['value']

        # Value range across all scenarios
        all_vals = [all_results[aid][s]['value'] for s in SCENARIOS]
        v_range = max(all_vals) - min(all_vals)

        # Classify personality
        atk_neutral = rn['act_probs'][0]
        coop_neutral = rn['act_probs'][3]
        give_neutral = rn['act_probs'][1]

        # Check discrimination
        ff_ally = all_results[aid]['_ff_ally']
        ff_foe = all_results[aid]['_ff_foe']
        discriminates = (ff_foe['act_probs'][0] - ff_ally['act_probs'][0]) > 0.05 or \
                       (ff_ally['act_probs'][3] - ff_foe['act_probs'][3]) > 0.05

        if discriminates:
            ptype = 'DISCRIMINATOR'
        elif atk_neutral > 0.5:
            ptype = 'Aggressive'
        elif coop_neutral > 0.3:
            ptype = 'Cooperative'
        elif give_neutral > 0.1:
            ptype = 'Altruist'
        elif atk_neutral < 0.05 and coop_neutral > 0.15:
            ptype = 'Peaceful'
        else:
            ptype = 'Undifferentiated'

        print(f'  A{aid:<3d} | {neutral_best:>9} | {enemy_best:>9} | {ally_best:>9}'
              f' | {v_neutral:+10.2f} | {v_range:8.3f} | {ptype}')

    # ============================================================
    # OVERALL VERDICT
    # ============================================================
    print()
    print('=' * 100)
    print('OVERALL VERDICT')
    print('=' * 100)

    n_discriminators = 0
    n_value_diff = 0
    for aid in range(n):
        ra = all_results[aid]['_ff_ally']
        re = all_results[aid]['_ff_foe']
        if (re['act_probs'][0] - ra['act_probs'][0]) > 0.05 or \
           (ra['act_probs'][3] - re['act_probs'][3]) > 0.05:
            n_discriminators += 1
        if abs(ra['value'] - re['value']) > 0.5:
            n_value_diff += 1

    print(f'\n  Agents with ACTION discrimination:  {n_discriminators}/{n}')
    print(f'  Agents with VALUE discrimination:   {n_value_diff}/{n}')

    avg_atk_diff = np.mean([
        all_results[aid]['_ff_foe']['act_probs'][0] - all_results[aid]['_ff_ally']['act_probs'][0]
        for aid in range(n)
    ])
    avg_val_diff = np.mean([
        all_results[aid]['_ff_ally']['value'] - all_results[aid]['_ff_foe']['value']
        for aid in range(n)
    ])

    print(f'  Avg attack diff (foe-ally):         {avg_atk_diff:+.2%}')
    print(f'  Avg value diff (ally-foe):           {avg_val_diff:+.3f}')
    print()


if __name__ == '__main__':
    main()
