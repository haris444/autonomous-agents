"""
Diagnose whether the network distinguishes friend vs foe in Phase 9.

Uses the exact ally/enemy profiles from scenario priming to test if:
1. Encoder produces different features
2. Value estimates differ (allies should be more valuable)
3. Action probabilities differ (attack enemies, cooperate with allies)
"""
import torch
import torch.nn.functional as F
from config import Config
from network import ActorCritic


# Phase 9 profiles (normalized values, will be scaled by 5x)
ALLY_PROFILE = {
    'damage_dealt': 0.0,
    'food_given': 0.5,      # Fed us
    'coop_count': 0.5,      # Cooperated with us
    'defense_score': 0.3,   # Defended us
}

ENEMY_PROFILE = {
    'damage_dealt': 0.55,   # Attacked us
    'food_given': 0.0,
    'coop_count': 0.05,     # Minimal cooperation
    'defense_score': 0.0,
}


def fourier_encode(dx, dy, device):
    """Encode position using Fourier features (matching environment)."""
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
    """Create observation with another agent having given social history.

    Token format (25 features):
        - fourier[16]: 4 freq bands × (sin_dx, cos_dx, sin_dy, cos_dy)
        - velocity[2]: (dv_x, dv_y)
        - type_onehot[2]: [is_food, is_agent]
        - value[1]: HP or quality
        - social[4]: [damage_dealt, food_given, coop_count, defense_score]
    """
    entity_tokens = torch.zeros(config.max_entities, config.entity_token_dim, device=device)
    entity_mask = torch.zeros(config.max_entities, device=device, dtype=torch.bool)

    # Agent 0 (self) at origin
    self_fourier = fourier_encode(0.0, 0.0, device)  # [16]
    entity_tokens[0, 0:16] = self_fourier            # fourier[16]
    entity_tokens[0, 16:18] = 0.0                    # velocity[2]
    entity_tokens[0, 18:20] = torch.tensor([0.0, 1.0], device=device)  # type: [is_food, is_agent]
    entity_tokens[0, 20] = 1.0                       # HP = full
    entity_tokens[0, 21:25] = 0.0                    # social[4]
    entity_mask[0] = True

    # Agent 1 (other) one cell to the right
    dx_norm = 0.0 / config.grid_size
    dy_norm = 1.0 / config.grid_size
    other_fourier = fourier_encode(dx_norm, dy_norm, device)  # [16]
    entity_tokens[1, 0:16] = other_fourier           # fourier[16]
    entity_tokens[1, 16:18] = 0.0                    # velocity[2]
    entity_tokens[1, 18:20] = torch.tensor([0.0, 1.0], device=device)  # type: [is_food, is_agent]
    entity_tokens[1, 20] = 1.0                       # HP = full
    # Social features (scaled by 5x as in environment)
    entity_tokens[1, 21] = social_profile['damage_dealt'] * 5.0
    entity_tokens[1, 22] = social_profile['food_given'] * 5.0
    entity_tokens[1, 23] = social_profile['coop_count'] * 5.0
    entity_tokens[1, 24] = social_profile['defense_score'] * 5.0
    entity_mask[1] = True

    return {
        'entity_tokens': entity_tokens.unsqueeze(0),
        'entity_mask': entity_mask.unsqueeze(0),
        'signals': torch.zeros(1, config.n_agents, device=device),
        'self_hp': torch.ones(1, 1, device=device),
        'self_inventory': torch.zeros(1, 1, device=device),
        'agent_id': torch.tensor([0], device=device)
    }


def analyze(model, obs_ally, obs_enemy):
    """Compare network outputs for ally vs enemy."""
    with torch.no_grad():
        # Get encoder features
        feat_ally = model._encode(obs_ally)
        feat_enemy = model._encode(obs_enemy)

        # Value estimates
        val_ally = model.value_head(feat_ally).item()
        val_enemy = model.value_head(feat_enemy).item()

        # Action probabilities
        ally_interact = F.softmax(model.interact_type_head(feat_ally), dim=-1).squeeze()
        enemy_interact = F.softmax(model.interact_type_head(feat_enemy), dim=-1).squeeze()

        # Feature similarity
        feat_ally_norm = F.normalize(feat_ally, dim=1)
        feat_enemy_norm = F.normalize(feat_enemy, dim=1)
        cosine_sim = (feat_ally_norm @ feat_enemy_norm.T).item()

        # L2 distance
        l2_dist = (feat_ally - feat_enemy).norm().item()

    return {
        'feat_ally': feat_ally.squeeze(),
        'feat_enemy': feat_enemy.squeeze(),
        'val_ally': val_ally,
        'val_enemy': val_enemy,
        'ally_interact': ally_interact.cpu().numpy(),
        'enemy_interact': enemy_interact.cpu().numpy(),
        'cosine_sim': cosine_sim,
        'l2_dist': l2_dist,
    }


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    # Find checkpoint
    paths = [args.model] if args.model else []
    paths += [
        'pretrain_phase9/checkpoint_latest.pt',
        'pretrain_phase9/pretrained.pt',
        'runs/latest/final_model.pt',
    ]

    model = None
    for path in paths:
        if path and os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state = ckpt.get('network_state_dicts', [ckpt])[0]
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']

            # Auto-detect n_agents
            if 'encoder.signal_encoder.0.weight' in state:
                config.n_agents = state['encoder.signal_encoder.0.weight'].shape[1]
                config.__post_init__()

            model = ActorCritic(config).to(device)
            model.load_state_dict(state)
            model.eval()
            print(f"Loaded: {path}")
            break

    if model is None:
        print("No checkpoint found!")
        return

    # Create observations
    obs_ally = create_obs(config, ALLY_PROFILE, device)
    obs_enemy = create_obs(config, ENEMY_PROFILE, device)

    # Analyze
    r = analyze(model, obs_ally, obs_enemy)

    # Print results
    print("\n" + "=" * 70)
    print("PHASE 9 FRIEND/FOE DISCRIMINATION TEST")
    print("=" * 70)

    print("\n--- INPUT SIGNALS (scaled by 5x) ---")
    print(f"ALLY:  dmg={ALLY_PROFILE['damage_dealt']*5:.2f} food={ALLY_PROFILE['food_given']*5:.2f} "
          f"coop={ALLY_PROFILE['coop_count']*5:.2f} def={ALLY_PROFILE['defense_score']*5:.2f}")
    print(f"ENEMY: dmg={ENEMY_PROFILE['damage_dealt']*5:.2f} food={ENEMY_PROFILE['food_given']*5:.2f} "
          f"coop={ENEMY_PROFILE['coop_count']*5:.2f} def={ENEMY_PROFILE['defense_score']*5:.2f}")

    print("\n--- FEATURE ANALYSIS ---")
    print(f"Cosine similarity:  {r['cosine_sim']:.4f}  (1.0=identical, 0.0=orthogonal)")
    print(f"L2 distance:        {r['l2_dist']:.4f}")

    if r['cosine_sim'] > 0.99:
        print("  >> WARNING: Features nearly identical! Network NOT discriminating.")
    elif r['cosine_sim'] > 0.95:
        print("  >> WEAK: Features very similar, minimal discrimination.")
    elif r['cosine_sim'] > 0.8:
        print("  >> MODERATE: Some feature differentiation.")
    else:
        print("  >> GOOD: Clear feature differentiation!")

    print("\n--- VALUE ESTIMATES ---")
    print(f"V(ally):  {r['val_ally']:+.3f}")
    print(f"V(enemy): {r['val_enemy']:+.3f}")
    print(f"Difference: {r['val_ally'] - r['val_enemy']:+.3f}")

    if r['val_ally'] > r['val_enemy'] + 0.1:
        print("  >> GOOD: Allies valued higher than enemies!")
    elif r['val_ally'] > r['val_enemy']:
        print("  >> WEAK: Allies slightly valued more.")
    else:
        print("  >> BAD: Enemies valued same or higher than allies.")

    print("\n--- ACTION PROBABILITIES ---")
    interact_names = ['ATTACK', 'GIVE', 'SIGNAL', 'COOP', 'IDLE']
    print(f"{'Action':<8} | {'ALLY':>8} | {'ENEMY':>8} | {'DIFF':>8}")
    print("-" * 40)
    for i, name in enumerate(interact_names):
        a, e = r['ally_interact'][i], r['enemy_interact'][i]
        print(f"{name:<8} | {a:>7.1%} | {e:>7.1%} | {a-e:>+7.1%}")

    # Key metrics
    attack_diff = r['enemy_interact'][0] - r['ally_interact'][0]
    coop_diff = r['ally_interact'][3] - r['enemy_interact'][3]
    give_diff = r['ally_interact'][1] - r['enemy_interact'][1]

    print("\n--- BEHAVIORAL SUMMARY ---")
    print(f"Attack enemy MORE than ally: {attack_diff:+.1%}  {'GOOD' if attack_diff > 0.05 else 'WEAK' if attack_diff > 0 else 'BAD'}")
    print(f"Coop with ally MORE than enemy: {coop_diff:+.1%}  {'GOOD' if coop_diff > 0.05 else 'WEAK' if coop_diff > 0 else 'BAD'}")
    print(f"Give to ally MORE than enemy: {give_diff:+.1%}  {'GOOD' if give_diff > 0.05 else 'WEAK' if give_diff > 0 else 'BAD'}")

    # Overall verdict
    print("\n" + "=" * 70)
    score = 0
    if r['cosine_sim'] < 0.95: score += 1
    if r['val_ally'] > r['val_enemy']: score += 1
    if attack_diff > 0: score += 1
    if coop_diff > 0 or give_diff > 0: score += 1

    if score >= 3:
        print("VERDICT: Network IS discriminating friend from foe!")
    elif score >= 2:
        print("VERDICT: Network shows SOME discrimination, but weak.")
    else:
        print("VERDICT: Network NOT discriminating friend from foe.")
    print("=" * 70)


if __name__ == '__main__':
    main()
