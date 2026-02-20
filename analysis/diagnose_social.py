"""
Diagnose whether the network learns meaningful social features.

Creates synthetic observations with different social histories between agents
and checks if the network responds differently (attacks enemies, cooperates with allies).
"""
import torch
import torch.nn.functional as F
from core.config import Config
from agents.network import ActorCritic


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


def create_synthetic_obs(config, other_agent_social, device):
    """
    Create a synthetic observation with another agent nearby with specific social history.

    other_agent_social: dict with keys 'damage_dealt', 'food_given', 'coop_count', 'defense_score'
                        These are the OTHER agent's history WITH US (what they did to us)

    Setup: Agent 0 (self) at center, Agent 1 adjacent to the right

    Token format (25 features):
        - fourier[16]: 4 freq bands × (sin_dx, cos_dx, sin_dy, cos_dy)
        - velocity[2]: (dv_x, dv_y)
        - type_onehot[2]: [is_food, is_agent]
        - value[1]: HP or quality
        - social[4]: [damage_dealt, food_given, coop_count, defense_score]
    """
    entity_tokens = torch.zeros(config.max_entities, config.entity_token_dim, device=device)
    entity_mask = torch.zeros(config.max_entities, device=device, dtype=torch.bool)

    # Agent 0 (self) at origin - always present
    # Fourier encode (0, 0) position
    self_fourier = fourier_encode(0.0, 0.0, device)  # [16]
    entity_tokens[0, 0:16] = self_fourier            # fourier[16]
    entity_tokens[0, 16:18] = 0.0                    # velocity[2]
    entity_tokens[0, 18:20] = torch.tensor([0.0, 1.0], device=device)  # type: [is_food, is_agent]
    entity_tokens[0, 20] = 1.0                       # HP (full)
    entity_tokens[0, 21:25] = 0.0                    # social[4] (self has no social history)
    entity_mask[0] = True

    # Agent 1 (other) one cell to the right
    # Position normalized: dx=0, dy=1/grid_size
    dx_norm = 0.0 / config.grid_size
    dy_norm = 1.0 / config.grid_size
    other_fourier = fourier_encode(dx_norm, dy_norm, device)  # [16]
    entity_tokens[1, 0:16] = other_fourier           # fourier[16]
    entity_tokens[1, 16:18] = 0.0                    # velocity[2]
    entity_tokens[1, 18:20] = torch.tensor([0.0, 1.0], device=device)  # type: [is_food, is_agent]
    entity_tokens[1, 20] = 1.0                       # HP (full)

    # Social features scaled by 5x to match environment scaling
    social_scale = 5.0
    entity_tokens[1, 21] = other_agent_social.get('damage_dealt', 0.0) * social_scale
    entity_tokens[1, 22] = other_agent_social.get('food_given', 0.0) * social_scale
    entity_tokens[1, 23] = other_agent_social.get('coop_count', 0.0) * social_scale
    entity_tokens[1, 24] = other_agent_social.get('defense_score', 0.0) * social_scale
    entity_mask[1] = True

    signals = torch.zeros(config.n_agents, device=device)
    self_hp = torch.tensor([1.0], device=device)  # Full HP
    self_inventory = torch.tensor([0.5], device=device)  # Some inventory
    agent_id = torch.tensor([0], device=device)

    return {
        'entity_tokens': entity_tokens.unsqueeze(0),  # [1, max_entities, 25]
        'entity_mask': entity_mask.unsqueeze(0),      # [1, max_entities]
        'signals': signals.unsqueeze(0),              # [1, N]
        'self_hp': self_hp.unsqueeze(0),              # [1, 1]
        'self_inventory': self_inventory.unsqueeze(0), # [1, 1]
        'agent_id': agent_id                          # [1]
    }


def get_action_probs(model, obs):
    """Get all action probabilities from the model."""
    with torch.no_grad():
        features = model.encoder(
            obs['entity_tokens'],
            obs['entity_mask'],
            obs['signals'],
            obs['self_hp'],
            obs['agent_id']
        )

        hidden = model.shared(features)

        # Move probabilities
        move_logits = model.move_head(hidden)
        move_probs = F.softmax(move_logits, dim=-1).squeeze()

        # Interact type probabilities (ATTACK, GIVE, SIGNAL, COOPERATE, IDLE)
        interact_type_logits = model.interact_type_head(hidden)
        interact_type_probs = F.softmax(interact_type_logits, dim=-1).squeeze()

        # Direction probabilities (for ATTACK and GIVE)
        direction_logits = model.direction_head(hidden)
        direction_probs = F.softmax(direction_logits, dim=-1).squeeze()

    return {
        'move': move_probs.cpu().numpy(),
        'interact_type': interact_type_probs.cpu().numpy(),
        'direction': direction_probs.cpu().numpy(),
        'features': features.squeeze().cpu()
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--n-agents', '-n', type=int, default=None,
                        help='Number of agents (auto-detected if not specified)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    import os
    checkpoint_paths = [args.model] if args.model else []
    checkpoint_paths += [
        'multiagent_run4/checkpoint_170.pt',  # Latest multi-agent
        'multiagent_run2/checkpoint_110.pt',
        'runs/latest/final_model.pt',
        'pretrain_curriculum_v2/checkpoint_190.pt',
    ]

    # First, detect n_agents from checkpoint
    loaded = False
    for path in checkpoint_paths:
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if 'network_state_dicts' in checkpoint:
                state_dict = checkpoint['network_state_dicts'][0]
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Auto-detect n_agents from signal_encoder weight shape
            if args.n_agents:
                config.n_agents = args.n_agents
            elif 'encoder.signal_encoder.0.weight' in state_dict:
                detected_n = state_dict['encoder.signal_encoder.0.weight'].shape[1]
                config.n_agents = detected_n
                print(f"Auto-detected n_agents={detected_n} from checkpoint")

            # Recompute derived values
            config.__post_init__()

            # Now create model with correct config
            model = ActorCritic(config).to(device)
            model.load_state_dict(state_dict)
            print(f"Loaded model from {path}")
            loaded = True
            break

    if not loaded:
        print("No checkpoint found - using random weights (untrained)")
        model = ActorCritic(config).to(device)

    model.eval()

    # Define social scenarios to test
    # Values are normalized (0-1 scale as in ledger)
    scenarios = {
        'neutral': {
            'damage_dealt': 0.0,
            'food_given': 0.0,
            'coop_count': 0.0,
            'defense_score': 0.0
        },
        'enemy_attacked_me': {
            'damage_dealt': 0.8,  # They dealt lots of damage to me
            'food_given': 0.0,
            'coop_count': 0.0,
            'defense_score': 0.0
        },
        'ally_defended_me': {
            'damage_dealt': 0.0,
            'food_given': 0.0,
            'coop_count': 0.0,
            'defense_score': 0.8  # They defended me a lot
        },
        'ally_fed_me': {
            'damage_dealt': 0.0,
            'food_given': 0.8,  # They gave me lots of food
            'coop_count': 0.0,
            'defense_score': 0.0
        },
        'ally_cooperated': {
            'damage_dealt': 0.0,
            'food_given': 0.0,
            'coop_count': 0.8,  # We cooperated on rich food many times
            'defense_score': 0.0
        },
        'strong_ally': {
            'damage_dealt': 0.0,
            'food_given': 0.5,
            'coop_count': 0.5,
            'defense_score': 0.5  # Good relationship across the board
        },
        'strong_enemy': {
            'damage_dealt': 1.0,  # Maximum damage
            'food_given': 0.0,
            'coop_count': 0.0,
            'defense_score': 0.0
        },
        'mixed_history': {
            'damage_dealt': 0.3,
            'food_given': 0.3,
            'coop_count': 0.2,
            'defense_score': 0.2  # Complex relationship
        }
    }

    print("=" * 80)
    print("SOCIAL FEATURE DIAGNOSIS")
    print("=" * 80)
    print("\nSetup: Agent 0 (self) with Agent 1 adjacent to the RIGHT")
    print("Testing how social history affects action selection")
    print()

    # Collect results
    results = {}
    for name, social in scenarios.items():
        obs = create_synthetic_obs(config, social, device)
        probs = get_action_probs(model, obs)
        results[name] = probs

    # Print interact type probabilities for each scenario
    interact_names = ['ATTACK', 'GIVE', 'SIGNAL', 'COOP', 'IDLE']
    dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    print("=" * 80)
    print("INTERACT TYPE PROBABILITIES")
    print("=" * 80)
    print(f"{'Scenario':<20} | " + " | ".join([f"{n:>7}" for n in interact_names]))
    print("-" * 80)

    for name, probs in results.items():
        interact_probs = probs['interact_type']
        prob_str = " | ".join([f"{p:>7.2%}" for p in interact_probs])
        print(f"{name:<20} | {prob_str}")

    # Print direction probabilities (relevant for ATTACK)
    print("\n" + "=" * 80)
    print("DIRECTION PROBABILITIES (for ATTACK/GIVE - target is to the RIGHT)")
    print("=" * 80)
    print(f"{'Scenario':<20} | " + " | ".join([f"{n:>7}" for n in dir_names]))
    print("-" * 80)

    for name, probs in results.items():
        dir_probs = probs['direction']
        prob_str = " | ".join([f"{p:>7.2%}" for p in dir_probs])
        print(f"{name:<20} | {prob_str}")

    # Key comparisons
    print("\n" + "=" * 80)
    print("KEY BEHAVIORAL COMPARISONS")
    print("=" * 80)

    # Attack probability: enemy vs ally
    attack_neutral = results['neutral']['interact_type'][0]
    attack_enemy = results['enemy_attacked_me']['interact_type'][0]
    attack_ally_def = results['ally_defended_me']['interact_type'][0]
    attack_ally_fed = results['ally_fed_me']['interact_type'][0]

    print("\n1. ATTACK probability:")
    print(f"   Neutral:              {attack_neutral:>7.2%}")
    print(f"   Enemy (attacked me):  {attack_enemy:>7.2%}  {'<-- MORE' if attack_enemy > attack_neutral else ''}")
    print(f"   Ally (defended me):   {attack_ally_def:>7.2%}  {'<-- LESS' if attack_ally_def < attack_neutral else ''}")
    print(f"   Ally (fed me):        {attack_ally_fed:>7.2%}  {'<-- LESS' if attack_ally_fed < attack_neutral else ''}")

    # Give probability: ally vs enemy
    give_neutral = results['neutral']['interact_type'][1]
    give_enemy = results['enemy_attacked_me']['interact_type'][1]
    give_ally_def = results['ally_defended_me']['interact_type'][1]
    give_ally_fed = results['ally_fed_me']['interact_type'][1]

    print("\n2. GIVE probability:")
    print(f"   Neutral:              {give_neutral:>7.2%}")
    print(f"   Enemy (attacked me):  {give_enemy:>7.2%}  {'<-- LESS' if give_enemy < give_neutral else ''}")
    print(f"   Ally (defended me):   {give_ally_def:>7.2%}  {'<-- MORE' if give_ally_def > give_neutral else ''}")
    print(f"   Ally (fed me):        {give_ally_fed:>7.2%}  {'<-- MORE' if give_ally_fed > give_neutral else ''}")

    # Cooperate probability
    coop_neutral = results['neutral']['interact_type'][3]
    coop_enemy = results['enemy_attacked_me']['interact_type'][3]
    coop_ally = results['ally_cooperated']['interact_type'][3]

    print("\n3. COOPERATE probability:")
    print(f"   Neutral:              {coop_neutral:>7.2%}")
    print(f"   Enemy (attacked me):  {coop_enemy:>7.2%}  {'<-- LESS' if coop_enemy < coop_neutral else ''}")
    print(f"   Ally (cooperated):    {coop_ally:>7.2%}  {'<-- MORE' if coop_ally > coop_neutral else ''}")

    # Feature diversity analysis
    print("\n" + "=" * 80)
    print("FEATURE DIVERSITY ANALYSIS")
    print("=" * 80)

    all_features = torch.stack([results[name]['features'] for name in scenarios.keys()])
    all_features_norm = F.normalize(all_features, dim=1)
    similarity_matrix = all_features_norm @ all_features_norm.T

    scenario_names = list(scenarios.keys())

    print("\nKey feature similarities:")
    pairs = [
        ('neutral', 'enemy_attacked_me', "Neutral vs Enemy"),
        ('neutral', 'ally_defended_me', "Neutral vs Ally (defended)"),
        ('enemy_attacked_me', 'ally_defended_me', "Enemy vs Ally"),
        ('strong_ally', 'strong_enemy', "Strong ally vs Strong enemy"),
    ]

    for s1, s2, desc in pairs:
        i1, i2 = scenario_names.index(s1), scenario_names.index(s2)
        sim = similarity_matrix[i1, i2].item()
        print(f"   {desc:<30}: {sim:.4f}")

    avg_sim = (similarity_matrix.sum() - len(scenario_names)) / (len(scenario_names) * (len(scenario_names) - 1))
    print(f"\n   Average pairwise similarity: {avg_sim:.4f}")

    if avg_sim > 0.99:
        print("\n   WARNING: Features are nearly identical!")
        print("   The network is NOT learning social structure.")
    elif avg_sim > 0.95:
        print("\n   Features show minimal social discrimination.")
    elif avg_sim > 0.8:
        print("\n   Features show moderate social discrimination.")
    else:
        print("\n   Features show good social discrimination.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Check if behavior makes sense
    attack_diff = attack_enemy - attack_ally_def
    give_diff = give_ally_def - give_enemy

    print(f"\nAttack enemy MORE than ally: {attack_diff:+.2%} {'GOOD' if attack_diff > 0.05 else 'WEAK' if attack_diff > 0 else 'BAD'}")
    print(f"Give to ally MORE than enemy: {give_diff:+.2%} {'GOOD' if give_diff > 0.05 else 'WEAK' if give_diff > 0 else 'BAD'}")

    if attack_diff <= 0 and give_diff <= 0:
        print("\nThe network does NOT differentiate based on social history.")
        print("This could mean:")
        print("  1. Insufficient multi-agent training")
        print("  2. Social features not being attended to")
        print("  3. Need stronger reward signals for social behavior")
    elif attack_diff > 0.1 or give_diff > 0.1:
        print("\nThe network DOES respond to social history!")
        print("Social learning appears to be working.")


def gradient_test(model, config, device):
    """Test if social values have a smooth gradient effect on behavior."""
    print("\n" + "=" * 80)
    print("GRADIENT TEST: Does increasing social values smoothly change behavior?")
    print("=" * 80)

    # Test damage_dealt gradient (0.0 to 1.0)
    print("\nDamage dealt (enemy intensity) gradient:")
    print(f"{'Value':<8} | {'ATTACK':<8} | {'GIVE':<8} | {'Direction->RIGHT':<16}")
    print("-" * 50)

    for damage_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        social = {'damage_dealt': damage_val, 'food_given': 0, 'coop_count': 0, 'defense_score': 0}
        obs = create_synthetic_obs(config, social, device)
        probs = get_action_probs(model, obs)
        attack_p = probs['interact_type'][0]
        give_p = probs['interact_type'][1]
        right_p = probs['direction'][3]
        print(f"{damage_val:<8.1f} | {attack_p:<8.2%} | {give_p:<8.2%} | {right_p:<16.2%}")

    # Test defense_score gradient (ally intensity)
    print("\nDefense score (ally intensity) gradient:")
    print(f"{'Value':<8} | {'ATTACK':<8} | {'GIVE':<8} | {'COOP':<8}")
    print("-" * 50)

    for def_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        social = {'damage_dealt': 0, 'food_given': 0, 'coop_count': 0, 'defense_score': def_val}
        obs = create_synthetic_obs(config, social, device)
        probs = get_action_probs(model, obs)
        attack_p = probs['interact_type'][0]
        give_p = probs['interact_type'][1]
        coop_p = probs['interact_type'][3]
        print(f"{def_val:<8.1f} | {attack_p:<8.2%} | {give_p:<8.2%} | {coop_p:<8.2%}")

    # Test coop_count gradient
    print("\nCoop count (cooperation history) gradient:")
    print(f"{'Value':<8} | {'ATTACK':<8} | {'GIVE':<8} | {'COOP':<8}")
    print("-" * 50)

    for coop_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        social = {'damage_dealt': 0, 'food_given': 0, 'coop_count': coop_val, 'defense_score': 0}
        obs = create_synthetic_obs(config, social, device)
        probs = get_action_probs(model, obs)
        attack_p = probs['interact_type'][0]
        give_p = probs['interact_type'][1]
        coop_p = probs['interact_type'][3]
        print(f"{coop_val:<8.1f} | {attack_p:<8.2%} | {give_p:<8.2%} | {coop_p:<8.2%}")


if __name__ == '__main__':
    # Run main which loads the model, then run gradient test
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--n-agents', '-n', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    checkpoint_paths = [args.model] if args.model else []
    checkpoint_paths += [
        'multiagent_run4/checkpoint_170.pt',
        'multiagent_run2/checkpoint_110.pt',
    ]

    model = None
    for path in checkpoint_paths:
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if 'network_state_dicts' in checkpoint:
                state_dict = checkpoint['network_state_dicts'][0]
            else:
                state_dict = checkpoint

            # Auto-detect n_agents
            if args.n_agents:
                config.n_agents = args.n_agents
            elif 'encoder.signal_encoder.0.weight' in state_dict:
                config.n_agents = state_dict['encoder.signal_encoder.0.weight'].shape[1]

            config.__post_init__()
            model = ActorCritic(config).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded: {path} (n_agents={config.n_agents})")
            break

    if model is None:
        print("No checkpoint - using random weights")
        model = ActorCritic(config).to(device)
        model.eval()

    # Run main analysis
    main.__code__ = (lambda: None).__code__  # Skip main, we loaded manually

    # Run all tests
    from diagnose_social import get_action_probs, create_synthetic_obs

    # === Run scenario tests ===
    scenarios = {
        'neutral': {'damage_dealt': 0.0, 'food_given': 0.0, 'coop_count': 0.0, 'defense_score': 0.0},
        'enemy_attacked_me': {'damage_dealt': 0.8, 'food_given': 0.0, 'coop_count': 0.0, 'defense_score': 0.0},
        'ally_defended_me': {'damage_dealt': 0.0, 'food_given': 0.0, 'coop_count': 0.0, 'defense_score': 0.8},
        'ally_fed_me': {'damage_dealt': 0.0, 'food_given': 0.8, 'coop_count': 0.0, 'defense_score': 0.0},
        'ally_cooperated': {'damage_dealt': 0.0, 'food_given': 0.0, 'coop_count': 0.8, 'defense_score': 0.0},
        'strong_ally': {'damage_dealt': 0.0, 'food_given': 0.5, 'coop_count': 0.5, 'defense_score': 0.5},
        'strong_enemy': {'damage_dealt': 1.0, 'food_given': 0.0, 'coop_count': 0.0, 'defense_score': 0.0},
    }

    print("=" * 80)
    print("SOCIAL FEATURE DIAGNOSIS")
    print("=" * 80)

    results = {}
    for name, social in scenarios.items():
        obs = create_synthetic_obs(config, social, device)
        probs = get_action_probs(model, obs)
        results[name] = probs

    interact_names = ['ATTACK', 'GIVE', 'SIGNAL', 'COOP', 'IDLE']
    print(f"\n{'Scenario':<20} | " + " | ".join([f"{n:>7}" for n in interact_names]))
    print("-" * 80)
    for name, probs in results.items():
        prob_str = " | ".join([f"{p:>7.2%}" for p in probs['interact_type']])
        print(f"{name:<20} | {prob_str}")

    # Key comparisons
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)
    attack_enemy = results['enemy_attacked_me']['interact_type'][0]
    attack_ally = results['ally_defended_me']['interact_type'][0]
    give_enemy = results['enemy_attacked_me']['interact_type'][1]
    give_ally = results['ally_defended_me']['interact_type'][1]

    print(f"\nAttack enemy vs ally: {attack_enemy:.2%} vs {attack_ally:.2%} (diff: {attack_enemy - attack_ally:+.2%})")
    print(f"Give to ally vs enemy: {give_ally:.2%} vs {give_enemy:.2%} (diff: {give_ally - give_enemy:+.2%})")

    # Run gradient test
    gradient_test(model, config, device)
