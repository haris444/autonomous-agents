"""
Diagnose whether the network learns meaningful food-position features.

Creates synthetic observations with food at different positions and checks
if the network responds differently (moves toward food).
"""
import torch
import torch.nn.functional as F
from config import Config
from network import ActorCritic

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


def create_synthetic_obs(config, food_pos, device):
    """
    Create a synthetic observation with food at a specific position.

    food_pos: (dy, dx) relative to agent, e.g., (0, 3) = 3 cells to the right
              Normalized by grid_size for Fourier encoding.

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
    self_fourier = fourier_encode(0.0, 0.0, device)  # [16]
    entity_tokens[0, 0:16] = self_fourier            # fourier[16]
    entity_tokens[0, 16:18] = 0.0                    # velocity[2]
    entity_tokens[0, 18:20] = torch.tensor([0.0, 1.0], device=device)  # type: [is_food, is_agent]
    entity_tokens[0, 20] = 1.0                       # HP (full)
    entity_tokens[0, 21:25] = 0.0                    # social[4]
    entity_mask[0] = True

    # Food token at specified position (in first food slot after agents)
    # food_pos format: (dy, dx) where dy=vertical (row), dx=horizontal (col)
    food_idx = config.n_agents  # First food slot
    food_row_norm = food_pos[0] / config.grid_size  # Normalize to [-1, 1] range
    food_col_norm = food_pos[1] / config.grid_size
    food_fourier = fourier_encode(food_row_norm, food_col_norm, device)  # [16]
    entity_tokens[food_idx, 0:16] = food_fourier     # fourier[16]
    entity_tokens[food_idx, 16:18] = 0.0             # velocity[2]
    entity_tokens[food_idx, 18:20] = torch.tensor([1.0, 0.0], device=device)  # type: [is_food, is_agent]
    entity_tokens[food_idx, 20] = 0.5                # quality (poor food)
    entity_tokens[food_idx, 21:25] = 0.0             # social[4]
    entity_mask[food_idx] = True

    signals = torch.zeros(config.n_agents, device=device)
    self_hp = torch.tensor([1.0], device=device)  # Full HP
    self_inventory = torch.tensor([0.0], device=device)  # No inventory
    agent_id = torch.tensor([0], device=device)

    return {
        'entity_tokens': entity_tokens.unsqueeze(0),  # [1, max_entities, 25]
        'entity_mask': entity_mask.unsqueeze(0),      # [1, max_entities]
        'signals': signals.unsqueeze(0),              # [1, N]
        'self_hp': self_hp.unsqueeze(0),              # [1, 1]
        'self_inventory': self_inventory.unsqueeze(0), # [1, 1]
        'agent_id': agent_id                          # [1]
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    # Load trained model
    model = ActorCritic(config).to(device)

    # Try to load checkpoint - use the test model we just trained
    import os
    checkpoint_paths = [
        'pretrain_test_unified/final_model.pt',
        'pretrain_test_unified/pretrained.pt',
    ]

    loaded = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if 'network_state_dicts' in checkpoint:
                state_dict = checkpoint['network_state_dicts'][0]
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            print(f"Loaded model from {path}")
            loaded = True
            break

    if not loaded:
        print("No checkpoint found - using random weights (untrained)")

    model.eval()

    # Test positions: food at different locations relative to agent
    # Format: (dy, dx) where positive dx = right, positive dy = down
    test_positions = [
        (0, 0),   # On top of agent
        (0, 1),   # 1 right
        (0, 2),   # 2 right
        (0, 3),   # 3 right
        (0, -1),  # 1 left
        (0, -2),  # 2 left
        (0, -3),  # 3 left
        (1, 0),   # 1 down
        (-1, 0),  # 1 up
        (2, 2),   # diagonal (down-right)
    ]

    print("=" * 60)
    print("ENTITY TOKEN FEATURE DIAGNOSIS")
    print("=" * 60)
    print("\nTesting if network responds differently to food at different positions")
    print("(Food position encoded using Fourier features in tokens)\n")

    # Collect features and actions for each position
    results = []

    with torch.no_grad():
        for pos in test_positions:
            obs = create_synthetic_obs(config, pos, device)

            # Get raw features from encoder
            features = model.encoder(
                obs['entity_tokens'],
                obs['entity_mask'],
                obs['signals'],
                obs['self_hp'],
                obs['agent_id']
            )

            # Get action logits
            hidden = model.shared(features)
            move_logits = model.move_head(hidden)

            # Get move probabilities
            move_probs = F.softmax(move_logits, dim=-1).squeeze()

            results.append({
                'pos': pos,
                'feat_mean': features.mean().item(),
                'feat_std': features.std().item(),
                'feat_norm': features.norm().item(),
                'move_probs': move_probs.cpu().numpy(),
                'features': features.squeeze().cpu()
            })

    # Print results
    print("Position  | Features (mean/std/norm)     | Move Probs (UP, DOWN, LEFT, RIGHT, STAY)")
    print("-" * 90)

    move_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']

    for r in results:
        pos_str = f"({r['pos'][0]:+d}, {r['pos'][1]:+d})"
        feat_str = f"{r['feat_mean']:+.3f} / {r['feat_std']:.3f} / {r['feat_norm']:.2f}"

        # Format move probs
        probs = r['move_probs']
        prob_str = " ".join([f"{move_names[i]}:{probs[i]:.2f}" for i in range(5)])

        print(f"{pos_str:10s} | {feat_str:28s} | {prob_str}")

    # Check feature diversity
    print("\n" + "=" * 60)
    print("FEATURE DIVERSITY ANALYSIS")
    print("=" * 60)

    # Stack all features
    all_features = torch.stack([r['features'] for r in results])

    # Compute pairwise cosine similarity
    all_features_norm = F.normalize(all_features, dim=1)
    similarity_matrix = all_features_norm @ all_features_norm.T

    print("\nCosine similarity between features for different food positions:")
    print("(1.0 = identical features, 0.0 = orthogonal)")
    print()

    # Print similarity for key pairs
    pairs = [
        (0, 1, "center vs 1-right"),
        (0, 4, "center vs 1-left"),
        (1, 4, "1-right vs 1-left"),
        (1, 2, "1-right vs 2-right"),
        (1, 7, "1-right vs 1-down"),
    ]

    for i, j, desc in pairs:
        sim = similarity_matrix[i, j].item()
        print(f"  {desc:25s}: {sim:.4f}")

    avg_sim = (similarity_matrix.sum() - len(results)) / (len(results) * (len(results) - 1))
    print(f"\n  Average pairwise similarity: {avg_sim:.4f}")

    if avg_sim > 0.95:
        print("\n  WARNING: Features are nearly identical regardless of food position!")
        print("     The network is NOT learning spatial structure.")
    elif avg_sim > 0.8:
        print("\n  Features are quite similar - limited spatial discrimination")
    else:
        print("\n  Features show reasonable diversity across positions")

    # Check if actions make sense
    print("\n" + "=" * 60)
    print("ACTION SANITY CHECK")
    print("=" * 60)
    print("\nDoes the network prefer to move TOWARD food?")

    # Food 1 right -> should prefer RIGHT (index 3)
    r_right = results[1]
    right_prob = r_right['move_probs'][3]
    print(f"  Food 1-right: P(RIGHT) = {right_prob:.2f}", "GOOD" if right_prob > 0.3 else "BAD")

    # Food 1 left -> should prefer LEFT (index 2)
    r_left = results[4]
    left_prob = r_left['move_probs'][2]
    print(f"  Food 1-left:  P(LEFT)  = {left_prob:.2f}", "GOOD" if left_prob > 0.3 else "BAD")

    # Food 1 up -> should prefer UP (index 0)
    r_up = results[8]
    up_prob = r_up['move_probs'][0]
    print(f"  Food 1-up:    P(UP)    = {up_prob:.2f}", "GOOD" if up_prob > 0.3 else "BAD")

    # Food 1 down -> should prefer DOWN (index 1)
    r_down = results[7]
    down_prob = r_down['move_probs'][1]
    print(f"  Food 1-down:  P(DOWN)  = {down_prob:.2f}", "GOOD" if down_prob > 0.3 else "BAD")


if __name__ == '__main__':
    main()
