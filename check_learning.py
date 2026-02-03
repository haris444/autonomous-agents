"""
Quick diagnostic to check if trained model moves toward food.
Run this while training to see if it's learning spatial awareness.

Usage: python check_learning.py <checkpoint_path>
"""
import sys
import torch
import torch.nn.functional as F
from config import Config
from network import ActorCritic

def check_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    model = ActorCritic(config).to(device)

    print(f'Loading: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'network_state_dicts' in ckpt:
        model.load_state_dict(ckpt['network_state_dicts'][0])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    def fourier_encode(dx, dy):
        """Encode position using Fourier features (matching environment)."""
        bands = [1.0, 2.0, 4.0, 8.0]
        features = []
        for freq in bands:
            features.extend([
                torch.sin(torch.tensor(freq * torch.pi * dx)),
                torch.cos(torch.tensor(freq * torch.pi * dx)),
                torch.sin(torch.tensor(freq * torch.pi * dy)),
                torch.cos(torch.tensor(freq * torch.pi * dy)),
            ])
        return torch.stack(features)

    def create_obs(food_dx, food_dy):
        """Create observation with Fourier positional encoding."""
        tokens = torch.zeros(1, config.max_entities, config.entity_token_dim, device=device)
        mask = torch.zeros(1, config.max_entities, device=device, dtype=torch.bool)

        # Self token (agent at 0,0):
        # fourier[16] + velocity[2] + type[2] + hp[1] + social[4] = 25
        self_fourier = fourier_encode(0.0, 0.0).to(device)
        self_token = torch.cat([
            self_fourier,                              # 16: fourier
            torch.zeros(2, device=device),             # 2: velocity
            torch.tensor([0.0, 1.0], device=device),   # 2: type [is_food, is_agent]
            torch.tensor([1.0], device=device),        # 1: hp
            torch.zeros(4, device=device),             # 4: social
        ])
        tokens[0, 0, :] = self_token
        mask[0, 0] = True

        # Food token:
        food_fourier = fourier_encode(food_dx, food_dy).to(device)
        food_token = torch.cat([
            food_fourier,                              # 16: fourier
            torch.zeros(2, device=device),             # 2: velocity
            torch.tensor([1.0, 0.0], device=device),   # 2: type [is_food, is_agent]
            torch.tensor([0.5], device=device),        # 1: quality
            torch.zeros(4, device=device),             # 4: social
        ])
        tokens[0, config.n_agents, :] = food_token
        mask[0, config.n_agents] = True

        return {
            'entity_tokens': tokens,
            'entity_mask': mask,
            'signals': torch.zeros(1, config.n_agents, device=device),
            'self_hp': torch.ones(1, 1, device=device),
            'self_inventory': torch.zeros(1, 1, device=device),
            'agent_id': torch.zeros(1, dtype=torch.long, device=device)
        }

    print()
    print('DOES MODEL MOVE TOWARD FOOD?')
    print('=' * 60)

    # Token format: [row_diff, col_diff, ...] where:
    #   row_diff > 0 = food is DOWN (higher row)
    #   col_diff > 0 = food is RIGHT (higher col)
    # Distance 1 in normalized coords: 1/grid_size (before 5x scaling)
    gs = config.grid_size
    dist1 = 1.0 / gs
    tests = [
        (0.0, dist1, 'RIGHT', 3),   # Food right (col+) -> should move RIGHT (idx 3)
        (0.0, -dist1, 'LEFT', 2),   # Food left (col-) -> should move LEFT (idx 2)
        (dist1, 0.0, 'DOWN', 1),    # Food down (row+) -> should move DOWN (idx 1)
        (-dist1, 0.0, 'UP', 0),     # Food up (row-) -> should move UP (idx 0)
    ]

    move_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
    correct = 0

    for dx, dy, name, expected_idx in tests:
        with torch.no_grad():
            obs = create_obs(dx, dy)
            hidden = model._encode(obs)
            logits = model.move_head(hidden)
            probs = F.softmax(logits, dim=-1).squeeze()

        expected_prob = probs[expected_idx].item()
        best_action = probs.argmax().item()

        is_correct = best_action == expected_idx
        correct += is_correct

        status = "GOOD" if is_correct else "BAD"
        print(f'Food {name:5s}: P({move_names[expected_idx]})={expected_prob:.2f}  '
              f'Best={move_names[best_action]:5s}  {status}  '
              f'[{", ".join(f"{p:.2f}" for p in probs.tolist())}]')

    print()
    print(f'Score: {correct}/4 correct')

    # Feature diversity check
    print()
    print('FEATURE DIVERSITY:')
    features = []
    for dx, dy, name, _ in tests:
        obs = create_obs(dx, dy)
        with torch.no_grad():
            feat = model.encoder(
                obs['entity_tokens'], obs['entity_mask'],
                obs['signals'], obs['self_hp'], obs['agent_id']
            )
        features.append(feat.squeeze())

    feat_stack = torch.stack(features)
    feat_norm = F.normalize(feat_stack, dim=1)
    sim = feat_norm @ feat_norm.T

    print(f'  RIGHT vs LEFT similarity: {sim[0,1]:.3f} (want < 0.5)')
    print(f'  UP vs DOWN similarity:    {sim[2,3]:.3f} (want < 0.5)')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python check_learning.py <checkpoint_path>')
        print('Example: python check_learning.py pretrain_output/checkpoint_100.pt')
    else:
        check_model(sys.argv[1])
