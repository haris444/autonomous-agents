"""
Visualize what influences Agent 0's decisions.

Shows:
- Attention weights over entities (which agents/food the agent focuses on)
- Feature importance breakdown
- Combined action values with influence indicators
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.colors import LinearSegmentedColormap
import argparse

from core.config import Config
from agents.network import SharedTrunkActorCritic


def load_episode(path: str):
    """Load a saved episode."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data['steps'], data['ledger_snapshots'], data['config']


def load_model(checkpoint_path: str, config: Config, agent_id: int = 0):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model = SharedTrunkActorCritic(config)
    if 'network_state_dict' in ckpt:
        model.load_state_dict(ckpt['network_state_dict'])
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {ckpt.keys()}")
    model.eval()
    return model


def fourier_encode(dx, dy):
    """Encode position using Fourier features (matching environment)."""
    bands = [1.0, 2.0, 4.0, 8.0]
    features = []
    for freq in bands:
        features.extend([
            np.sin(freq * np.pi * dx),
            np.cos(freq * np.pi * dx),
            np.sin(freq * np.pi * dy),
            np.cos(freq * np.pi * dy),
        ])
    return np.array(features, dtype=np.float32)


def reconstruct_observation(step, ledger, config, agent_id: int = 0):
    """Reconstruct observation tensor from saved step data.

    Token format (25 features):
        - fourier[16]: 4 freq bands × (sin_dx, cos_dx, sin_dy, cos_dy)
        - velocity[2]: (dv_x, dv_y)
        - type_onehot[2]: [is_food, is_agent]
        - value[1]: HP or quality
        - social[4]: [damage_dealt, food_given, coop_count, defense_score]
    """
    n_agents = config.n_agents
    gs = config.grid_size
    max_food = config.max_food_tokens
    max_entities = n_agents + max_food
    token_dim = config.entity_token_dim  # 25

    entity_tokens = torch.zeros(max_entities, token_dim)
    entity_mask = torch.zeros(max_entities, dtype=torch.bool)

    # Agent position (for relative positions)
    if not step.alive[agent_id]:
        return None

    my_pos = step.positions[agent_id]

    # Fill agent tokens (slots 0 to n_agents-1)
    for i in range(n_agents):
        if step.alive[i]:
            rel_pos = (step.positions[i] - my_pos).astype(float)
            rel_pos_norm = rel_pos / gs  # Normalized for Fourier encoding

            hp_norm = float(step.hp[i] / config.max_hp)

            # Social history (what agent i did to agent_id)
            social = ledger[i, agent_id, :] / 100.0 * 5.0  # Normalize and scale

            # Fourier encode position
            fourier = fourier_encode(rel_pos_norm[0], rel_pos_norm[1])
            entity_tokens[i, 0:16] = torch.from_numpy(fourier)  # fourier[16]
            entity_tokens[i, 16:18] = 0.0                       # velocity[2]
            entity_tokens[i, 18:20] = torch.tensor([0.0, 1.0])  # type: [is_food, is_agent]
            entity_tokens[i, 20] = hp_norm                      # value
            entity_tokens[i, 21:25] = torch.from_numpy(social.astype(np.float32))  # social[4]

            entity_mask[i] = True

    # Fill food tokens (slots n_agents to max_entities)
    food_idx = n_agents

    # Get all food positions
    poor_food_pos = list(zip(*np.where(step.poor_food)))
    rich_food_pos = list(zip(*np.where(step.rich_food)))

    # Sort by distance to agent
    def dist_to_agent(pos):
        return abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1])

    all_food = [(pos, 0.5) for pos in poor_food_pos] + [(pos, 1.0) for pos in rich_food_pos]
    all_food.sort(key=lambda x: dist_to_agent(x[0]))

    for (r, c), quality in all_food[:max_food]:
        if food_idx >= max_entities:
            break

        rel_pos = np.array([r - my_pos[0], c - my_pos[1]], dtype=float)
        rel_pos_norm = rel_pos / gs

        # Fourier encode position
        fourier = fourier_encode(rel_pos_norm[0], rel_pos_norm[1])
        entity_tokens[food_idx, 0:16] = torch.from_numpy(fourier)  # fourier[16]
        entity_tokens[food_idx, 16:18] = 0.0                       # velocity[2]
        entity_tokens[food_idx, 18:20] = torch.tensor([1.0, 0.0])  # type: [is_food, is_agent]
        entity_tokens[food_idx, 20] = float(quality)               # value
        entity_tokens[food_idx, 21:25] = 0.0                       # social[4]

        entity_mask[food_idx] = True
        food_idx += 1

    # Signals
    signals = torch.tensor(step.signals, dtype=torch.float32)

    # Self state
    self_hp = torch.tensor([step.hp[agent_id] / config.max_hp])
    self_inventory = torch.tensor([0.0])  # Not saved in episode data
    agent_id_tensor = torch.tensor([agent_id])

    return {
        'entity_tokens': entity_tokens.unsqueeze(0),
        'entity_mask': entity_mask.unsqueeze(0),
        'signals': signals.unsqueeze(0),
        'self_hp': self_hp.unsqueeze(0),
        'self_inventory': self_inventory.unsqueeze(0),
        'agent_id': agent_id_tensor
    }


def get_attention_weights(model, obs):
    """Get attention weights from the model."""
    with torch.no_grad():
        enc = model.encoder
        my_tokens = obs['entity_tokens']  # [1, max_entities, token_dim]
        f = enc.fourier_dim
        fourier_proj = enc.fourier_embed(my_tokens[..., :f])
        velocity_proj = enc.velocity_embed(my_tokens[..., f:f+2])
        type_proj = enc.type_embed(my_tokens[..., f+2:f+4])
        value_proj = enc.value_embed(my_tokens[..., f+4:f+5])
        social_proj = enc.social_embed(my_tokens[..., f+5:f+9])
        grouped = torch.cat([fourier_proj, velocity_proj, type_proj, value_proj, social_proj], dim=-1)
        tokens = enc.entity_embed(grouped)

        attn_mask = ~obs['entity_mask']
        _, attn_weights = enc.entity_attention(
            tokens, tokens, tokens,
            key_padding_mask=attn_mask,
            average_attn_weights=True
        )

        return attn_weights[0].numpy()  # [max_entities, max_entities]


def visualize_influence(episode_path: str, checkpoint_path: str, frame_idx: int = 64):
    """Visualize what influences Agent 0's decisions."""

    steps, ledger_snapshots, config = load_episode(episode_path)
    step = steps[frame_idx]
    ledger = ledger_snapshots[frame_idx]

    # Load model
    model = load_model(checkpoint_path, config, agent_id=0)

    # Reconstruct observation
    obs = reconstruct_observation(step, ledger, config, agent_id=0)
    if obs is None:
        print("Agent 0 is dead at this frame")
        return

    # Get attention weights
    attn_weights = get_attention_weights(model, obs)

    # Get action probabilities
    with torch.no_grad():
        dir_logits, act_logits, value = model.forward(obs, agent_indices=0)
        dir_probs = torch.softmax(dir_logits, dim=-1)[0].numpy()
        act_probs = torch.softmax(act_logits, dim=-1)[0].numpy()

    # Create visualization
    gs = config.grid_size
    n_agents = config.n_agents
    agent_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Agent 0 Decision Influences - Step {frame_idx}', fontsize=14, fontweight='bold')

    # === TOP LEFT: Grid with attention-weighted connections ===
    ax_grid = fig.add_axes([0.02, 0.45, 0.35, 0.50])
    ax_grid.set_xlim(-0.5, gs - 0.5)
    ax_grid.set_ylim(-0.5, gs - 0.5)
    ax_grid.set_aspect('equal')
    ax_grid.invert_yaxis()
    ax_grid.set_title('Attention to Entities\n(line thickness = attention)', fontsize=11, fontweight='bold')

    # Grid lines
    for i in range(gs + 1):
        ax_grid.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.2)
        ax_grid.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.2)

    my_pos = step.positions[0]

    # Draw food with attention-based opacity
    food_attentions = []
    for row in range(gs):
        for col in range(gs):
            if step.rich_food[row, col]:
                ax_grid.plot(col, row, '*', color='gold', markersize=18, alpha=0.9)

    # Calculate attention from agent 0 (token 0) to other entities
    # Sum attention across all heads (already averaged)
    attn_from_agent0 = attn_weights[0, :]  # [max_entities]

    # Draw agents with attention lines
    for i in range(n_agents):
        if step.alive[i]:
            row, col = step.positions[i]
            color = agent_colors[i % len(agent_colors)]
            hp_frac = step.hp[i] / config.max_hp

            circle = Circle((col, row), 0.38, color=color, ec='black', linewidth=2, alpha=0.9)
            ax_grid.add_patch(circle)
            ax_grid.text(col, row, str(i), ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')

            # Draw attention line from agent 0 to this agent
            if i != 0 and obs['entity_mask'][0, i]:
                attn = attn_from_agent0[i]
                if attn > 0.01:
                    ax_grid.plot([my_pos[1], col], [my_pos[0], row],
                               color=color, linewidth=1 + attn * 15, alpha=0.3 + attn * 0.6)
                    # Attention value label
                    mid_x = (my_pos[1] + col) / 2
                    mid_y = (my_pos[0] + row) / 2
                    ax_grid.text(mid_x, mid_y, f'{attn:.0%}', fontsize=8,
                               ha='center', va='center', color=color, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

    # === TOP MIDDLE: Attention heatmap ===
    ax_attn = fig.add_axes([0.40, 0.55, 0.25, 0.40])

    # Show attention from token 0 (agent 0) to all tokens
    valid_mask = obs['entity_mask'][0].numpy()
    n_valid = valid_mask.sum()

    # Create labels for entities
    entity_labels = []
    for i in range(config.n_agents):
        if valid_mask[i]:
            entity_labels.append(f'A{i}')
    for i in range(config.n_agents, len(valid_mask)):
        if valid_mask[i]:
            entity_labels.append(f'F{i-config.n_agents}')

    # Extract valid attention values
    valid_attn = attn_from_agent0[valid_mask]

    # Bar chart of attention
    bars = ax_attn.bar(range(len(valid_attn)), valid_attn,
                       color=['#3498db' if l.startswith('A') else '#f1c40f' for l in entity_labels],
                       edgecolor='black')
    ax_attn.set_xticks(range(len(entity_labels)))
    ax_attn.set_xticklabels(entity_labels, fontsize=9)
    ax_attn.set_ylabel('Attention Weight')
    ax_attn.set_title('Attention Distribution\n(A=Agent, F=Food)', fontsize=11, fontweight='bold')
    ax_attn.set_ylim(0, max(0.5, valid_attn.max() * 1.2))

    # Highlight max attention
    max_idx = np.argmax(valid_attn)
    bars[max_idx].set_edgecolor('lime')
    bars[max_idx].set_linewidth(3)

    # === TOP RIGHT: Feature breakdown ===
    ax_feat = fig.add_axes([0.68, 0.55, 0.30, 0.40])
    ax_feat.axis('off')
    ax_feat.set_title('Observation Features', fontsize=11, fontweight='bold')

    lines = []
    lines.append("What Agent 0 Observes:")
    lines.append("=" * 45)
    lines.append("")
    lines.append("SELF STATE:")
    lines.append(f"  HP: {step.hp[0]:.0f}/{config.max_hp:.0f} ({step.hp[0]/config.max_hp:.0%})")
    lines.append("")
    lines.append("OTHER AGENTS:")

    for i in range(1, n_agents):
        if step.alive[i]:
            rel = step.positions[i] - my_pos
            dist = abs(rel[0]) + abs(rel[1])
            attn = attn_from_agent0[i] if i < len(attn_from_agent0) else 0

            # Social scores
            dmg = ledger[i, 0, 0]
            food = ledger[i, 0, 1]
            coop = ledger[i, 0, 2]
            defense = ledger[i, 0, 3]

            lines.append(f"  Agent {i}: dist={dist}, attn={attn:.0%}")
            lines.append(f"    Social: dmg={dmg:.0f} food={food:.0f} coop={coop:.0f} def={defense:.0f}")

    lines.append("")
    lines.append("FOOD:")
    rich_count = step.rich_food.sum()
    poor_count = step.poor_food.sum()
    lines.append(f"  Rich: {rich_count}, Poor: {poor_count}")

    ax_feat.text(0.02, 0.95, '\n'.join(lines), transform=ax_feat.transAxes,
                fontfamily='monospace', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))

    # === BOTTOM LEFT: Direction probs with influence ===
    ax_dir = fig.add_axes([0.02, 0.08, 0.28, 0.32])
    dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
    colors_dir = ['#e74c3c', '#e74c3c', '#3498db', '#3498db', '#95a5a6']
    bars_dir = ax_dir.bar(dir_names, dir_probs, color=colors_dir, edgecolor='black', linewidth=1.5)
    ax_dir.set_ylabel('Probability')
    ax_dir.set_title('Direction Probabilities', fontsize=11, fontweight='bold')
    ax_dir.set_ylim(0, 1)

    for bar, prob in zip(bars_dir, dir_probs):
        if prob > 0.01:
            ax_dir.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{prob:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    chosen_dir = step.actions[0][0] if step.actions and 0 in step.actions else -1
    if chosen_dir >= 0:
        bars_dir[chosen_dir].set_edgecolor('lime')
        bars_dir[chosen_dir].set_linewidth(4)

    # === BOTTOM MIDDLE: Action type probs ===
    ax_act = fig.add_axes([0.36, 0.08, 0.28, 0.32])
    act_names = ['MOVE', 'ATTACK', 'GIVE', 'SIGNAL', 'COOP']
    colors_act = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars_act = ax_act.bar(act_names, act_probs, color=colors_act, edgecolor='black', linewidth=1.5)
    ax_act.set_ylabel('Probability')
    ax_act.set_title('Action Type Probabilities', fontsize=11, fontweight='bold')
    ax_act.set_ylim(0, 1)

    for bar, prob in zip(bars_act, act_probs):
        if prob > 0.01:
            ax_act.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{prob:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    chosen_act = step.actions[0][1] if step.actions and 0 in step.actions else -1
    if chosen_act >= 0:
        bars_act[chosen_act].set_edgecolor('lime')
        bars_act[chosen_act].set_linewidth(4)

    # === BOTTOM RIGHT: Decision summary ===
    ax_summary = fig.add_axes([0.70, 0.08, 0.28, 0.32])
    ax_summary.axis('off')
    ax_summary.set_title('Decision Analysis', fontsize=11, fontweight='bold')

    summary = []
    summary.append("WHY THIS DECISION?")
    summary.append("=" * 40)
    summary.append("")

    # Find what agent is paying most attention to
    max_attn_idx = np.argmax(attn_from_agent0[1:n_agents]) + 1 if n_agents > 1 else -1
    if max_attn_idx > 0 and step.alive[max_attn_idx]:
        summary.append(f"Most attention on: Agent {max_attn_idx}")
        summary.append(f"  Attention weight: {attn_from_agent0[max_attn_idx]:.0%}")

        # Check relationship
        score = ledger[max_attn_idx, 0, 1] + ledger[max_attn_idx, 0, 2] + ledger[max_attn_idx, 0, 3] - ledger[max_attn_idx, 0, 0]
        if score > 20:
            summary.append(f"  Relationship: FRIEND (score={score:.0f})")
        elif score < -20:
            summary.append(f"  Relationship: ENEMY (score={score:.0f})")
        else:
            summary.append(f"  Relationship: NEUTRAL (score={score:.0f})")

    summary.append("")

    if chosen_act == 4:  # COOP
        summary.append("Action: COOPERATE")
        summary.append("  -> Likely near rich food with ally")
        summary.append(f"  -> COOP prob: {act_probs[4]:.0%}")
    elif chosen_act == 0:  # MOVE
        summary.append(f"Action: MOVE {dir_names[chosen_dir]}")
        summary.append("  -> Moving toward objective")
    elif chosen_act == 1:  # ATTACK
        summary.append(f"Action: ATTACK {dir_names[chosen_dir]}")
        summary.append("  -> Engaging target")

    summary.append("")
    summary.append(f"Value estimate: {value[0].item():.2f}")

    ax_summary.text(0.02, 0.95, '\n'.join(summary), transform=ax_summary.transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.savefig('agent_influence.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved agent_influence.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=str, help='Path to episode file')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--frame', '-f', type=int, default=64, help='Frame to visualize')
    args = parser.parse_args()

    visualize_influence(args.episode, args.checkpoint, args.frame)
