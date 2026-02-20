"""
Analyze how different token features influence attention scores.

Shows:
- Token features breakdown (position, type, HP, social)
- Correlation between distance and attention
- Ablation: attention with/without position info

Unified Playback Mode (--playback):
- Grid with agents, food labels, attention lines, relationship lines, action arrows
- Direction & action type probability matrices
- 4 ledger heatmaps (Damage, Food Given, Coop Count, Defense) with kill markers
- Attention scatter plot and distribution bars
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
import argparse

from core.config import Config
from agents.network import ActorCritic
from analysis.visualize import (
    classify_relationship, _get_target_from_direction,
    DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT, DIR_STAY,
    ACT_MOVE, ACT_ATTACK, ACT_GIVE, ACT_SIGNAL, ACT_COOPERATE,
    DIR_DELTAS
)


def load_episode(path: str):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data['steps'], data['ledger_snapshots'], data['config']


def load_model(checkpoint_path: str, config: Config, agent_id: int = 0):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ActorCritic(config)
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'network_state_dicts' in ckpt:
        state_dict = ckpt['network_state_dicts'][agent_id]
    else:
        raise KeyError(f"Checkpoint missing model state dict. Keys: {ckpt.keys()}")

    model.load_state_dict(state_dict, strict=False)
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

    if not step.alive[agent_id]:
        return None, None

    my_pos = step.positions[agent_id]

    # Store raw info for analysis
    entity_info = []

    # Fill agent tokens
    for i in range(n_agents):
        if step.alive[i]:
            rel_pos = (step.positions[i] - my_pos).astype(float)
            distance = abs(rel_pos[0]) + abs(rel_pos[1])
            rel_pos_norm = rel_pos / gs  # Normalized for Fourier encoding
            hp_norm = float(step.hp[i] / config.max_hp)
            social = ledger[i, agent_id, :] / 100.0 * 5.0

            # Fourier encode position
            fourier = fourier_encode(rel_pos_norm[0], rel_pos_norm[1])
            entity_tokens[i, 0:16] = torch.from_numpy(fourier)  # fourier[16]
            entity_tokens[i, 16:18] = 0.0                       # velocity[2]
            entity_tokens[i, 18:20] = torch.tensor([0.0, 1.0])  # type: [is_food, is_agent]
            entity_tokens[i, 20] = hp_norm                      # value
            entity_tokens[i, 21:25] = torch.from_numpy(social.astype(np.float32))  # social[4]
            entity_mask[i] = True

            entity_info.append({
                'idx': i,
                'label': f'A{i}',
                'type': 'agent',
                'distance': distance,
                'dx': float(rel_pos_norm[1]),  # For display (col direction)
                'dy': float(rel_pos_norm[0]),  # For display (row direction)
                'hp': hp_norm,
                'social': social.tolist(),
                'social_score': float(social[1] + social[2] + social[3] - social[0])  # food+coop+def-dmg
            })

    # Fill food tokens
    food_idx = n_agents
    poor_food_pos = list(zip(*np.where(step.poor_food)))
    rich_food_pos = list(zip(*np.where(step.rich_food)))

    def dist_to_agent(pos):
        return abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1])

    all_food = [(pos, 0.5) for pos in poor_food_pos] + [(pos, 1.0) for pos in rich_food_pos]
    all_food.sort(key=lambda x: dist_to_agent(x[0]))

    food_count = 0
    for (r, c), quality in all_food[:max_food]:
        if food_idx >= max_entities:
            break
        rel_pos = np.array([r - my_pos[0], c - my_pos[1]], dtype=float)
        distance = abs(rel_pos[0]) + abs(rel_pos[1])
        rel_pos_norm = rel_pos / gs

        # Fourier encode position
        fourier = fourier_encode(rel_pos_norm[0], rel_pos_norm[1])
        entity_tokens[food_idx, 0:16] = torch.from_numpy(fourier)  # fourier[16]
        entity_tokens[food_idx, 16:18] = 0.0                       # velocity[2]
        entity_tokens[food_idx, 18:20] = torch.tensor([1.0, 0.0])  # type: [is_food, is_agent]
        entity_tokens[food_idx, 20] = float(quality)               # value
        entity_tokens[food_idx, 21:25] = 0.0                       # social[4]
        entity_mask[food_idx] = True

        entity_info.append({
            'idx': food_idx,
            'label': f'F{food_count}',
            'type': 'food',
            'distance': distance,
            'dx': float(rel_pos_norm[1]),
            'dy': float(rel_pos_norm[0]),
            'quality': quality,
            'social': [0, 0, 0, 0],
            'social_score': 0
        })

        food_idx += 1
        food_count += 1

    signals = torch.tensor(step.signals, dtype=torch.float32)
    self_hp = torch.tensor([step.hp[agent_id] / config.max_hp])
    self_inventory = torch.tensor([0.0])
    agent_id_tensor = torch.tensor([agent_id])

    obs = {
        'entity_tokens': entity_tokens.unsqueeze(0),
        'entity_mask': entity_mask.unsqueeze(0),
        'signals': signals.unsqueeze(0),
        'self_hp': self_hp.unsqueeze(0),
        'self_inventory': self_inventory.unsqueeze(0),
        'agent_id': agent_id_tensor
    }

    return obs, entity_info


def get_attention_weights(model, obs):
    """Get attention weights from the model."""
    with torch.no_grad():
        tokens = model.encoder.entity_embed(obs['entity_tokens'])
        attn_mask = ~obs['entity_mask']
        _, attn_weights = model.encoder.entity_attention(
            tokens, tokens, tokens,
            key_padding_mask=attn_mask,
            average_attn_weights=True
        )
        return attn_weights[0].numpy()


def get_attention_ablated(model, obs, ablate_position=False, ablate_social=False, ablate_type=False, ablate_velocity=False, ablate_value=False):
    """Get attention with certain features zeroed out.

    Token format (25 features):
        - fourier[16]: indices 0-15
        - velocity[2]: indices 16-17
        - type_onehot[2]: indices 18-19
        - value[1]: index 20
        - social[4]: indices 21-24
    """
    tokens_modified = obs['entity_tokens'].clone()

    if ablate_position:
        tokens_modified[:, :, 0:16] = 0  # Zero fourier features
    if ablate_velocity:
        tokens_modified[:, :, 16:18] = 0  # Zero velocity
    if ablate_type:
        tokens_modified[:, :, 18:20] = 0.5  # Neutral type
    if ablate_value:
        tokens_modified[:, :, 20] = 0.5  # Neutral value (0.5 HP/Quality)
    if ablate_social:
        tokens_modified[:, :, 21:25] = 0  # Zero social channels

    obs_modified = {**obs, 'entity_tokens': tokens_modified}

    with torch.no_grad():
        tokens = model.encoder.entity_embed(obs_modified['entity_tokens'])
        attn_mask = ~obs_modified['entity_mask']
        _, attn_weights = model.encoder.entity_attention(
            tokens, tokens, tokens,
            key_padding_mask=attn_mask,
            average_attn_weights=True
        )
        return attn_weights[0].numpy()


def analyze_attention_features(episode_path: str, checkpoint_path: str, frame_idx: int = 64):
    """Analyze how token features influence attention (static frame)."""

    steps, ledger_snapshots, config = load_episode(episode_path)
    model = load_model(checkpoint_path, config, agent_id=0)

    step = steps[frame_idx]
    ledger = ledger_snapshots[frame_idx]

    obs, entity_info = reconstruct_observation(step, ledger, config, agent_id=0)
    if obs is None:
        print("Agent 0 is dead")
        return

    # Get attention weights
    attn_normal = get_attention_weights(model, obs)
    attn_no_pos = get_attention_ablated(model, obs, ablate_position=True)
    attn_no_social = get_attention_ablated(model, obs, ablate_social=True)
    attn_no_velocity = get_attention_ablated(model, obs, ablate_velocity=True)
    attn_no_value = get_attention_ablated(model, obs, ablate_value=True)

    # Extract attention from agent 0's perspective
    attn_from_a0 = attn_normal[0, :]
    attn_from_a0_no_pos = attn_no_pos[0, :]
    attn_from_a0_no_social = attn_no_social[0, :]
    attn_from_a0_no_velocity = attn_no_velocity[0, :]
    attn_from_a0_no_value = attn_no_value[0, :]

    # Filter to valid entities
    valid_mask = obs['entity_mask'][0].numpy()
    valid_info = [e for e in entity_info if valid_mask[e['idx']]]
    valid_attn = [attn_from_a0[e['idx']] for e in valid_info]
    valid_attn_no_pos = [attn_from_a0_no_pos[e['idx']] for e in valid_info]
    valid_attn_no_social = [attn_from_a0_no_social[e['idx']] for e in valid_info]
    valid_attn_no_velocity = [attn_from_a0_no_velocity[e['idx']] for e in valid_info]
    valid_attn_no_value = [attn_from_a0_no_value[e['idx']] for e in valid_info]

    # Create visualization
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'Attention Feature Analysis - Step {frame_idx}', fontsize=14, fontweight='bold')

    # === TOP LEFT: Token features table ===
    ax_table = fig.add_axes([0.02, 0.55, 0.45, 0.40])
    ax_table.axis('off')
    ax_table.set_title('Entity Token Features (what the network sees)', fontsize=12, fontweight='bold')

    # Create table data
    headers = ['Entity', 'dx', 'dy', 'Dist', 'Type', 'HP/Qual', 'Social', 'Attn']
    table_data = []
    for i, e in enumerate(valid_info):
        if e['type'] == 'agent':
            social_str = f"{e['social_score']:.1f}"
            hp_qual = f"{e['hp']:.0%}"
        else:
            social_str = "N/A"
            hp_qual = "Rich" if e.get('quality', 0) > 0.5 else "Poor"

        table_data.append([
            e['label'],
            f"{e['dx']:+.2f}",
            f"{e['dy']:+.2f}",
            f"{e['distance']:.0f}",
            e['type'].upper(),
            hp_qual,
            social_str,
            f"{valid_attn[i]:.1%}"
        ])

    table = ax_table.table(cellText=table_data, colLabels=headers,
                           loc='center', cellLoc='center',
                           colColours=['lightblue']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Highlight row with max attention
    max_attn_idx = np.argmax(valid_attn)
    for j in range(len(headers)):
        table[(max_attn_idx + 1, j)].set_facecolor('lightgreen')

    # === TOP RIGHT: Distance vs Attention scatter ===
    ax_scatter = fig.add_axes([0.55, 0.55, 0.40, 0.40])

    distances = [e['distance'] for e in valid_info]
    colors = ['#3498db' if e['type'] == 'agent' else '#f1c40f' for e in valid_info]

    ax_scatter.scatter(distances, valid_attn, c=colors, s=200, edgecolors='black', linewidth=2, alpha=0.8)

    # Add labels
    for i, e in enumerate(valid_info):
        ax_scatter.annotate(e['label'], (distances[i], valid_attn[i]),
                           textcoords="offset points", xytext=(5, 5), fontsize=10, fontweight='bold')

    # Fit line
    if len(distances) > 1:
        z = np.polyfit(distances, valid_attn, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(distances), max(distances), 100)
        ax_scatter.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend (slope={z[0]:.3f})')

        # Correlation
        corr = np.corrcoef(distances, valid_attn)[0, 1]
        ax_scatter.text(0.95, 0.95, f'Correlation: {corr:.2f}', transform=ax_scatter.transAxes,
                       ha='right', va='top', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_scatter.set_xlabel('Distance from Agent 0', fontsize=11)
    ax_scatter.set_ylabel('Attention Weight', fontsize=11)
    ax_scatter.set_title('Distance vs Attention\n(Does closer = more attention?)', fontsize=11, fontweight='bold')
    ax_scatter.legend(loc='upper right')
    ax_scatter.grid(True, alpha=0.3)

    # === BOTTOM LEFT: Ablation comparison ===
    ax_ablation = fig.add_axes([0.02, 0.08, 0.45, 0.40])

    x = np.arange(len(valid_info))
    width = 0.15

    bars1 = ax_ablation.bar(x - 2*width, valid_attn, width, label='Normal', color='#3498db', edgecolor='black')
    bars2 = ax_ablation.bar(x - width, valid_attn_no_pos, width, label='No Position', color='#e74c3c', edgecolor='black')
    bars3 = ax_ablation.bar(x, valid_attn_no_social, width, label='No Social', color='#2ecc71', edgecolor='black')
    bars4 = ax_ablation.bar(x + width, valid_attn_no_velocity, width, label='No Velocity', color='#9b59b6', edgecolor='black')
    bars5 = ax_ablation.bar(x + 2*width, valid_attn_no_value, width, label='No Value', color='#f1c40f', edgecolor='black')

    ax_ablation.set_xlabel('Entity')
    ax_ablation.set_ylabel('Attention Weight')
    ax_ablation.set_title('Ablation Study: Impact of removing features', fontsize=11, fontweight='bold')
    ax_ablation.set_xticks(x)
    ax_ablation.set_xticklabels([e['label'] for e in valid_info])
    ax_ablation.legend(loc='upper right', ncol=2, fontsize=9)
    # Calculate max height properly dealing with empty lists
    max_height = 0
    if valid_attn:
         max_height = max(max(valid_attn), 
                          max(valid_attn_no_pos), 
                          max(valid_attn_no_social),
                          max(valid_attn_no_velocity),
                          max(valid_attn_no_value))
    ax_ablation.set_ylim(0, max_height * 1.3)

    # === BOTTOM RIGHT: Feature importance summary ===
    ax_summary = fig.add_axes([0.55, 0.08, 0.40, 0.40])
    ax_summary.axis('off')
    ax_summary.set_title('Feature Influence Analysis', fontsize=12, fontweight='bold')

    # Calculate feature importance
    pos_influence = np.mean(np.abs(np.array(valid_attn) - np.array(valid_attn_no_pos)))
    social_influence = np.mean(np.abs(np.array(valid_attn) - np.array(valid_attn_no_social)))
    vel_influence = np.mean(np.abs(np.array(valid_attn) - np.array(valid_attn_no_velocity)))
    val_influence = np.mean(np.abs(np.array(valid_attn) - np.array(valid_attn_no_value)))

    # Distance correlation
    dist_corr = np.corrcoef(distances, valid_attn)[0, 1] if len(distances) > 1 else 0

    lines = []
    lines.append("FEATURE IMPORTANCE RANKING (Avg Attn Change):")
    lines.append("=" * 50)
    
    influences = [
        ('Position', pos_influence),
        ('Social', social_influence),
        ('Velocity', vel_influence),
        ('Value (HP/Qual)', val_influence)
    ]
    influences.sort(key=lambda x: x[1], reverse=True)

    for name, inf in influences:
        lines.append(f"{name:<15}: {inf:.4f}")

    lines.append("")
    lines.append("-" * 50)
    lines.append("INSIGHTS:")
    
    # Generate insights based on ranking
    top_feature = influences[0][0]
    lines.append(f"• Primary driver: {top_feature.upper()}")
    
    if abs(dist_corr) > 0.4:
        lines.append(f"• Distance dependent: {'YES' if dist_corr < 0 else 'YES (Inverse)'} (Corr: {dist_corr:.2f})")
    else:
        lines.append(f"• Distance dependent: NO (Corr: {dist_corr:.2f})")

    if social_influence > 0.01:
        lines.append("• Agent considers relationship history")
    
    if vel_influence > 0.01:
        lines.append("• Agent considers movement (velocity)")

    ax_summary.text(0.05, 0.95, '\n'.join(lines), transform=ax_summary.transAxes,
                   fontfamily='monospace', fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.savefig('attention_feature_analysis.png', dpi=150, bbox_inches='tight')
    print('Saved attention_feature_analysis.png')
    plt.show()


def playback_attention_features(episode_path: str, checkpoint_path: str, fps: int = 10):
    """Unified playback combining attention analysis with action/ledger visualization.

    Controls:
        SPACE - Pause/Resume
        LEFT/RIGHT - Skip backward/forward 1 frame (when paused)
        UP/DOWN - Skip backward/forward 10 frames (when paused)
        Q - Quit

    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Unified Playback - Step X                     │
    ├────────────────────┬────────────────────┬───────────────────────┤
    │                    │  Distance vs Attn  │   Attention Bars      │
    │     GRID VIEW      │    (scatter)       │   (horizontal)        │
    │  (with attention   ├────────────────────┼───────────────────────┤
    │   lines + actions) │  Dir Probs Matrix  │  Action Type Matrix   │
    ├────────────────────┴────────────────────┴───────────────────────┤
    │  Damage  │  Food Given  │  Coop Count  │  Defense  │   Info     │
    │  Heatmap │   Heatmap    │   Heatmap    │  Heatmap  │   Panel    │
    └──────────┴──────────────┴──────────────┴───────────┴────────────┘
    """
    from matplotlib.animation import FuncAnimation

    steps, ledger_snapshots, config = load_episode(episode_path)
    model = load_model(checkpoint_path, config, agent_id=0)
    n_frames = len(steps)
    gs = config.grid_size
    n_agents = config.n_agents
    agent_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    # Health colormap (green -> yellow -> red)
    colors_health = ['#ff0000', '#ff8800', '#ffff00', '#88ff00', '#00ff00']
    health_cmap = LinearSegmentedColormap.from_list('health', colors_health, N=256)

    # Check if we have action probabilities
    has_probs = (steps[0].direction_probs is not None and
                 steps[0].action_type_probs is not None)

    # Pre-compute cumulative rewards
    cumulative_rewards = np.zeros((n_frames, n_agents))
    for frame_idx in range(n_frames):
        if frame_idx > 0:
            cumulative_rewards[frame_idx] = cumulative_rewards[frame_idx - 1].copy()
        if steps[frame_idx].rewards:
            for agent_id, reward in steps[frame_idx].rewards.items():
                cumulative_rewards[frame_idx, agent_id] += reward

    # Pre-compute kills
    kill_matrix = np.zeros((n_agents, n_agents), dtype=bool)
    for frame_idx in range(1, n_frames):
        prev_alive = steps[frame_idx - 1].alive
        curr_alive = steps[frame_idx].alive
        for victim_id in range(n_agents):
            if prev_alive[victim_id] and not curr_alive[victim_id]:
                damage_to_victim = ledger_snapshots[frame_idx][:, victim_id, 0]
                if damage_to_victim.max() > 0:
                    killer_id = int(np.argmax(damage_to_victim))
                    kill_matrix[killer_id, victim_id] = True

    # Playback state
    state = {'paused': False, 'frame': 0}

    # Create figure with unified layout
    fig = plt.figure(figsize=(22, 14))

    # Top row: Grid (left), Scatter + Attention bars (right)
    ax_grid = fig.add_axes([0.02, 0.38, 0.32, 0.57])
    ax_scatter = fig.add_axes([0.36, 0.62, 0.30, 0.33])
    ax_attn_bars = fig.add_axes([0.68, 0.62, 0.30, 0.33])

    # Middle row: Direction probs + Action type probs (under scatter/bars)
    ax_dir_probs = fig.add_axes([0.36, 0.38, 0.30, 0.20])
    ax_act_probs = fig.add_axes([0.68, 0.38, 0.30, 0.20])

    # Bottom row: 4 ledger heatmaps + info panel
    ax_ledger = [
        fig.add_axes([0.02, 0.04, 0.17, 0.28]),   # Damage
        fig.add_axes([0.21, 0.04, 0.17, 0.28]),   # Food Given
        fig.add_axes([0.40, 0.04, 0.17, 0.28]),   # Coop Count
        fig.add_axes([0.59, 0.04, 0.17, 0.28]),   # Defense Score
    ]
    ax_info = fig.add_axes([0.78, 0.04, 0.20, 0.28])

    ledger_names = ['Damage Dealt', 'Food Given', 'Coop Count', 'Defense Score']
    ledger_cmaps = ['Reds', 'Greens', 'Blues', 'Purples']

    def on_key(event):
        if event.key == ' ':
            state['paused'] = not state['paused']
            if state['paused']:
                anim.event_source.stop()
                print(f"PAUSED at frame {state['frame']}")
            else:
                anim.event_source.start()
                print("RESUMED")
        elif event.key == 'right' and state['paused']:
            state['frame'] = min(state['frame'] + 1, n_frames - 1)
            update(state['frame'])
            fig.canvas.draw()
            print(f"Frame {state['frame']}")
        elif event.key == 'left' and state['paused']:
            state['frame'] = max(state['frame'] - 1, 0)
            update(state['frame'])
            fig.canvas.draw()
            print(f"Frame {state['frame']}")
        elif event.key == 'up' and state['paused']:
            state['frame'] = min(state['frame'] + 10, n_frames - 1)
            update(state['frame'])
            fig.canvas.draw()
            print(f"Frame {state['frame']}")
        elif event.key == 'down' and state['paused']:
            state['frame'] = max(state['frame'] - 10, 0)
            update(state['frame'])
            fig.canvas.draw()
            print(f"Frame {state['frame']}")
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame_idx):
        state['frame'] = frame_idx
        # Clear all axes
        ax_grid.clear()
        ax_scatter.clear()
        ax_attn_bars.clear()
        ax_dir_probs.clear()
        ax_act_probs.clear()
        ax_info.clear()
        for ax in ax_ledger:
            ax.clear()

        step = steps[frame_idx]
        ledger = ledger_snapshots[frame_idx]

        obs, entity_info = reconstruct_observation(step, ledger, config, agent_id=0)

        if obs is None:
            ax_grid.text(0.5, 0.5, 'Agent 0 DEAD', ha='center', va='center',
                        fontsize=20, color='red', transform=ax_grid.transAxes)
            fig.suptitle(f'Unified Playback - Step {frame_idx}/{n_frames}', fontsize=14, fontweight='bold')
            return

        # Get attention weights
        attn_normal = get_attention_weights(model, obs)
        attn_no_pos = get_attention_ablated(model, obs, ablate_position=True)
        attn_no_social = get_attention_ablated(model, obs, ablate_social=True)

        attn_from_a0 = attn_normal[0, :]
        attn_from_a0_no_pos = attn_no_pos[0, :]
        attn_from_a0_no_social = attn_no_social[0, :]

        valid_mask = obs['entity_mask'][0].numpy()
        valid_info = [e for e in entity_info if valid_mask[e['idx']]]
        valid_attn = [attn_from_a0[e['idx']] for e in valid_info]
        valid_attn_no_pos = [attn_from_a0_no_pos[e['idx']] for e in valid_info]
        valid_attn_no_social = [attn_from_a0_no_social[e['idx']] for e in valid_info]

        # ========== GRID VIEW ==========
        ax_grid.set_xlim(-0.5, gs - 0.5)
        ax_grid.set_ylim(-0.5, gs - 0.5)
        ax_grid.set_aspect('equal')
        ax_grid.invert_yaxis()
        ax_grid.set_title(f'Grid View - Step {frame_idx}', fontsize=11, fontweight='bold')

        # Grid lines
        for i in range(gs + 1):
            ax_grid.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)
            ax_grid.axvline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)

        # Draw food with labels
        poor_food_pos = list(zip(*np.where(step.poor_food)))
        rich_food_pos = list(zip(*np.where(step.rich_food)))
        my_pos = step.positions[0] if step.alive[0] else None

        # Build food label lookup from valid_info
        food_labels = {}
        for e in valid_info:
            if e['type'] == 'food' and my_pos is not None:
                abs_r = int(my_pos[0] + e['dy'] * gs)
                abs_c = int(my_pos[1] + e['dx'] * gs)
                food_labels[(abs_r, abs_c)] = (e['label'], attn_from_a0[e['idx']])

        for r, c in poor_food_pos:
            ax_grid.plot(c, r, 's', color='yellowgreen', markersize=12, alpha=0.6)
            if (r, c) in food_labels:
                label, attn = food_labels[(r, c)]
                ax_grid.text(c, r, label, ha='center', va='center', fontsize=7,
                            fontweight='bold', color='black')

        for r, c in rich_food_pos:
            ax_grid.plot(c, r, '*', color='gold', markersize=20)
            if (r, c) in food_labels:
                label, attn = food_labels[(r, c)]
                ax_grid.text(c, r + 0.35, label, ha='center', va='center', fontsize=8,
                            fontweight='bold', color='darkgoldenrod',
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

        # Draw relationship lines between agents (friend=green, enemy=red)
        for viewer_id in range(n_agents):
            if not step.alive[viewer_id]:
                continue
            viewer_row, viewer_col = step.positions[viewer_id]

            for target_id in range(viewer_id + 1, n_agents):  # Only draw once per pair
                if not step.alive[target_id]:
                    continue

                rel, strength = classify_relationship(ledger, viewer_id, target_id)
                target_row, target_col = step.positions[target_id]

                if rel == 'friend':
                    color = 'limegreen'
                    linewidth = min(1 + strength / 20, 4)
                    ax_grid.plot([viewer_col, target_col], [viewer_row, target_row],
                                color=color, linewidth=linewidth, alpha=0.5,
                                linestyle='-', zorder=1)
                elif rel == 'enemy':
                    color = 'red'
                    linewidth = min(1 + strength / 20, 4)
                    ax_grid.plot([viewer_col, target_col], [viewer_row, target_row],
                                color=color, linewidth=linewidth, alpha=0.5,
                                linestyle='-', zorder=1)

        # Draw agents with health-based coloring
        for i in range(n_agents):
            if step.alive[i]:
                r, c = step.positions[i]
                health_frac = step.hp[i] / config.max_hp
                color = health_cmap(max(0, min(1, health_frac)))

                # Find attention weight for this agent (from Agent 0's view)
                attn_weight = 0
                for e in valid_info:
                    if e['type'] == 'agent' and e['label'] == f'A{i}':
                        attn_weight = attn_from_a0[e['idx']]
                        break

                # Draw agent circle
                circle = Circle((c, r), 0.35, color=color, ec='black', linewidth=2, alpha=0.9)
                ax_grid.add_patch(circle)
                ax_grid.text(c, r, str(i), ha='center', va='center', fontsize=10,
                            fontweight='bold', color='black')

                # Draw attention line from Agent 0 (purple dashed)
                if i != 0 and my_pos is not None and attn_weight > 0.05:
                    ax_grid.plot([my_pos[1], c], [my_pos[0], r],
                               color='purple', linewidth=attn_weight * 8, alpha=0.4,
                               linestyle='--', zorder=2)

        # Draw action arrows (attack/give/cooperate)
        if step.actions:
            for agent_id, action_tuple in step.actions.items():
                if not step.alive[agent_id]:
                    continue

                direction, action_type = action_tuple
                agent_pos = step.positions[agent_id]
                agent_row, agent_col = int(agent_pos[0]), int(agent_pos[1])

                # Handle COOPERATE (non-directional)
                if action_type == ACT_COOPERATE:
                    coop_circle = Circle((agent_col, agent_row), 0.45,
                                        fill=False, color='blue', linewidth=3, alpha=0.8)
                    ax_grid.add_patch(coop_circle)
                    ax_grid.text(agent_col + 0.35, agent_row - 0.35, 'C',
                                fontsize=9, color='blue', fontweight='bold')
                    continue

                # Skip SIGNAL and MOVE
                if action_type in (ACT_SIGNAL, ACT_MOVE):
                    continue

                # Handle ATTACK and GIVE (directional)
                target_pos = _get_target_from_direction(agent_pos, direction, gs)
                if target_pos is None:
                    continue

                target_row, target_col = target_pos

                # Check if there's actually a living agent at target
                target_agent_id = None
                for other_id in range(n_agents):
                    if other_id == agent_id or not step.alive[other_id]:
                        continue
                    other_row, other_col = step.positions[other_id]
                    if int(other_row) == target_row and int(other_col) == target_col:
                        target_agent_id = other_id
                        break

                if target_agent_id is None:
                    continue

                if action_type == ACT_ATTACK:
                    ax_grid.annotate('',
                        xy=(target_col, target_row),
                        xytext=(agent_col, agent_row),
                        arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8))
                    ax_grid.text(target_col + 0.3, target_row + 0.3, '⚔',
                                fontsize=10, color='red', fontweight='bold')

                elif action_type == ACT_GIVE:
                    ax_grid.annotate('',
                        xy=(target_col, target_row),
                        xytext=(agent_col, agent_row),
                        arrowprops=dict(arrowstyle='->', color='limegreen', lw=3, alpha=0.8))
                    ax_grid.text(target_col + 0.3, target_row + 0.3, '♥',
                                fontsize=10, color='green', fontweight='bold')

        # ========== DISTANCE VS ATTENTION ==========
        distances = [e['distance'] for e in valid_info]
        colors_scatter = ['#3498db' if e['type'] == 'agent' else '#f1c40f' for e in valid_info]

        ax_scatter.scatter(distances, valid_attn, c=colors_scatter, s=150,
                          edgecolors='black', linewidth=2, alpha=0.8)
        for i, e in enumerate(valid_info):
            ax_scatter.annotate(e['label'], (distances[i], valid_attn[i]),
                               textcoords="offset points", xytext=(5, 5), fontsize=9)

        if len(distances) > 1 and len(set(distances)) > 1:
            corr = np.corrcoef(distances, valid_attn)[0, 1]
            ax_scatter.text(0.95, 0.95, f'Corr: {corr:.2f}', transform=ax_scatter.transAxes,
                           ha='right', va='top', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_scatter.set_xlabel('Distance')
        ax_scatter.set_ylabel('Attention')
        ax_scatter.set_title('Distance vs Attention', fontsize=10, fontweight='bold')
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.set_ylim(0, max(valid_attn) * 1.3 if valid_attn else 1)

        # ========== ATTENTION BAR CHART ==========
        if valid_info:
            labels = [e['label'] for e in valid_info]
            bar_colors = ['#3498db' if e['type'] == 'agent' else '#f1c40f' for e in valid_info]
            bars = ax_attn_bars.barh(labels, valid_attn, color=bar_colors, edgecolor='black')

            for bar, val in zip(bars, valid_attn):
                ax_attn_bars.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                 f'{val:.1%}', va='center', fontsize=8)

            ax_attn_bars.set_xlim(0, max(valid_attn) * 1.3 if valid_attn else 1)
            ax_attn_bars.set_xlabel('Attention Weight')
            ax_attn_bars.set_title('Attention Distribution', fontsize=10, fontweight='bold')

        # ========== DIRECTION PROBABILITY MATRIX ==========
        if has_probs and step.direction_probs is not None:
            dir_names = ['UP', 'DN', 'LT', 'RT', 'ST']
            dir_data = step.direction_probs.copy()

            # Gray out dead agents
            for i in range(n_agents):
                if not step.alive[i]:
                    dir_data[i, :] = 0

            im_dir = ax_dir_probs.imshow(dir_data, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            ax_dir_probs.set_title('Direction Probs', fontsize=10, fontweight='bold')
            ax_dir_probs.set_xticks(range(5))
            ax_dir_probs.set_xticklabels(dir_names, fontsize=8)
            ax_dir_probs.set_yticks(range(n_agents))
            ax_dir_probs.set_yticklabels([f'A{i}' for i in range(n_agents)], fontsize=8)

            for i in range(n_agents):
                for j in range(5):
                    val = dir_data[i, j]
                    color = 'white' if val > 0.5 else 'black'
                    ax_dir_probs.text(j, i, f'{val:.2f}', ha='center', va='center',
                                      fontsize=7, color=color)
        else:
            ax_dir_probs.text(0.5, 0.5, 'No direction probs', ha='center', va='center',
                             transform=ax_dir_probs.transAxes, fontsize=10)
            ax_dir_probs.set_title('Direction Probs', fontsize=10, fontweight='bold')

        # ========== ACTION TYPE PROBABILITY MATRIX ==========
        if has_probs and step.action_type_probs is not None:
            act_names = ['MOV', 'ATK', 'GIV', 'SIG', 'COP']
            act_data = step.action_type_probs.copy()

            for i in range(n_agents):
                if not step.alive[i]:
                    act_data[i, :] = 0

            im_act = ax_act_probs.imshow(act_data, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
            ax_act_probs.set_title('Action Type Probs', fontsize=10, fontweight='bold')
            ax_act_probs.set_xticks(range(5))
            ax_act_probs.set_xticklabels(act_names, fontsize=8)
            ax_act_probs.set_yticks(range(n_agents))
            ax_act_probs.set_yticklabels([f'A{i}' for i in range(n_agents)], fontsize=8)

            for i in range(n_agents):
                for j in range(5):
                    val = act_data[i, j]
                    color = 'white' if val > 0.5 else 'black'
                    ax_act_probs.text(j, i, f'{val:.2f}', ha='center', va='center',
                                      fontsize=7, color=color)
        else:
            ax_act_probs.text(0.5, 0.5, 'No action probs', ha='center', va='center',
                             transform=ax_act_probs.transAxes, fontsize=10)
            ax_act_probs.set_title('Action Type Probs', fontsize=10, fontweight='bold')

        # ========== LEDGER HEATMAPS ==========
        for idx, (ax, name, cmap) in enumerate(zip(ax_ledger, ledger_names, ledger_cmaps)):
            channel_data = ledger[:, :, idx]

            im = ax.imshow(channel_data, cmap=cmap, aspect='equal', vmin=0)
            ax.set_title(name, fontsize=9, fontweight='bold')
            ax.set_xlabel('Target', fontsize=8)
            ax.set_ylabel('Source', fontsize=8)
            ax.set_xticks(range(n_agents))
            ax.set_yticks(range(n_agents))
            ax.set_xticklabels([f'{i}' for i in range(n_agents)], fontsize=7)
            ax.set_yticklabels([f'{i}' for i in range(n_agents)], fontsize=7)

            # Add kill markers (X) on Damage Dealt heatmap
            if idx == 0:
                for killer_id in range(n_agents):
                    for victim_id in range(n_agents):
                        if kill_matrix[killer_id, victim_id]:
                            ax.plot(victim_id, killer_id, 'X', markersize=12,
                                   color='black', markeredgecolor='white', markeredgewidth=1.5)

        # ========== INFO PANEL ==========
        ax_info.axis('off')
        pos_influence = np.mean(np.abs(np.array(valid_attn) - np.array(valid_attn_no_pos))) if valid_attn else 0
        social_influence = np.mean(np.abs(np.array(valid_attn) - np.array(valid_attn_no_social))) if valid_attn else 0

        paused_str = " [PAUSED]" if state['paused'] else ""
        info_lines = [
            f"Step: {frame_idx}/{n_frames}{paused_str}",
            f"Entities visible: {len(valid_info)}",
            "",
        ]

        # Show per-agent info: cumulative reward, value estimate, aux values
        has_values = step.values is not None
        has_aux = step.aux_values is not None

        info_lines.append("Agent  CumR    V(s)   [Surv  Res   Soc]")
        info_lines.append("─" * 42)
        for i in range(n_agents):
            status = "X" if not step.alive[i] else " "
            cum_r = cumulative_rewards[frame_idx, i]

            # Value estimate
            v_str = f"{step.values[i]:+5.1f}" if has_values else "  N/A"

            # Auxiliary values
            if has_aux:
                v_surv = step.aux_values['survival'][i]
                v_res = step.aux_values['resource'][i]
                v_soc = step.aux_values['social'][i]
                aux_str = f"[{v_surv:+4.1f} {v_res:+4.1f} {v_soc:+4.1f}]"
            else:
                aux_str = ""

            info_lines.append(f"A{i}{status} {cum_r:+6.1f} {v_str} {aux_str}")

        info_lines.extend([
            "",
            f"Pos influence: {pos_influence:.3f}",
            f"Social influence: {social_influence:.3f}",
            "",
            "─── Controls ───",
            "SPACE: Pause/Resume",
            "←/→: ±1 | ↑/↓: ±10 | Q: Quit",
        ])
        ax_info.text(0.05, 0.98, '\n'.join(info_lines), transform=ax_info.transAxes,
                    fontfamily='monospace', fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        fig.suptitle(f'Unified Playback - {episode_path}', fontsize=12, fontweight='bold')

    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, repeat=True)

    plt.show()
    print(f"Playback complete ({n_frames} frames)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=str, help='Path to episode file')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--frame', '-f', type=int, default=64, help='Frame to analyze (static mode)')
    parser.add_argument('--playback', '-p', action='store_true', help='Run animated playback')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for playback')
    args = parser.parse_args()

    if args.playback:
        playback_attention_features(args.episode, args.checkpoint, args.fps)
    else:
        analyze_attention_features(args.episode, args.checkpoint, args.frame)
