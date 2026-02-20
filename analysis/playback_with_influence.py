"""
Animated playback showing Agent 0's decision influences over time.

Shows:
- Grid with attention-weighted connections
- Attention distribution (which entities agent focuses on)
- Action probabilities
- Decision analysis
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
import argparse

from core.config import Config
from agents.network import ActorCritic


def load_episode(path: str):
    """Load a saved episode."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data['steps'], data['ledger_snapshots'], data['config']


def load_model(checkpoint_path: str, config: Config, agent_id: int = 0):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ActorCritic(config)
    
    if 'network_state_dicts' in ckpt:
        model.load_state_dict(ckpt['network_state_dicts'][agent_id])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {ckpt.keys()}")
        
    model.eval()
    return model


def fourier_encode(dx, dy, n_bands=4):
    """Encode position using Fourier features (matching environment)."""
    bands = [2.0 ** i for i in range(n_bands)]
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

    Token format (config.entity_token_dim features):
        - fourier[4*bands]: fourier_bands freq bands × (sin_dx, cos_dx, sin_dy, cos_dy)
        - velocity[2]: (dv_x, dv_y)
        - type_onehot[2]: [is_food, is_agent]
        - value[1]: HP or quality
        - social[4]: [damage_dealt, food_given, coop_count, defense_score]
    """
    n_agents = config.n_agents
    gs = config.grid_size
    n_bands = config.fourier_bands
    fourier_dim = 4 * n_bands
    max_food = config.max_food_tokens
    max_entities = n_agents + max_food
    token_dim = config.entity_token_dim

    # Compute offsets: fourier | velocity | type | value | social
    o_vel = fourier_dim
    o_type = o_vel + 2
    o_val = o_type + 2
    o_social = o_val + 1

    entity_tokens = torch.zeros(max_entities, token_dim)
    entity_mask = torch.zeros(max_entities, dtype=torch.bool)

    if not step.alive[agent_id]:
        return None

    my_pos = step.positions[agent_id]

    # Fill agent tokens
    for i in range(n_agents):
        if step.alive[i]:
            rel_pos = (step.positions[i] - my_pos).astype(float)
            rel_pos_norm = rel_pos / gs
            hp_norm = float(step.hp[i] / config.max_hp)

            damage = ledger[i, agent_id, 0] / 100.0
            food = ledger[i, agent_id, 1] / 5.0
            coop = ledger[i, agent_id, 2] / 10.0
            defense = ledger[i, agent_id, 3] / 5.0
            social_features = torch.tensor([damage, food, coop, defense], dtype=torch.float32)

            fourier = fourier_encode(rel_pos_norm[0], rel_pos_norm[1], n_bands)
            entity_tokens[i, 0:fourier_dim] = torch.from_numpy(fourier)
            entity_tokens[i, o_vel:o_vel+2] = 0.0
            entity_tokens[i, o_type:o_type+2] = torch.tensor([0.0, 1.0])
            entity_tokens[i, o_val] = hp_norm
            entity_tokens[i, o_social:o_social+4] = social_features
            entity_mask[i] = True

    # Fill food tokens
    food_idx = n_agents
    poor_food_pos = list(zip(*np.where(step.poor_food)))
    rich_food_pos = list(zip(*np.where(step.rich_food)))

    def dist_to_agent(pos):
        return abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1])

    all_food = [(pos, 0.5) for pos in poor_food_pos] + [(pos, 1.0) for pos in rich_food_pos]
    all_food.sort(key=lambda x: dist_to_agent(x[0]))

    for (r, c), quality in all_food[:max_food]:
        if food_idx >= max_entities:
            break
        rel_pos = np.array([r - my_pos[0], c - my_pos[1]], dtype=float)
        rel_pos_norm = rel_pos / gs

        fourier = fourier_encode(rel_pos_norm[0], rel_pos_norm[1], n_bands)
        entity_tokens[food_idx, 0:fourier_dim] = torch.from_numpy(fourier)
        entity_tokens[food_idx, o_vel:o_vel+2] = 0.0
        entity_tokens[food_idx, o_type:o_type+2] = torch.tensor([1.0, 0.0])
        entity_tokens[food_idx, o_val] = float(quality)
        entity_tokens[food_idx, o_social:o_social+4] = 0.0
        entity_mask[food_idx] = True
        food_idx += 1

    signals = torch.tensor(step.signals, dtype=torch.float32)
    self_hp = torch.tensor([step.hp[agent_id] / config.max_hp])
    self_inventory = torch.tensor([0.0])
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
        return attn_weights[0].numpy()


def playback_with_influence(episode_path: str, checkpoint_path: str, speed: float = 1.0):
    """Animated playback showing decision influences."""

    steps, ledger_snapshots, config = load_episode(episode_path)
    model = load_model(checkpoint_path, config, agent_id=0)

    gs = config.grid_size
    n_agents = config.n_agents
    agent_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
    act_names = ['MOVE', 'ATTACK', 'GIVE', 'SIGNAL', 'COOP']

    # Create figure
    fig = plt.figure(figsize=(20, 11))
    fig.suptitle('Agent 0 Decision Influences', fontsize=14, fontweight='bold')

    ax_grid = fig.add_axes([0.02, 0.40, 0.32, 0.55])
    ax_attn = fig.add_axes([0.36, 0.55, 0.28, 0.40])
    ax_info = fig.add_axes([0.66, 0.55, 0.32, 0.40])
    ax_dir = fig.add_axes([0.02, 0.08, 0.28, 0.28])
    ax_act = fig.add_axes([0.36, 0.08, 0.28, 0.28])
    ax_summary = fig.add_axes([0.70, 0.08, 0.28, 0.28])

    # Pre-compute cumulative rewards per agent
    cumulative_rewards = np.zeros((len(steps), n_agents))
    step_rewards = np.zeros((len(steps), n_agents))
    for fi in range(len(steps)):
        if fi > 0:
            cumulative_rewards[fi] = cumulative_rewards[fi - 1].copy()
        if steps[fi].rewards:
            for aid, rew in steps[fi].rewards.items():
                cumulative_rewards[fi, aid] += rew
                step_rewards[fi, aid] = rew

    # Animation state
    is_paused = [False]
    current_frame = [0]

    def draw_frame(frame_idx):
        step = steps[frame_idx]
        ledger = ledger_snapshots[frame_idx]

        # Clear all axes
        for ax in [ax_grid, ax_attn, ax_info, ax_dir, ax_act, ax_summary]:
            ax.clear()

        # Check if agent 0 is alive
        if not step.alive[0]:
            ax_grid.text(0.5, 0.5, 'Agent 0 DEAD', ha='center', va='center',
                        fontsize=20, color='red', transform=ax_grid.transAxes)
            return

        my_pos = step.positions[0]

        # Reconstruct observation and get model outputs
        obs = reconstruct_observation(step, ledger, config, agent_id=0)
        attn_weights = get_attention_weights(model, obs)
        attn_from_agent0 = attn_weights[0, :]

        with torch.no_grad():
            result = model(obs)
            dir_logits, act_logits, value = result[0], result[1], result[2]
            dir_probs = torch.softmax(dir_logits, dim=-1)[0].numpy()
            act_probs = torch.softmax(act_logits, dim=-1)[0].numpy()

        # === GRID VIEW with attention lines ===
        ax_grid.set_xlim(-0.5, gs - 0.5)
        ax_grid.set_ylim(-0.5, gs - 0.5)
        ax_grid.set_aspect('equal')
        ax_grid.invert_yaxis()
        ax_grid.set_title(f'Step {frame_idx} - Attention to Entities', fontsize=11, fontweight='bold')

        for i in range(gs + 1):
            ax_grid.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.2)
            ax_grid.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.2)

        # Draw food
        for row in range(gs):
            for col in range(gs):
                if step.rich_food[row, col]:
                    ax_grid.plot(col, row, '*', color='gold', markersize=18, alpha=0.9)
                elif step.poor_food[row, col]:
                    ax_grid.plot(col, row, 'o', color='lightgreen', markersize=10, alpha=0.7)

        # Draw agents with attention lines
        for i in range(n_agents):
            if step.alive[i]:
                row, col = step.positions[i]
                color = agent_colors[i % len(agent_colors)]

                circle = Circle((col, row), 0.38, color=color, ec='black', linewidth=2, alpha=0.9)
                ax_grid.add_patch(circle)
                ax_grid.text(col, row, str(i), ha='center', va='center',
                            fontsize=11, fontweight='bold', color='white')

                # HP bar
                hp_frac = step.hp[i] / config.max_hp
                bar_w = 0.7
                ax_grid.add_patch(Rectangle((col - bar_w/2, row - 0.55), bar_w, 0.12, color='darkgray', alpha=0.5))
                hp_color = 'lime' if hp_frac > 0.5 else 'red'
                ax_grid.add_patch(Rectangle((col - bar_w/2, row - 0.55), bar_w * hp_frac, 0.12, color=hp_color))

                # Attention line from agent 0
                if i != 0 and obs['entity_mask'][0, i]:
                    attn = attn_from_agent0[i]
                    if attn > 0.05:
                        ax_grid.plot([my_pos[1], col], [my_pos[0], row],
                                   color=color, linewidth=2 + attn * 20, alpha=0.4 + attn * 0.5)
                        mid_x, mid_y = (my_pos[1] + col) / 2, (my_pos[0] + row) / 2
                        ax_grid.text(mid_x, mid_y, f'{attn:.0%}', fontsize=9, ha='center', va='center',
                                   color=color, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

        # === ATTENTION BAR CHART ===
        valid_mask = obs['entity_mask'][0].numpy()
        entity_labels = []
        for i in range(n_agents):
            if valid_mask[i]:
                entity_labels.append(f'A{i}')
        for i in range(n_agents, len(valid_mask)):
            if valid_mask[i]:
                entity_labels.append(f'F{i-n_agents}')

        valid_attn = attn_from_agent0[valid_mask]
        colors_attn = ['#3498db' if l.startswith('A') else '#f1c40f' for l in entity_labels]
        bars_attn = ax_attn.bar(range(len(valid_attn)), valid_attn, color=colors_attn, edgecolor='black')
        ax_attn.set_xticks(range(len(entity_labels)))
        ax_attn.set_xticklabels(entity_labels, fontsize=8)
        ax_attn.set_ylabel('Attention')
        ax_attn.set_title('Attention Distribution (A=Agent, F=Food)', fontsize=10, fontweight='bold')
        ax_attn.set_ylim(0, max(0.4, valid_attn.max() * 1.3))

        if len(valid_attn) > 0:
            max_idx = np.argmax(valid_attn)
            bars_attn[max_idx].set_edgecolor('lime')
            bars_attn[max_idx].set_linewidth(3)

        # === INFO PANEL ===
        ax_info.axis('off')
        ax_info.set_title('What Agent 0 Observes', fontsize=10, fontweight='bold')

        lines = []
        lines.append(f"SELF: HP={step.hp[0]:.0f}/{config.max_hp:.0f}")
        lines.append("")
        lines.append("OTHER AGENTS:")
        for i in range(1, n_agents):
            if step.alive[i]:
                rel = step.positions[i] - my_pos
                dist = abs(rel[0]) + abs(rel[1])
                attn = attn_from_agent0[i] if i < len(attn_from_agent0) else 0
                score = ledger[i, 0, 1] + ledger[i, 0, 2] + ledger[i, 0, 3] - ledger[i, 0, 0]
                rel_type = "FRIEND" if score > 20 else ("ENEMY" if score < -20 else "NEUTRAL")
                lines.append(f"  A{i}: dist={dist} attn={attn:.0%} [{rel_type}]")
        lines.append("")
        lines.append(f"FOOD: {step.rich_food.sum()} rich, {step.poor_food.sum()} poor")

        ax_info.text(0.05, 0.95, '\n'.join(lines), transform=ax_info.transAxes,
                    fontfamily='monospace', fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))

        # === DIRECTION PROBS ===
        colors_dir = ['#e74c3c', '#e74c3c', '#3498db', '#3498db', '#95a5a6']
        bars_dir = ax_dir.bar(dir_names, dir_probs, color=colors_dir, edgecolor='black', linewidth=1.5)
        ax_dir.set_ylabel('Probability')
        ax_dir.set_title('Direction', fontsize=10, fontweight='bold')
        ax_dir.set_ylim(0, 1)

        for bar, prob in zip(bars_dir, dir_probs):
            if prob > 0.05:
                ax_dir.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{prob:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        chosen_dir = step.actions[0][0] if step.actions and 0 in step.actions else -1
        if chosen_dir >= 0:
            bars_dir[chosen_dir].set_edgecolor('lime')
            bars_dir[chosen_dir].set_linewidth(4)

        # === ACTION TYPE PROBS ===
        colors_act = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        bars_act = ax_act.bar(act_names, act_probs, color=colors_act, edgecolor='black', linewidth=1.5)
        ax_act.set_ylabel('Probability')
        ax_act.set_title('Action Type', fontsize=10, fontweight='bold')
        ax_act.set_ylim(0, 1)

        for bar, prob in zip(bars_act, act_probs):
            if prob > 0.05:
                ax_act.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{prob:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        chosen_act = step.actions[0][1] if step.actions and 0 in step.actions else -1
        if chosen_act >= 0:
            bars_act[chosen_act].set_edgecolor('lime')
            bars_act[chosen_act].set_linewidth(4)

        # === REWARD PLOT ===
        ax_summary.set_title('Cumulative Rewards', fontsize=10, fontweight='bold')
        for aid in range(n_agents):
            if any(steps[f].alive[aid] for f in range(min(frame_idx + 1, len(steps)))):
                color = agent_colors[aid % len(agent_colors)]
                ax_summary.plot(range(frame_idx + 1), cumulative_rewards[:frame_idx + 1, aid],
                               color=color, linewidth=1.5, alpha=0.8, label=f'A{aid}')
        ax_summary.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax_summary.axvline(x=frame_idx, color='black', linestyle=':', alpha=0.3)
        ax_summary.set_xlim(0, len(steps) - 1)
        ax_summary.set_xlabel('Step', fontsize=8)
        ax_summary.set_ylabel('Cumulative Reward', fontsize=8)
        ax_summary.legend(fontsize=7, loc='upper left', ncol=2)
        ax_summary.grid(True, alpha=0.2)

        # Add chosen action + value as text annotation
        action_text = ""
        if chosen_dir >= 0 and chosen_act >= 0:
            action_text = f"{dir_names[chosen_dir]}+{act_names[chosen_act]}  V={value[0].item():.1f}"
            r0 = step_rewards[frame_idx, 0]
            action_text += f"  r={r0:+.2f}"
        ax_summary.text(0.98, 0.02, action_text, transform=ax_summary.transAxes,
                       fontfamily='monospace', fontsize=8, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        fig.canvas.draw_idle()

    # Animation
    interval = int(300 / speed)

    def animate(frame_idx):
        current_frame[0] = frame_idx
        draw_frame(frame_idx)
        return []

    anim = animation.FuncAnimation(fig, animate, frames=len(steps),
                                   interval=interval, blit=False, repeat=True)

    # Controls
    def on_key(event):
        if event.key == ' ':
            if is_paused[0]:
                anim.resume()
                is_paused[0] = False
            else:
                anim.pause()
                is_paused[0] = True
        elif event.key == 'left' and is_paused[0]:
            new_frame = max(0, current_frame[0] - 1)
            draw_frame(new_frame)
            current_frame[0] = new_frame
        elif event.key == 'right' and is_paused[0]:
            new_frame = min(len(steps) - 1, current_frame[0] + 1)
            draw_frame(new_frame)
            current_frame[0] = new_frame

    fig.canvas.mpl_connect('key_press_event', on_key)

    ax_pause = fig.add_axes([0.45, 0.01, 0.1, 0.03])
    btn_pause = Button(ax_pause, 'Pause')

    def toggle_pause(event):
        if is_paused[0]:
            anim.resume()
            is_paused[0] = False
            btn_pause.label.set_text('Pause')
        else:
            anim.pause()
            is_paused[0] = True
            btn_pause.label.set_text('Resume')

    btn_pause.on_clicked(toggle_pause)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=str, help='Path to episode file')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--speed', type=float, default=0.5, help='Playback speed')
    args = parser.parse_args()

    playback_with_influence(args.episode, args.checkpoint, args.speed)
