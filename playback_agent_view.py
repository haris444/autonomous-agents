"""
Enhanced episode playback showing Agent 0's observation of other agents.

Shows:
- Grid view with agents and food
- Agent 0's egocentric view (relative positions)
- What Agent 0 observes about each other agent (position, HP, social history)
- Social ledger evolution
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, Slider
from typing import List, Optional
import argparse


def load_episode(path: str):
    """Load a saved episode."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data['steps'], data['ledger_snapshots'], data['config']


def playback_with_agent_view(episode_path: str, speed: float = 1.0, save_gif: bool = False):
    """
    Playback episode with Agent 0's observation details.

    Shows what Agent 0 "sees" about other agents at each timestep.
    """
    steps, ledger_snapshots, config = load_episode(episode_path)

    gs = config.grid_size
    n_agents = config.n_agents

    # Health colormap
    colors_health = ['#ff0000', '#ff8800', '#ffff00', '#88ff00', '#00ff00']
    health_cmap = LinearSegmentedColormap.from_list('health', colors_health, N=256)

    # Agent colors
    agent_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12',
                    '#1abc9c', '#e91e63', '#795548']

    # Create figure with panels
    fig = plt.figure(figsize=(20, 11))
    fig.suptitle('Agent 0 Observation Playback', fontsize=14, fontweight='bold')

    # Layout:
    # [Grid View] [Egocentric View] [Observation Details]
    # [Social Ledger Heatmap] [HP Over Time] [Social Score Evolution]

    ax_grid = fig.add_axes([0.02, 0.40, 0.30, 0.55])
    ax_ego = fig.add_axes([0.34, 0.40, 0.30, 0.55])
    ax_obs = fig.add_axes([0.66, 0.40, 0.32, 0.55])
    ax_ledger = fig.add_axes([0.02, 0.08, 0.28, 0.28])
    ax_hp = fig.add_axes([0.36, 0.08, 0.28, 0.28])
    ax_social = fig.add_axes([0.70, 0.08, 0.28, 0.28])

    # Pre-compute data for plots
    hp_history = np.array([s.hp for s in steps])

    # Social score evolution (what each agent's score is toward agent 0)
    social_scores = np.zeros((len(steps), n_agents))
    for t, ledger in enumerate(ledger_snapshots):
        for i in range(n_agents):
            if i != 0:
                # Score = food_given + coop + defense - damage
                social_scores[t, i] = (ledger[i, 0, 1] + ledger[i, 0, 2] +
                                       ledger[i, 0, 3] - ledger[i, 0, 0])

    # Animation state
    current_frame = [0]
    is_paused = [False]

    def draw_frame(frame_idx):
        step = steps[frame_idx]
        ledger = ledger_snapshots[frame_idx]

        # === Grid View ===
        ax_grid.clear()
        ax_grid.set_xlim(-0.5, gs - 0.5)
        ax_grid.set_ylim(-0.5, gs - 0.5)
        ax_grid.set_aspect('equal')
        ax_grid.invert_yaxis()
        ax_grid.set_title(f'Grid View (Step {frame_idx})', fontsize=11, fontweight='bold')

        # Grid lines
        for i in range(gs + 1):
            ax_grid.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            ax_grid.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        # Draw food
        for row in range(gs):
            for col in range(gs):
                if step.poor_food[row, col]:
                    ax_grid.plot(col, row, 'o', color='lightgreen', markersize=10, alpha=0.7)
                elif step.rich_food[row, col]:
                    ax_grid.plot(col, row, '*', color='gold', markersize=16, alpha=0.9)

        # Draw agents
        for i in range(n_agents):
            if step.alive[i]:
                row, col = step.positions[i]
                hp_frac = step.hp[i] / config.max_hp
                color = agent_colors[i % len(agent_colors)]

                # Agent circle
                circle = plt.Circle((col, row), 0.38, color=color, ec='black', linewidth=2, alpha=0.9)
                ax_grid.add_patch(circle)
                ax_grid.text(col, row, str(i), ha='center', va='center',
                           fontsize=11, fontweight='bold', color='white')

                # HP bar above
                bar_width = 0.7
                ax_grid.add_patch(plt.Rectangle((col - bar_width/2, row - 0.55),
                                                 bar_width, 0.12, color='darkgray', alpha=0.5))
                hp_color = 'lime' if hp_frac > 0.5 else ('yellow' if hp_frac > 0.25 else 'red')
                ax_grid.add_patch(plt.Rectangle((col - bar_width/2, row - 0.55),
                                                 bar_width * hp_frac, 0.12, color=hp_color))

                # Signal indicator
                if step.signals[i]:
                    ax_grid.add_patch(plt.Circle((col, row), 0.5, fill=False,
                                                  color='yellow', linewidth=3))

        # === Egocentric View (Agent 0's perspective) ===
        ax_ego.clear()
        ax_ego.set_xlim(-gs, gs)
        ax_ego.set_ylim(-gs, gs)
        ax_ego.set_aspect('equal')
        ax_ego.invert_yaxis()
        ax_ego.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax_ego.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
        ax_ego.set_title('Agent 0\'s Egocentric View', fontsize=11, fontweight='bold')
        ax_ego.set_xlabel('Δ Column')
        ax_ego.set_ylabel('Δ Row')

        if step.alive[0]:
            pos_0 = step.positions[0]

            # Draw Agent 0 at center
            ax_ego.add_patch(plt.Circle((0, 0), 0.45, color=agent_colors[0],
                                         ec='black', linewidth=2))
            ax_ego.text(0, 0, '0', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')

            # Draw other agents relative to Agent 0
            for i in range(1, n_agents):
                if step.alive[i]:
                    rel = step.positions[i] - pos_0
                    dr, dc = rel[0], rel[1]
                    dist = abs(dr) + abs(dc)
                    color = agent_colors[i % len(agent_colors)]

                    ax_ego.add_patch(plt.Circle((dc, dr), 0.38, color=color,
                                                 ec='black', linewidth=2, alpha=0.9))
                    ax_ego.text(dc, dr, str(i), ha='center', va='center',
                               fontsize=11, fontweight='bold', color='white')

                    # Distance label
                    ax_ego.annotate(f'd={dist}', (dc, dr), textcoords="offset points",
                                   xytext=(12, 12), fontsize=9, color=color, fontweight='bold')

                    # Draw line to agent
                    ax_ego.plot([0, dc], [0, dr], color=color, linewidth=1, alpha=0.4, linestyle='--')

            # Draw food relative to Agent 0
            for row in range(gs):
                for col in range(gs):
                    if step.poor_food[row, col]:
                        ax_ego.plot(col - pos_0[1], row - pos_0[0], 'o',
                                   color='lightgreen', markersize=6, alpha=0.6)
                    elif step.rich_food[row, col]:
                        ax_ego.plot(col - pos_0[1], row - pos_0[0], '*',
                                   color='gold', markersize=10, alpha=0.8)
        else:
            ax_ego.text(0, 0, 'Agent 0\nDEAD', ha='center', va='center',
                       fontsize=16, color='red', fontweight='bold')

        # === Observation Details ===
        ax_obs.clear()
        ax_obs.axis('off')
        ax_obs.set_title('Agent 0\'s Observation Tokens', fontsize=11, fontweight='bold')

        lines = []
        lines.append(f"Step: {frame_idx}/{len(steps)-1}")
        lines.append(f"Grid: {gs}×{gs}, Agents: {n_agents}")
        lines.append("=" * 55)

        if step.alive[0]:
            pos_0 = step.positions[0]
            hp_0 = step.hp[0]
            lines.append(f"SELF (Agent 0): pos=({pos_0[0]}, {pos_0[1]}) HP={hp_0:.0f}/{config.max_hp:.0f}")
            lines.append("")
            lines.append("OTHER AGENTS (what Agent 0 observes):")
            lines.append("-" * 55)

            for i in range(1, n_agents):
                if step.alive[i]:
                    rel = step.positions[i] - pos_0
                    dist = abs(rel[0]) + abs(rel[1])
                    hp_pct = step.hp[i] / config.max_hp * 100

                    # Social history from ledger
                    dmg = ledger[i, 0, 0]
                    food = ledger[i, 0, 1]
                    coop = ledger[i, 0, 2]
                    defense = ledger[i, 0, 3]
                    score = food + coop + defense - dmg

                    # Relationship indicator
                    if score > 20:
                        rel_str = "FRIEND 🟢"
                    elif score < -20:
                        rel_str = "ENEMY 🔴"
                    else:
                        rel_str = "NEUTRAL ⚪"

                    lines.append(f"Agent {i}: {rel_str}")
                    lines.append(f"  Position: Δ=({rel[0]:+d}, {rel[1]:+d}) dist={dist}")
                    lines.append(f"  Health: {hp_pct:.0f}%")
                    lines.append(f"  Social: dmg={dmg:.1f} food={food:.1f} coop={coop:.1f} def={defense:.1f}")
                    lines.append(f"  Score: {score:+.1f}")
                    lines.append("")
                else:
                    lines.append(f"Agent {i}: DEAD (masked)")
                    lines.append("")
        else:
            lines.append("Agent 0 is DEAD - no observations")

        # Actions
        if step.actions:
            lines.append("-" * 55)
            lines.append("ACTIONS THIS STEP:")
            dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
            act_names = ['MOVE', 'ATTACK', 'GIVE', 'SIGNAL', 'COOP']
            for i, (d, a) in step.actions.items():
                if step.alive[i]:
                    lines.append(f"  Agent {i}: {dir_names[d]:5} + {act_names[a]}")

        ax_obs.text(0.02, 0.98, '\n'.join(lines), transform=ax_obs.transAxes,
                   fontfamily='monospace', fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # === Social Ledger Heatmap ===
        ax_ledger.clear()
        social_score_matrix = (ledger[:, :, 1] + ledger[:, :, 2] +
                               ledger[:, :, 3] - ledger[:, :, 0])
        np.fill_diagonal(social_score_matrix, np.nan)

        vmax = max(50, np.nanmax(np.abs(social_score_matrix)))
        im = ax_ledger.imshow(social_score_matrix, cmap='RdYlGn', aspect='equal',
                              vmin=-vmax, vmax=vmax)
        ax_ledger.set_title('Social Ledger', fontsize=10, fontweight='bold')
        ax_ledger.set_xlabel('Target')
        ax_ledger.set_ylabel('Source')
        ax_ledger.set_xticks(range(n_agents))
        ax_ledger.set_yticks(range(n_agents))

        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    val = social_score_matrix[i, j]
                    color = 'white' if abs(val) > vmax/2 else 'black'
                    ax_ledger.text(j, i, f'{val:.0f}', ha='center', va='center',
                                  color=color, fontsize=7)

        # === HP Over Time ===
        ax_hp.clear()
        for i in range(n_agents):
            ax_hp.plot(hp_history[:frame_idx+1, i], color=agent_colors[i % len(agent_colors)],
                      label=f'A{i}', alpha=0.8, linewidth=1.5)
        ax_hp.axvline(x=frame_idx, color='black', linestyle='--', alpha=0.5)
        ax_hp.set_xlabel('Step')
        ax_hp.set_ylabel('HP')
        ax_hp.set_title('HP Over Time', fontsize=10, fontweight='bold')
        ax_hp.legend(loc='upper right', fontsize=7, ncol=2)
        ax_hp.set_xlim(0, len(steps))
        ax_hp.set_ylim(0, config.max_hp * 1.1)
        ax_hp.grid(True, alpha=0.3)

        # === Social Score Evolution (toward Agent 0) ===
        ax_social.clear()
        for i in range(1, n_agents):
            ax_social.plot(social_scores[:frame_idx+1, i], color=agent_colors[i % len(agent_colors)],
                          label=f'A{i}→A0', alpha=0.8, linewidth=1.5)
        ax_social.axvline(x=frame_idx, color='black', linestyle='--', alpha=0.5)
        ax_social.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax_social.set_xlabel('Step')
        ax_social.set_ylabel('Social Score')
        ax_social.set_title('Social Score → Agent 0', fontsize=10, fontweight='bold')
        ax_social.legend(loc='upper left', fontsize=7)
        ax_social.set_xlim(0, len(steps))
        ax_social.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    # Animation
    interval = int(200 / speed)  # ms between frames

    def animate(frame_idx):
        current_frame[0] = frame_idx
        draw_frame(frame_idx)
        return []

    anim = animation.FuncAnimation(fig, animate, frames=len(steps),
                                   interval=interval, blit=False, repeat=True)

    # Pause/Resume controls
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

    # Pause button
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

    if save_gif:
        print("Saving GIF...")
        anim.save('episode_agent_view.gif', writer='pillow', fps=5)
        print("Saved to episode_agent_view.gif")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Playback episode with Agent 0 view')
    parser.add_argument('episode', type=str, help='Path to episode .pt file')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed')
    parser.add_argument('--save-gif', action='store_true', help='Save as GIF')
    args = parser.parse_args()

    playback_with_agent_view(args.episode, speed=args.speed, save_gif=args.save_gif)
