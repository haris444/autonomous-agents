"""
Visualize what an agent sees about other agents from a saved episode.

Shows:
1. Grid view with agent positions at a specific timestep
2. Agent 0's observation of each other agent
3. Social ledger evolution over time
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import argparse


def load_episode(path: str):
    """Load a saved episode."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data['steps'], data['ledger_snapshots'], data['config']


def visualize_timestep(steps, ledger_snapshots, config, timestep: int = 0):
    """Visualize a single timestep from the episode."""
    step = steps[timestep]
    ledger = ledger_snapshots[timestep]

    gs = config.grid_size
    n_agents = config.n_agents

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Episode View - Timestep {timestep}/{len(steps)-1}', fontsize=14, fontweight='bold')

    # === SUBPLOT 1: Grid view ===
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_xlim(-0.5, gs - 0.5)
    ax1.set_ylim(-0.5, gs - 0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Grid View (Step {timestep})', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.invert_yaxis()

    # Draw poor food (green dots)
    poor_pos = np.where(step.poor_food)
    for r, c in zip(poor_pos[0], poor_pos[1]):
        ax1.add_patch(plt.Circle((c, r), 0.2, color='lightgreen', alpha=0.8))

    # Draw rich food (gold stars)
    rich_pos = np.where(step.rich_food)
    for r, c in zip(rich_pos[0], rich_pos[1]):
        ax1.add_patch(plt.Circle((c, r), 0.3, color='gold', alpha=0.8))
        ax1.text(c, r, '★', ha='center', va='center', fontsize=8, color='darkgoldenrod')

    # Draw agents
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    for i in range(n_agents):
        if step.alive[i]:
            r, c = step.positions[i]
            color = colors[i % len(colors)]
            hp_frac = step.hp[i] / config.max_hp

            # Agent circle
            circle = plt.Circle((c, r), 0.35, color=color, alpha=0.8)
            ax1.add_patch(circle)
            ax1.text(c, r, str(i), ha='center', va='center', color='white', fontweight='bold', fontsize=10)

            # HP bar
            bar_width = 0.6
            ax1.add_patch(plt.Rectangle((c - bar_width/2, r - 0.55), bar_width, 0.12,
                                         color='darkgray', alpha=0.5))
            ax1.add_patch(plt.Rectangle((c - bar_width/2, r - 0.55), bar_width * hp_frac, 0.12,
                                         color='lime' if hp_frac > 0.5 else 'red'))

            # Signal indicator
            if step.signals[i]:
                ax1.add_patch(plt.Circle((c, r), 0.45, fill=False, color='yellow', linewidth=2))

    # === SUBPLOT 2: Agent 0's egocentric view ===
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_xlim(-gs, gs)
    ax2.set_ylim(-gs, gs)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_title('Agent 0\'s Egocentric View', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Δ Column')
    ax2.set_ylabel('Δ Row')
    ax2.invert_yaxis()

    if step.alive[0]:
        # Draw agent 0 at center
        ax2.add_patch(plt.Circle((0, 0), 0.4, color='blue', alpha=0.9))
        ax2.text(0, 0, '0', ha='center', va='center', color='white', fontweight='bold', fontsize=12)

        pos_0 = step.positions[0]

        # Draw other agents relative to agent 0
        for i in range(1, n_agents):
            if step.alive[i]:
                rel_pos = step.positions[i] - pos_0
                dr, dc = rel_pos[0], rel_pos[1]
                color = colors[i % len(colors)]

                ax2.add_patch(plt.Circle((dc, dr), 0.35, color=color, alpha=0.8))
                ax2.text(dc, dr, str(i), ha='center', va='center', color='white', fontweight='bold')

                # Distance and direction
                dist = abs(dr) + abs(dc)
                ax2.annotate(f'd={int(dist)}', (dc, dr), textcoords="offset points",
                            xytext=(8, 8), fontsize=8, color=color)

        # Draw food relative to agent 0
        for r, c in zip(poor_pos[0], poor_pos[1]):
            ax2.add_patch(plt.Circle((c - pos_0[1], r - pos_0[0]), 0.15, color='lightgreen', alpha=0.7))
        for r, c in zip(rich_pos[0], rich_pos[1]):
            ax2.add_patch(plt.Circle((c - pos_0[1], r - pos_0[0]), 0.25, color='gold', alpha=0.7))
    else:
        ax2.text(0, 0, 'Agent 0 DEAD', ha='center', va='center', fontsize=14, color='red')

    # === SUBPLOT 3: Observation token details ===
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis('off')
    ax3.set_title('Agent 0 Observation Details', fontsize=11, fontweight='bold')

    lines = []
    lines.append(f"Timestep: {timestep}")
    lines.append(f"Grid: {gs}x{gs}, Agents: {n_agents}")
    lines.append("-" * 50)

    if step.alive[0]:
        pos_0 = step.positions[0]
        lines.append(f"Agent 0: pos=({pos_0[0]}, {pos_0[1]}) HP={step.hp[0]:.0f}")
        lines.append("")
        lines.append("Other agents visible to Agent 0:")

        for i in range(1, n_agents):
            if step.alive[i]:
                rel = step.positions[i] - pos_0
                dist = abs(rel[0]) + abs(rel[1])
                hp_pct = step.hp[i] / config.max_hp * 100

                # Social channels from ledger: [damage_dealt, food_given, coop_count, defense]
                social = ledger[i, 0, :]  # What agent i has done TO agent 0

                lines.append(f"  Agent {i}: Δ=({rel[0]:+d}, {rel[1]:+d}) dist={dist} HP={hp_pct:.0f}%")
                lines.append(f"    Social: dmg={social[0]:.1f} food={social[1]:.1f} coop={social[2]:.1f} def={social[3]:.1f}")
            else:
                lines.append(f"  Agent {i}: DEAD")
    else:
        lines.append("Agent 0 is DEAD")

    lines.append("")
    lines.append("-" * 50)
    lines.append("Actions this step:")
    for i, (direction, action_type) in step.actions.items():
        dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        act_names = ['MOVE', 'ATTACK', 'GIVE', 'SIGNAL', 'COOP']
        if step.alive[i]:
            lines.append(f"  Agent {i}: {dir_names[direction]} + {act_names[action_type]}")

    ax3.text(0.02, 0.98, '\n'.join(lines), transform=ax3.transAxes,
             fontfamily='monospace', fontsize=9, verticalalignment='top')

    # === SUBPLOT 4: Social ledger heatmap ===
    ax4 = fig.add_subplot(2, 3, 4)

    # Combined social score: positive = friend, negative = foe
    social_score = (ledger[:, :, 1] + ledger[:, :, 2] + ledger[:, :, 3] - ledger[:, :, 0])
    np.fill_diagonal(social_score, np.nan)

    vmax = max(50, np.nanmax(np.abs(social_score)))
    im = ax4.imshow(social_score, cmap='RdYlGn', aspect='equal', vmin=-vmax, vmax=vmax)
    ax4.set_title('Social Ledger (Green=Friend)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Target Agent')
    ax4.set_ylabel('Source Agent')
    ax4.set_xticks(range(n_agents))
    ax4.set_yticks(range(n_agents))

    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                val = social_score[i, j]
                color = 'white' if abs(val) > vmax/2 else 'black'
                ax4.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax4, label='Social Score')

    # === SUBPLOT 5: HP over time ===
    ax5 = fig.add_subplot(2, 3, 5)

    hp_history = np.array([s.hp for s in steps[:timestep+1]])
    for i in range(n_agents):
        ax5.plot(hp_history[:, i], color=colors[i % len(colors)], label=f'Agent {i}', alpha=0.8)

    ax5.axvline(x=timestep, color='black', linestyle='--', alpha=0.5, label='Current')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('HP')
    ax5.set_title('HP Over Time', fontsize=11, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.set_xlim(0, len(steps))
    ax5.grid(True, alpha=0.3)

    # === SUBPLOT 6: Damage dealt evolution ===
    ax6 = fig.add_subplot(2, 3, 6)

    # Track damage from each agent to agent 0 over time
    damage_to_0 = np.array([snap[:, 0, 0] for snap in ledger_snapshots[:timestep+1]])  # [time, n_agents]

    for i in range(1, n_agents):
        ax6.plot(damage_to_0[:, i], color=colors[i % len(colors)], label=f'From Agent {i}', alpha=0.8)

    ax6.axvline(x=timestep, color='black', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Cumulative Damage')
    ax6.set_title('Damage Dealt TO Agent 0', fontsize=11, fontweight='bold')
    ax6.legend(loc='upper left', fontsize=8)
    ax6.set_xlim(0, len(steps))
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_animation_frames(episode_path: str, output_dir: str = 'episode_frames'):
    """Create frames for all timesteps."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    steps, ledger_snapshots, config = load_episode(episode_path)

    for t in range(len(steps)):
        fig = visualize_timestep(steps, ledger_snapshots, config, t)
        fig.savefig(f'{output_dir}/frame_{t:03d}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        if t % 10 == 0:
            print(f'Saved frame {t}/{len(steps)-1}')

    print(f'\nFrames saved to {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize agent view from saved episode')
    parser.add_argument('episode', type=str, help='Path to episode .pt file')
    parser.add_argument('--timestep', '-t', type=int, default=0, help='Timestep to visualize')
    parser.add_argument('--animate', action='store_true', help='Create frames for all timesteps')
    args = parser.parse_args()

    if args.animate:
        create_animation_frames(args.episode)
    else:
        steps, ledger_snapshots, config = load_episode(args.episode)
        print(f"Loaded episode: {len(steps)} steps, {config.n_agents} agents, {config.grid_size}x{config.grid_size} grid")

        fig = visualize_timestep(steps, ledger_snapshots, config, args.timestep)
        plt.savefig('episode_agent_view.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nSaved to 'episode_agent_view.png'")
