"""
Visualize Agent 0's action values and probabilities.

Shows what the agent "thinks" about each possible action.
Can show a single frame or generate an animated GIF.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import argparse
import os


def setup_figure(config):
    """Create figure and axes layout."""
    gs = config.grid_size
    fig = plt.figure(figsize=(18, 10))
    
    # === LEFT: Grid with movement arrows ===
    ax_grid = fig.add_axes([0.02, 0.35, 0.35, 0.60])
    
    # === TOP RIGHT: Direction probabilities ===
    ax_dir = fig.add_axes([0.42, 0.55, 0.25, 0.40])
    
    # === MIDDLE RIGHT: Action type probabilities ===
    ax_act = fig.add_axes([0.72, 0.55, 0.25, 0.40])
    
    # === BOTTOM LEFT: Value estimates ===
    ax_val = fig.add_axes([0.02, 0.08, 0.30, 0.22])
    ax_val.axis('off')
    
    # === BOTTOM MIDDLE: Action interpretation ===
    ax_info = fig.add_axes([0.36, 0.08, 0.28, 0.22])
    ax_info.axis('off')
    
    # === BOTTOM RIGHT: Combined action probability grid ===
    ax_comb = fig.add_axes([0.68, 0.08, 0.30, 0.22])
    
    return fig, (ax_grid, ax_dir, ax_act, ax_val, ax_info, ax_comb)


def draw_frame(frame_idx, step, config, fig, axes):
    """Draw a single frame on the provided axes."""
    ax_grid, ax_dir, ax_act, ax_val, ax_info, ax_comb = axes
    gs = config.grid_size
    n_agents = config.n_agents
    agent_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    # Clear axes
    for ax in axes:
        ax.clear()

    # Re-setup Grid
    ax_grid.set_xlim(-0.5, gs - 0.5)
    ax_grid.set_ylim(-0.5, gs - 0.5)
    ax_grid.set_aspect('equal')
    ax_grid.invert_yaxis()
    ax_grid.set_title(f'Agent 0 Action Values - Step {frame_idx}', fontsize=12, fontweight='bold')
    
    for i in range(gs + 1):
        ax_grid.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        ax_grid.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        
    # Draw food
    for row in range(gs):
        for col in range(gs):
            if step.rich_food[row, col]:
                ax_grid.plot(col, row, '*', color='gold', markersize=20, alpha=0.8)
            elif step.poor_food[row, col]:
                ax_grid.plot(col, row, 'o', color='lightgreen', markersize=10, alpha=0.6)

    # Draw agents
    for i in range(n_agents):
        if step.alive[i]:
            row, col = step.positions[i]
            color = agent_colors[i % len(agent_colors)]
            circle = Circle((col, row), 0.35, color=color, ec='black', linewidth=2, alpha=0.9)
            ax_grid.add_patch(circle)
            ax_grid.text(col, row, str(i), ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')

    # Draw movement probability arrows for Agent 0
    chosen_dir = -1
    chosen_act = -1
    dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
    act_names = ['MOVE', 'ATTACK', 'GIVE', 'SIGNAL', 'COOP']
    
    if step.alive[0]:
        pos_0 = step.positions[0]
        row, col = pos_0[0], pos_0[1]
        
        # Check if we have probabilities
        if step.direction_probs is not None:
            dir_probs = step.direction_probs[0]  # [UP, DOWN, LEFT, RIGHT, STAY]
            
            # Arrow directions: UP=(-1,0), DOWN=(1,0), LEFT=(0,-1), RIGHT=(0,1)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            prob_cmap = LinearSegmentedColormap.from_list('prob', ['#ffffff', '#3498db'], N=256)
            
            for idx, (dr, dc) in enumerate(directions):
                prob = dir_probs[idx]
                if prob > 0.01:
                    arrow_len = 0.6 + prob * 0.8
                    start_row = row + dr * 0.4
                    start_col = col + dc * 0.4
                    end_row = row + dr * arrow_len
                    end_col = col + dc * arrow_len
                    
                    color = prob_cmap(prob)
                    ax_grid.annotate('', xy=(end_col, end_row), xytext=(start_col, start_row),
                                   arrowprops=dict(arrowstyle='->', color=color,
                                                  lw=2 + prob*4, mutation_scale=15 + prob*20))
                    
                    label_row = row + dr * (arrow_len + 0.3)
                    label_col = col + dc * (arrow_len + 0.3)
                    ax_grid.text(label_col, label_row, f'{prob:.1%}', ha='center', va='center',
                                fontsize=9, fontweight='bold', color='darkblue',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            stay_prob = dir_probs[4]
            ax_grid.text(col, row + 0.55, f'STAY:{stay_prob:.1%}', ha='center', va='top',
                        fontsize=8, color='gray', fontweight='bold')

            # === TOP RIGHT: Direction probabilities ===
            colors_dir = ['#e74c3c', '#e74c3c', '#3498db', '#3498db', '#95a5a6']
            bars = ax_dir.bar(dir_names, dir_probs, color=colors_dir, edgecolor='black', linewidth=1.5)
            ax_dir.set_ylabel('Probability', fontsize=10)
            ax_dir.set_title('Direction Probabilities', fontsize=11, fontweight='bold')
            ax_dir.set_ylim(0, 1)
            ax_dir.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
            
            for bar, prob in zip(bars, dir_probs):
                ax_dir.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{prob:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            chosen_dir = step.actions[0][0] if step.actions and 0 in step.actions else -1
            if chosen_dir >= 0:
                bars[chosen_dir].set_edgecolor('lime')
                bars[chosen_dir].set_linewidth(4)
                ax_dir.text(chosen_dir, dir_probs[chosen_dir] + 0.08, 'CHOSEN', 
                           ha='center', fontsize=8, color='green', fontweight='bold')

        if step.action_type_probs is not None:
            act_probs = step.action_type_probs[0]
            colors_act = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            bars_act = ax_act.bar(act_names, act_probs, color=colors_act, edgecolor='black', linewidth=1.5)
            ax_act.set_ylabel('Probability', fontsize=10)
            ax_act.set_title('Action Type Probabilities', fontsize=11, fontweight='bold')
            ax_act.set_ylim(0, 1)
            ax_act.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
            
            for bar, prob in zip(bars_act, act_probs):
                if prob > 0.001:
                    ax_act.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{prob:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            chosen_act = step.actions[0][1] if step.actions and 0 in step.actions else -1
            if chosen_act >= 0:
                bars_act[chosen_act].set_edgecolor('lime')
                bars_act[chosen_act].set_linewidth(4)
                ax_act.text(chosen_act, act_probs[chosen_act] + 0.08, 'CHOSEN',
                           ha='center', fontsize=8, color='green', fontweight='bold')

            # === BOTTOM RIGHT: Combined ===
            if step.direction_probs is not None:
                combined_probs = np.outer(dir_probs, act_probs)
                im = ax_comb.imshow(combined_probs.T, cmap='Blues', aspect='auto', vmin=0, vmax=0.3)
                ax_comb.set_xticks(range(5))
                ax_comb.set_yticks(range(5))
                ax_comb.set_xticklabels(dir_names, fontsize=8)
                ax_comb.set_yticklabels(act_names, fontsize=8)
                ax_comb.set_xlabel('Direction')
                ax_comb.set_ylabel('Action Type')
                ax_comb.set_title('Combined Action Probs', fontsize=10, fontweight='bold')
                
                if chosen_dir >= 0 and chosen_act >= 0:
                    ax_comb.add_patch(Rectangle((chosen_dir-0.5, chosen_act-0.5), 1, 1,
                                                 fill=False, edgecolor='lime', linewidth=3))

    else:
        ax_grid.text(0.5, 0.5, 'AGENT 0 DEAD', transform=ax_grid.transAxes, 
                    ha='center', va='center', fontsize=30, color='red', alpha=0.5)

    # === BOTTOM LEFT: Value Estimates ===
    ax_val.axis('off')
    ax_val.set_title('Value Estimates', fontsize=11, fontweight='bold')
    val_text = []
    if step.values is not None:
        val_text.append(f'Main Value: {step.values[0]:.2f}')
    val_text.append('')
    if step.aux_values:
        val_text.append('Auxiliary Values:')
        val_text.append(f'  Survival:  {step.aux_values["survival"][0]:.2f}')
        val_text.append(f'  Resource:  {step.aux_values["resource"][0]:.2f}')
        val_text.append(f'  Social:    {step.aux_values["social"][0]:.2f}')
    
    ax_val.text(0.1, 0.9, '\n'.join(val_text), transform=ax_val.transAxes,
               fontfamily='monospace', fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # === BOTTOM MIDDLE: Interpretation ===
    ax_info.axis('off')
    ax_info.set_title('Action Interpretation', fontsize=11, fontweight='bold')
    info_text = []
    if chosen_dir >= 0 and chosen_act >= 0:
        info_text.append(f'Chosen: {dir_names[chosen_dir]} + {act_names[chosen_act]}')
        info_text.append('')
        info_text.append('What this means:')
        if chosen_act == 4:  # COOP
            info_text.append('  Agent 0 is trying to COOPERATE')
        elif chosen_act == 0:  # MOVE
            info_text.append(f'  Agent 0 is MOVING {dir_names[chosen_dir]}')
        elif chosen_act == 1:  # ATTACK
            info_text.append(f'  Agent 0 is ATTACKING {dir_names[chosen_dir]}')
        elif chosen_act == 2:  # GIVE
            info_text.append(f'  Agent 0 is GIVING food {dir_names[chosen_dir]}')
        elif chosen_act == 3:  # SIGNAL
            info_text.append(f'  Agent 0 is SIGNALING')
        
    ax_info.text(0.1, 0.9, '\n'.join(info_text), transform=ax_info.transAxes,
               fontfamily='monospace', fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))


def show_action_values(episode_path: str, frame_idx: int = 64, output_file: str = 'agent_action_values.png'):
    """Visualize action values for Agent 0 at a single frame."""
    data = torch.load(episode_path, map_location='cpu', weights_only=False)
    steps = data['steps']
    config = data['config']
    
    # Handle negative frame index (end of episode)
    if frame_idx < 0:
        frame_idx = len(steps) + frame_idx

    step = steps[frame_idx]
    fig, axes = setup_figure(config)
    
    draw_frame(frame_idx, step, config, fig, axes)
    
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.show()
    print(f'Saved {output_file}')


def animate_action_values(episode_path: str, output_file: str = 'agent_action_values.gif', fps: int = 5):
    """Animate action values for the entire episode."""
    data = torch.load(episode_path, map_location='cpu', weights_only=False)
    steps = data['steps']
    config = data['config']
    
    fig, axes = setup_figure(config)
    
    def update(frame_idx):
        draw_frame(frame_idx, steps[frame_idx], config, fig, axes)
        if frame_idx % 10 == 0:
            print(f"Rendered frame {frame_idx}/{len(steps)}")
    
    print(f"Creating animation ({len(steps)} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=200)
    
    writer = animation.PillowWriter(fps=fps)
    ani.save(output_file, writer=writer)
    print(f'Saved animation to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=str, help='Path to episode file')
    parser.add_argument('--frame', '-f', type=int, default=64, help='Frame to visualize (single frame mode)')
    parser.add_argument('--animate', '-a', action='store_true', help='Generate animated GIF of all frames')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output filename')
    parser.add_argument('--fps', type=int, default=5, help='FPS for animation')
    args = parser.parse_args()

    if args.animate:
        output = args.output if args.output else 'agent_action_values.gif'
        animate_action_values(args.episode, output, args.fps)
    else:
        output = args.output if args.output else f'agent_action_values_f{args.frame}.png'
        show_action_values(args.episode, args.frame, output)
