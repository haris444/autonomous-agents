"""
Visualize what an agent sees about other agents in its observation.

Shows:
1. Grid view with agent positions
2. Agent 0's observation of each other agent (position, HP, social history)
3. Entity token values for debugging
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from environment import GridWorld
from config import Config
from scenarios import CoopFoodLowHPScenario

def visualize_agent_view(phase: int = 9, n_agents: int = 4):
    """Visualize what agent 0 sees about other agents."""

    # Create environment
    config = Config(
        grid_size=15,
        n_agents=n_agents,
        max_hp=100.0
    )

    # Create environment
    device = torch.device('cpu')
    env = GridWorld(config, device)
    env.reset()

    # Apply scenario for the specified phase
    if phase == 9:
        scenario = CoopFoodLowHPScenario(distance=3, hp_fraction=0.5, n_rich_food=5)
        env.apply_scenario(scenario)

    # Get observations
    obs = env._get_all_observations()
    entity_tokens = obs['entity_tokens']  # [n_agents, max_entities, 25]
    entity_mask = obs['entity_mask']      # [n_agents, max_entities]
    signals = obs['signals']              # [n_agents, n_agents]

    # Focus on agent 0's view
    agent_0_tokens = entity_tokens[0]  # [max_entities, 25]
    agent_0_mask = entity_mask[0]      # [max_entities]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # === SUBPLOT 1: Grid view with all agents ===
    ax1 = fig.add_subplot(2, 2, 1)

    # Draw grid
    gs = config.grid_size
    ax1.set_xlim(-0.5, gs - 0.5)
    ax1.set_ylim(-0.5, gs - 0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Grid View (True Positions)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.invert_yaxis()  # Row 0 at top

    # Draw rich food
    rich_food_pos = torch.where(env.rich_food)
    for r, c in zip(rich_food_pos[0].tolist(), rich_food_pos[1].tolist()):
        ax1.add_patch(plt.Circle((c, r), 0.3, color='gold', alpha=0.7))
        ax1.text(c, r, '★', ha='center', va='center', fontsize=10)

    # Draw agents
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    for i in range(n_agents):
        if env.agent_alive[i]:
            r, c = env.agent_positions[i].tolist()
            color = colors[i % len(colors)]
            hp_frac = env.agent_hp[i].item() / config.max_hp

            # Agent circle (size based on HP)
            circle = plt.Circle((c, r), 0.35, color=color, alpha=0.8)
            ax1.add_patch(circle)
            ax1.text(c, r, str(i), ha='center', va='center', color='white', fontweight='bold')

            # HP bar above agent
            bar_width = 0.6
            ax1.add_patch(plt.Rectangle((c - bar_width/2, r - 0.6), bar_width, 0.15,
                                         color='darkgray', alpha=0.5))
            ax1.add_patch(plt.Rectangle((c - bar_width/2, r - 0.6), bar_width * hp_frac, 0.15,
                                         color='lime' if hp_frac > 0.5 else 'red'))

    # === SUBPLOT 2: Agent 0's relative view ===
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlim(-gs/2, gs/2)
    ax2.set_ylim(-gs/2, gs/2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_title('Agent 0\'s Egocentric View\n(Self at origin)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Δ Column (relative)')
    ax2.set_ylabel('Δ Row (relative)')
    ax2.invert_yaxis()

    # Draw agent 0 at center
    ax2.add_patch(plt.Circle((0, 0), 0.4, color='blue', alpha=0.9))
    ax2.text(0, 0, '0', ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    # Draw other agents from agent 0's perspective
    pos_0 = env.agent_positions[0].float()
    for i in range(1, n_agents):
        if env.agent_alive[i]:
            rel_pos = env.agent_positions[i].float() - pos_0
            dc, dr = rel_pos[1].item(), rel_pos[0].item()  # col, row
            color = colors[i % len(colors)]
            hp_frac = env.agent_hp[i].item() / config.max_hp

            ax2.add_patch(plt.Circle((dc, dr), 0.35, color=color, alpha=0.8))
            ax2.text(dc, dr, str(i), ha='center', va='center', color='white', fontweight='bold')

            # Distance annotation
            dist = abs(dr) + abs(dc)
            ax2.annotate(f'd={int(dist)}', (dc, dr), textcoords="offset points",
                        xytext=(10, 10), fontsize=8, color=color)

    # Draw rich food relative to agent 0
    for r, c in zip(rich_food_pos[0].tolist(), rich_food_pos[1].tolist()):
        rel_r = r - pos_0[0].item()
        rel_c = c - pos_0[1].item()
        ax2.add_patch(plt.Circle((rel_c, rel_r), 0.25, color='gold', alpha=0.7))

    # === SUBPLOT 3: Entity token values for agents ===
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    ax3.set_title('Agent 0\'s Observation Tokens (Other Agents)', fontsize=12, fontweight='bold')

    # Token format (25 features):
    # fourier[16] + velocity[2] + type_onehot[2] + value[1] + social[4]
    # - fourier[0:16]: 4 freq bands × (sin_dx, cos_dx, sin_dy, cos_dy)
    # - velocity[16:18]: (dv_x, dv_y)
    # - type[18:20]: [is_food, is_agent]
    # - value[20]: HP or quality
    # - social[21:25]: [dmg_dealt, food_given, coop_count, defense]

    text_lines = []
    text_lines.append("Token format: fourier[16] + velocity[2] + type[2] + value[1] + social[4]")
    text_lines.append("HP = normalized health, social[4] = interaction history (scaled 5x)")
    text_lines.append("-" * 80)

    # Show first n_agents tokens (agent tokens)
    for i in range(1, n_agents):  # Skip self (token 0 is self-placeholder)
        token = agent_0_tokens[i]
        mask = agent_0_mask[i].item()

        if mask:  # Valid token
            values = token.tolist()
            # Extract key features from new format
            hp_pct = values[20] * 100  # Convert to percentage
            is_agent = values[19]
            # Social features are at indices 21-24
            dmg, food, coop, defense = values[21], values[22], values[23], values[24]

            # Get actual position from environment for display
            rel_pos = env.agent_positions[i].float() - env.agent_positions[0].float()
            dx_grid, dy_grid = rel_pos[1].item(), rel_pos[0].item()

            line = f"Agent {i}: pos=({dx_grid:+.1f}, {dy_grid:+.1f}) HP={hp_pct:.0f}%"
            line += f" | dmg={dmg:.2f} food={food:.2f} coop={coop:.2f} def={defense:.2f}"
            text_lines.append(line)
        else:
            text_lines.append(f"Agent {i}: [NOT VISIBLE / DEAD]")

    text_lines.append("-" * 80)
    text_lines.append("Token key values (first 4 agent tokens):")
    text_lines.append("  Format: [fourier_sample, velocity, type, HP, social...]")
    for i in range(min(4, n_agents)):
        token = agent_0_tokens[i].tolist()
        # Show: first 2 fourier, velocity, type, HP, social
        summary = f"[f={token[0]:+.2f},{token[1]:+.2f}.. v={token[16]:+.2f},{token[17]:+.2f} "
        summary += f"t={token[18]:.0f},{token[19]:.0f} HP={token[20]:.2f} "
        summary += f"s={token[21]:.2f},{token[22]:.2f},{token[23]:.2f},{token[24]:.2f}]"
        text_lines.append(f"  Token[{i}]: {summary}")

    ax3.text(0.02, 0.98, '\n'.join(text_lines), transform=ax3.transAxes,
             fontfamily='monospace', fontsize=9, verticalalignment='top')

    # === SUBPLOT 4: Social history heatmap ===
    ax4 = fig.add_subplot(2, 2, 4)

    # Get ledger data
    ledger_data = env.ledger.tensor.cpu().numpy()  # [n, n, 4]

    # Create a combined social score (positive = friend, negative = foe)
    # Score = food_given + coop_count + defense - damage_dealt
    social_score = (ledger_data[:, :, 1] + ledger_data[:, :, 2] + ledger_data[:, :, 3]
                   - ledger_data[:, :, 0])

    # Mask diagonal (self)
    np.fill_diagonal(social_score, np.nan)

    im = ax4.imshow(social_score[:n_agents, :n_agents], cmap='RdYlGn', aspect='equal',
                    vmin=-50, vmax=50)
    ax4.set_title('Social Ledger\n(Green=Friend, Red=Foe)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Target Agent')
    ax4.set_ylabel('Source Agent')
    ax4.set_xticks(range(n_agents))
    ax4.set_yticks(range(n_agents))

    # Add value annotations
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                val = social_score[i, j]
                color = 'white' if abs(val) > 25 else 'black'
                ax4.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax4, label='Social Score')

    plt.tight_layout()
    plt.savefig('agent_view_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nVisualization saved to 'agent_view_visualization.png'")
    print(f"\nAgent positions:")
    for i in range(n_agents):
        pos = env.agent_positions[i].tolist()
        hp = env.agent_hp[i].item()
        print(f"  Agent {i}: pos=({pos[0]}, {pos[1]}) HP={hp:.0f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize agent observation')
    parser.add_argument('--phase', type=int, default=9, help='Curriculum phase')
    parser.add_argument('--n_agents', type=int, default=4, help='Number of agents')
    args = parser.parse_args()

    visualize_agent_view(phase=args.phase, n_agents=args.n_agents)
