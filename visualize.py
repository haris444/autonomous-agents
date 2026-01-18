"""
Visualization and Observability tools for Multi-Agent RL.

Features:
- GridRenderer: Render grid world state with agents and food
- Ledger heatmaps: Visualize agent-to-agent relationships
- TrainingLogger: Track and plot training metrics
- EpisodeRecorder: Record and replay episodes
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

import torch

from config import Config

# Action constants (must match environment.py)
# Attack directions: 0-7, Give directions: 8-15, Signal: 16, Cooperate: 17, Idle: 18
ATTACK_UP, ATTACK_DOWN, ATTACK_LEFT, ATTACK_RIGHT = 0, 1, 2, 3
ATTACK_UP_LEFT, ATTACK_UP_RIGHT, ATTACK_DOWN_LEFT, ATTACK_DOWN_RIGHT = 4, 5, 6, 7
GIVE_UP, GIVE_DOWN, GIVE_LEFT, GIVE_RIGHT = 8, 9, 10, 11
GIVE_UP_LEFT, GIVE_UP_RIGHT, GIVE_DOWN_LEFT, GIVE_DOWN_RIGHT = 12, 13, 14, 15
INTERACT_SIGNAL = 16
INTERACT_COOPERATE = 17
INTERACT_IDLE = 18

# Direction deltas: UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT
INTERACT_DIR_DELTAS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),      # Cardinal
    (-1, -1), (-1, 1), (1, -1), (1, 1)     # Diagonal
]


def _is_attack_action(action):
    """Check if action is an attack (0-7)."""
    return 0 <= action <= 7


def _is_give_action(action):
    """Check if action is a give (8-15)."""
    return 8 <= action <= 15


def _get_target_from_action(agent_pos, action, grid_size):
    """Get target position from directional action. Returns None if out of bounds."""
    if _is_attack_action(action):
        direction = action  # 0-7
    elif _is_give_action(action):
        direction = action - 8  # 8-15 -> 0-7
    else:
        return None

    delta_row, delta_col = INTERACT_DIR_DELTAS[direction]
    target_row = agent_pos[0] + delta_row
    target_col = agent_pos[1] + delta_col

    # Bounds check
    if 0 <= target_row < grid_size and 0 <= target_col < grid_size:
        return (target_row, target_col)
    return None


@dataclass
class StepData:
    """Data for a single timestep."""
    positions: np.ndarray      # [n_agents, 2]
    hp: np.ndarray             # [n_agents]
    alive: np.ndarray          # [n_agents]
    poor_food: np.ndarray      # [grid_size, grid_size]
    rich_food: np.ndarray      # [grid_size, grid_size]
    signals: np.ndarray        # [n_agents]
    actions: Dict[int, Tuple[int, int]]
    rewards: Dict[int, float]


class GridRenderer:
    """
    Renders the grid world state using matplotlib.

    Shows agents (colored by health), food, and signals.
    """

    def __init__(self, config: Config, figsize: Tuple[int, int] = (10, 10)):
        self.config = config
        self.figsize = figsize

        # Health colormap (green -> yellow -> red)
        colors = ['#ff0000', '#ff8800', '#ffff00', '#88ff00', '#00ff00']
        self.health_cmap = LinearSegmentedColormap.from_list('health', colors, N=256)

        # Agent colors (distinct for each agent)
        self.agent_colors = plt.cm.tab10(np.linspace(0, 1, config.n_agents))

    def render(self, env, ax: Optional[plt.Axes] = None, show: bool = True) -> plt.Figure:
        """
        Render current environment state.

        Args:
            env: GridWorld environment
            ax: Optional axes to draw on
            show: Whether to display immediately

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        else:
            fig = ax.figure

        ax.clear()
        gs = self.config.grid_size

        # Draw grid
        ax.set_xlim(-0.5, gs - 0.5)
        ax.set_ylim(-0.5, gs - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Row 0 at top

        # Grid lines
        for i in range(gs + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)

        # Convert tensors to numpy
        poor_food = env.poor_food.cpu().numpy()
        rich_food = env.rich_food.cpu().numpy()
        positions = env.agent_positions.cpu().numpy()
        hp = env.agent_hp.cpu().numpy()
        alive = env.agent_alive.cpu().numpy()
        signals = env.signals.cpu().numpy()

        # Draw food
        for row in range(gs):
            for col in range(gs):
                if poor_food[row, col]:
                    ax.plot(col, row, 'go', markersize=12, alpha=0.7)
                elif rich_food[row, col]:
                    ax.plot(col, row, 'y*', markersize=18, alpha=0.9)

        # Draw agents
        for agent_id in range(self.config.n_agents):
            if not alive[agent_id]:
                continue

            row, col = positions[agent_id]
            health_frac = hp[agent_id] / self.config.max_hp

            # Agent circle (color based on health)
            color = self.health_cmap(health_frac)
            circle = plt.Circle((col, row), 0.35, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)

            # Agent ID label
            ax.text(col, row, str(agent_id), ha='center', va='center',
                   fontsize=10, fontweight='bold', color='black')

            # Signal indicator
            if signals[agent_id]:
                ax.text(col + 0.35, row - 0.35, '~', fontsize=12, color='blue', fontweight='bold')

        # Title and labels
        ax.set_title(f'Multi-Agent Grid World (Step {env.step_count})', fontsize=14)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        # Legend
        legend_elements = [
            mpatches.Patch(color='green', alpha=0.7, label='Poor Food'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow',
                      markersize=12, label='Rich Food'),
            mpatches.Patch(color='limegreen', label='Healthy Agent'),
            mpatches.Patch(color='red', label='Low HP Agent'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1))

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def render_with_health_bars(self, env, show: bool = True) -> plt.Figure:
        """Render grid with health bars below."""
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] + 2),
                                  gridspec_kw={'height_ratios': [4, 1]})

        # Render grid
        self.render(env, ax=axes[0], show=False)

        # Health bars
        ax = axes[1]
        hp = env.agent_hp.cpu().numpy()
        alive = env.agent_alive.cpu().numpy()

        bar_width = 0.8
        for i in range(self.config.n_agents):
            health_frac = hp[i] / self.config.max_hp if alive[i] else 0
            color = self.health_cmap(health_frac) if alive[i] else 'gray'

            # Background bar
            ax.barh(i, 1.0, bar_width, color='lightgray', alpha=0.5)
            # Health bar
            ax.barh(i, health_frac, bar_width, color=color)
            # Label
            ax.text(0.5, i, f'A{i}: {hp[i]:.0f}' if alive[i] else f'A{i}: DEAD',
                   ha='center', va='center', fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, self.config.n_agents - 0.5)
        ax.set_xlabel('Health')
        ax.set_yticks([])
        ax.set_title('Agent Health')

        plt.tight_layout()

        if show:
            plt.show()

        return fig


def render_ledger_heatmaps(ledger_tensor: torch.Tensor,
                           show: bool = True,
                           figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Render 4 heatmaps showing agent-to-agent relationships.

    Args:
        ledger_tensor: [n_agents, n_agents, 4] tensor
        show: Whether to display immediately
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if isinstance(ledger_tensor, torch.Tensor):
        data = ledger_tensor.cpu().numpy()
    else:
        data = ledger_tensor

    n_agents = data.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    channel_names = ['Damage Dealt', 'Food Given', 'Coop Count', 'Defense Score']
    cmaps = ['Reds', 'Greens', 'Blues', 'Purples']

    for idx, (ax, name, cmap) in enumerate(zip(axes.flat, channel_names, cmaps)):
        channel_data = data[:, :, idx]

        im = ax.imshow(channel_data, cmap=cmap, aspect='equal')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Agent')
        ax.set_ylabel('Source Agent')

        # Tick labels
        ax.set_xticks(range(n_agents))
        ax.set_yticks(range(n_agents))
        ax.set_xticklabels([f'A{i}' for i in range(n_agents)])
        ax.set_yticklabels([f'A{i}' for i in range(n_agents)])

        # Colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate cells with values
        for i in range(n_agents):
            for j in range(n_agents):
                val = channel_data[i, j]
                if val > 0:
                    text_color = 'white' if val > channel_data.max() * 0.5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                           fontsize=8, color=text_color)

    plt.suptitle('Agent Interaction Ledger', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show()

    return fig


class TrainingLogger:
    """
    Tracks and visualizes training metrics.

    Stores episode returns, losses, entropy, etc.
    """

    def __init__(self):
        self.episode_returns: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.total_losses: List[float] = []
        self.entropies: List[float] = []
        self.kl_divs: List[float] = []
        self.clip_fractions: List[float] = []
        self.steps: List[int] = []

    def log(self, step: int, episode_return: Optional[float] = None,
            metrics: Optional[Dict[str, float]] = None) -> None:
        """Log training data."""
        self.steps.append(step)

        if episode_return is not None:
            self.episode_returns.append(episode_return)

        if metrics is not None:
            self.policy_losses.append(metrics.get('policy_loss', 0))
            self.value_losses.append(metrics.get('value_loss', 0))
            self.total_losses.append(metrics.get('total_loss', 0))
            self.entropies.append(metrics.get('entropy', 0))
            self.kl_divs.append(metrics.get('approx_kl', 0))
            self.clip_fractions.append(metrics.get('clip_fraction', 0))

    def plot(self, window: int = 100, show: bool = True,
             figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot training curves.

        Args:
            window: Smoothing window size
            show: Whether to display immediately
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        def smooth(data, w):
            if len(data) < w:
                return data
            return np.convolve(data, np.ones(w)/w, mode='valid')

        # Episode returns
        if self.episode_returns:
            ax = axes[0, 0]
            ax.plot(self.episode_returns, alpha=0.3, color='blue')
            if len(self.episode_returns) >= window:
                smoothed = smooth(self.episode_returns, window)
                ax.plot(range(window-1, len(self.episode_returns)), smoothed,
                       color='blue', linewidth=2)
            ax.set_title('Episode Returns')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.grid(True, alpha=0.3)

        # Policy loss
        if self.policy_losses:
            ax = axes[0, 1]
            ax.plot(self.policy_losses, alpha=0.5, color='red')
            ax.set_title('Policy Loss')
            ax.set_xlabel('Update')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

        # Value loss
        if self.value_losses:
            ax = axes[0, 2]
            ax.plot(self.value_losses, alpha=0.5, color='green')
            ax.set_title('Value Loss')
            ax.set_xlabel('Update')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

        # Entropy
        if self.entropies:
            ax = axes[1, 0]
            ax.plot(self.entropies, color='purple')
            ax.set_title('Policy Entropy')
            ax.set_xlabel('Update')
            ax.set_ylabel('Entropy')
            ax.grid(True, alpha=0.3)

        # KL divergence
        if self.kl_divs:
            ax = axes[1, 1]
            ax.plot(self.kl_divs, color='orange')
            ax.set_title('Approx KL Divergence')
            ax.set_xlabel('Update')
            ax.set_ylabel('KL')
            ax.grid(True, alpha=0.3)

        # Clip fraction
        if self.clip_fractions:
            ax = axes[1, 2]
            ax.plot(self.clip_fractions, color='brown')
            ax.set_title('Clip Fraction')
            ax.set_xlabel('Update')
            ax.set_ylabel('Fraction')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Training Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def save(self, filepath: str) -> None:
        """Save logged data to file."""
        data = {
            'episode_returns': self.episode_returns,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'total_losses': self.total_losses,
            'entropies': self.entropies,
            'kl_divs': self.kl_divs,
            'clip_fractions': self.clip_fractions,
            'steps': self.steps
        }
        torch.save(data, filepath)

    def load(self, filepath: str) -> None:
        """Load logged data from file."""
        data = torch.load(filepath, weights_only=False)
        self.episode_returns = data['episode_returns']
        self.policy_losses = data['policy_losses']
        self.value_losses = data['value_losses']
        self.total_losses = data['total_losses']
        self.entropies = data['entropies']
        self.kl_divs = data['kl_divs']
        self.clip_fractions = data['clip_fractions']
        self.steps = data['steps']


class EpisodeRecorder:
    """
    Records episode data for replay.

    Stores agent positions, HP, food, actions, rewards per step.
    """

    def __init__(self, config: Config):
        self.config = config
        self.steps: List[StepData] = []
        self.ledger_snapshots: List[np.ndarray] = []

    def reset(self) -> None:
        """Clear recorded data."""
        self.steps = []
        self.ledger_snapshots = []

    def record(self, env, actions: Dict[int, Tuple[int, int]] = None,
               rewards: Dict[int, float] = None) -> None:
        """Record current environment state."""
        step_data = StepData(
            positions=env.agent_positions.cpu().numpy().copy(),
            hp=env.agent_hp.cpu().numpy().copy(),
            alive=env.agent_alive.cpu().numpy().copy(),
            poor_food=env.poor_food.cpu().numpy().copy(),
            rich_food=env.rich_food.cpu().numpy().copy(),
            signals=env.signals.cpu().numpy().copy(),
            actions=actions or {},
            rewards=rewards or {}
        )
        self.steps.append(step_data)
        self.ledger_snapshots.append(env.ledger.tensor.cpu().numpy().copy())

    def get_recording(self) -> List[StepData]:
        """Return recorded steps."""
        return self.steps

    def save(self, filepath: str) -> None:
        """Save recording to file."""
        data = {
            'steps': self.steps,
            'ledger_snapshots': self.ledger_snapshots,
            'config': self.config
        }
        torch.save(data, filepath)

    def load(self, filepath: str) -> None:
        """Load recording from file."""
        data = torch.load(filepath, weights_only=False)
        self.steps = data['steps']
        self.ledger_snapshots = data['ledger_snapshots']
        self.config = data['config']


def replay_episode(recording: List[StepData], config: Config,
                   speed: float = 1.0, save_gif: bool = False,
                   gif_path: str = 'episode.gif',
                   ledger_snapshots: List[np.ndarray] = None,
                   show_ledger: bool = True) -> None:
    """
    Replay a recorded episode with animation.

    Args:
        recording: List of StepData from EpisodeRecorder
        config: Environment config
        speed: Playback speed multiplier
        save_gif: Whether to save as GIF
        gif_path: Path for GIF output
        ledger_snapshots: Optional list of ledger tensors per step
        show_ledger: Whether to show ledger heatmaps (if available)
    """
    if not recording:
        print("No recording to replay")
        return

    # Health colormap
    colors = ['#ff0000', '#ff8800', '#ffff00', '#88ff00', '#00ff00']
    health_cmap = LinearSegmentedColormap.from_list('health', colors, N=256)

    gs = config.grid_size
    n_agents = config.n_agents

    # Create figure layout based on whether we have ledger data
    has_ledger = ledger_snapshots is not None and len(ledger_snapshots) > 0 and show_ledger

    if has_ledger:
        # Grid on left, 4 ledger heatmaps on right (2x2)
        fig = plt.figure(figsize=(18, 10))
        ax_grid = fig.add_subplot(1, 2, 1)
        ax_ledger = [
            fig.add_subplot(2, 4, 3),  # Damage
            fig.add_subplot(2, 4, 4),  # Food Given
            fig.add_subplot(2, 4, 7),  # Coop
            fig.add_subplot(2, 4, 8),  # Defense
        ]
        ledger_names = ['Damage Dealt', 'Food Given', 'Coop Count', 'Defense Score']
        ledger_cmaps = ['Reds', 'Greens', 'Blues', 'Purples']
    else:
        fig, ax_grid = plt.subplots(1, 1, figsize=(10, 10))
        ax_ledger = None

    def animate(frame_idx: int):
        ax_grid.clear()
        step_data = recording[frame_idx]

        # Draw grid
        ax_grid.set_xlim(-0.5, gs - 0.5)
        ax_grid.set_ylim(-0.5, gs - 0.5)
        ax_grid.set_aspect('equal')
        ax_grid.invert_yaxis()

        # Grid lines
        for i in range(gs + 1):
            ax_grid.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax_grid.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)

        # Draw food
        for row in range(gs):
            for col in range(gs):
                if step_data.poor_food[row, col]:
                    ax_grid.plot(col, row, 'go', markersize=12, alpha=0.7)
                elif step_data.rich_food[row, col]:
                    ax_grid.plot(col, row, 'y*', markersize=18, alpha=0.9)

        # Draw agents
        for agent_id in range(n_agents):
            if not step_data.alive[agent_id]:
                continue

            row, col = step_data.positions[agent_id]
            health_frac = step_data.hp[agent_id] / config.max_hp

            color = health_cmap(max(0, min(1, health_frac)))
            circle = plt.Circle((col, row), 0.35, color=color, ec='black', linewidth=2)
            ax_grid.add_patch(circle)

            ax_grid.text(col, row, str(agent_id), ha='center', va='center',
                        fontsize=10, fontweight='bold', color='black')

            if step_data.signals[agent_id]:
                ax_grid.text(col + 0.35, row - 0.35, '~', fontsize=12, color='blue', fontweight='bold')

        # Draw attack, give, and cooperate interactions
        if step_data.actions:
            for agent_id, (move_act, interact_act) in step_data.actions.items():
                if not step_data.alive[agent_id]:
                    continue

                agent_pos = step_data.positions[agent_id]
                agent_row, agent_col = int(agent_pos[0]), int(agent_pos[1])

                # Handle COOPERATE action (non-directional)
                if interact_act == INTERACT_COOPERATE:
                    # Blue circle around agent for cooperation
                    coop_circle = plt.Circle((agent_col, agent_row), 0.45,
                                            fill=False, color='blue', linewidth=3, alpha=0.8)
                    ax_grid.add_patch(coop_circle)
                    # "C" symbol for cooperate
                    ax_grid.text(agent_col + 0.35, agent_row - 0.35, 'C',
                                fontsize=9, color='blue', fontweight='bold')
                    continue

                # Get target position from directional action (attack/give)
                target_pos = _get_target_from_action(agent_pos, interact_act, gs)
                if target_pos is None:
                    continue

                target_row, target_col = target_pos

                # CHECK: Is there actually a living agent at the target position?
                target_agent_id = None
                for other_id in range(n_agents):
                    if other_id == agent_id:
                        continue
                    if not step_data.alive[other_id]:
                        continue
                    other_row, other_col = step_data.positions[other_id]
                    if int(other_row) == target_row and int(other_col) == target_col:
                        target_agent_id = other_id
                        break

                if target_agent_id is None:
                    continue  # No agent at target - don't draw arrow

                if _is_attack_action(interact_act):
                    # Red arrow for attack
                    ax_grid.annotate('',
                        xy=(target_col, target_row),
                        xytext=(agent_col, agent_row),
                        arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8))
                    # Attack symbol at target
                    ax_grid.text(target_col + 0.3, target_row + 0.3, '⚔',
                                fontsize=10, color='red', fontweight='bold')

                elif _is_give_action(interact_act):
                    # Green arrow for food giving
                    ax_grid.annotate('',
                        xy=(target_col, target_row),
                        xytext=(agent_col, agent_row),
                        arrowprops=dict(arrowstyle='->', color='limegreen', lw=3, alpha=0.8))
                    # Heart symbol at target
                    ax_grid.text(target_col + 0.3, target_row + 0.3, '♥',
                                fontsize=10, color='green', fontweight='bold')

        ax_grid.set_title(f'Episode Replay - Step {frame_idx + 1}/{len(recording)}', fontsize=14)
        ax_grid.set_xlabel('Column')
        ax_grid.set_ylabel('Row')

        # Draw ledger heatmaps
        if has_ledger and ax_ledger is not None:
            ledger_data = ledger_snapshots[frame_idx]

            for idx, (ax, name, cmap) in enumerate(zip(ax_ledger, ledger_names, ledger_cmaps)):
                ax.clear()
                channel_data = ledger_data[:, :, idx]

                im = ax.imshow(channel_data, cmap=cmap, aspect='equal', vmin=0)
                ax.set_title(name, fontsize=10, fontweight='bold')
                ax.set_xlabel('Target')
                ax.set_ylabel('Source')
                ax.set_xticks(range(n_agents))
                ax.set_yticks(range(n_agents))
                ax.set_xticklabels([f'{i}' for i in range(n_agents)], fontsize=8)
                ax.set_yticklabels([f'{i}' for i in range(n_agents)], fontsize=8)

        return []

    interval = int(200 / speed)  # Base 200ms between frames
    anim = animation.FuncAnimation(fig, animate, frames=len(recording),
                                   interval=interval, blit=False, repeat=True)

    if save_gif:
        try:
            anim.save(gif_path, writer='pillow', fps=int(5 * speed))
            print(f"Saved animation to {gif_path}")
        except Exception as e:
            print(f"Could not save GIF: {e}")

    plt.tight_layout()
    plt.show()


def visualize_trained_agent(model_path: str, config: Config = None,
                            num_episodes: int = 1) -> None:
    """
    Load a trained model and visualize its behavior.

    Args:
        model_path: Path to saved model checkpoint
        config: Config (if None, loads from checkpoint)
        num_episodes: Number of episodes to visualize
    """
    from environment import GridWorld
    from network import ActorCritic
    from utils import get_device

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if config is None:
        config = checkpoint.get('config', Config())

    device = get_device()

    # Initialize
    env = GridWorld(config, device)
    network = ActorCritic(config).to(device)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()

    renderer = GridRenderer(config)
    recorder = EpisodeRecorder(config)

    for ep in range(num_episodes):
        obs = env.reset()
        recorder.reset()
        done = False

        while not done:
            recorder.record(env)

            # Stack observations
            stacked_obs = {
                'spatial': torch.stack([obs[i]['spatial'] for i in range(config.n_agents)]),
                'ledger': torch.stack([obs[i]['ledger'] for i in range(config.n_agents)]),
                'signals': torch.stack([obs[i]['signals'] for i in range(config.n_agents)]),
                'self_hp': torch.stack([obs[i]['self_hp'] for i in range(config.n_agents)]),
                'agent_id': torch.stack([obs[i]['agent_id'] for i in range(config.n_agents)])
            }

            with torch.no_grad():
                action_masks = env.get_action_masks()
                move_actions, interact_actions, _, _, _ = network.get_action_and_value(
                    stacked_obs, action_masks=action_masks
                )

            actions = {
                i: (move_actions[i].item(), interact_actions[i].item())
                for i in range(config.n_agents)
            }

            obs, rewards, dones, _ = env.step(actions)
            done = all(dones.values())

        print(f"Episode {ep + 1} finished after {env.step_count} steps")

        # Replay episode
        replay_episode(recorder.get_recording(), config, speed=2.0)

        # Show final ledger
        render_ledger_heatmaps(env.ledger.get_tensor())


if __name__ == "__main__":
    # Demo with random environment
    from environment import GridWorld
    from utils import set_seed, get_device

    set_seed(42)
    config = Config()
    device = get_device()

    env = GridWorld(config, device)
    env.reset()

    # Run a few random steps
    for _ in range(10):
        actions = {
            i: (np.random.randint(5), np.random.randint(4))
            for i in range(config.n_agents)
        }
        env.step(actions)

    # Render
    renderer = GridRenderer(config)
    renderer.render_with_health_bars(env)

    # Show ledger
    render_ledger_heatmaps(env.ledger.get_tensor())
