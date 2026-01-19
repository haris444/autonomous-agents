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
# Attack directions: 0-3, Give directions: 4-7, Signal: 8, Cooperate: 9, Idle: 10
ATTACK_UP, ATTACK_DOWN, ATTACK_LEFT, ATTACK_RIGHT = 0, 1, 2, 3
GIVE_UP, GIVE_DOWN, GIVE_LEFT, GIVE_RIGHT = 4, 5, 6, 7
INTERACT_SIGNAL = 8
INTERACT_COOPERATE = 9
INTERACT_IDLE = 10

# Direction deltas: UP, DOWN, LEFT, RIGHT
INTERACT_DIR_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_attack_action(action):
    """Check if action is an attack (0-3)."""
    return 0 <= action <= 3


def _is_give_action(action):
    """Check if action is a give (4-7)."""
    return 4 <= action <= 7


def classify_relationship(ledger, viewer_id, target_id):
    """
    Classify how viewer sees target: 'friend', 'enemy', or 'neutral'.

    Args:
        ledger: Ledger snapshot array [n_agents, n_agents, 4]
        viewer_id: Agent doing the viewing
        target_id: Agent being observed

    Returns:
        (relationship, strength): ('friend'/'enemy'/'neutral', score)
    """
    # What target has done TO viewer (from viewer's perspective)
    dmg = ledger[target_id, viewer_id, 0]     # DAMAGE_DEALT channel
    food = ledger[target_id, viewer_id, 1]    # FOOD_GIVEN channel
    coop = ledger[viewer_id, target_id, 2]    # COOP_COUNT (symmetric)
    defense = ledger[target_id, viewer_id, 3] # DEFENSE_SCORE channel

    friend_score = food + coop * 2 + defense * 1.5
    enemy_score = dmg * 2

    if enemy_score > friend_score + 10:
        return 'enemy', enemy_score
    elif friend_score > enemy_score + 10:
        return 'friend', friend_score
    else:
        return 'neutral', 0


def _get_target_from_action(agent_pos, action, grid_size):
    """Get target position from directional action. Returns None if out of bounds."""
    if _is_attack_action(action):
        direction = action  # 0-3
    elif _is_give_action(action):
        direction = action - 4  # 4-7 -> 0-3
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
    actions: Dict[int, int]    # Unified action per agent (0-14)
    rewards: Dict[int, float]
    values: Optional[np.ndarray] = None  # [n_agents] value estimates
    # Action probabilities for visualization (unified action space)
    action_probs: Optional[np.ndarray] = None     # [n_agents, 15] unified action probs


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
        # Per-agent returns: [episode][agent_id]
        self.agent_returns: List[List[float]] = []

    def log(self, step: int, episode_return: Optional[float] = None,
            agent_returns: Optional[List[float]] = None,
            metrics: Optional[Dict[str, float]] = None) -> None:
        """Log training data."""
        self.steps.append(step)

        if episode_return is not None:
            self.episode_returns.append(episode_return)

        if agent_returns is not None:
            self.agent_returns.append(agent_returns)

        if metrics is not None:
            self.policy_losses.append(metrics.get('policy_loss', 0))
            self.value_losses.append(metrics.get('value_loss', 0))
            self.total_losses.append(metrics.get('total_loss', 0))
            self.entropies.append(metrics.get('entropy', 0))
            self.kl_divs.append(metrics.get('approx_kl', 0))
            self.clip_fractions.append(metrics.get('clip_fraction', 0))

    def plot(self, window: int = 100, show: bool = True,
             figsize: Tuple[int, int] = (14, 12)) -> plt.Figure:
        """
        Plot training curves.

        Args:
            window: Smoothing window size
            show: Whether to display immediately
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)

        def smooth(data, w):
            if len(data) < w:
                return data
            return np.convolve(data, np.ones(w)/w, mode='valid')

        # Episode returns (average)
        if self.episode_returns:
            ax = axes[0, 0]
            ax.plot(self.episode_returns, alpha=0.3, color='blue')
            if len(self.episode_returns) >= window:
                smoothed = smooth(self.episode_returns, window)
                ax.plot(range(window-1, len(self.episode_returns)), smoothed,
                       color='blue', linewidth=2)
            ax.set_title('Episode Returns (Avg)')
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

        # Per-agent returns (new plot spanning bottom row)
        if self.agent_returns:
            ax = axes[2, 0]
            # Hide the other two bottom subplots, use first one for wide plot
            axes[2, 1].axis('off')
            axes[2, 2].axis('off')

            # Reposition to span full width
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width * 3.2, pos.height])

            agent_data = np.array(self.agent_returns)  # [n_episodes, n_agents]
            n_agents = agent_data.shape[1]

            # Use distinct colors for each agent
            colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

            for agent_id in range(n_agents):
                agent_returns = agent_data[:, agent_id]
                # Light raw data
                ax.plot(agent_returns, alpha=0.2, color=colors[agent_id])
                # Bold smoothed line
                if len(agent_returns) >= window:
                    smoothed = smooth(agent_returns.tolist(), window)
                    ax.plot(range(window-1, len(agent_returns)), smoothed,
                           color=colors[agent_id], linewidth=2, label=f'Agent {agent_id}')
                else:
                    ax.plot(agent_returns, color=colors[agent_id], linewidth=1.5,
                           label=f'Agent {agent_id}')

            ax.set_title('Per-Agent Returns')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.legend(loc='upper left', ncol=4, fontsize=8)
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
            'steps': self.steps,
            'agent_returns': self.agent_returns
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
        # Backward compat: older logs may not have agent_returns
        self.agent_returns = data.get('agent_returns', [])


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

    def record(self, env, actions: Dict[int, int] = None,
               rewards: Dict[int, float] = None,
               values: np.ndarray = None,
               action_probs: np.ndarray = None) -> None:
        """Record current environment state."""
        step_data = StepData(
            positions=env.agent_positions.cpu().numpy().copy(),
            hp=env.agent_hp.cpu().numpy().copy(),
            alive=env.agent_alive.cpu().numpy().copy(),
            poor_food=env.poor_food.cpu().numpy().copy(),
            rich_food=env.rich_food.cpu().numpy().copy(),
            signals=env.signals.cpu().numpy().copy(),
            actions=actions or {},
            rewards=rewards or {},
            values=values.copy() if values is not None else None,
            action_probs=action_probs.copy() if action_probs is not None else None
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

    # Agent colors for the info box
    agent_colors = plt.cm.tab10(np.linspace(0, 1, config.n_agents))

    gs = config.grid_size
    n_agents = config.n_agents

    # Pre-compute cumulative rewards for each frame
    cumulative_rewards = np.zeros((len(recording), n_agents))
    for frame_idx in range(len(recording)):
        if frame_idx > 0:
            cumulative_rewards[frame_idx] = cumulative_rewards[frame_idx - 1].copy()
        step_data = recording[frame_idx]
        if step_data.rewards:
            for agent_id, reward in step_data.rewards.items():
                cumulative_rewards[frame_idx, agent_id] += reward

    # Check if we have value estimates
    has_values = recording[0].values is not None

    # Check if we have action probabilities (unified action space)
    has_probs = recording[0].action_probs is not None

    # Pre-compute kills: track (killer, victim) pairs by checking when agents die
    # and who dealt damage to them (from ledger snapshots)
    kill_matrix = np.zeros((n_agents, n_agents), dtype=bool)  # kill_matrix[killer, victim] = True
    if ledger_snapshots is not None and len(ledger_snapshots) > 1:
        for frame_idx in range(1, len(recording)):
            prev_alive = recording[frame_idx - 1].alive
            curr_alive = recording[frame_idx].alive
            for victim_id in range(n_agents):
                if prev_alive[victim_id] and not curr_alive[victim_id]:
                    # Agent died - find who dealt the most damage to them
                    damage_to_victim = ledger_snapshots[frame_idx][:, victim_id, 0]  # Damage Dealt channel
                    if damage_to_victim.max() > 0:
                        killer_id = int(np.argmax(damage_to_victim))
                        kill_matrix[killer_id, victim_id] = True

    # Create figure layout based on whether we have ledger data and probabilities
    has_ledger = ledger_snapshots is not None and len(ledger_snapshots) > 0 and show_ledger

    if has_ledger and has_probs:
        # Grid on left, probability matrices top-right, all 4 ledger heatmaps in bottom row
        fig = plt.figure(figsize=(20, 12))
        # Grid on left (main view)
        ax_grid = fig.add_axes([0.02, 0.35, 0.38, 0.60])
        # Probability matrices top-right (compact)
        ax_move_probs = fig.add_axes([0.42, 0.55, 0.27, 0.40])
        ax_interact_probs = fig.add_axes([0.71, 0.55, 0.27, 0.40])
        # ALL 4 ledger heatmaps in bottom row
        ax_ledger = [
            fig.add_axes([0.02, 0.05, 0.22, 0.28]),   # Damage
            fig.add_axes([0.26, 0.05, 0.22, 0.28]),   # Food Given
            fig.add_axes([0.52, 0.05, 0.22, 0.28]),   # Coop Count
            fig.add_axes([0.76, 0.05, 0.22, 0.28]),   # Defense Score
        ]
        ledger_names = ['Damage Dealt', 'Food Given', 'Coop Count', 'Defense Score']
        ledger_cmaps = ['Reds', 'Greens', 'Blues', 'Purples']
    elif has_ledger:
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
        ax_move_probs = None
        ax_interact_probs = None
    elif has_probs:
        # Grid on left, probability panels on right
        fig = plt.figure(figsize=(18, 10))
        ax_grid = fig.add_axes([0.05, 0.12, 0.50, 0.83])
        ax_move_probs = fig.add_axes([0.58, 0.12, 0.18, 0.83])
        ax_interact_probs = fig.add_axes([0.79, 0.12, 0.18, 0.83])
        ax_ledger = None
        ledger_names = None
        ledger_cmaps = None
    else:
        fig, ax_grid = plt.subplots(1, 1, figsize=(12, 10))
        ax_ledger = None
        ax_move_probs = None
        ax_interact_probs = None
        ledger_names = None
        ledger_cmaps = None

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

        # Draw relationship lines between agents (friend=green, enemy=red)
        relationship_info = {}  # Store for agent info text
        if has_ledger:
            ledger_data = ledger_snapshots[frame_idx]

            for viewer_id in range(n_agents):
                if not step_data.alive[viewer_id]:
                    continue

                relationship_info[viewer_id] = []
                viewer_row, viewer_col = step_data.positions[viewer_id]

                for target_id in range(n_agents):
                    if target_id == viewer_id or not step_data.alive[target_id]:
                        continue

                    rel, strength = classify_relationship(ledger_data, viewer_id, target_id)
                    relationship_info[viewer_id].append((target_id, rel))

                    # Only draw line once per pair (from lower to higher id)
                    if viewer_id < target_id:
                        target_row, target_col = step_data.positions[target_id]

                        if rel == 'friend':
                            color = 'limegreen'
                            linewidth = min(1 + strength / 20, 4)
                            alpha = 0.6
                        elif rel == 'enemy':
                            color = 'red'
                            linewidth = min(1 + strength / 20, 4)
                            alpha = 0.6
                        else:
                            continue  # Don't draw neutral lines

                        ax_grid.plot([viewer_col, target_col], [viewer_row, target_row],
                                    color=color, linewidth=linewidth, alpha=alpha,
                                    linestyle='-', zorder=1)

        # Draw attack, give, and cooperate interactions
        if step_data.actions:
            for agent_id, action_val in step_data.actions.items():
                if not step_data.alive[agent_id]:
                    continue

                # Handle both action formats:
                # Old format: (move_act, interact_act) tuple
                # New unified format: single int (0-14)
                if isinstance(action_val, tuple):
                    move_act, interact_act = action_val
                else:
                    # Unified action: 0-4 move, 5-8 atk, 9-12 give, 13 sig, 14 coop
                    unified_act = action_val
                    if unified_act < 5:
                        # Move action - no interaction to draw
                        continue
                    elif unified_act < 9:
                        # Attack: 5-8 -> interact_act 0-3
                        interact_act = unified_act - 5
                    elif unified_act < 13:
                        # Give: 9-12 -> interact_act 4-7
                        interact_act = unified_act - 5  # 9->4, 10->5, 11->6, 12->7
                    elif unified_act == 13:
                        # Signal
                        interact_act = INTERACT_SIGNAL
                    else:
                        # Cooperate (14)
                        interact_act = INTERACT_COOPERATE

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

        # Draw per-agent info box in top-right corner
        info_lines = []
        for agent_id in range(n_agents):
            cum_reward = cumulative_rewards[frame_idx, agent_id]

            # Build relationship string (e.g., "A1:F A2:E")
            rel_str = ""
            if agent_id in relationship_info:
                rel_parts = []
                for target_id, rel in relationship_info[agent_id]:
                    if rel == 'friend':
                        rel_parts.append(f"A{target_id}:F")
                    elif rel == 'enemy':
                        rel_parts.append(f"A{target_id}:E")
                if rel_parts:
                    rel_str = " | " + " ".join(rel_parts)

            if has_values and step_data.values is not None:
                value = step_data.values[agent_id]
                info_lines.append(f'A{agent_id}: R={cum_reward:+.1f}  V={value:.2f}{rel_str}')
            else:
                info_lines.append(f'A{agent_id}: R={cum_reward:+.1f}{rel_str}')

        info_text = '\n'.join(info_lines)
        # Position in upper right of grid (using axes coordinates)
        ax_grid.text(0.98, 0.98, info_text, transform=ax_grid.transAxes,
                    fontsize=8, fontfamily='monospace',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

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

                # Add X markers on Damage Dealt heatmap for kills
                if idx == 0:  # Damage Dealt
                    for killer_id in range(n_agents):
                        for victim_id in range(n_agents):
                            if kill_matrix[killer_id, victim_id]:
                                ax.plot(victim_id, killer_id, 'X', markersize=15,
                                       color='black', markeredgecolor='white', markeredgewidth=1.5)

        # Draw action probability MATRICES (heatmap style)
        # Unified action space: 0-4 move, 5-8 attack, 9-12 give, 13 signal, 14 coop
        if has_probs and ax_move_probs is not None and step_data.action_probs is not None:
            ax_move_probs.clear()
            ax_interact_probs.clear()

            move_names = ['UP', 'DN', 'LT', 'RT', 'ST']
            interact_names = ['ATK', 'GIV', 'SIG', 'COP']

            # Extract move probabilities from unified action probs (actions 0-4)
            move_data = step_data.action_probs[:, :5].copy()
            # Gray out dead agents
            for i in range(n_agents):
                if not step_data.alive[i]:
                    move_data[i, :] = 0

            im_move = ax_move_probs.imshow(move_data, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            ax_move_probs.set_title('Move Probabilities', fontsize=10, fontweight='bold')
            ax_move_probs.set_xticks(range(5))
            ax_move_probs.set_xticklabels(move_names, fontsize=8)
            ax_move_probs.set_yticks(range(n_agents))
            ax_move_probs.set_yticklabels([f'A{i}' for i in range(n_agents)], fontsize=8)

            # Annotate cells with probability values
            for i in range(n_agents):
                for j in range(5):
                    val = move_data[i, j]
                    color = 'white' if val > 0.5 else 'black'
                    ax_move_probs.text(j, i, f'{val:.2f}', ha='center', va='center',
                                      fontsize=7, color=color)

            # Aggregate interact probabilities from unified actions
            # ATK: sum of actions 5-8, GIV: sum of 9-12, SIG: 13, COP: 14
            interact_data = np.zeros((n_agents, 4))
            interact_data[:, 0] = step_data.action_probs[:, 5:9].sum(axis=1)   # ATK
            interact_data[:, 1] = step_data.action_probs[:, 9:13].sum(axis=1)  # GIV
            interact_data[:, 2] = step_data.action_probs[:, 13]                # SIG
            interact_data[:, 3] = step_data.action_probs[:, 14]                # COP
            for i in range(n_agents):
                if not step_data.alive[i]:
                    interact_data[i, :] = 0

            im_interact = ax_interact_probs.imshow(interact_data, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
            ax_interact_probs.set_title('Interact Probabilities', fontsize=10, fontweight='bold')
            ax_interact_probs.set_xticks(range(4))
            ax_interact_probs.set_xticklabels(interact_names, fontsize=8)
            ax_interact_probs.set_yticks(range(n_agents))
            ax_interact_probs.set_yticklabels([f'A{i}' for i in range(n_agents)], fontsize=8)

            # Annotate cells
            for i in range(n_agents):
                for j in range(4):
                    val = interact_data[i, j]
                    color = 'white' if val > 0.5 else 'black'
                    ax_interact_probs.text(j, i, f'{val:.2f}', ha='center', va='center',
                                          fontsize=7, color=color)

        return []

    interval = int(200 / speed)  # Base 200ms between frames
    anim = animation.FuncAnimation(fig, animate, frames=len(recording),
                                   interval=interval, blit=False, repeat=True)

    # === Pause/Resume Controls ===
    is_paused = [False]  # Use list for nonlocal mutation

    def on_key(event):
        """Handle spacebar to toggle pause/resume."""
        if event.key == ' ':
            if is_paused[0]:
                anim.resume()
                is_paused[0] = False
            else:
                anim.pause()
                is_paused[0] = True

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Add pause button widget
    from matplotlib.widgets import Button
    ax_pause = fig.add_axes([0.45, 0.02, 0.1, 0.04])
    btn_pause = Button(ax_pause, 'Pause')

    def toggle_pause(event):
        """Toggle pause/resume on button click."""
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
    # Demo with random environment - animated episode replay
    import torch
    from environment import GridWorld
    from utils import set_seed, get_device

    set_seed(42)
    config = Config()
    device = get_device()

    env = GridWorld(config, device)
    env.reset()

    # Record an episode with random actions
    recorder = EpisodeRecorder(config)
    recorder.record(env)  # Record initial state

    num_steps = 50  # Run 50 steps for demo
    for step in range(num_steps):
        actions = torch.randint(0, 15, (config.n_agents,), device=device)
        rewards, dones, info = env.step(actions)

        # Convert to dicts for recording
        actions_dict = {i: actions[i].item() for i in range(config.n_agents)}
        rewards_dict = {i: rewards[i].item() for i in range(config.n_agents)}

        recorder.record(env, actions=actions_dict, rewards=rewards_dict)

    # Replay the recorded episode with animation
    print(f"Replaying {len(recorder.steps)} frames...")
    replay_episode(
        recording=recorder.get_recording(),
        config=config,
        speed=1.0,
        ledger_snapshots=recorder.ledger_snapshots,
        show_ledger=True
    )
