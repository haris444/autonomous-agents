"""
Main training loop for Multi-Agent RL.

Orchestrates rollout collection, GAE computation, and PPO updates.

Usage:
    python train.py                    # Train without visualization
    python train.py --visualize        # Train with live visualization
    python train.py --visualize --render-every 50  # Render grid every 50 updates
"""
from typing import Dict, Optional
import time
import argparse

import torch
import matplotlib.pyplot as plt

from config import Config
from environment import GridWorld
from network import ActorCritic
from buffer import RolloutBuffer
from ppo import PPO
from utils import set_seed, get_device


def stack_observations(obs: Dict[int, Dict[str, torch.Tensor]], n_agents: int) -> Dict[str, torch.Tensor]:
    """Stack per-agent observations into batched tensors (for dict-based API)."""
    return {
        'spatial': torch.stack([obs[i]['spatial'] for i in range(n_agents)]),
        'ledger': torch.stack([obs[i]['ledger'] for i in range(n_agents)]),
        'signals': torch.stack([obs[i]['signals'] for i in range(n_agents)]),
        'self_hp': torch.stack([obs[i]['self_hp'] for i in range(n_agents)])
    }


def convert_obs_dict_to_batched(obs: Dict[int, Dict[str, torch.Tensor]], n_agents: int) -> Dict[str, torch.Tensor]:
    """Convert dict-of-dicts to batched tensor dict (for backward compat)."""
    return stack_observations(obs, n_agents)


def train(config: Config = None, visualize: bool = False,
          render_every: int = 50, plot_every: int = 10, record_every: int = 100,
          output_dir: str = "viz_output"):
    """
    Main training function.

    Args:
        config: Training configuration
        visualize: Enable live visualization
        render_every: Render grid every N updates
        plot_every: Update training curves every N updates
        record_every: Record episode every N updates
        output_dir: Directory for visualization output and checkpoints
    """
    if config is None:
        config = Config()

    # Setup
    device = get_device()
    set_seed(config.seed)
    print(f"Training on {device}")
    print(f"Config: {config.n_agents} agents, {config.grid_size}x{config.grid_size} grid")

    # Initialize components
    env = GridWorld(config, device)
    network = ActorCritic(config).to(device)
    buffer = RolloutBuffer(config, device)
    ppo = PPO(config, network, device)

    # Visualization components (lazy import to avoid matplotlib issues when not visualizing)
    renderer = None
    logger = None
    recorder = None
    fig_grid = None
    fig_curves = None

    if visualize:
        import os
        from visualize import GridRenderer, TrainingLogger, EpisodeRecorder

        # Create output directory
        viz_dir = output_dir
        os.makedirs(viz_dir, exist_ok=True)

        renderer = GridRenderer(config)
        logger = TrainingLogger()
        recorder = EpisodeRecorder(config)

        # Use non-interactive backend to avoid Windows crashes
        plt.switch_backend('Agg')
        print(f"Visualization enabled: saving to {viz_dir}/")
        print(f"  Grid every {render_every}, curves every {plot_every}, record every {record_every}")

    # Training tracking
    global_step = 0
    num_updates = config.total_timesteps // config.batch_size
    start_time = time.time()

    # Episode tracking
    episode_returns = []
    current_episode_rewards = {i: 0.0 for i in range(config.n_agents)}

    print(f"Starting training for {config.total_timesteps} timesteps ({num_updates} updates)")
    print("-" * 60)

    for update in range(num_updates):
        # === ROLLOUT PHASE ===
        obs = env.reset()  # Now returns batched Dict[str, Tensor]
        buffer.reset()

        # Reset episode tracking (tensor-based)
        current_episode_rewards = torch.zeros(config.n_agents, device=device)

        for step in range(config.num_steps):
            global_step += config.n_agents

            # obs is already batched from vectorized env
            # Get actions from policy
            with torch.no_grad():
                move_actions, interact_actions, log_probs, _, values = \
                    network.get_action_and_value(obs)

            # Environment step (tensor-based API)
            next_obs, rewards, dones, infos = env.step(move_actions, interact_actions)

            # Track episode rewards (tensor)
            current_episode_rewards += rewards

            # Store transition (convert to dict format for buffer compatibility)
            obs_dict = {i: {k: obs[k][i] for k in obs} for i in range(config.n_agents)}
            rewards_dict = {i: rewards[i].item() for i in range(config.n_agents)}
            dones_dict = {i: dones[i].item() for i in range(config.n_agents)}

            buffer.store(
                obs_dict, move_actions, interact_actions,
                log_probs, rewards_dict, dones_dict, values
            )

            obs = next_obs

            # Check for episode end
            if dones.all():
                avg_return = current_episode_rewards.mean().item()
                episode_returns.append(avg_return)

                # Reset for next episode within rollout
                obs = env.reset()
                current_episode_rewards.zero_()

        # === ADVANTAGE COMPUTATION ===
        with torch.no_grad():
            # obs is already batched from vectorized env
            next_value = network.get_value(obs).squeeze(-1)
            next_done = dones.float()

        buffer.compute_gae(next_value, next_done)

        # === PPO UPDATE PHASE ===
        metrics = ppo.update(buffer)

        # === LOGGING ===
        if update % 10 == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed if elapsed > 0 else 0

            avg_return = sum(episode_returns[-10:]) / max(1, len(episode_returns[-10:]))

            print(f"Update {update:4d} | "
                  f"Step {global_step:8d} | "
                  f"Return {avg_return:7.2f} | "
                  f"Loss {metrics['total_loss']:7.4f} | "
                  f"Entropy {metrics['entropy']:5.3f} | "
                  f"KL {metrics['approx_kl']:6.4f} | "
                  f"SPS {sps:5.0f}")

        # === VISUALIZATION ===
        if visualize:
            # Log metrics
            if episode_returns:
                logger.log(global_step, episode_return=episode_returns[-1], metrics=metrics)
            else:
                logger.log(global_step, metrics=metrics)

            # Render grid snapshot (save to file)
            if update % render_every == 0:
                fig_grid = renderer.render(env, show=False)
                fig_grid.suptitle(f'Update {update} | Step {global_step}')
                grid_path = f"{viz_dir}/grid_update_{update:05d}.png"
                fig_grid.savefig(grid_path, dpi=100, bbox_inches='tight')
                plt.close(fig_grid)
                print(f"  Saved grid -> {grid_path}")

            # Update training curves (save to file, overwrite)
            if update % plot_every == 0 and update > 0:
                fig_curves = logger.plot(window=50, show=False)
                curves_path = f"{viz_dir}/training_curves.png"
                fig_curves.savefig(curves_path, dpi=100, bbox_inches='tight')
                plt.close(fig_curves)

            # Record episode for replay
            if update % record_every == 0:
                recorder.reset()
                # Run a full evaluation episode
                eval_obs = env.reset()
                for _ in range(config.max_steps_per_episode):
                    with torch.no_grad():
                        m_act, i_act, _, _, _ = network.get_action_and_value(eval_obs)
                    # Record state + actions before stepping
                    actions_dict = {i: (m_act[i].item(), i_act[i].item()) for i in range(config.n_agents)}
                    recorder.record(env, actions=actions_dict)
                    eval_obs, _, eval_dones, _ = env.step(m_act, i_act)
                    if eval_dones.all():
                        break
                rec_path = f"{viz_dir}/episode_update_{update}.pt"
                recorder.save(rec_path)
                print(f"  Recorded episode ({len(recorder.steps)} steps) -> {rec_path}")
                # Reset env state for next rollout
                obs = env.reset()

        # === CHECKPOINTING ===
        if update % 100 == 0 and update > 0:
            import os
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = f"{output_dir}/checkpoint_{update}.pt"
            torch.save({
                'update': update,
                'global_step': global_step,
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final save
    import os
    os.makedirs(output_dir, exist_ok=True)
    final_path = f"{output_dir}/final_model.pt"
    torch.save({
        'network_state_dict': network.state_dict(),
        'config': config
    }, final_path)
    print(f"Training complete! Saved {final_path}")

    # Save training logs if visualizing
    if visualize and logger is not None:
        logs_path = f"{viz_dir}/training_logs.pt"
        logger.save(logs_path)
        print(f"Saved training logs to {logs_path}")

        # Save final training curves
        fig_final = logger.plot(window=100, show=False)
        final_path = f"{viz_dir}/training_curves_final.png"
        fig_final.savefig(final_path, dpi=150, bbox_inches='tight')
        plt.close(fig_final)
        print(f"Saved final curves to {final_path}")

    return network, episode_returns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Multi-Agent RL system')

    # Visualization
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Enable live visualization during training')
    parser.add_argument('--render-every', type=int, default=50,
                        help='Render grid every N updates (default: 50)')
    parser.add_argument('--plot-every', type=int, default=10,
                        help='Update training curves every N updates (default: 10)')
    parser.add_argument('--record-every', type=int, default=100,
                        help='Record episode every N updates (default: 100)')

    # Training
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')

    # Output
    parser.add_argument('--output-dir', '-o', type=str, default='viz_output',
                        help='Output directory for visualizations and checkpoints (default: viz_output)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create config with any overrides
    config = Config()
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps
    if args.seed is not None:
        config.seed = args.seed

    train(
        config=config,
        visualize=args.visualize,
        render_every=args.render_every,
        plot_every=args.plot_every,
        record_every=args.record_every,
        output_dir=args.output_dir
    )
