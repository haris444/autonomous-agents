"""
Vectorized Training Script - Uses parallel environments for faster training.

This script uses VecEnv to run N environments in parallel, significantly
speeding up training by batching operations across environments.

Usage:
    python train_vec.py --n_envs 8 --pretrain
    python train_vec.py --n_envs 16 --batch_size 1024  # Fixed batch, more frequent updates
"""
import argparse
import csv
import os
import time
import torch
import torch.nn as nn

from config import Config
from batched_env import BatchedGridWorld as VecEnv
from ppo import VmapPPO
from buffer import VecBuffer
from scenarios import CURRICULUM, THRESHOLDS


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_vec(
    config: Config = None,
    n_envs: int = None,
    pretrain: bool = False,
    total_timesteps: int = None,
    target_batch_size: int = None,
    csv_log_path: str = None,
    output_dir: str = "checkpoints_vec"
):
    """
    Train using vectorized environments.

    Args:
        config: Training configuration
        n_envs: Number of parallel environments (overrides config)
        pretrain: If True, run curriculum pretraining
        total_timesteps: Override total timesteps
        target_batch_size: If set, collect this many transitions per update
                          (adjusts num_steps dynamically based on n_envs)
        csv_log_path: If set, log metrics to this CSV file
        output_dir: Directory for checkpoint saves
    """
    if config is None:
        config = Config()

    if pretrain:
        config.pretrain_mode = True
        config.curriculum_enabled = True

    if n_envs is not None:
        config.n_envs = n_envs
    n_envs = config.n_envs

    if total_timesteps is not None:
        config.total_timesteps = total_timesteps

    # Compute num_steps based on target batch size
    if target_batch_size is not None:
        # transitions_per_step = n_envs * n_agents
        # num_steps = target_batch_size / transitions_per_step
        transitions_per_step = n_envs * config.n_agents
        num_steps = max(1, target_batch_size // transitions_per_step)
        config.num_steps = num_steps
        actual_batch_size = num_steps * transitions_per_step
        print(f"Target batch size: {target_batch_size}")
        print(f"Adjusted num_steps: {num_steps} (actual batch: {actual_batch_size})")
    
    # Setup
    device = get_device()
    set_seed(config.seed)
    print(f"Training on {device}")
    print(f"Config: {config.n_agents} agents, {config.grid_size}x{config.grid_size} grid")
    print(f"Parallel environments: {n_envs}")
    print(f"Batch size: {config.num_steps} steps × {n_envs} envs × {config.n_agents} agents = {config.num_steps * n_envs * config.n_agents}")

    # Initialize vectorized environment
    vec_env = VecEnv(config, device, n_envs=n_envs)

    # Initialize PPO
    multi_agent = VmapPPO(config, device)

    # Sync curriculum phase
    if config.pretrain_mode and config.curriculum_enabled:
        multi_agent.set_curriculum_phase(vec_env.get_curriculum_phase())

    # Create vectorized buffer
    buffer = VecBuffer(config, device, n_envs=n_envs)

    # Training tracking
    global_step = 0
    effective_batch_size = config.num_steps * n_envs * config.n_agents
    num_updates = config.total_timesteps // effective_batch_size
    start_time = time.time()

    # Episode tracking (per env)
    episode_rewards = torch.zeros(n_envs, config.n_agents, device=device)
    episode_steps = torch.zeros(n_envs, dtype=torch.long, device=device)
    completed_episodes = 0
    total_returns = []

    # CSV logging
    csv_file = None
    csv_writer = None
    if csv_log_path:
        csv_file = open(csv_log_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['update', 'global_step', 'episode_return', 'episodes_completed',
                             'curriculum_phase', 'sps', 'wall_time', 'total_loss'])

    print(f"Starting training for {config.total_timesteps} timesteps ({num_updates} updates)")
    print("-" * 60)

    # Initial reset (only once!)
    obs = vec_env.reset()

    for update in range(num_updates):
        # Get current phase and scenario
        phase = vec_env.get_curriculum_phase()
        scenario = CURRICULUM.get(phase) if config.pretrain_mode else None

        if scenario is not None:
            scenario_config = scenario.get_config()
            multi_agent.set_n_active(scenario_config.n_active_agents)
            multi_agent.set_clone_mode(scenario_config.clone_mode)
        else:
            scenario_config = None

        # === ROLLOUT PHASE ===
        buffer.reset()  # Reset buffer, but NOT the environments!

        # Apply scenario (only needed for pretraining)
        if scenario is not None and update == 0:
            vec_env.apply_scenario(scenario)
            obs = vec_env.get_observations()

        for step in range(config.num_steps):
            global_step += n_envs * config.n_agents

            with torch.no_grad():
                # Get action masks [n_envs, n_agents, 5]
                direction_mask, action_type_mask = vec_env.get_action_masks()

                # Mask out attacks in coop phases
                if scenario_config is not None and scenario_config.partner_mode == "always_coop":
                    action_type_mask[:, 0, 1] = False  # Disable ATTACK for agent 0
                    action_type_mask[:, 1, 1] = False  # Disable ATTACK for agent 1

                # Get actions and values [n_envs, n_agents]
                directions, action_types, log_probs, _, values = multi_agent.vec_get_actions_and_values(
                    obs, direction_mask=direction_mask, action_type_mask=action_type_mask
                )

                # Get auxiliary values [n_envs, n_agents]
                aux_values = multi_agent.vec_get_auxiliary_values(obs)

            # Environment step
            next_obs, rewards, dones, infos = vec_env.step(directions, action_types)

            # Track episode rewards
            episode_rewards += rewards
            episode_steps += 1

            # Store in buffer
            buffer.store(
                obs, directions, action_types, log_probs,
                rewards, dones, values,
                direction_mask, action_type_mask,
                reward_survival=infos['reward_survival'],
                reward_resource=infos['reward_resource'],
                reward_social=infos['reward_social'],
                value_survival=aux_values['survival'],
                value_resource=aux_values['resource'],
                value_social=aux_values['social']
            )

            # Check for completed episodes (all agents done in an env)
            env_dones = dones.all(dim=1)  # [n_envs]
            if env_dones.any():
                for e in range(n_envs):
                    if env_dones[e]:
                        # Record episode return (agent 0)
                        ret = episode_rewards[e, 0].item()
                        total_returns.append(ret)
                        completed_episodes += 1

                        # Save checkpoint every 10 episodes
                        if completed_episodes % 10 == 0:
                            os.makedirs(output_dir, exist_ok=True)
                            checkpoint_path = f"{output_dir}/checkpoint_ep{completed_episodes}.pt"
                            torch.save({
                                'episode': completed_episodes,
                                'model_state_dict': multi_agent.networks[0].state_dict(),
                                'optimizer_state_dict': multi_agent.optimizers[0].state_dict(),
                                'avg_return': sum(total_returns[-10:]) / len(total_returns[-10:]),
                                'global_step': global_step,
                            }, checkpoint_path)
                            print(f"  [Saved checkpoint: {checkpoint_path}]")

                        # Reset tracking for this env
                        episode_rewards[e] = 0
                        episode_steps[e] = 0

            obs = next_obs

        # === COMPUTE GAE ===
        with torch.no_grad():
            next_values = multi_agent.vec_get_values(obs)
            next_aux = multi_agent.vec_get_auxiliary_values(obs)
            next_dones = torch.zeros(n_envs, config.n_agents, device=device)  # Not terminal yet

        buffer.compute_gae(
            next_values, next_dones,
            next_value_survival=next_aux['survival'],
            next_value_resource=next_aux['resource'],
            next_value_social=next_aux['social']
        )

        # === PPO UPDATE ===
        metrics = multi_agent.update_from_vec_buffer(buffer)

        # === LOGGING ===
        elapsed = time.time() - start_time
        sps = global_step / elapsed
        avg_return = sum(total_returns[-100:]) / max(1, len(total_returns[-100:]))

        print(f"Update {update}/{num_updates} | "
              f"Phase {phase} | "
              f"Ep: {completed_episodes} | "
              f"Avg Return: {avg_return:.1f} | "
              f"SPS: {sps:.0f} | "
              f"Loss: {metrics.get('total_loss', 0):.3f}")

        if csv_writer:
            loss_val = metrics.get('total_loss', 0)
            if hasattr(loss_val, 'item'):
                loss_val = loss_val.item()
            csv_writer.writerow([
                update, global_step, f'{avg_return:.2f}', completed_episodes,
                phase, f'{sps:.0f}', f'{elapsed:.1f}', f'{loss_val:.4f}'
            ])
            csv_file.flush()

    # Final summary
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Training complete!")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total steps: {global_step:,}")
    print(f"Steps/sec: {global_step / elapsed:.0f}")
    print(f"Episodes completed: {completed_episodes}")
    print(f"PPO updates: {num_updates}")
    if total_returns:
        print(f"Final avg return: {sum(total_returns[-100:]) / len(total_returns[-100:]):.1f}")

    if csv_file:
        csv_file.close()
        print(f"CSV log saved to {csv_log_path}")


def main():
    parser = argparse.ArgumentParser(description='Vectorized RL Training')
    parser.add_argument('--n_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--pretrain', action='store_true', help='Run curriculum pretraining')
    parser.add_argument('--timesteps', type=int, default=500_000, help='Total timesteps')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Target batch size per update (adjusts num_steps automatically)')
    parser.add_argument('--grid_size', type=int, default=None, help='Grid size (e.g. 8 for 8x8)')
    parser.add_argument('--n_agents', type=int, default=None, help='Number of agents')
    parser.add_argument('--episode_length', type=int, default=None,
                        help='Max steps per episode (overrides config)')
    parser.add_argument('--fourier-bands', type=int, default=None,
                        help='Number of Fourier frequency bands (default: 4, produces 4*bands position dims)')
    parser.add_argument('--csv-log', type=str, default=None,
                        help='Path to CSV log file for metrics tracking')
    parser.add_argument('--output-dir', type=str, default='checkpoints_vec',
                        help='Directory for checkpoint saves (default: checkpoints_vec)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    args = parser.parse_args()

    config = Config()
    if args.fourier_bands is not None:
        config.fourier_bands = args.fourier_bands
        config.__post_init__()  # Recompute entity_token_dim
    if args.grid_size:
        config.grid_size = args.grid_size
    if args.n_agents:
        config.n_agents = args.n_agents
    if args.seed is not None:
        config.seed = args.seed
    if args.episode_length:
        config.max_steps_per_episode = args.episode_length
        # Sync num_steps if batch_size not explicitly set to override it later
        if args.batch_size is None:
            config.num_steps = args.episode_length

    train_vec(
        config=config,
        n_envs=args.n_envs,
        pretrain=args.pretrain,
        total_timesteps=args.timesteps,
        target_batch_size=args.batch_size,
        csv_log_path=args.csv_log,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
