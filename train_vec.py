"""
Vectorized Training Script - Uses parallel environments for faster training.

This script uses VecEnv to run N environments in parallel, significantly
speeding up training by batching operations across environments.

Usage:
    python train_vec.py --n_envs 8 --pretrain
    python train_vec.py --n_envs 16 --batch_size 1024  # Fixed batch, more frequent updates
    python train_vec.py --n_envs 8 --pretrain --visualize  # With visualization
"""
import argparse
import csv
import os
import time
import torch
import torch.nn as nn

from config import Config
from batched_env import BatchedGridWorld as VecEnv
from ppo import VmapPPO, load_state_dict_flexible
from buffer import VecBuffer
from scenarios import CURRICULUM, THRESHOLDS
from train import apply_scripted_partner


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
    output_dir: str = "checkpoints_vec",
    visualize: bool = False,
    render_every: int = 50,
    record_every: int = 100,
    checkpoint_path: str = None,
    start_phase: int = None,
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
        visualize: Enable visualization (grid snapshots, training curves, episode recording)
        render_every: Save grid snapshot every N updates
        record_every: Record episode for replay every N episodes
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
    total_batch = config.num_steps * n_envs * config.n_agents
    print(f"Batch size: {config.num_steps} steps x {n_envs} envs x {config.n_agents} agents = {total_batch}")
    print(f"Minibatch size: {total_batch // config.num_minibatches} ({config.num_minibatches} minibatches)")

    # Initialize vectorized environment
    vec_env = VecEnv(config, device, n_envs=n_envs)

    # Initialize PPO
    multi_agent = VmapPPO(config, device)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if 'network_state_dicts' in ckpt:
            # New format: per-agent state dicts
            weights_resized = False
            for i, net in enumerate(multi_agent.networks):
                if i < len(ckpt['network_state_dicts']):
                    weights_resized |= load_state_dict_flexible(net, ckpt['network_state_dicts'][i])
                else:
                    weights_resized |= load_state_dict_flexible(net, ckpt['network_state_dicts'][0])
            if weights_resized:
                print("  Skipping optimizer state (weight shapes changed)")
            elif 'optimizer_state_dicts' in ckpt:
                for i, opt in enumerate(multi_agent.optimizers):
                    if i < len(ckpt['optimizer_state_dicts']):
                        opt.load_state_dict(ckpt['optimizer_state_dicts'][i])
        else:
            # Legacy format: single state dict → replicate to all networks
            state_dict = ckpt['model_state_dict']
            weights_resized = False
            for net in multi_agent.networks:
                weights_resized |= load_state_dict_flexible(net, state_dict)
            if not weights_resized:
                multi_agent.optimizers[0].load_state_dict(ckpt['optimizer_state_dict'])
            else:
                print("  Skipping optimizer state (weight shapes changed)")
        # Restore curriculum phase if saved
        if 'curriculum_phase' in ckpt:
            vec_env.set_curriculum_phase(ckpt['curriculum_phase'])
            print(f"  Resumed from episode {ckpt['episode']}, phase {ckpt['curriculum_phase']}, "
                  f"global_step {ckpt['global_step']}, avg_return {ckpt['avg_return']:.1f}")
        else:
            print(f"  Resumed from episode {ckpt['episode']}, global_step {ckpt['global_step']}, "
                  f"avg_return {ckpt['avg_return']:.1f}")
            print(f"  Warning: checkpoint has no curriculum_phase, starting at phase {vec_env.get_curriculum_phase()}")

    # Override curriculum phase if specified
    if start_phase is not None:
        vec_env.set_curriculum_phase(start_phase)
        print(f"  Overriding curriculum phase to {start_phase}")

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
    env0_episodes = 0
    next_record_at = record_every  # trigger first recording after this many total episodes
    total_returns = []

    # CSV logging
    csv_file = None
    csv_writer = None
    if csv_log_path:
        csv_file = open(csv_log_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['update', 'global_step', 'episode_return', 'episodes_completed',
                             'curriculum_phase', 'sps', 'wall_time', 'total_loss'])

    # === VISUALIZATION SETUP ===
    renderer = None
    logger = None
    recorder = None
    recording_active = False  # True when we're recording env 0's current episode

    if visualize:
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        from visualize import GridRenderer, TrainingLogger, EpisodeRecorder, render_ledger_heatmaps

        os.makedirs(output_dir, exist_ok=True)
        renderer = GridRenderer(config)
        logger = TrainingLogger()
        recorder = EpisodeRecorder(config)
        print(f"Visualization enabled: saving to {output_dir}/")
        print(f"  Grid every {render_every} updates, record every {record_every} episodes")

    print(f"Starting training for {config.total_timesteps} timesteps ({num_updates} updates)")
    print("-" * 60)

    # Initial reset (only once!)
    obs = vec_env.reset()
    prev_phase = None
    last_partner_info = None
    from ledger import Ledger

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

        # Apply scenario on first update or whenever phase changes
        if scenario is not None and (prev_phase is None or phase != prev_phase):
            vec_env.apply_scenario(scenario)
            obs = vec_env.get_observations()
        prev_phase = phase

        for step in range(config.num_steps):
            global_step += n_envs * config.n_agents

            with torch.no_grad():
                # Get action masks [n_envs, n_agents, 5]
                direction_mask, action_type_mask = vec_env.get_action_masks()

                # Get actions, values, and aux values in single forward pass
                directions, action_types, log_probs, _, values, aux_values = multi_agent.vec_get_actions_and_values(
                    obs, direction_mask=direction_mask, action_type_mask=action_type_mask
                )

            # Apply scripted partner behavior (phase 10+: per-env relationship)
            if scenario_config is not None and scenario_config.partner_mode == "scripted":
                shadow = vec_env._shadow_env
                for e in range(n_envs):
                    rel = vec_env.partner_relationships[e]
                    if rel is not None:
                        shadow.agent_positions = vec_env.agent_positions[e]
                        shadow.agent_alive = vec_env.agent_alive[e]
                        shadow.rich_food = vec_env.rich_food[e]
                        shadow.ledger.tensor = vec_env.ledger_tensor[e]
                        dirs_e, acts_e = apply_scripted_partner(
                            shadow, directions[e], action_types[e], relationship=rel
                        )
                        directions[e] = dirs_e
                        action_types[e] = acts_e

            # Record env 0 state before step (if recording active)
            if recording_active:
                env0 = vec_env.get_first_env()
                actions_dict = {
                    i: (directions[0, i].item(), action_types[0, i].item())
                    for i in range(config.n_agents)
                }
                values_np = values[0].cpu().numpy()
                aux_np = {
                    'survival': aux_values['survival'][0].cpu().numpy(),
                    'resource': aux_values['resource'][0].cpu().numpy(),
                    'social': aux_values['social'][0].cpu().numpy(),
                }
                recorder.record(env0, actions=actions_dict, values=values_np,
                                aux_values=aux_np)

            # Environment step
            next_obs, rewards, dones, infos = vec_env.step(directions, action_types)

            # Update recorded step with rewards
            if recording_active and recorder.steps:
                recorder.steps[-1].rewards = {
                    i: rewards[0, i].item() for i in range(config.n_agents)
                }

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

                        # Capture partner info from terminal ledger (env 0 only, phase 9+)
                        if e == 0 and phase >= 9 and infos.get('terminal_ledger') is not None:
                            ledger_data = infos['terminal_ledger'][0]  # [n_agents, n_agents, 4]
                            rel = infos['terminal_relationships'][0] or 'unknown'
                            last_partner_info = {
                                'rel': rel,
                                'dmg': ledger_data[1:, 0, Ledger.DAMAGE_DEALT].sum().item(),
                                'food': ledger_data[1:, 0, Ledger.FOOD_GIVEN].sum().item(),
                                'coop': ledger_data[1:, 0, Ledger.COOP_COUNT].sum().item(),
                                'def': ledger_data[1:, 0, Ledger.DEFENSE_SCORE].sum().item(),
                            }

                        # Track env 0 episodes separately for recording
                        if e == 0:
                            env0_episodes += 1

                            # Save recording if we were recording
                            if recording_active:
                                recording_active = False
                                replay_path = f"{output_dir}/replay_ep{env0_episodes}.pt"
                                recorder.save(replay_path)
                                print(f"  [Saved replay: {replay_path} ({len(recorder.steps)} steps)]")

                            # Start new recording if it's time (based on total episodes)
                            if visualize and completed_episodes >= next_record_at:
                                recording_active = True
                                recorder.reset()
                                next_record_at = completed_episodes + record_every

                        # Save checkpoint every 100 episodes
                        if completed_episodes % 100 == 0:
                            os.makedirs(output_dir, exist_ok=True)
                            checkpoint_path = f"{output_dir}/checkpoint_ep{completed_episodes}.pt"
                            torch.save({
                                'episode': completed_episodes,
                                'model_state_dict': multi_agent.networks[0].state_dict(),
                                'network_state_dicts': [net.state_dict() for net in multi_agent.networks],
                                'optimizer_state_dict': multi_agent.optimizers[0].state_dict(),
                                'optimizer_state_dicts': [opt.state_dict() for opt in multi_agent.optimizers],
                                'avg_return': sum(total_returns[-10:]) / len(total_returns[-10:]),
                                'global_step': global_step,
                                'curriculum_phase': vec_env.get_curriculum_phase(),
                                'config': config.to_dict(),
                            }, checkpoint_path)
                            print(f"  [Saved checkpoint: {checkpoint_path}]")

                        # Reset tracking for this env
                        episode_rewards[e] = 0
                        episode_steps[e] = 0

            obs = next_obs

        # === COMPUTE GAE ===
        with torch.no_grad():
            _, _, _, _, next_values, next_aux = multi_agent.vec_get_actions_and_values(obs)
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

        # Build phase type string
        if config.pretrain_mode:
            thresh = THRESHOLDS.get(phase, 0)
            if phase >= 9:
                phase_type = "Social"
            elif phase >= 6:
                phase_type = "Coop"
            else:
                phase_type = "Solo"
            phase_str = f"Phase {phase} {phase_type} ({avg_return:.0f}/{thresh})"
        else:
            phase_str = f"Phase {phase}"

        # Build partner info string
        partner_str = ""
        if phase >= 9 and last_partner_info is not None:
            p = last_partner_info
            rel = p['rel'].upper()[:3]
            partner_str = f" | Partner[{rel}]: food={p['food']:.0f} coop={p['coop']:.0f} def={p['def']:.0f} dmg={p['dmg']:.0f}"

        print(f"Update {update}/{num_updates} | "
              f"{phase_str} | "
              f"Ep: {completed_episodes} | "
              f"Avg Return: {avg_return:.1f} | "
              f"SPS: {sps:.0f} | "
              f"Loss: {metrics.get('total_loss', 0):.3f}"
              f"{partner_str}")

        if csv_writer:
            loss_val = metrics.get('total_loss', 0)
            if hasattr(loss_val, 'item'):
                loss_val = loss_val.item()
            csv_writer.writerow([
                update, global_step, f'{avg_return:.2f}', completed_episodes,
                phase, f'{sps:.0f}', f'{elapsed:.1f}', f'{loss_val:.4f}'
            ])
            csv_file.flush()

        # === VISUALIZATION ===
        if visualize:
            import matplotlib.pyplot as plt

            # Log metrics
            logger.log(global_step, episode_return=avg_return, metrics=metrics)

            # Grid snapshot
            if update % render_every == 0:
                env0 = vec_env.get_first_env()
                fig = renderer.render(env0, show=False)
                fig.suptitle(f'Update {update} | Step {global_step} | Phase {phase}')
                grid_path = f"{output_dir}/grid_update_{update:05d}.png"
                fig.savefig(grid_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                print(f"  [Saved grid: {grid_path}]")

                # Ledger heatmap
                fig_ledger = render_ledger_heatmaps(vec_env.ledger.tensor, show=False)
                fig_ledger.suptitle(f'Ledger - Update {update}')
                ledger_path = f"{output_dir}/ledger_update_{update:05d}.png"
                fig_ledger.savefig(ledger_path, dpi=100, bbox_inches='tight')
                plt.close(fig_ledger)

            # Training curves (overwrite each time)
            if update % 10 == 0 and update > 0:
                fig_curves = logger.plot(window=50, show=False)
                curves_path = f"{output_dir}/training_curves.png"
                fig_curves.savefig(curves_path, dpi=100, bbox_inches='tight')
                plt.close(fig_curves)

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

    if visualize and logger is not None:
        logger.save(f"{output_dir}/training_log.pt")
        print(f"Training log saved to {output_dir}/training_log.pt")


def main():
    parser = argparse.ArgumentParser(description='Vectorized RL Training')
    parser.add_argument('--n_envs', type=int, default=None, help='Number of parallel environments (default: 8)')
    parser.add_argument('--pretrain', action='store_true', help='Run curriculum pretraining')
    parser.add_argument('--timesteps', type=int, default=None, help='Total timesteps (default: 500000)')
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
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Enable visualization (grid snapshots, curves, episode recording)')
    parser.add_argument('--render-every', type=int, default=50,
                        help='Save grid snapshot every N updates (default: 50)')
    parser.add_argument('--record-every', type=int, default=100,
                        help='Record episode for replay every N episodes (default: 100)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pt file to resume from')
    parser.add_argument('--num_minibatches', type=int, default=None,
                        help='Number of minibatches per PPO epoch (default: 1)')
    parser.add_argument('--update_epochs', type=int, default=None,
                        help='Number of PPO epochs per update (default: 4)')
    parser.add_argument('--phase', type=int, default=None,
                        help='Override starting curriculum phase')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Path to experiment YAML config file')
    parser.add_argument('--ablate-ledger', action='store_true',
                        help='Zero out social features in observations (control experiment)')
    args = parser.parse_args()

    # Build config: YAML base (if provided) with CLI overrides on top
    if args.experiment:
        from experiment_config import load_yaml, build_config
        yaml_dict = load_yaml(args.experiment)
        cli_overrides = {}
        if args.grid_size is not None:
            cli_overrides['grid_size'] = args.grid_size
        if args.n_agents is not None:
            cli_overrides['n_agents'] = args.n_agents
        if args.seed is not None:
            cli_overrides['seed'] = args.seed
        if args.fourier_bands is not None:
            cli_overrides['fourier_bands'] = args.fourier_bands
        if args.num_minibatches is not None:
            cli_overrides['num_minibatches'] = args.num_minibatches
        if args.update_epochs is not None:
            cli_overrides['update_epochs'] = args.update_epochs
        if args.episode_length is not None:
            cli_overrides['max_steps_per_episode'] = args.episode_length
            if args.batch_size is None:
                cli_overrides['num_steps'] = args.episode_length
        config = build_config(yaml_dict, cli_overrides)
    else:
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
            if args.batch_size is None:
                config.num_steps = args.episode_length
        if args.num_minibatches is not None:
            config.num_minibatches = args.num_minibatches
        if args.update_epochs is not None:
            config.update_epochs = args.update_epochs

    if args.ablate_ledger:
        config.ablate_ledger = True

    # Resolve training params from YAML training section, CLI overrides on top
    pretrain = args.pretrain
    start_phase = args.phase
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint
    if args.experiment:
        training = yaml_dict.get('training', {})
        if not pretrain and training.get('pretrain', False):
            pretrain = True
        if start_phase is None and 'start_phase' in training:
            start_phase = training['start_phase']
        if output_dir == 'checkpoints_vec' and 'output_dir' in training:
            output_dir = training['output_dir']
        if checkpoint_path is None and 'checkpoint' in training:
            checkpoint_path = training['checkpoint']

    # Apply old CLI defaults when no experiment YAML is used
    n_envs = args.n_envs if args.n_envs is not None else (None if args.experiment else 8)
    timesteps = args.timesteps if args.timesteps is not None else (None if args.experiment else 500_000)

    train_vec(
        config=config,
        n_envs=n_envs,
        pretrain=pretrain,
        total_timesteps=timesteps,
        target_batch_size=args.batch_size,
        csv_log_path=args.csv_log,
        output_dir=output_dir,
        visualize=args.visualize,
        render_every=args.render_every,
        record_every=args.record_every,
        checkpoint_path=checkpoint_path,
        start_phase=start_phase,
    )


if __name__ == '__main__':
    main()
