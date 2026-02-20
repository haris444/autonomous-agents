"""
SAC Training Script - Discrete Soft Actor-Critic for Multi-Agent RL.

Off-policy training with replay buffer, twin Q-networks, and automatic
entropy tuning. Uses the same BatchedGridWorld as train_vec.py.

Usage:
    python train_sac.py --experiment experiments/configs/sac.yaml --visualize
    python train_sac.py --n_envs 8 --timesteps 1000000
"""
import argparse
import csv
import os
import time
import torch

from core.config import Config
from env.batched_env import BatchedGridWorld as VecEnv
from agents.sac import IndependentSAC
from agents.replay_buffer import ReplayBuffer
from training.scenarios import CURRICULUM, THRESHOLDS


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


def train_sac(
    config: Config = None,
    n_envs: int = None,
    pretrain: bool = False,
    total_timesteps: int = None,
    csv_log_path: str = None,
    output_dir: str = "checkpoints_sac",
    visualize: bool = False,
    render_every: int = 50,
    record_every: int = 100,
    ckpt_every: int = 100,
    checkpoint_path: str = None,
    start_phase: int = None,
):
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

    # Setup
    device = get_device()
    set_seed(config.seed)
    print(f"Training SAC on {device}")
    print(f"Config: {config.n_agents} agents, {config.grid_size}x{config.grid_size} grid")
    print(f"Parallel environments: {n_envs}")
    transitions_per_step = n_envs * config.n_agents
    print(f"Transitions per env step: {transitions_per_step}")
    print(f"Replay buffer capacity: {config.sac_buffer_size:,}")
    print(f"Batch size: {config.sac_batch_size}")
    print(f"Learning starts: {config.sac_learning_starts}")

    # Initialize vectorized environment
    vec_env = VecEnv(config, device, n_envs=n_envs)

    # Initialize SAC
    sac = IndependentSAC(config, device)

    # Training tracking (defaults, overridden by checkpoint below)
    start_env_step = 0
    completed_episodes = 0
    env0_episodes = 0
    update_count = 0
    resumed = False

    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'sac_state' in ckpt:
            sac.load_state_dict(ckpt['sac_state'])
            print("  Loaded SAC state (actors, critics, targets, alphas)")
        elif 'actor_state_dicts' in ckpt:
            sac.load_state_dict(ckpt)
            print("  Loaded SAC state from flat checkpoint")
        if 'curriculum_phase' in ckpt:
            vec_env.set_curriculum_phase(ckpt['curriculum_phase'])
        # Restore counters
        completed_episodes = ckpt.get('episode', 0)
        ckpt_global_step = ckpt.get('global_step', 0)
        start_env_step = ckpt_global_step // transitions_per_step
        update_count = ckpt.get('update_count', 0)
        env0_episodes = ckpt.get('env0_episodes', completed_episodes // max(1, n_envs))
        resumed = True
        print(f"  Resumed: episode={completed_episodes}, global_step={ckpt_global_step}, "
              f"update_count={update_count}, env0_episodes={env0_episodes}")

    # Override curriculum phase if specified
    if start_phase is not None:
        vec_env.set_curriculum_phase(start_phase)
        print(f"  Overriding curriculum phase to {start_phase}")

    # Create replay buffer
    replay_buffer = ReplayBuffer(config, device)

    # Training tracking
    global_step = 0
    start_time = time.time()

    # Episode tracking (per env)
    episode_rewards = torch.zeros(n_envs, config.n_agents, device=device)
    episode_steps = torch.zeros(n_envs, dtype=torch.long, device=device)
    next_record_at = completed_episodes + record_every
    total_returns = []

    # CSV logging
    csv_file = None
    csv_writer = None
    if csv_log_path:
        csv_file = open(csv_log_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'global_step', 'episode_return', 'episodes_completed',
                             'curriculum_phase', 'sps', 'wall_time',
                             'critic_loss', 'actor_loss', 'alpha_dir', 'alpha_act', 'q1_mean'])

    # === VISUALIZATION SETUP ===
    renderer = None
    logger = None
    recorder = None
    recording_active = False

    if visualize:
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        from analysis.visualize import GridRenderer, TrainingLogger, EpisodeRecorder, render_ledger_heatmaps

        os.makedirs(output_dir, exist_ok=True)
        renderer = GridRenderer(config)
        logger = TrainingLogger()
        recorder = EpisodeRecorder(config)
        print(f"Visualization enabled: saving to {output_dir}/")
        print(f"  Grid every {render_every} updates, record every {record_every} episodes")

    total_env_steps = config.total_timesteps // transitions_per_step
    print(f"Starting SAC training for {config.total_timesteps:,} timesteps ({total_env_steps:,} env steps)")
    print("-" * 60, flush=True)

    # Initial reset
    obs = vec_env.reset()
    print("Environment reset complete.", flush=True)

    # Pre-fill replay buffer on resume (avoid random warmup)
    if resumed and config.sac_learning_starts > 0:
        warmup_needed = config.sac_learning_starts
        print(f"  Pre-filling replay buffer with {warmup_needed} transitions using loaded policy...", flush=True)
        while len(replay_buffer) < warmup_needed:
            with torch.no_grad():
                dirs, acts = sac.get_actions(obs)
            next_obs, rewards, dones, infos = vec_env.step(dirs, acts)
            env_dones = dones.all(dim=1)
            dones_expanded = env_dones.unsqueeze(1).expand_as(rewards)
            replay_buffer.add_batch(obs, dirs, acts, rewards, next_obs, dones_expanded)
            obs = next_obs
        print(f"  Buffer pre-filled: {len(replay_buffer)} transitions", flush=True)

    print("Starting loop...", flush=True)
    prev_phase = None
    last_metrics = {}

    for env_step in range(start_env_step, total_env_steps):
        global_step = env_step * transitions_per_step

        # Get current phase and scenario
        phase = vec_env.get_curriculum_phase()
        scenario = CURRICULUM.get(phase) if config.pretrain_mode else None

        if scenario is not None and (prev_phase is None or phase != prev_phase):
            vec_env.apply_scenario(scenario)
            obs = vec_env.get_observations()
        prev_phase = phase

        # === ACTION SELECTION ===
        if global_step < config.sac_learning_starts:
            # Random exploration
            directions = torch.randint(0, config.n_directions, (n_envs, config.n_agents), device=device)
            action_types = torch.randint(0, config.n_action_types, (n_envs, config.n_agents), device=device)
        else:
            directions, action_types = sac.get_actions(obs)

        # Record env 0 state before step (if recording active)
        if recording_active:
            env0 = vec_env.get_first_env()
            actions_dict = {
                i: (directions[0, i].item(), action_types[0, i].item())
                for i in range(config.n_agents)
            }
            recorder.record(env0, actions=actions_dict)

        # Environment step
        next_obs, rewards, dones, infos = vec_env.step(directions, action_types)

        # Update recorded step with rewards
        if recording_active and recorder.steps:
            recorder.steps[-1].rewards = {
                i: rewards[0, i].item() for i in range(config.n_agents)
            }

        # Store transitions in replay buffer
        # dones for replay: use per-env done (all agents share same done flag)
        env_dones = dones.all(dim=1)  # [n_envs]
        dones_expanded = env_dones.unsqueeze(1).expand_as(rewards)  # [n_envs, n_agents]
        replay_buffer.add_batch(obs, directions, action_types, rewards, next_obs, dones_expanded)

        # Track episode rewards
        episode_rewards += rewards
        episode_steps += 1

        # Check for completed episodes
        if env_dones.any():
            for e in range(n_envs):
                if env_dones[e]:
                    ret = episode_rewards[e, 0].item()
                    total_returns.append(ret)
                    completed_episodes += 1

                    if e == 0:
                        env0_episodes += 1

                        # Save recording if active
                        if recording_active:
                            recording_active = False
                            replay_path = f"{output_dir}/replay_ep{env0_episodes}.pt"
                            recorder.save(replay_path)
                            print(f"  [Saved replay: {replay_path} ({len(recorder.steps)} steps)]")

                        # Start new recording if it's time
                        if visualize and completed_episodes >= next_record_at:
                            recording_active = True
                            recorder.reset()
                            next_record_at = completed_episodes + record_every

                    # Save checkpoint every N episodes
                    if completed_episodes % ckpt_every == 0:
                        os.makedirs(output_dir, exist_ok=True)
                        ckpt_path = f"{output_dir}/checkpoint_ep{completed_episodes}.pt"
                        torch.save({
                            'episode': completed_episodes,
                            'global_step': global_step,
                            'sac_state': sac.state_dict(),
                            'avg_return': sum(total_returns[-10:]) / max(1, len(total_returns[-10:])),
                            'curriculum_phase': vec_env.get_curriculum_phase(),
                            'config': config.to_dict(),
                            'update_count': update_count,
                            'env0_episodes': env0_episodes,
                        }, ckpt_path)
                        print(f"  [Saved checkpoint: {ckpt_path}]")

                    episode_rewards[e] = 0
                    episode_steps[e] = 0

        obs = next_obs

        # === SAC UPDATE ===
        if (global_step >= config.sac_learning_starts
                and env_step % config.sac_update_frequency == 0):
            last_metrics = sac.update(replay_buffer, config.sac_batch_size)
            update_count += 1

        # === LOGGING (every 10 env steps) ===
        if env_step % 10 == 0 and env_step > 0:
            elapsed = time.time() - start_time
            sps = global_step / max(1, elapsed)
            avg_return = sum(total_returns[-100:]) / max(1, len(total_returns[-100:]))
            buf_size = len(replay_buffer)

            # Phase info
            if config.pretrain_mode:
                thresh = THRESHOLDS.get(phase, 0)
                phase_str = f"Phase {phase} ({avg_return:.0f}/{thresh})"
            else:
                phase_str = f"Phase {phase}"

            print(f"Step {global_step:,} | "
                  f"{phase_str} | "
                  f"Ep: {completed_episodes} | "
                  f"Avg Return: {avg_return:.1f} | "
                  f"SPS: {sps:.0f} | "
                  f"Buffer: {buf_size:,} | "
                  f"Updates: {update_count} | "
                  f"Q: {last_metrics.get('q1_mean', 0):.2f} | "
                  f"a_d: {last_metrics.get('alpha_dir', 0):.3f} a_a: {last_metrics.get('alpha_act', 0):.3f}",
                  flush=True)

            if csv_writer:
                csv_writer.writerow([
                    env_step, global_step, f'{avg_return:.2f}', completed_episodes,
                    phase, f'{sps:.0f}', f'{elapsed:.1f}',
                    f"{last_metrics.get('critic_loss', 0):.4f}",
                    f"{last_metrics.get('actor_loss', 0):.4f}",
                    f"{last_metrics.get('alpha_dir', 0):.4f}",
                    f"{last_metrics.get('alpha_act', 0):.4f}",
                    f"{last_metrics.get('q1_mean', 0):.4f}",
                ])
                csv_file.flush()

        # === VISUALIZATION ===
        if visualize and env_step % 10 == 0 and env_step > 0:
            import matplotlib.pyplot as plt

            avg_return = sum(total_returns[-100:]) / max(1, len(total_returns[-100:]))
            logger.log(global_step, episode_return=avg_return, metrics=last_metrics)

            # Grid snapshot
            if update_count % render_every == 0 and update_count > 0:
                env0 = vec_env.get_first_env()
                fig = renderer.render(env0, show=False)
                fig.suptitle(f'Step {global_step:,} | Phase {phase} | SAC')
                grid_path = f"{output_dir}/grid_step_{global_step:08d}.png"
                fig.savefig(grid_path, dpi=100, bbox_inches='tight')
                plt.close(fig)

                # Ledger heatmap
                fig_ledger = render_ledger_heatmaps(vec_env.ledger.tensor, show=False)
                fig_ledger.suptitle(f'Ledger - Step {global_step:,}')
                ledger_path = f"{output_dir}/ledger_step_{global_step:08d}.png"
                fig_ledger.savefig(ledger_path, dpi=100, bbox_inches='tight')
                plt.close(fig_ledger)

            # Training curves
            if update_count % 10 == 0 and update_count > 0:
                fig_curves = logger.plot(window=50, show=False)
                curves_path = f"{output_dir}/training_curves.png"
                fig_curves.savefig(curves_path, dpi=100, bbox_inches='tight')
                plt.close(fig_curves)

        # === CURRICULUM ADVANCEMENT ===
        if (config.pretrain_mode and config.curriculum_enabled
                and completed_episodes > 0 and completed_episodes % 50 == 0):
            avg_return = sum(total_returns[-50:]) / max(1, len(total_returns[-50:]))
            thresh = THRESHOLDS.get(phase, None)
            if thresh is not None and avg_return >= thresh:
                new_phase = phase + 1
                if new_phase in CURRICULUM:
                    vec_env.set_curriculum_phase(new_phase)
                    print(f"  >>> Advancing to phase {new_phase} (avg return {avg_return:.1f} >= {thresh})")

    # Final summary
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"SAC Training complete!")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total steps: {global_step:,}")
    print(f"Steps/sec: {global_step / max(1, elapsed):.0f}")
    print(f"Episodes completed: {completed_episodes}")
    print(f"SAC updates: {update_count}")
    if total_returns:
        print(f"Final avg return: {sum(total_returns[-100:]) / len(total_returns[-100:]):.1f}")

    # Final checkpoint
    if completed_episodes > 0:
        os.makedirs(output_dir, exist_ok=True)
        final_path = f"{output_dir}/checkpoint_final.pt"
        torch.save({
            'episode': completed_episodes,
            'global_step': global_step,
            'sac_state': sac.state_dict(),
            'avg_return': sum(total_returns[-10:]) / max(1, len(total_returns[-10:])),
            'curriculum_phase': vec_env.get_curriculum_phase(),
            'config': config.to_dict(),
            'update_count': update_count,
            'env0_episodes': env0_episodes,
        }, final_path)
        print(f"Final checkpoint: {final_path}")

    if csv_file:
        csv_file.close()
        print(f"CSV log saved to {csv_log_path}")

    if visualize and logger is not None:
        logger.save(f"{output_dir}/training_log.pt")
        print(f"Training log saved to {output_dir}/training_log.pt")


def main():
    parser = argparse.ArgumentParser(description='SAC Multi-Agent Training')
    parser.add_argument('--n_envs', type=int, default=None, help='Number of parallel environments')
    parser.add_argument('--pretrain', action='store_true', help='Run curriculum pretraining')
    parser.add_argument('--timesteps', type=int, default=None, help='Total timesteps')
    parser.add_argument('--grid_size', type=int, default=None, help='Grid size')
    parser.add_argument('--n_agents', type=int, default=None, help='Number of agents')
    parser.add_argument('--episode_length', type=int, default=None, help='Max steps per episode')
    parser.add_argument('--fourier-bands', type=int, default=None, help='Number of Fourier frequency bands')
    parser.add_argument('--csv-log', type=str, default=None, help='Path to CSV log file')
    parser.add_argument('--output-dir', type=str, default='checkpoints_sac', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--visualize', '-v', action='store_true', help='Enable visualization')
    parser.add_argument('--render-every', type=int, default=50, help='Grid snapshot every N updates')
    parser.add_argument('--record-every', type=int, default=100, help='Record episode every N episodes')
    parser.add_argument('--ckpt-every', type=int, default=100, help='Save checkpoint every N episodes')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--phase', type=int, default=None, help='Override starting curriculum phase')
    parser.add_argument('--experiment', type=str, default=None, help='Path to experiment YAML config')
    parser.add_argument('--ablate-ledger', action='store_true', help='Zero out social features')
    args = parser.parse_args()

    # Build config: YAML base (if provided) with CLI overrides on top
    if args.experiment:
        from experiments.experiment_config import load_yaml, build_config
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
        if args.episode_length is not None:
            cli_overrides['max_steps_per_episode'] = args.episode_length
        config = build_config(yaml_dict, cli_overrides)
    else:
        config = Config()
        if args.fourier_bands is not None:
            config.fourier_bands = args.fourier_bands
            config.__post_init__()
        if args.grid_size:
            config.grid_size = args.grid_size
        if args.n_agents:
            config.n_agents = args.n_agents
        if args.seed is not None:
            config.seed = args.seed
        if args.episode_length:
            config.max_steps_per_episode = args.episode_length

    if args.ablate_ledger:
        config.ablate_ledger = True

    # Resolve training params from YAML
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
        if output_dir == 'checkpoints_sac' and 'output_dir' in training:
            output_dir = training['output_dir']
        if checkpoint_path is None and 'checkpoint' in training:
            checkpoint_path = training['checkpoint']

    n_envs = args.n_envs if args.n_envs is not None else (None if args.experiment else 8)
    timesteps = args.timesteps if args.timesteps is not None else (None if args.experiment else 1_000_000)

    train_sac(
        config=config,
        n_envs=n_envs,
        pretrain=pretrain,
        total_timesteps=timesteps,
        csv_log_path=args.csv_log,
        output_dir=output_dir,
        visualize=args.visualize,
        render_every=args.render_every,
        record_every=args.record_every,
        ckpt_every=args.ckpt_every,
        checkpoint_path=checkpoint_path,
        start_phase=start_phase,
    )


if __name__ == '__main__':
    main()
