"""
Experiment runner — orchestrates sweeps from YAML configs.

Usage:
    python run_experiment.py experiments/my_experiment.yaml
    python run_experiment.py experiments/my_experiment.yaml --dry-run
    python run_experiment.py experiments/my_experiment.yaml --n_envs 16
    python run_experiment.py experiments/my_experiment.yaml --timesteps 1000
"""
import argparse
import csv
import os
import time

import torch

from config import Config
from experiment_config import (
    load_yaml,
    validate_yaml_keys,
    build_config,
    generate_sweep_configs,
    save_resolved_config,
    create_experiment_dir,
    copy_source_yaml,
)
from train_vec import train_vec


def main():
    parser = argparse.ArgumentParser(description='Run YAML-configured experiments')
    parser.add_argument('yaml_path', type=str, help='Path to experiment YAML file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print all runs without executing')
    parser.add_argument('--n_envs', type=int, default=None,
                        help='Override number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override total timesteps')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base directory for experiment outputs (default: results)')
    args = parser.parse_args()

    # Load and validate YAML
    yaml_dict = load_yaml(args.yaml_path)
    validate_yaml_keys(yaml_dict)

    # Get experiment metadata
    experiment = yaml_dict.get('experiment', {})
    exp_name = experiment.get('name', 'experiment')
    exp_desc = experiment.get('description', '')
    training = yaml_dict.get('training', {})
    pretrain = training.get('pretrain', False)
    start_phase = training.get('start_phase', None)
    checkpoint_path = training.get('checkpoint', None)
    base_output_dir = training.get('output_dir', None)

    # Apply CLI overrides to YAML
    cli_overrides = {}
    if args.n_envs is not None:
        cli_overrides['n_envs'] = args.n_envs
    if args.timesteps is not None:
        cli_overrides['total_timesteps'] = args.timesteps

    # Generate sweep runs
    runs = generate_sweep_configs(yaml_dict)

    print(f"Experiment: {exp_name}")
    if exp_desc:
        print(f"Description: {exp_desc}")
    print(f"Total runs: {len(runs)}")
    print("-" * 60)

    for i, (overrides, label) in enumerate(runs):
        merged = {**{k: v for k, v in yaml_dict.items()
                     if k not in {'experiment', 'training', 'reward_presets', 'sweep', 'reward_preset'}},
                  **overrides, **cli_overrides}
        print(f"  Run {i+1}/{len(runs)}: {label}")
        if args.dry_run:
            # Show key overrides
            show_keys = ['seed', 'learning_rate', 'r_coop_attempt', 'r_attack_mult',
                         'n_envs', 'total_timesteps']
            shown = {k: merged.get(k) for k in show_keys if k in merged}
            print(f"    Config: {shown}")

    if args.dry_run:
        print("\n(Dry run — no experiments executed)")
        return

    # Create experiment directory
    exp_dir = create_experiment_dir(args.results_dir, exp_name)
    print(f"\nOutput directory: {exp_dir}")
    copy_source_yaml(args.yaml_path, exp_dir)

    # Save base resolved config (without per-run overrides)
    base_config = build_config(yaml_dict, cli_overrides)
    save_resolved_config(base_config, os.path.join(exp_dir, 'config.yaml'))

    # Run each configuration
    summary_rows = []
    for i, (overrides, label) in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(runs)}: {label}")
        print(f"{'='*60}")

        # Build config for this run: YAML base + run overrides + CLI overrides
        run_yaml = dict(yaml_dict)
        run_yaml.update(overrides)
        config = build_config(run_yaml, cli_overrides)

        # Create run directory
        run_dir = os.path.join(exp_dir, label)
        os.makedirs(run_dir, exist_ok=True)
        save_resolved_config(config, os.path.join(run_dir, 'config.yaml'))

        # CSV log for this run
        csv_path = os.path.join(run_dir, 'metrics.csv')

        # Use YAML output_dir if set (override experiment dir structure)
        effective_output_dir = base_output_dir if base_output_dir else run_dir

        # Run training
        wall_start = time.time()
        train_vec(
            config=config,
            pretrain=pretrain,
            csv_log_path=csv_path,
            output_dir=effective_output_dir,
            start_phase=start_phase,
            checkpoint_path=checkpoint_path,
        )
        wall_time = time.time() - wall_start

        # Read final metrics from CSV
        final_return = 0.0
        final_phase = 0
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    final_return = float(last.get('episode_return', 0))
                    final_phase = int(last.get('curriculum_phase', 0))

        summary_rows.append({
            'run': label,
            'seed': config.seed,
            'final_return': f'{final_return:.2f}',
            'final_phase': final_phase,
            'wall_time': f'{wall_time:.1f}',
        })

        # Free GPU memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write summary CSV
    summary_path = os.path.join(exp_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'seed', 'final_return',
                                                'final_phase', 'wall_time'])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print(f"All runs complete. Summary saved to {summary_path}")
    print(f"{'='*60}")
    for row in summary_rows:
        print(f"  {row['run']}: return={row['final_return']}, "
              f"phase={row['final_phase']}, time={row['wall_time']}s")


if __name__ == '__main__':
    main()
