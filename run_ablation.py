"""
Ablation Study: Generic A/B comparison runner.

Runs two train_vec.py processes IN PARALLEL with different settings,
logging metrics to separate CSVs, then produces a comparison plot.

All extra arguments are forwarded to both training runs.

Usage:
    # Default fourier ablation (backward compatible):
    python run_ablation.py --pretrain --timesteps 500000

    # Custom A/B comparison (e.g. parallelism ablation):
    python run_ablation.py \\
        --label-a "32-envs" --extra-a "--n_envs 32" \\
        --csv-a ablation_32envs.csv --dir-a ablation_output_32envs \\
        --label-b "2-envs" --extra-b "--n_envs 2" \\
        --csv-b ablation_2envs.csv --dir-b ablation_output_2envs \\
        --pretrain --timesteps 20000000

    python run_ablation.py --plot-only
"""
import argparse
import subprocess
import sys
import threading
import time


def stream_output(proc, prefix):
    """Read and print lines from a subprocess stdout with a prefix."""
    for line in iter(proc.stdout.readline, ''):
        if line:
            print(f"[{prefix}] {line}", end='', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Ablation Study: A/B comparison runner',
        add_help=False  # We'll handle --help manually so passthrough works
    )
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training, just plot from existing CSVs')
    # Variant A settings
    parser.add_argument('--label-a', type=str, default='16-dim',
                        help='Label for variant A (default: 16-dim)')
    parser.add_argument('--extra-a', type=str, default='--fourier-bands 4',
                        help='Extra args for variant A (default: --fourier-bands 4)')
    parser.add_argument('--csv-a', type=str, default='ablation_16dim.csv',
                        help='CSV path for variant A (default: ablation_16dim.csv)')
    parser.add_argument('--dir-a', type=str, default='ablation_output_16dim',
                        help='Output dir for variant A (default: ablation_output_16dim)')
    # Variant B settings
    parser.add_argument('--label-b', type=str, default='8-dim',
                        help='Label for variant B (default: 8-dim)')
    parser.add_argument('--extra-b', type=str, default='--fourier-bands 2',
                        help='Extra args for variant B (default: --fourier-bands 2)')
    parser.add_argument('--csv-b', type=str, default='ablation_8dim.csv',
                        help='CSV path for variant B (default: ablation_8dim.csv)')
    parser.add_argument('--dir-b', type=str, default='ablation_output_8dim',
                        help='Output dir for variant B (default: ablation_output_8dim)')
    # Backward compat aliases
    parser.add_argument('--csv-16', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--csv-8', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--help', '-h', action='store_true')

    args, passthrough = parser.parse_known_args()

    if args.help:
        parser.print_help()
        print("\nAll other arguments are forwarded to train_vec.py (e.g. --pretrain, --timesteps, --n_envs)")
        return

    # Support old --csv-16/--csv-8 flags
    csv_a = args.csv_16 if args.csv_16 else args.csv_a
    csv_b = args.csv_8 if args.csv_8 else args.csv_b
    label_a = args.label_a
    label_b = args.label_b

    if not args.plot_only:
        # Build commands for both runs
        base_cmd = [sys.executable, '-u', 'train_vec.py'] + passthrough

        cmd_a = base_cmd + args.extra_a.split() + [
            '--csv-log', csv_a,
            '--output-dir', args.dir_a,
        ]
        cmd_b = base_cmd + args.extra_b.split() + [
            '--csv-log', csv_b,
            '--output-dir', args.dir_b,
        ]

        print("=" * 60)
        print(f"ABLATION STUDY: {label_a} vs {label_b}")
        print("=" * 60)
        print(f"  [{label_a}] cmd: {' '.join(cmd_a)}")
        print(f"  [{label_b}] cmd: {' '.join(cmd_b)}")
        print(f"  Passthrough args: {passthrough}")
        print("=" * 60)
        print("Launching both processes in parallel...\n")

        # Launch both processes
        proc_a = subprocess.Popen(
            cmd_a, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        proc_b = subprocess.Popen(
            cmd_b, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )

        # Stream output from both in parallel threads
        ta = threading.Thread(target=stream_output, args=(proc_a, label_a), daemon=True)
        tb = threading.Thread(target=stream_output, args=(proc_b, label_b), daemon=True)
        ta.start()
        tb.start()

        # Wait for both to finish
        rc_a = proc_a.wait()
        rc_b = proc_b.wait()
        ta.join(timeout=5)
        tb.join(timeout=5)

        print("\n" + "=" * 60)
        print(f"[{label_a}] process exited with code {rc_a}")
        print(f"[{label_b}] process exited with code {rc_b}")
        print("=" * 60)

        if rc_a != 0 or rc_b != 0:
            print("One or both runs failed. Check output above.")
            return

    # Plot comparison
    print("\nGenerating comparison plot...")
    from plot_ablation import plot_comparison
    plot_comparison(csv_a, csv_b, label_a=label_a, label_b=label_b,
                    title=f'Ablation Study: {label_a} vs {label_b}')


if __name__ == '__main__':
    main()
