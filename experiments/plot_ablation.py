"""
Plot ablation study comparison from CSV logs.

Produces:
1. Episode return vs global_step (sample efficiency) - both configs overlaid
2. Episode return vs wall_time in minutes (wall-clock efficiency)
3. Curriculum phase progression vs global_step
4. Table of time-to-phase for each phase

Usage:
    python plot_ablation.py ablation_16dim.csv ablation_8dim.csv
    python plot_ablation.py ablation_32envs.csv ablation_2envs.csv "32-envs" "2-envs"
    python plot_ablation.py  # uses default filenames
"""
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    """Load CSV log into dict of lists."""
    data = {'update': [], 'global_step': [], 'episode_return': [],
            'episodes_completed': [], 'curriculum_phase': [], 'sps': [],
            'wall_time': [], 'total_loss': []}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))
    return data


def smooth(values, window=50):
    """Simple moving average."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i + 1]))
    return result


def plot_comparison(csv_a, csv_b, output_path='ablation_comparison.png',
                    label_a='16-dim (4 bands)', label_b='8-dim (2 bands)',
                    title='Ablation Study'):
    """Plot comparison of two ablation runs."""
    da = load_csv(csv_a)
    db = load_csv(csv_b)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    # Plot 1: Episode return vs global_step (sample efficiency)
    ax = axes[0]
    ax.plot(da['global_step'], smooth(da['episode_return']),
            label=label_a, alpha=0.8)
    ax.plot(db['global_step'], smooth(db['episode_return']),
            label=label_b, alpha=0.8)
    ax.set_ylabel('Avg Episode Return (smoothed)')
    ax.set_xlabel('Global Step')
    ax.set_title('Sample Efficiency: Episode Returns vs Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Episode return vs wall_time (wall-clock efficiency)
    ax = axes[1]
    wall_a = [t / 60.0 for t in da['wall_time']]  # seconds -> minutes
    wall_b = [t / 60.0 for t in db['wall_time']]
    ax.plot(wall_a, smooth(da['episode_return']),
            label=label_a, alpha=0.8)
    ax.plot(wall_b, smooth(db['episode_return']),
            label=label_b, alpha=0.8)
    ax.set_ylabel('Avg Episode Return (smoothed)')
    ax.set_xlabel('Wall Time (minutes)')
    ax.set_title('Wall-Clock Efficiency: Episode Returns vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Curriculum phase vs global_step
    ax = axes[2]
    ax.plot(da['global_step'], da['curriculum_phase'],
            label=label_a, alpha=0.8)
    ax.plot(db['global_step'], db['curriculum_phase'],
            label=label_b, alpha=0.8)
    ax.set_ylabel('Curriculum Phase')
    ax.set_xlabel('Global Step')
    ax.set_title('Curriculum Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    max_phase = max(max(da['curriculum_phase']), max(db['curriculum_phase']))
    ax.set_yticks(range(1, int(max_phase) + 2))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")

    # Print phase timing table
    print("\nPhase Advancement Timing:")
    print(f"{'Phase':<8} {label_a + ' (steps)':<22} {label_b + ' (steps)':<22} {'Faster'}")
    print("-" * 70)
    for phase in range(2, 13):
        step_a = next((s for s, p in zip(da['global_step'], da['curriculum_phase']) if p >= phase), None)
        step_b = next((s for s, p in zip(db['global_step'], db['curriculum_phase']) if p >= phase), None)
        sa = f"{int(step_a):,}" if step_a else "N/A"
        sb = f"{int(step_b):,}" if step_b else "N/A"
        if step_a and step_b:
            faster = label_a if step_a < step_b else label_b
        else:
            faster = "-"
        print(f"{phase:<8} {sa:<22} {sb:<22} {faster}")


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        csv_a, csv_b = sys.argv[1], sys.argv[2]
        label_a = sys.argv[3] if len(sys.argv) >= 4 else '16-dim (4 bands)'
        label_b = sys.argv[4] if len(sys.argv) >= 5 else '8-dim (2 bands)'
        plot_comparison(csv_a, csv_b, label_a=label_a, label_b=label_b)
    else:
        plot_comparison('ablation_16dim.csv', 'ablation_8dim.csv')
