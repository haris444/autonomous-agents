"""
Ablation Study: BatchedGridWorld vs VecEnv Performance Comparison.

Benchmarks:
1. Raw env.step() throughput at varying n_envs
2. Full training loop throughput (env + network + PPO)
3. Scaling plots saved to ablation_env_speed.png
"""
import time
import gc
import statistics
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from config import Config
from vec_env import VecEnv
from batched_env import BatchedGridWorld
from ppo import VmapPPO
from buffer import VecBuffer


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def flush_print(msg):
    print(msg)
    sys.stdout.flush()


# =====================================================================
# Part 1: Raw step benchmark
# =====================================================================

def benchmark_raw_steps(env_class, config, device, n_envs, n_steps=200, warmup=20):
    """Time raw env.step() calls. Returns steps/second."""
    env = env_class(config, device, n_envs=n_envs)
    n = config.n_agents

    env.reset()

    # Warmup
    for _ in range(warmup):
        dirs = torch.randint(0, 5, (n_envs, n), device=device)
        acts = torch.randint(0, 5, (n_envs, n), device=device)
        env.step(dirs, acts)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed run
    t0 = time.perf_counter()
    for _ in range(n_steps):
        dirs = torch.randint(0, 5, (n_envs, n), device=device)
        acts = torch.randint(0, 5, (n_envs, n), device=device)
        env.step(dirs, acts)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    transitions = n_steps * n_envs * n
    sps = transitions / elapsed

    del env
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return sps, elapsed


# =====================================================================
# Part 2: Full training benchmark
# =====================================================================

def benchmark_training(env_class, config, device, n_envs, n_updates=3):
    """Time full training loop (env + network + PPO). Returns SPS."""
    cfg = Config()
    cfg.n_envs = n_envs
    cfg.max_steps_per_episode = 128
    cfg.num_steps = 64

    env = env_class(cfg, device, n_envs=n_envs)
    ppo = VmapPPO(cfg, device)
    buf = VecBuffer(cfg, device, n_envs=n_envs)

    obs = env.reset()
    n = cfg.n_agents
    global_step = 0

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for update in range(n_updates):
        buf.reset()

        for step in range(cfg.num_steps):
            global_step += n_envs * n

            with torch.no_grad():
                dir_mask, act_mask = env.get_action_masks()
                directions, action_types, log_probs, _, values, aux_values = ppo.vec_get_actions_and_values(
                    obs, direction_mask=dir_mask, action_type_mask=act_mask
                )

            next_obs, rewards, dones, infos = env.step(directions, action_types)

            buf.store(
                obs, directions, action_types, log_probs,
                rewards, dones, values,
                dir_mask, act_mask,
                reward_survival=infos['reward_survival'],
                reward_resource=infos['reward_resource'],
                reward_social=infos['reward_social'],
                value_survival=aux_values['survival'],
                value_resource=aux_values['resource'],
                value_social=aux_values['social']
            )

            obs = next_obs

        with torch.no_grad():
            next_values = ppo.vec_get_values(obs)
            next_aux = ppo.vec_get_auxiliary_values(obs)
            next_dones = torch.zeros(n_envs, n, device=device)

        buf.compute_gae(
            next_values, next_dones,
            next_value_survival=next_aux['survival'],
            next_value_resource=next_aux['resource'],
            next_value_social=next_aux['social']
        )

        ppo.update_from_vec_buffer(buf)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sps = global_step / elapsed

    del env, ppo, buf
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return sps, elapsed


# =====================================================================
# Main
# =====================================================================

def main():
    device = get_device()
    flush_print(f"Device: {device}")

    config = Config()
    config.max_steps_per_episode = 500

    # === Part 1: Raw step throughput ===
    n_envs_list = [1, 2, 4, 8]
    n_trials = 3

    vec_sps = {}
    bat_sps = {}

    flush_print("\n" + "=" * 60)
    flush_print("PART 1: Raw env.step() throughput (200 steps, 3 trials)")
    flush_print("=" * 60)

    for ne in n_envs_list:
        flush_print(f"\nn_envs = {ne}:")

        # VecEnv
        trials = []
        for t in range(n_trials):
            sps, elapsed = benchmark_raw_steps(VecEnv, config, device, ne)
            trials.append(sps)
            flush_print(f"  VecEnv      trial {t+1}: {sps:>10,.0f} SPS  ({elapsed:.2f}s)")
        vec_sps[ne] = statistics.median(trials)

        # BatchedGridWorld
        trials = []
        for t in range(n_trials):
            sps, elapsed = benchmark_raw_steps(BatchedGridWorld, config, device, ne)
            trials.append(sps)
            flush_print(f"  Batched     trial {t+1}: {sps:>10,.0f} SPS  ({elapsed:.2f}s)")
        bat_sps[ne] = statistics.median(trials)

        speedup = bat_sps[ne] / vec_sps[ne]
        flush_print(f"  >> Median: VecEnv={vec_sps[ne]:,.0f}  Batched={bat_sps[ne]:,.0f}  Speedup={speedup:.2f}x")

    # === Part 2: Full training throughput ===
    train_n_envs = [4, 8]
    vec_train_sps = {}
    bat_train_sps = {}

    flush_print("\n" + "=" * 60)
    flush_print("PART 2: Full training loop throughput (3 PPO updates)")
    flush_print("=" * 60)

    for ne in train_n_envs:
        flush_print(f"\nn_envs = {ne}:")

        sps, elapsed = benchmark_training(VecEnv, config, device, ne, n_updates=3)
        vec_train_sps[ne] = sps
        flush_print(f"  VecEnv:   {sps:>10,.0f} SPS  ({elapsed:.2f}s)")

        sps, elapsed = benchmark_training(BatchedGridWorld, config, device, ne, n_updates=3)
        bat_train_sps[ne] = sps
        flush_print(f"  Batched:  {sps:>10,.0f} SPS  ({elapsed:.2f}s)")

        speedup = bat_train_sps[ne] / vec_train_sps[ne]
        flush_print(f"  >> Speedup: {speedup:.2f}x")

    # === Part 3: Plots ===
    flush_print("\n" + "=" * 60)
    flush_print("Generating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Raw SPS vs n_envs
    ax = axes[0]
    x_raw = sorted(vec_sps.keys())
    ax.plot(x_raw, [vec_sps[k] for k in x_raw], 'o-', color='#e74c3c', linewidth=2,
            markersize=8, label='VecEnv (sequential)')
    ax.plot(x_raw, [bat_sps[k] for k in x_raw], 's-', color='#2ecc71', linewidth=2,
            markersize=8, label='BatchedGridWorld (tensor)')
    ax.set_xlabel('Number of Parallel Environments', fontsize=12)
    ax.set_ylabel('Steps per Second', fontsize=12)
    ax.set_title('Raw env.step() Throughput', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(x_raw)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Plot 2: Speedup ratio
    ax = axes[1]
    speedups = [bat_sps[k] / vec_sps[k] for k in x_raw]
    bars = ax.bar(range(len(x_raw)), speedups, color='#3498db', edgecolor='#2c3e50', linewidth=1.5)
    ax.set_xticks(range(len(x_raw)))
    ax.set_xticklabels([str(k) for k in x_raw])
    ax.set_xlabel('Number of Parallel Environments', fontsize=12)
    ax.set_ylabel('Speedup (Batched / VecEnv)', fontsize=12)
    ax.set_title('Speedup Ratio', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Full training SPS
    ax = axes[2]
    x_train = sorted(vec_train_sps.keys())
    ax.plot(x_train, [vec_train_sps[k] for k in x_train], 'o-', color='#e74c3c',
            linewidth=2, markersize=8, label='VecEnv (sequential)')
    ax.plot(x_train, [bat_train_sps[k] for k in x_train], 's-', color='#2ecc71',
            linewidth=2, markersize=8, label='BatchedGridWorld (tensor)')
    ax.set_xlabel('Number of Parallel Environments', fontsize=12)
    ax.set_ylabel('Steps per Second', fontsize=12)
    ax.set_title('Full Training Loop Throughput', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(x_train)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig('ablation_env_speed.png', dpi=150, bbox_inches='tight')
    flush_print("Saved to ablation_env_speed.png")

    # === Summary table ===
    flush_print("\n" + "=" * 60)
    flush_print("SUMMARY")
    flush_print("=" * 60)
    flush_print(f"\nRaw env.step():")
    flush_print(f"{'n_envs':>8} {'VecEnv SPS':>14} {'Batched SPS':>14} {'Speedup':>10}")
    flush_print("-" * 48)
    for ne in x_raw:
        sp = bat_sps[ne] / vec_sps[ne]
        flush_print(f"{ne:>8} {vec_sps[ne]:>14,.0f} {bat_sps[ne]:>14,.0f} {sp:>9.2f}x")

    flush_print(f"\nFull training:")
    flush_print(f"{'n_envs':>8} {'VecEnv SPS':>14} {'Batched SPS':>14} {'Speedup':>10}")
    flush_print("-" * 48)
    for ne in x_train:
        sp = bat_train_sps[ne] / vec_train_sps[ne]
        flush_print(f"{ne:>8} {vec_train_sps[ne]:>14,.0f} {bat_train_sps[ne]:>14,.0f} {sp:>9.2f}x")


if __name__ == '__main__':
    main()
