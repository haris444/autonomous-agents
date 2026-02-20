"""Quick benchmark: where does time go in one training step?"""
import torch, time
from core.config import Config
from env.batched_env import BatchedGridWorld

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

cfg = Config(grid_size=15, n_agents=8, n_predators=0)
env = BatchedGridWorld(cfg, device, n_envs=8)
env.reset()

dirs = torch.randint(0, 5, (8, 8), device=device)
acts = torch.randint(0, 5, (8, 8), device=device)

# Warmup
for _ in range(20):
    env.step(dirs, acts)

# Time full step
times = []
for _ in range(200):
    t0 = time.perf_counter()
    env.step(dirs, acts)
    times.append((time.perf_counter() - t0) * 1000)
avg = sum(times) / len(times)
print(f"\nFull step: {avg:.3f} ms  ({8*8*1000/avg:.0f} agent-steps/sec)")

# Now profile each piece inside step()
import types

def profile_methods(env, dirs, acts):
    """Monkey-patch to time each sub-method."""
    results = {}
    is_move = (acts == 0)
    is_attack = (acts == 1)
    is_give = (acts == 2)
    is_signal = (acts == 3)

    env.reset()
    env.prev_agent_positions = env.agent_positions.clone()

    methods = [
        ("_get_all_observations", lambda: env._get_all_observations()),
        ("_resolve_movement+occ", lambda: (env._resolve_movement(dirs, is_move), env._update_occupancy())),
        ("_resolve_interactions", lambda: env._resolve_interactions(dirs, acts, is_attack, is_give, is_signal)),
        ("_compute_nearest_food", lambda: env._compute_nearest_food_info()),
        ("_process_food_eating", lambda: env._process_food_eating(acts)),
        ("_spawn_food", lambda: env._spawn_food()),
        ("_check_deaths+occ", lambda: (env._check_deaths(), env._update_occupancy())),
        ("_intrinsic_coop", lambda: env._compute_intrinsic_coop_rewards(acts)),
        ("_hierarchy_rewards", lambda: env._compute_hierarchy_rewards()),
    ]

    for name, fn in methods:
        for _ in range(10):  # warmup
            fn()
        ts = []
        for _ in range(200):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        results[name] = sum(ts) / len(ts)

    # Sort by time
    print(f"\n{'Method':<30} {'Time (ms)':>10} {'% of step':>10}")
    print("-" * 52)
    for name, t in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} {t:>8.3f} ms {t/avg*100:>8.1f}%")
    print(f"  {'SUM':<28} {sum(results.values()):>8.3f} ms")

profile_methods(env, dirs, acts)

# Scaling: how does n_envs affect per-env cost?
print(f"\n{'n_envs':<8} {'total ms':>10} {'ms/env':>10} {'agent-steps/s':>15}")
print("-" * 45)
for ne in [1, 4, 8, 16, 32, 64]:
    cfg2 = Config(grid_size=15, n_agents=8, n_predators=0)
    e = BatchedGridWorld(cfg2, device, n_envs=ne)
    e.reset()
    d = torch.randint(0, 5, (ne, 8), device=device)
    a = torch.randint(0, 5, (ne, 8), device=device)
    for _ in range(20):
        e.step(d, a)
    ts = []
    for _ in range(100):
        t0 = time.perf_counter()
        e.step(d, a)
        ts.append((time.perf_counter() - t0) * 1000)
    avg2 = sum(ts) / len(ts)
    print(f"  {ne:<6} {avg2:>8.3f} ms {avg2/ne:>8.3f} ms {ne*8*1000/avg2:>12.0f}")
