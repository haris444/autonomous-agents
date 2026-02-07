"""
Correctness test: Compare BatchedGridWorld vs VecEnv step-by-step.

Strategy:
1. Force both to identical state
2. Run identical deterministic actions (MOVE only — no attacks to avoid deaths)
3. After each step, re-sync food grids (spawning RNG differs)
4. Compare rewards, dones, HP, positions, inventory, ledger

Then run a second pass with attacks to verify combat mechanics.
"""
import torch
from config import Config
from vec_env import VecEnv
from batched_env import BatchedGridWorld

N_ENVS = 4
N_STEPS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sync_state(vec: VecEnv, batched: BatchedGridWorld):
    for i, env in enumerate(vec.envs):
        batched.agent_positions[i] = env.agent_positions.clone()
        batched.prev_agent_positions[i] = env.prev_agent_positions.clone()
        batched.agent_hp[i] = env.agent_hp.clone()
        batched.agent_alive[i] = env.agent_alive.clone()
        batched.agent_inventory[i] = env.agent_inventory.clone()
        batched.occupancy[i] = env.occupancy.clone()
        batched.poor_food[i] = env.poor_food.clone()
        batched.rich_food[i] = env.rich_food.clone()
        batched.signals[i] = env.signals.clone()
        batched.ledger_tensor[i] = env.ledger.tensor.clone()
        batched.step_count[i] = env.step_count
        batched.recent_attacks[i] = env.recent_attacks.clone()


def sync_food_only(vec: VecEnv, batched: BatchedGridWorld):
    for i, env in enumerate(vec.envs):
        batched.poor_food[i] = env.poor_food.clone()
        batched.rich_food[i] = env.rich_food.clone()


def compare_state(vec, batched, n_envs):
    hp_diff = inv_diff = pos_diff = ledger_diff = 0.0
    for i, env in enumerate(vec.envs):
        hp_diff = max(hp_diff, (env.agent_hp - batched.agent_hp[i]).abs().max().item())
        inv_diff = max(inv_diff, (env.agent_inventory - batched.agent_inventory[i]).abs().max().item())
        pos_diff = max(pos_diff, (env.agent_positions - batched.agent_positions[i]).abs().max().item())
        ledger_diff = max(ledger_diff, (env.ledger.tensor - batched.ledger_tensor[i]).abs().max().item())
    return hp_diff, inv_diff, pos_diff, ledger_diff


def run_test(label, action_fn, n_steps):
    config = Config()
    config.max_steps_per_episode = 500  # Very long to avoid time-based resets
    n = config.n_agents

    torch.manual_seed(42)
    vec = VecEnv(config, DEVICE, n_envs=N_ENVS)
    vec.reset()

    torch.manual_seed(99)
    batched = BatchedGridWorld(config, DEVICE, n_envs=N_ENVS)
    batched.reset()

    sync_state(vec, batched)

    print(f"\n{'='*60}")
    print(f"TEST: {label} ({n_steps} steps)")
    print(f"{'='*60}")

    max_r = max_hp = max_inv = max_pos = max_ledger = 0.0
    done_mm = 0
    steps_compared = 0

    for step in range(1, n_steps + 1):
        directions, action_types = action_fn(step, N_ENVS, n, DEVICE)

        vec_obs, vec_r, vec_d, vec_info = vec.step(directions, action_types)
        bat_obs, bat_r, bat_d, bat_info = batched.step(directions, action_types)

        vec_done_all = vec_d.all(dim=1)
        bat_done_all = bat_d.all(dim=1)
        any_reset = vec_done_all.any() or bat_done_all.any()

        if not any_reset:
            steps_compared += 1
            r_diff = (vec_r - bat_r).abs().max().item()
            d_match = torch.equal(vec_d, bat_d)
            hp_diff, inv_diff, pos_diff, ledger_diff = compare_state(vec, batched, N_ENVS)

            max_r = max(max_r, r_diff)
            max_hp = max(max_hp, hp_diff)
            max_inv = max(max_inv, inv_diff)
            max_pos = max(max_pos, pos_diff)
            max_ledger = max(max_ledger, ledger_diff)
            if not d_match:
                done_mm += 1

            if r_diff > 0.01 or hp_diff > 0.01 or pos_diff > 0:
                print(f"  Step {step} DIFF: r={r_diff:.4f} hp={hp_diff:.4f} "
                      f"pos={pos_diff} ledger={ledger_diff:.4f}")

        sync_food_only(vec, batched)
        if any_reset:
            sync_state(vec, batched)

        if step % 25 == 0:
            print(f"  Step {step}: r={max_r:.4f} hp={max_hp:.4f} pos={max_pos:.0f} "
                  f"ledger={max_ledger:.4f} (compared {steps_compared} steps)")

    ok = max_r < 0.01 and max_hp < 0.01 and max_pos == 0 and done_mm == 0 and max_ledger < 0.01
    print(f"\nResults: reward={max_r:.6f} hp={max_hp:.6f} inv={max_inv:.6f} "
          f"pos={max_pos:.0f} ledger={max_ledger:.6f} dones_mm={done_mm}")
    print(f"{'PASS' if ok else 'FAIL'} ({steps_compared} steps compared, "
          f"{n_steps - steps_compared} skipped due to reset)")
    return ok


def move_only(step, ne, n, device):
    """Only MOVE actions, cycle through directions."""
    dirs = torch.zeros((ne, n), device=device, dtype=torch.long)
    acts = torch.zeros((ne, n), device=device, dtype=torch.long)  # ACT_MOVE=0
    for a in range(n):
        dirs[:, a] = (step + a) % 5
    return dirs, acts


def move_and_attack(step, ne, n, device):
    """Mix of MOVE and ATTACK actions."""
    dirs = torch.zeros((ne, n), device=device, dtype=torch.long)
    acts = torch.zeros((ne, n), device=device, dtype=torch.long)
    for a in range(n):
        dirs[:, a] = (step + a) % 5
        acts[:, a] = 1 if (step + a) % 4 == 0 else 0  # Attack every 4th step per agent
    return dirs, acts


def all_action_types(step, ne, n, device):
    """Cycle through all action types including GIVE and SIGNAL."""
    dirs = torch.zeros((ne, n), device=device, dtype=torch.long)
    acts = torch.zeros((ne, n), device=device, dtype=torch.long)
    for a in range(n):
        dirs[:, a] = (step + a) % 5
        acts[:, a] = (step + a) % 4  # MOVE, ATTACK, GIVE, SIGNAL (skip COOP)
    return dirs, acts


def main():
    results = []
    results.append(("Movement only", run_test("Movement only", move_only, N_STEPS)))
    results.append(("Move + Attack", run_test("Move + Attack", move_and_attack, 50)))
    results.append(("All actions", run_test("All action types", all_action_types, 50)))

    print(f"\n{'='*60}")
    print("SUMMARY:")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, ok in results)
    print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")


if __name__ == '__main__':
    main()
