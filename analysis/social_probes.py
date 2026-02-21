"""
Social Probes — Controlled behavioral experiments on trained agents.

Tests whether agents with access to a reputation ledger develop discriminating
social behaviors compared to agents trained without ledger information.

Probe scenarios inject synthetic ledger histories and measure behavioral responses:
- Baseline: blank ledger (no history)
- Ally: positive history (food given, cooperation, defense)
- Enemy: negative history (high damage dealt)
- Forgiveness: mixed history (damage followed by cooperation)
- Stranger vs Friend: one ally + one stranger, test discrimination

Usage:
    python social_probes.py --checkpoint models/with_ledger.pt
    python social_probes.py --compare models/with_ledger.pt models/without_ledger.pt
    python social_probes.py --checkpoint models/with_ledger.pt --save-gifs
"""
import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from core.config import Config
from env.environment import GridWorld
from core.ledger import Ledger
from agents.ppo import PPO
from training.scenarios import CoopFoodScenario
from analysis.visualize import EpisodeRecorder, replay_episode


# ─── Probe definitions ───────────────────────────────────────────────

@dataclass
class ProbeConfig:
    """Configuration for a single behavioral probe."""
    name: str
    n_agents: int
    n_rich_food: int = 3
    description: str = ""

    def inject_ledger(self, env: GridWorld) -> None:
        """Override in subclasses to inject specific ledger histories."""
        pass


class BaselineProbe(ProbeConfig):
    """Blank ledger — no history between agents."""

    def __init__(self):
        super().__init__(
            name="Baseline",
            n_agents=2,
            description="Blank ledger (all zeros). Default behavior with no history."
        )

    def inject_ledger(self, env: GridWorld) -> None:
        env.ledger.reset()


class AllyProbe(ProbeConfig):
    """Positive history — agent 1 has been helpful to agent 0."""

    def __init__(self):
        super().__init__(
            name="Ally",
            n_agents=2,
            description="Positive history: food_given=50, coop=5, defense=30."
        )

    def inject_ledger(self, env: GridWorld) -> None:
        env.ledger.reset()
        # Agent 1 helped agent 0
        env.ledger.tensor[1, 0, Ledger.FOOD_GIVEN] = 50.0
        env.ledger.tensor[1, 0, Ledger.COOP_COUNT] = 5.0
        env.ledger.tensor[1, 0, Ledger.DEFENSE_SCORE] = 30.0
        env.ledger.tensor[1, 0, Ledger.DAMAGE_DEALT] = 0.0
        # Agent 0 also helped agent 1 (mutual)
        env.ledger.tensor[0, 1, Ledger.FOOD_GIVEN] = 30.0
        env.ledger.tensor[0, 1, Ledger.COOP_COUNT] = 5.0
        env.ledger.tensor[0, 1, Ledger.DEFENSE_SCORE] = 20.0
        env.ledger.tensor[0, 1, Ledger.DAMAGE_DEALT] = 0.0


class EnemyProbe(ProbeConfig):
    """Negative history — agent 1 has attacked agent 0."""

    def __init__(self):
        super().__init__(
            name="Enemy",
            n_agents=2,
            description="Negative history: damage_dealt=60, no positive signals."
        )

    def inject_ledger(self, env: GridWorld) -> None:
        env.ledger.reset()
        # Agent 1 attacked agent 0
        env.ledger.tensor[1, 0, Ledger.DAMAGE_DEALT] = 60.0
        env.ledger.tensor[1, 0, Ledger.FOOD_GIVEN] = 0.0
        env.ledger.tensor[1, 0, Ledger.COOP_COUNT] = 0.0
        env.ledger.tensor[1, 0, Ledger.DEFENSE_SCORE] = 0.0
        # Mutual hostility
        env.ledger.tensor[0, 1, Ledger.DAMAGE_DEALT] = 40.0
        env.ledger.tensor[0, 1, Ledger.FOOD_GIVEN] = 0.0
        env.ledger.tensor[0, 1, Ledger.COOP_COUNT] = 0.0
        env.ledger.tensor[0, 1, Ledger.DEFENSE_SCORE] = 0.0


class ForgivenessProbe(ProbeConfig):
    """Mixed history — initial hostility followed by cooperation."""

    def __init__(self):
        super().__init__(
            name="Forgiveness",
            n_agents=2,
            description="Mixed: damage_dealt=40, then food_given=30, coop=3."
        )

    def inject_ledger(self, env: GridWorld) -> None:
        env.ledger.reset()
        # Agent 1 was hostile but then reformed
        env.ledger.tensor[1, 0, Ledger.DAMAGE_DEALT] = 40.0
        env.ledger.tensor[1, 0, Ledger.FOOD_GIVEN] = 30.0
        env.ledger.tensor[1, 0, Ledger.COOP_COUNT] = 3.0
        env.ledger.tensor[1, 0, Ledger.DEFENSE_SCORE] = 10.0
        # Agent 0 has been cautiously cooperative
        env.ledger.tensor[0, 1, Ledger.DAMAGE_DEALT] = 20.0
        env.ledger.tensor[0, 1, Ledger.FOOD_GIVEN] = 15.0
        env.ledger.tensor[0, 1, Ledger.COOP_COUNT] = 3.0
        env.ledger.tensor[0, 1, Ledger.DEFENSE_SCORE] = 5.0


class StrangerVsFriendProbe(ProbeConfig):
    """3-agent probe: agent 1 = ally, agent 2 = stranger (blank history)."""

    def __init__(self):
        super().__init__(
            name="StrangerVsFriend",
            n_agents=3,
            n_rich_food=3,
            description="Agent 1 = ally history, Agent 2 = blank history (stranger)."
        )

    def inject_ledger(self, env: GridWorld) -> None:
        env.ledger.reset()
        # Agent 1 is a known ally
        env.ledger.tensor[1, 0, Ledger.FOOD_GIVEN] = 50.0
        env.ledger.tensor[1, 0, Ledger.COOP_COUNT] = 5.0
        env.ledger.tensor[1, 0, Ledger.DEFENSE_SCORE] = 30.0
        env.ledger.tensor[1, 0, Ledger.DAMAGE_DEALT] = 0.0
        env.ledger.tensor[0, 1, Ledger.FOOD_GIVEN] = 30.0
        env.ledger.tensor[0, 1, Ledger.COOP_COUNT] = 5.0
        env.ledger.tensor[0, 1, Ledger.DEFENSE_SCORE] = 20.0
        env.ledger.tensor[0, 1, Ledger.DAMAGE_DEALT] = 0.0
        # Agent 2 is a stranger — all zeros (already reset above)


ALL_PROBES = [
    BaselineProbe(),
    AllyProbe(),
    EnemyProbe(),
    ForgivenessProbe(),
    StrangerVsFriendProbe(),
]


# ─── Probe result data structure ─────────────────────────────────────

@dataclass
class ProbeResult:
    """Results from running a probe over multiple episodes."""
    probe_name: str
    n_episodes: int = 0
    # Per-agent metrics accumulated across episodes (agent 0 is the focal agent)
    coop_rates: List[float] = field(default_factory=list)
    attack_rates: List[float] = field(default_factory=list)
    give_rates: List[float] = field(default_factory=list)
    move_rates: List[float] = field(default_factory=list)
    avg_returns: List[float] = field(default_factory=list)
    rich_food_eaten: List[int] = field(default_factory=list)
    avg_distances: List[float] = field(default_factory=list)  # avg distance to other agents
    steps_alive: List[int] = field(default_factory=list)
    # For StrangerVsFriend: distance to ally vs stranger
    dist_to_ally: List[float] = field(default_factory=list)
    dist_to_stranger: List[float] = field(default_factory=list)
    # Best episode recording (for GIF)
    best_recording: Optional[object] = field(default=None, repr=False)
    best_ledger_snapshots: Optional[list] = field(default=None, repr=False)
    best_predator_ledger_snapshots: Optional[list] = field(default=None, repr=False)

    @property
    def mean_coop_rate(self) -> float:
        return np.mean(self.coop_rates) if self.coop_rates else 0.0

    @property
    def mean_attack_rate(self) -> float:
        return np.mean(self.attack_rates) if self.attack_rates else 0.0

    @property
    def mean_give_rate(self) -> float:
        return np.mean(self.give_rates) if self.give_rates else 0.0

    @property
    def mean_return(self) -> float:
        return np.mean(self.avg_returns) if self.avg_returns else 0.0

    @property
    def mean_rich_food(self) -> float:
        return np.mean(self.rich_food_eaten) if self.rich_food_eaten else 0.0

    @property
    def mean_distance(self) -> float:
        return np.mean(self.avg_distances) if self.avg_distances else 0.0

    @property
    def std_coop_rate(self) -> float:
        return np.std(self.coop_rates) if len(self.coop_rates) > 1 else 0.0

    @property
    def std_attack_rate(self) -> float:
        return np.std(self.attack_rates) if len(self.attack_rates) > 1 else 0.0


# ─── Main evaluation runner ──────────────────────────────────────────

ACT_MOVE, ACT_ATTACK, ACT_GIVE, ACT_SIGNAL, ACT_COOPERATE = 0, 1, 2, 3, 4


def load_checkpoint(path: str, device: torch.device) -> Tuple[PPO, Config]:
    """Load a trained checkpoint and return (ppo, config)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    raw_cfg = ckpt['config']
    if isinstance(raw_cfg, dict):
        config = Config.from_dict(raw_cfg)
    else:
        config = raw_cfg

    ppo = PPO(config, device)
    ppo.network.load_state_dict(ckpt['network_state_dict'])
    ppo.network.eval()

    print(f"Loaded checkpoint: {path}")
    print(f"  {config.n_agents} agents, {config.grid_size}x{config.grid_size} grid")
    if 'curriculum_phase' in ckpt:
        print(f"  Phase {ckpt['curriculum_phase']}, ep {ckpt.get('episode', '?')}, "
              f"avg_return {ckpt.get('avg_return', 0):.1f}")

    return ppo, config


def _get_actions(ppo, obs, dir_mask, act_mask):
    """Get actions from the network."""
    dirs, acts, _, _, _ = ppo.get_actions_and_values(obs, dir_mask, act_mask)
    return dirs, acts


def run_probe(
    ppo: PPO,
    config: Config,
    probe: ProbeConfig,
    device: torch.device,
    n_episodes: int = 50,
    record_best: bool = False,
) -> ProbeResult:
    """Run a single probe scenario over multiple episodes."""
    result = ProbeResult(probe_name=probe.name, n_episodes=n_episodes)
    env = GridWorld(config, device)

    # Use a coop food scenario as the base environment setup
    base_scenario = CoopFoodScenario(distance=3, n_rich_food=probe.n_rich_food)
    n_active = probe.n_agents

    ppo.set_n_active(n_active)
    ppo.set_clone_mode(False)

    best_return = float('-inf')
    recorder = EpisodeRecorder(config) if record_best else None

    for ep in range(n_episodes):
        obs = env.reset()
        env.apply_scenario(base_scenario)

        # Override ledger with probe-specific history
        probe.inject_ledger(env)

        # Activate correct number of agents
        env.agent_alive[:n_active] = True
        env.agent_hp[:n_active] = config.max_hp
        env.agent_alive[n_active:] = False
        env.agent_hp[n_active:] = 0
        env._update_occupancy()

        # Re-fetch observations after ledger injection
        obs = env._get_all_observations()

        # Episode tracking
        action_counts = [0] * 5  # MOVE, ATTACK, GIVE, SIGNAL, COOP for agent 0
        total_return = 0.0
        total_steps = 0
        rich_food_count = 0
        total_distance = 0.0
        dist_ally_total = 0.0
        dist_stranger_total = 0.0

        is_recording = (record_best and recorder is not None)
        if is_recording:
            recorder.reset()

        for step in range(config.max_steps_per_episode):
            dir_mask, act_mask = env.get_action_masks()

            with torch.no_grad():
                dirs, acts = _get_actions(ppo, obs, dir_mask, act_mask)

            # Record step before env.step
            if is_recording:
                actions_dict = {i: (dirs[i].item(), acts[i].item()) for i in range(n_active)}
                recorder.record(env, actions=actions_dict)

            obs, rewards, dones, infos = env.step(dirs, acts)

            # Update recording with rewards
            if is_recording and recorder.steps:
                recorder.steps[-1].rewards = {i: rewards[i].item() for i in range(n_active)}

            if env.agent_alive[0]:
                total_steps += 1
                act_type = acts[0].item()
                action_counts[act_type] += 1
                total_return += rewards[0].item()

                # Track rich food (successful coop harvests)
                rich_food_count += infos.get('rich_food_eaten', {}).get(0, 0) if isinstance(infos.get('rich_food_eaten'), dict) else 0

                # Distance to other alive agents
                pos0 = env.agent_positions[0].float()
                for i in range(1, n_active):
                    if env.agent_alive[i]:
                        pos_i = env.agent_positions[i].float()
                        d = (pos0 - pos_i).abs().sum().item()
                        total_distance += d

                        if probe.name == "StrangerVsFriend":
                            if i == 1:  # ally
                                dist_ally_total += d
                            elif i == 2:  # stranger
                                dist_stranger_total += d

            if dones.all():
                break

        # Compute rates
        total_acts = max(total_steps, 1)
        result.coop_rates.append(action_counts[ACT_COOPERATE] / total_acts)
        result.attack_rates.append(action_counts[ACT_ATTACK] / total_acts)
        result.give_rates.append(action_counts[ACT_GIVE] / total_acts)
        result.move_rates.append(action_counts[ACT_MOVE] / total_acts)
        result.avg_returns.append(total_return)
        result.rich_food_eaten.append(rich_food_count)
        result.steps_alive.append(total_steps)

        # Average distance per step
        avg_dist = total_distance / max(total_steps, 1) / max(n_active - 1, 1)
        result.avg_distances.append(avg_dist)

        if probe.name == "StrangerVsFriend":
            result.dist_to_ally.append(dist_ally_total / max(total_steps, 1))
            result.dist_to_stranger.append(dist_stranger_total / max(total_steps, 1))

        # Track best episode for GIF
        if record_best and total_return > best_return:
            best_return = total_return
            result.best_recording = recorder.steps.copy()
            result.best_ledger_snapshots = recorder.ledger_snapshots.copy()
            result.best_predator_ledger_snapshots = (
                recorder.predator_ledger_snapshots.copy()
                if recorder.predator_ledger_snapshots else None
            )

    return result


# ─── Visualization ────────────────────────────────────────────────────

def plot_comparison(results_a: Dict[str, ProbeResult],
                    results_b: Dict[str, ProbeResult],
                    label_a: str = "With Ledger",
                    label_b: str = "Without Ledger",
                    output_dir: str = "probe_results") -> None:
    """Generate side-by-side comparison bar charts."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    probe_names = list(results_a.keys())
    x = np.arange(len(probe_names))
    width = 0.35

    # --- Cooperation Rate ---
    fig, ax = plt.subplots(figsize=(10, 5))
    coop_a = [results_a[p].mean_coop_rate for p in probe_names]
    coop_b = [results_b[p].mean_coop_rate for p in probe_names]
    err_a = [results_a[p].std_coop_rate for p in probe_names]
    err_b = [results_b[p].std_coop_rate for p in probe_names]
    ax.bar(x - width/2, coop_a, width, label=label_a, yerr=err_a, capsize=4, color='#2196F3')
    ax.bar(x + width/2, coop_b, width, label=label_b, yerr=err_b, capsize=4, color='#FF9800')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('Cooperation Rate by Probe Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison_coop_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir}/comparison_coop_rate.png")

    # --- Attack Rate ---
    fig, ax = plt.subplots(figsize=(10, 5))
    atk_a = [results_a[p].mean_attack_rate for p in probe_names]
    atk_b = [results_b[p].mean_attack_rate for p in probe_names]
    err_a = [results_a[p].std_attack_rate for p in probe_names]
    err_b = [results_b[p].std_attack_rate for p in probe_names]
    ax.bar(x - width/2, atk_a, width, label=label_a, yerr=err_a, capsize=4, color='#2196F3')
    ax.bar(x + width/2, atk_b, width, label=label_b, yerr=err_b, capsize=4, color='#FF9800')
    ax.set_ylabel('Attack Rate')
    ax.set_title('Attack Rate by Probe Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison_attack_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir}/comparison_attack_rate.png")

    # --- Rich Food Eaten ---
    fig, ax = plt.subplots(figsize=(10, 5))
    rich_a = [results_a[p].mean_rich_food for p in probe_names]
    rich_b = [results_b[p].mean_rich_food for p in probe_names]
    ax.bar(x - width/2, rich_a, width, label=label_a, color='#2196F3')
    ax.bar(x + width/2, rich_b, width, label=label_b, color='#FF9800')
    ax.set_ylabel('Rich Food Eaten (avg)')
    ax.set_title('Cooperative Harvesting by Probe Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison_rich_food.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir}/comparison_rich_food.png")

    # --- Average Return ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ret_a = [results_a[p].mean_return for p in probe_names]
    ret_b = [results_b[p].mean_return for p in probe_names]
    ax.bar(x - width/2, ret_a, width, label=label_a, color='#2196F3')
    ax.bar(x + width/2, ret_b, width, label=label_b, color='#FF9800')
    ax.set_ylabel('Average Return (Agent 0)')
    ax.set_title('Average Return by Probe Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison_return.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir}/comparison_return.png")


def plot_discrimination(results: Dict[str, ProbeResult],
                        label: str = "With Ledger",
                        output_dir: str = "probe_results") -> None:
    """
    The key figure: cooperation + attack rates across all probes for ONE model.
    Shows whether the agent discriminates based on ledger content.
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    probe_names = list(results.keys())
    x = np.arange(len(probe_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    coop = [results[p].mean_coop_rate for p in probe_names]
    attack = [results[p].mean_attack_rate for p in probe_names]
    give = [results[p].mean_give_rate for p in probe_names]

    ax.bar(x - width, coop, width, label='Cooperate', color='#4CAF50')
    ax.bar(x, attack, width, label='Attack', color='#F44336')
    ax.bar(x + width, give, width, label='Give', color='#2196F3')

    ax.set_ylabel('Action Rate (fraction of steps)')
    ax.set_title(f'Social Discrimination Test — {label}')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    safe_label = label.replace(' ', '_').lower()
    path = f"{output_dir}/discrimination_{safe_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_approach_avoidance(results: Dict[str, ProbeResult],
                            label: str = "With Ledger",
                            output_dir: str = "probe_results") -> None:
    """Plot average inter-agent distance across probes (approach vs avoidance)."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    probe_names = list(results.keys())
    x = np.arange(len(probe_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    dists = [results[p].mean_distance for p in probe_names]
    ax.bar(x, dists, color='#9C27B0')
    ax.set_ylabel('Avg Distance to Other Agents')
    ax.set_title(f'Approach vs Avoidance — {label}\n(lower = more approach)')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=15)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    safe_label = label.replace(' ', '_').lower()
    path = f"{output_dir}/approach_avoidance_{safe_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_gifs(results: Dict[str, ProbeResult], config: Config,
              output_dir: str = "probe_results") -> None:
    """Save GIF replays of the best episode per probe."""
    import matplotlib
    matplotlib.use('Agg')

    os.makedirs(output_dir, exist_ok=True)
    for probe_name, result in results.items():
        if result.best_recording is not None:
            gif_path = f"{output_dir}/replay_{probe_name.lower()}.gif"
            print(f"  Generating GIF: {gif_path} ({len(result.best_recording)} steps)")
            try:
                replay_episode(
                    result.best_recording, config,
                    save_gif=True, gif_path=gif_path,
                    ledger_snapshots=result.best_ledger_snapshots,
                    predator_ledger_snapshots=result.best_predator_ledger_snapshots,
                    show_ledger=True
                )
            except Exception as e:
                print(f"    Warning: GIF generation failed for {probe_name}: {e}")


# ─── Report ───────────────────────────────────────────────────────────

def print_report(results: Dict[str, ProbeResult], label: str = "Model") -> None:
    """Print a text summary of probe results."""
    print(f"\n{'=' * 70}")
    print(f"  SOCIAL PROBES REPORT — {label}")
    print(f"{'=' * 70}")
    print()
    print(f"{'Probe':<20} {'Coop%':>7} {'Atk%':>7} {'Give%':>7} {'Move%':>7} "
          f"{'Return':>8} {'Dist':>6} {'Alive':>6}")
    print(f"{'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*6}")

    for name, r in results.items():
        print(f"{name:<20} "
              f"{r.mean_coop_rate*100:>6.1f}% "
              f"{r.mean_attack_rate*100:>6.1f}% "
              f"{r.mean_give_rate*100:>6.1f}% "
              f"{np.mean(r.move_rates)*100:>6.1f}% "
              f"{r.mean_return:>8.1f} "
              f"{r.mean_distance:>6.1f} "
              f"{np.mean(r.steps_alive):>6.0f}")

    # Special: StrangerVsFriend discrimination
    if "StrangerVsFriend" in results:
        svf = results["StrangerVsFriend"]
        if svf.dist_to_ally and svf.dist_to_stranger:
            print(f"\n  StrangerVsFriend discrimination:")
            print(f"    Avg distance to ally:     {np.mean(svf.dist_to_ally):.1f}")
            print(f"    Avg distance to stranger: {np.mean(svf.dist_to_stranger):.1f}")
            diff = np.mean(svf.dist_to_stranger) - np.mean(svf.dist_to_ally)
            print(f"    Difference:               {diff:+.1f} "
                  f"({'approaches ally more' if diff > 0 else 'approaches stranger more'})")

    print()


def run_all_probes(
    checkpoint_path: str,
    device: torch.device,
    n_episodes: int = 50,
    output_dir: str = "probe_results",
    save_gifs_flag: bool = False,
    label: str = None,
) -> Dict[str, ProbeResult]:
    """Run all probes on a single checkpoint."""
    ppo, config = load_checkpoint(checkpoint_path, device)

    if label is None:
        label = os.path.basename(checkpoint_path)

    results = {}
    for probe in ALL_PROBES:
        print(f"\n  Running probe: {probe.name} ({n_episodes} episodes)...")
        result = run_probe(
            ppo, config, probe, device,
            n_episodes=n_episodes,
            record_best=save_gifs_flag,
        )
        results[probe.name] = result

    print_report(results, label)

    # Generate plots
    plot_discrimination(results, label=label, output_dir=output_dir)
    plot_approach_avoidance(results, label=label, output_dir=output_dir)

    if save_gifs_flag:
        print("\nGenerating GIF replays...")
        save_gifs(results, config, output_dir=output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description='Social Probes — Behavioral experiments on trained agents')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a single checkpoint to evaluate')
    parser.add_argument('--compare', nargs=2, metavar=('WITH_LEDGER', 'WITHOUT_LEDGER'),
                        help='Compare two checkpoints (with vs without ledger)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes per probe (default: 50)')
    parser.add_argument('--output-dir', type=str, default='probe_results',
                        help='Output directory for plots and GIFs')
    parser.add_argument('--save-gifs', action='store_true',
                        help='Save GIF replays of best episodes')
    parser.add_argument('--label', type=str, default=None,
                        help='Label for single-checkpoint mode')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.compare:
        path_a, path_b = args.compare
        label_a = "With Ledger"
        label_b = "Without Ledger"
        dir_a = f"{args.output_dir}/with_ledger"
        dir_b = f"{args.output_dir}/without_ledger"

        print(f"\n{'='*70}")
        print(f"  MODEL A: {label_a} ({path_a})")
        print(f"{'='*70}")
        results_a = run_all_probes(path_a, device, args.episodes, dir_a,
                                    args.save_gifs, label=label_a)

        print(f"\n{'='*70}")
        print(f"  MODEL B: {label_b} ({path_b})")
        print(f"{'='*70}")
        results_b = run_all_probes(path_b, device, args.episodes, dir_b,
                                    args.save_gifs, label=label_b)

        # Comparison charts
        print(f"\n{'='*70}")
        print(f"  GENERATING COMPARISON CHARTS")
        print(f"{'='*70}")
        plot_comparison(results_a, results_b, label_a, label_b, args.output_dir)

        # Print side-by-side summary
        print(f"\n{'='*70}")
        print(f"  SIDE-BY-SIDE COMPARISON")
        print(f"{'='*70}")
        print(f"\n{'Probe':<20} {'Coop A':>7} {'Coop B':>7} {'Atk A':>6} {'Atk B':>6} "
              f"{'Ret A':>7} {'Ret B':>7}")
        print(f"{'-'*20} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
        for name in results_a:
            ra, rb = results_a[name], results_b[name]
            print(f"{name:<20} "
                  f"{ra.mean_coop_rate*100:>6.1f}% "
                  f"{rb.mean_coop_rate*100:>6.1f}% "
                  f"{ra.mean_attack_rate*100:>5.1f}% "
                  f"{rb.mean_attack_rate*100:>5.1f}% "
                  f"{ra.mean_return:>7.1f} "
                  f"{rb.mean_return:>7.1f}")

    elif args.checkpoint:
        run_all_probes(
            args.checkpoint, device, args.episodes, args.output_dir,
            args.save_gifs, label=args.label
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
