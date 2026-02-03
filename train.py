"""
Main training loop for Multi-Agent RL.

Orchestrates rollout collection, GAE computation, and PPO updates.

Usage:
    python train.py                    # Train without visualization
    python train.py --visualize        # Train with live visualization
    python train.py --visualize --render-every 50  # Render grid every 50 updates
"""
from typing import Dict, Optional, Tuple
import time
import argparse

import torch
import matplotlib.pyplot as plt

from config import Config
from environment import GridWorld
from network import ActorCritic
from buffer import RolloutBuffer, SingleAgentBuffer
from ppo import PPO, IndependentPPO, VmapPPO
from utils import set_seed, get_device
from scenarios import CURRICULUM, THRESHOLDS
from ledger import Ledger


def apply_scripted_partner(
    env,
    directions: torch.Tensor,
    action_types: torch.Tensor,
    relationship: str = 'ally'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Override partner agents' actions based on relationship type.

    For allies (relationship='ally'):
        All partners (1, 2, ...) coop when near rich food with agent 0.

    For enemies (relationship='enemy'):
        All partners attack agent 0 when adjacent.

    For mixed (relationship='mixed'):
        Agent 1 is always ally (coops), agent 2+ are enemies (attack).

    This provides consistent partner behavior matching the injected history.

    Factored action space:
        directions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
        action_types: 0=MOVE, 1=ATTACK, 2=GIVE, 3=SIGNAL, 4=COOPERATE
    """
    # Direction constants
    DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT, DIR_STAY = 0, 1, 2, 3, 4
    # Action type constants
    ACT_MOVE, ACT_ATTACK, ACT_GIVE, ACT_SIGNAL, ACT_COOPERATE = 0, 1, 2, 3, 4

    pos0 = env.agent_positions[0].float()
    directions = directions.clone()
    action_types = action_types.clone()

    # Determine which agents are allies vs enemies
    n_active = env.agent_alive.sum().item()

    if relationship == 'ally':
        ally_agents = list(range(1, int(n_active)))
        enemy_agents = []
    elif relationship == 'enemy':
        ally_agents = []
        enemy_agents = list(range(1, int(n_active)))
    elif relationship == 'mixed':
        # Agent 1 is always ally, agent 2+ are enemies
        ally_agents = [1] if n_active > 1 else []
        enemy_agents = list(range(2, int(n_active)))
    elif relationship == 'neutral':
        # Neutral: responds to agent 0's behavior with reciprocity
        # - Positive: If agent 0 gives food or cooperates, neutral becomes friendly
        # - Negative: If agent 0 attacks, neutral retaliates
        # - Passive: If no interaction, block attacks only

        # Get rich food locations for positive reciprocity behavior
        rich_coords = torch.nonzero(env.rich_food, as_tuple=False)

        for agent_id in range(1, int(n_active)):
            if not env.agent_alive[agent_id]:
                continue

            # Check agent 0's behavior toward this neutral
            damage_dealt = env.ledger.tensor[0, agent_id, Ledger.DAMAGE_DEALT].item()
            food_given = env.ledger.tensor[0, agent_id, Ledger.FOOD_GIVEN].item()
            coop_count = env.ledger.tensor[0, agent_id, Ledger.COOP_COUNT].item()

            if damage_dealt > 0:
                # NEGATIVE RECIPROCITY: Agent 0 attacked -> neutral retaliates
                pos_n = env.agent_positions[agent_id].float()
                diff = pos0 - pos_n
                dist = diff.abs().sum()

                if dist == 1:  # Adjacent - attack
                    action_types[agent_id] = ACT_ATTACK
                    if diff[0] == -1:
                        directions[agent_id] = DIR_UP
                    elif diff[0] == 1:
                        directions[agent_id] = DIR_DOWN
                    elif diff[1] == -1:
                        directions[agent_id] = DIR_LEFT
                    else:
                        directions[agent_id] = DIR_RIGHT
                # else: keep learned policy action (move toward agent 0 to retaliate)

            elif food_given > 0 or coop_count > 0:
                # POSITIVE RECIPROCITY: Agent 0 was nice -> neutral becomes ally
                pos_n = env.agent_positions[agent_id].float()
                action_types[agent_id] = ACT_MOVE
                directions[agent_id] = DIR_STAY

                if rich_coords.numel() > 0:
                    # Find closest rich food
                    best_food = None
                    best_dist = float('inf')
                    for food_pos in rich_coords:
                        food_pos_f = food_pos.float()
                        dist_n = (pos_n - food_pos_f).abs().sum().item()
                        if dist_n < best_dist:
                            best_dist = dist_n
                            best_food = food_pos_f

                    if best_food is not None:
                        dist0 = (pos0 - best_food).abs().sum().item()

                        if dist0 <= 1 and best_dist <= 1:
                            # Both near food - cooperate
                            action_types[agent_id] = ACT_COOPERATE
                            directions[agent_id] = DIR_STAY
                        elif best_dist > 0:
                            # Move toward food
                            diff = best_food - pos_n
                            if abs(diff[0]) >= abs(diff[1]):
                                directions[agent_id] = DIR_UP if diff[0] < 0 else DIR_DOWN
                            else:
                                directions[agent_id] = DIR_LEFT if diff[1] < 0 else DIR_RIGHT
            else:
                # PASSIVE: No interaction yet - block attacks only
                if action_types[agent_id] == ACT_ATTACK:
                    action_types[agent_id] = ACT_MOVE
                    directions[agent_id] = DIR_STAY

        return directions, action_types
    else:
        return directions, action_types

    # Apply ally behavior: move towards rich food, coop when both agents adjacent
    rich_coords = torch.nonzero(env.rich_food, as_tuple=False)
    food_count = env.rich_food.sum().item()
    expected_foods = getattr(env, '_coop_n_foods', 1)  # Get from scenario or default

    for agent_id in ally_agents:
        if not env.agent_alive[agent_id]:
            continue
        pos_a = env.agent_positions[agent_id].float()

        # Default to MOVE + STAY
        action_types[agent_id] = ACT_MOVE
        directions[agent_id] = DIR_STAY

        # Check if we should flee (food was recently eaten)
        # "Flee mode": partner moves away from food nearest to agent 0
        flee_mode = False
        if food_count < expected_foods and rich_coords.numel() > 0:
            # Food was eaten - find food closest to agent 0
            best_food_for_agent0 = None
            best_dist_to_0 = float('inf')
            for food_pos in rich_coords:
                food_pos_f = food_pos.float()
                dist_to_0 = (pos0 - food_pos_f).abs().sum().item()
                if dist_to_0 < best_dist_to_0:
                    best_dist_to_0 = dist_to_0
                    best_food_for_agent0 = food_pos_f

            if best_food_for_agent0 is not None:
                dist_partner_to_that_food = (pos_a - best_food_for_agent0).abs().sum().item()
                # If partner is near that food, move AWAY from it
                if dist_partner_to_that_food <= 3:
                    flee_mode = True
                    # Move away = opposite direction from food
                    diff = pos_a - best_food_for_agent0  # Direction AWAY from food
                    action_types[agent_id] = ACT_MOVE
                    if abs(diff[0]) >= abs(diff[1]):
                        directions[agent_id] = DIR_DOWN if diff[0] > 0 else DIR_UP
                    else:
                        directions[agent_id] = DIR_RIGHT if diff[1] > 0 else DIR_LEFT

        if not flee_mode and rich_coords.numel() > 0:
            # Normal behavior: Find closest rich food and move toward it
            best_food = None
            best_dist = float('inf')
            for food_pos in rich_coords:
                food_pos_f = food_pos.float()
                dist_a = (pos_a - food_pos_f).abs().sum().item()
                if dist_a < best_dist:
                    best_dist = dist_a
                    best_food = food_pos_f

            if best_food is not None:
                dist0 = (pos0 - best_food).abs().sum().item()
                dist_a = best_dist

                if dist0 <= 1 and dist_a <= 1:
                    # Both near food - ally coops
                    action_types[agent_id] = ACT_COOPERATE
                    directions[agent_id] = DIR_STAY  # Direction ignored for COOP
                elif dist_a > 0:
                    # Not on top of food yet - move towards it
                    action_types[agent_id] = ACT_MOVE
                    diff = best_food - pos_a  # Direction to food
                    # Prioritize larger distance axis
                    if abs(diff[0]) >= abs(diff[1]):
                        if diff[0] < 0:
                            directions[agent_id] = DIR_UP
                        elif diff[0] > 0:
                            directions[agent_id] = DIR_DOWN
                    else:
                        if diff[1] < 0:
                            directions[agent_id] = DIR_LEFT
                        elif diff[1] > 0:
                            directions[agent_id] = DIR_RIGHT
                # else: adjacent to food but agent 0 not there yet - STAY

    # Apply enemy behavior: attack agent 0 when adjacent
    for agent_id in enemy_agents:
        if not env.agent_alive[agent_id]:
            continue
        pos_e = env.agent_positions[agent_id].float()
        diff = pos0 - pos_e  # Direction from enemy to agent 0
        dist = diff.abs().sum()

        if dist == 1:  # Adjacent
            action_types[agent_id] = ACT_ATTACK
            if diff[0] == -1:  # Agent 0 is above
                directions[agent_id] = DIR_UP
            elif diff[0] == 1:  # Agent 0 is below
                directions[agent_id] = DIR_DOWN
            elif diff[1] == -1:  # Agent 0 is left
                directions[agent_id] = DIR_LEFT
            elif diff[1] == 1:  # Agent 0 is right
                directions[agent_id] = DIR_RIGHT
        else:
            # Not adjacent - stay in place
            action_types[agent_id] = ACT_MOVE
            directions[agent_id] = DIR_STAY

    return directions, action_types


def save_pretrained(network, path: str, curriculum_phase: int = None):
    """Save pretrained weights from a single network."""
    data = {'state_dict': network.state_dict()}
    if curriculum_phase is not None:
        data['curriculum_phase'] = curriculum_phase
    torch.save(data, path)
    print(f"Saved pretrained weights to {path}")


def load_pretrained(networks, path: str, env=None):
    """Load pretrained weights into all networks. Handles both checkpoint and pretrained formats.

    If env is provided and checkpoint has curriculum_phase, restores the phase.
    Handles architecture mismatches by loading only compatible layers.
    """
    checkpoint = torch.load(path, weights_only=False)

    # Check if this is a recording file instead of a model checkpoint
    if 'steps' in checkpoint and 'ledger_snapshots' in checkpoint:
        raise ValueError(f"'{path}' is an episode recording file, not a model checkpoint. "
                        f"Use a checkpoint file (e.g., final_model.pt or checkpoint_*.pt) instead.")

    # Handle different checkpoint formats
    if 'network_state_dicts' in checkpoint:
        # Full checkpoint format (from training)
        state_dict = checkpoint['network_state_dicts'][0]
    elif 'state_dict' in checkpoint:
        # Pretrained format (from save_pretrained)
        state_dict = checkpoint['state_dict']
    else:
        # Raw state dict
        state_dict = checkpoint

    for i, net in enumerate(networks):
        # Try strict loading first, fall back to partial loading on mismatch
        try:
            net.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"  [Warning] Architecture mismatch, loading compatible layers only...")
                # Load only layers with matching shapes
                model_dict = net.state_dict()
                compatible = {}
                skipped = []
                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        compatible[k] = v
                    else:
                        skipped.append(k)
                model_dict.update(compatible)
                net.load_state_dict(model_dict)
                print(f"  Loaded {len(compatible)}/{len(state_dict)} layers, skipped: {skipped}")
            else:
                raise e

    print(f"Loaded pretrained weights from {path} into {len(networks)} networks")

    # Restore curriculum phase if available
    if env is not None and 'curriculum_phase' in checkpoint:
        phase = checkpoint['curriculum_phase']
        env.curriculum_phase = phase
        print(f"Restored curriculum phase: {phase}")


def stack_observations(obs: Dict[int, Dict[str, torch.Tensor]], n_agents: int) -> Dict[str, torch.Tensor]:
    """Stack per-agent observations into batched tensors (for dict-based API)."""
    return {
        'entity_tokens': torch.stack([obs[i]['entity_tokens'] for i in range(n_agents)]),
        'entity_mask': torch.stack([obs[i]['entity_mask'] for i in range(n_agents)]),
        'signals': torch.stack([obs[i]['signals'] for i in range(n_agents)]),
        'self_hp': torch.stack([obs[i]['self_hp'] for i in range(n_agents)]),
        'agent_id': torch.stack([obs[i]['agent_id'] for i in range(n_agents)])
    }


def convert_obs_dict_to_batched(obs: Dict[int, Dict[str, torch.Tensor]], n_agents: int) -> Dict[str, torch.Tensor]:
    """Convert dict-of-dicts to batched tensor dict (for backward compat)."""
    return stack_observations(obs, n_agents)


def test_direction_learning(model: 'ActorCritic', config: Config, device: torch.device) -> tuple:
    """
    Test if model moves toward food in all 4 directions.

    With factored action space, we check that the model selects the correct
    direction AND the MOVE action type when food is adjacent.

    Returns:
        (num_correct, details): num_correct out of 4, and list of (direction, expected, actual, prob)
    """
    import torch.nn.functional as F

    def fourier_encode(dx, dy):
        """Encode position using Fourier features (matching environment)."""
        bands = [1.0, 2.0, 4.0, 8.0]
        features = []
        for freq in bands:
            features.extend([
                torch.sin(torch.tensor(freq * torch.pi * dx)),
                torch.cos(torch.tensor(freq * torch.pi * dx)),
                torch.sin(torch.tensor(freq * torch.pi * dy)),
                torch.cos(torch.tensor(freq * torch.pi * dy)),
            ])
        return torch.stack(features)

    def create_obs(food_dx, food_dy):
        """Create observation with food at relative position using Fourier encoding."""
        tokens = torch.zeros(1, config.max_entities, config.entity_token_dim, device=device)
        mask = torch.zeros(1, config.max_entities, device=device, dtype=torch.bool)

        # Self token (agent 0 at position 0,0):
        # fourier[16]: sin(0)=0, cos(0)=1 for all bands
        # velocity[2]: [0, 0]
        # type[2]: [0, 1] (is_agent)
        # hp[1]: 1.0
        # social[4]: [0, 0, 0, 0]
        self_fourier = fourier_encode(0.0, 0.0).to(device)  # [16]
        self_token = torch.cat([
            self_fourier,                                      # 16: fourier
            torch.zeros(2, device=device),                     # 2: velocity
            torch.tensor([0.0, 1.0], device=device),           # 2: type [is_food, is_agent]
            torch.tensor([1.0], device=device),                # 1: hp
            torch.zeros(4, device=device),                     # 4: social
        ])
        tokens[0, 0, :] = self_token
        mask[0, 0] = True

        # Food token:
        # fourier[16]: encoded from (food_dx, food_dy)
        # velocity[2]: [0, 0] (food doesn't move)
        # type[2]: [1, 0] (is_food)
        # quality[1]: 0.5 (poor food)
        # social[4]: [0, 0, 0, 0]
        food_fourier = fourier_encode(food_dx, food_dy).to(device)  # [16]
        food_token = torch.cat([
            food_fourier,                                      # 16: fourier
            torch.zeros(2, device=device),                     # 2: velocity
            torch.tensor([1.0, 0.0], device=device),           # 2: type [is_food, is_agent]
            torch.tensor([0.5], device=device),                # 1: quality
            torch.zeros(4, device=device),                     # 4: social
        ])
        tokens[0, config.n_agents, :] = food_token
        mask[0, config.n_agents] = True

        return {
            'entity_tokens': tokens,
            'entity_mask': mask,
            'signals': torch.zeros(1, config.n_agents, device=device),
            'self_hp': torch.ones(1, 1, device=device),
            'self_inventory': torch.zeros(1, 1, device=device),
            'agent_id': torch.zeros(1, dtype=torch.long, device=device)
        }

    # Test cases: (row_diff, col_diff, direction_name, expected_direction)
    # Values are in grid cells (will be normalized by grid_size then scaled by 5x)
    # For distance 1: 1/15 * 5 = 0.333 in token
    # row_diff > 0 = food DOWN, col_diff > 0 = food RIGHT
    # Direction: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
    gs = config.grid_size
    dist1 = 1.0 / gs  # Distance 1 in normalized coords (before 5x scaling in create_obs)
    tests = [
        (0.0, dist1, 'RIGHT', 3),   # Food right -> move RIGHT (idx 3)
        (0.0, -dist1, 'LEFT', 2),   # Food left -> move LEFT (idx 2)
        (dist1, 0.0, 'DOWN', 1),    # Food down -> move DOWN (idx 1)
        (-dist1, 0.0, 'UP', 0),     # Food up -> move UP (idx 0)
    ]

    dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
    correct = 0
    details = []

    model.eval()
    with torch.no_grad():
        for dx, dy, name, expected_dir in tests:
            obs = create_obs(dx, dy)
            # Factored action heads return direction and action type logits
            direction_logits, action_type_logits, _ = model.forward(obs)
            # Get direction probabilities
            dir_probs = F.softmax(direction_logits, dim=-1).squeeze()

            best_dir = dir_probs.argmax().item()
            expected_prob = dir_probs[expected_dir].item()
            is_correct = best_dir == expected_dir

            if is_correct:
                correct += 1

            details.append((name, dir_names[expected_dir], dir_names[best_dir], expected_prob))

    model.train()
    return correct, details


def train(config: Config = None, visualize: bool = False,
          render_every: int = 50, plot_every: int = 10, record_every: int = 100,
          output_dir: str = "viz_output", pretrain: bool = False,
          load_pretrained_path: Optional[str] = None,
          num_teams: Optional[int] = None):
    """
    Main training function.

    Args:
        config: Training configuration
        visualize: Enable live visualization
        render_every: Render grid every N updates
        plot_every: Update training curves every N updates
        record_every: Record episode every N updates
        output_dir: Directory for visualization output and checkpoints
        pretrain: If True, save pretrained weights after training
        load_pretrained_path: Path to pretrained weights to load into all agents
    """
    if config is None:
        config = Config()

    # Setup
    device = get_device()
    set_seed(config.seed)
    print(f"Training on {device}")
    print(f"Config: {config.n_agents} agents, {config.grid_size}x{config.grid_size} grid")

    # Initialize components
    env = GridWorld(config, device)

    # Vmap PPO: each agent has its own network, forward passes parallelized via vmap
    multi_agent = VmapPPO(config, device)

    # Load pretrained weights if specified (also restores curriculum phase)
    if load_pretrained_path is not None:
        load_pretrained(multi_agent.networks, load_pretrained_path, env=env)

    # Sync curriculum phase to PPO (for clone mode in phases 6-8)
    if config.pretrain_mode and config.curriculum_enabled:
        multi_agent.set_curriculum_phase(env.get_curriculum_phase())

    # Apply torch.compile() for faster forward passes (PyTorch 2.0+)
    # NOTE: Disabled for debugging - may cause value function caching issues
    # if hasattr(torch, 'compile'):
    #     try:
    #         n_to_compile = multi_agent.n_active
    #         for i in range(n_to_compile):
    #             multi_agent.networks[i] = torch.compile(multi_agent.networks[i], mode='reduce-overhead')
    #         print(f"torch.compile() enabled for {n_to_compile} network(s) (reduce-overhead mode)")
    #     except Exception as e:
    #         print(f"torch.compile() unavailable: {e}")
    print("torch.compile() DISABLED for debugging")

    # Per-agent buffers
    buffers = [SingleAgentBuffer(config, device, i) for i in range(config.n_agents)]
    print(f"Using independent policies: {multi_agent.n_active} active of {config.n_agents} networks")

    # Visualization components (lazy import to avoid matplotlib issues when not visualizing)
    renderer = None
    logger = None
    recorder = None
    fig_grid = None
    fig_curves = None

    if visualize:
        import os
        from visualize import GridRenderer, TrainingLogger, EpisodeRecorder

        # Create output directory
        viz_dir = output_dir
        os.makedirs(viz_dir, exist_ok=True)

        renderer = GridRenderer(config)
        logger = TrainingLogger()
        recorder = EpisodeRecorder(config)

        # Use non-interactive backend to avoid Windows crashes
        plt.switch_backend('Agg')
        print(f"Visualization enabled: saving to {viz_dir}/")
        print(f"  Grid every {render_every}, curves every {plot_every}, record every {record_every}")

    # Training tracking
    global_step = 0
    num_updates = config.total_timesteps // config.batch_size
    start_time = time.time()

    # Episode tracking
    episode_returns = []
    episode_lengths = []
    current_episode_rewards = {i: 0.0 for i in range(config.n_agents)}
    last_per_agent_returns = None  # Store per-agent returns for logging
    last_partner_info = None  # Store partner ledger info for phase 9+ logging

    print(f"Starting training for {config.total_timesteps} timesteps ({num_updates} updates)")
    print("-" * 60)

    for update in range(num_updates):
        # Get current phase and scenario
        phase = env.get_curriculum_phase()
        scenario = CURRICULUM.get(phase)

        # Override scenario with multi-team training if specified
        if num_teams is not None and num_teams > 1:
            from scenarios import MultiTeamScenario
            scenario = MultiTeamScenario(n_agents=config.n_agents, num_teams=num_teams)

        # Apply scenario configuration to PPO
        # Apply when: (pretrain mode with curriculum) OR (multi-team mode)
        use_scenario = (config.pretrain_mode and config.curriculum_enabled and scenario is not None) or \
                       (num_teams is not None and num_teams > 1)
        if use_scenario and scenario is not None:
            scenario_config = scenario.get_config()
            multi_agent.set_n_active(scenario_config.n_active_agents)
            multi_agent.set_clone_mode(scenario_config.clone_mode)
        else:
            scenario_config = None

        # === ROLLOUT PHASE ===
        obs = env.reset()  # Now returns batched Dict[str, Tensor]

        # Apply scenario after reset (configures agents, food)
        if use_scenario and scenario is not None:
            env.apply_scenario(scenario)
            # Re-get observations after scenario applied
            obs = env._get_all_observations()

            # Capture partner info for logging (what agent 0 sees about agent 1)
            if phase >= 9:
                from ledger import Ledger
                ledger = env.ledger.tensor[0, 1]  # Agent 0's view of agent 1
                last_partner_info = {
                    'rel': env.partner_relationship or 'unknown',
                    'dmg': ledger[Ledger.DAMAGE_DEALT].item(),
                    'food': ledger[Ledger.FOOD_GIVEN].item(),
                    'coop': ledger[Ledger.COOP_COUNT].item(),
                    'def': ledger[Ledger.DEFENSE_SCORE].item(),
                }

                # Log normalized+scaled signals for debugging friend/foe discrimination
                if update % 10 == 0:
                    rel = env.partner_relationship or 'unknown'
                    # Raw ledger signals (normalized × 5)
                    social = env.ledger.get_normalized_tensor() * 5.0  # [n, n, 4]
                    sig = social[0, 1]  # Agent 0's view of agent 1 (matches entity token)
                    print(f"  [Signal] {rel.upper()}: dmg={sig[0]:.2f} food={sig[1]:.2f} coop={sig[2]:.2f} def={sig[3]:.2f}")
                    # Entity tokens (what transformer actually sees) - agent 0's view of agent 1
                    tok = obs['entity_tokens'][0, 1, 21:25]  # social channels from agent 1 token (indices 21-24)
                    print(f"  [Token]  {rel.upper()}: dmg={tok[0]:.2f} food={tok[1]:.2f} coop={tok[2]:.2f} def={tok[3]:.2f}")
                    # Encoder output (what network produces) - check if ally/enemy produce different features
                    with torch.no_grad():
                        obs_0 = {
                            'entity_tokens': obs['entity_tokens'][0:1],
                            'entity_mask': obs['entity_mask'][0:1],
                            'signals': obs['signals'][0:1],
                            'self_hp': obs['self_hp'][0:1],
                            'self_inventory': obs['self_inventory'][0:1],
                            'agent_id': torch.tensor([0], device=device)
                        }
                        net = multi_agent.networks[0]
                        feat = net._encode(obs_0)  # [1, 128]
                        value = net.value_head(feat).item()
                        # Show first 8 features + value estimate
                        f8 = feat[0, :8].tolist()
                        print(f"  [Encoder] {rel.upper()}: feat[:8]=[{', '.join(f'{v:.2f}' for v in f8)}] V={value:.2f}")

        for buf in buffers:
            buf.reset()

        # Reset episode tracking (tensor-based)
        current_episode_rewards = torch.zeros(config.n_agents, device=device)
        current_episode_steps = 0
        total_coop_successes = 0  # Track successful cooperations this rollout
        episodes_this_rollout = 0  # Track if any episodes completed

        # Recording: capture actual training episode (not separate eval)
        recording_this_update = visualize and (update % record_every == 0)
        if recording_this_update:
            recorder.reset()

        for step in range(config.num_steps):
            global_step += config.n_agents

            # obs is already batched from vectorized env
            # Get action masks and actions from each agent's policy
            with torch.no_grad():
                direction_mask, action_type_mask = env.get_action_masks()  # Returns two [n_agents, 5] tensors

                # In coop phases (6-8), mask out ATTACK action type for BOTH agents (learn to cooperate, not fight)
                # ACT_ATTACK = 1 in action type space
                if scenario_config is not None and scenario_config.partner_mode == "always_coop":
                    action_type_mask[0, 1] = False  # Disable ATTACK for agent 0
                    action_type_mask[1, 1] = False  # Disable ATTACK for agent 1 (partner)

                # In scripted social phases, mask based on relationship
                if scenario_config is not None and scenario_config.partner_mode == "scripted":
                    if env.partner_relationship == 'ally':
                        action_type_mask[0, 1] = False  # Disable ATTACK for agent 0
                        action_type_mask[1, 1] = False  # Disable ATTACK for agent 1 (partner)

                directions, action_types, log_probs, _, values = \
                    multi_agent.get_actions_and_values(obs, direction_mask=direction_mask, action_type_mask=action_type_mask)

                # Get auxiliary values for decomposed reward streams
                aux_values = multi_agent.get_auxiliary_values(obs)

            # Apply scripted partner behavior (phases 6-8: always coop)
            if scenario_config is not None and scenario_config.partner_mode == "always_coop":
                directions, action_types = apply_scripted_partner(env, directions, action_types, relationship='ally')

            # Apply scripted partner behavior (phase 9+: ally or enemy based on relationship)
            if scenario_config is not None and scenario_config.partner_mode == "scripted":
                directions, action_types = apply_scripted_partner(env, directions, action_types, relationship=env.partner_relationship)

            # Record training step (before env.step so we capture pre-step state)
            if recording_this_update and episodes_this_rollout == 0:
                # Factored actions - store direction and action_type per agent
                actions_dict = {i: (directions[i].item(), action_types[i].item()) for i in range(config.n_agents)}
                values_np = values.cpu().numpy()

                # Convert auxiliary values to numpy for recording
                aux_values_np = {
                    'survival': aux_values['survival'].cpu().numpy(),
                    'resource': aux_values['resource'].cpu().numpy(),
                    'social': aux_values['social'].cpu().numpy(),
                }

                # Capture action probabilities for visualization
                direction_probs, action_type_probs = multi_agent.get_action_probs(obs, direction_mask, action_type_mask)
                action_probs_np = (direction_probs.cpu().numpy(), action_type_probs.cpu().numpy())

                recorder.record(env, actions=actions_dict, values=values_np,
                               action_probs=action_probs_np, aux_values=aux_values_np)

            # Environment step (tensor-based API - factored actions)
            next_obs, rewards, dones, infos = env.step(directions, action_types)

            # Update recorded step with rewards
            if recording_this_update and episodes_this_rollout == 0 and recorder.steps:
                recorder.steps[-1].rewards = {i: rewards[i].item() for i in range(config.n_agents)}

            # Track successful cooperations
            if hasattr(env, 'last_coop_success_count'):
                total_coop_successes += env.last_coop_success_count

            # Track episode rewards (tensor)
            current_episode_rewards += rewards

            # Store per-agent to individual buffers (only active agents)
            for i in range(multi_agent.n_active):
                obs_i = {k: v[i] for k, v in obs.items()}
                dir_mask_i = direction_mask[i]
                act_mask_i = action_type_mask[i]
                buffers[i].store(
                    obs_i, directions[i], action_types[i],
                    log_probs[i], rewards[i], dones[i], values[i],
                    direction_mask=dir_mask_i, action_type_mask=act_mask_i,
                    # Decomposed rewards and auxiliary values
                    reward_survival=infos['reward_survival'][i],
                    reward_resource=infos['reward_resource'][i],
                    reward_social=infos['reward_social'][i],
                    value_survival=aux_values['survival'][i],
                    value_resource=aux_values['resource'][i],
                    value_social=aux_values['social'][i],
                )

            obs = next_obs
            current_episode_steps += 1

            # Check for episode end
            if dones.all():
                # Track agent 0's return (matches curriculum threshold tracking)
                agent0_return = current_episode_rewards[0].item()
                episode_returns.append(agent0_return)
                episode_lengths.append(current_episode_steps)
                episodes_this_rollout += 1

                # Save recording of first episode (before reset clears state)
                if recording_this_update and episodes_this_rollout == 1:
                    rec_path = f"{viz_dir}/episode_update_{update}.pt"
                    recorder.save(rec_path)
                    rel_str = f" [{env.partner_relationship}]" if hasattr(env, 'partner_relationship') and env.partner_relationship else ""
                    print(f"  Recorded training episode ({len(recorder.steps)} steps, R={agent0_return:.1f}){rel_str} -> {rec_path}")

                # Capture partner info from COMPLETED episode (before reset)
                if phase >= 9:
                    from ledger import Ledger
                    ledger = env.ledger.tensor[1, 0]
                    last_partner_info = {
                        'rel': env.partner_relationship or 'unknown',
                        'dmg': ledger[Ledger.DAMAGE_DEALT].item(),
                        'food': ledger[Ledger.FOOD_GIVEN].item(),
                        'coop': ledger[Ledger.COOP_COUNT].item(),
                        'def': ledger[Ledger.DEFENSE_SCORE].item(),
                    }

                # Store per-agent returns for logging
                last_per_agent_returns = current_episode_rewards.tolist()

                # Reset for next episode within rollout
                obs = env.reset()
                # Re-apply scenario after reset
                if use_scenario and scenario is not None:
                    env.apply_scenario(scenario)
                    obs = env._get_all_observations()
                current_episode_rewards.zero_()
                current_episode_steps = 0

        # === ADVANTAGE COMPUTATION ===
        with torch.no_grad():
            # Get value estimates from each agent's network
            next_values = multi_agent.get_values(obs)
            next_done = dones.float()

            # Get auxiliary next values for decomposed reward streams
            next_aux_values = multi_agent.get_auxiliary_values(obs)

        # Compute GAE for each active agent's buffer
        for i in range(multi_agent.n_active):
            buffers[i].compute_gae(
                next_values[i], next_done[i],
                next_value_survival=next_aux_values['survival'][i],
                next_value_resource=next_aux_values['resource'][i],
                next_value_social=next_aux_values['social'][i],
            )

        # If no episodes completed this rollout, use agent 0's rollout reward as proxy
        if episodes_this_rollout == 0:
            agent0_rollout_reward = buffers[0].rewards.sum().item()
            episode_returns.append(agent0_rollout_reward)
            episode_lengths.append(config.num_steps)

            # Save recording even if episode didn't complete
            if recording_this_update:
                rec_path = f"{viz_dir}/episode_update_{update}.pt"
                recorder.save(rec_path)
                rel_str = f" [{env.partner_relationship}]" if hasattr(env, 'partner_relationship') and env.partner_relationship else ""
                print(f"  Recorded training rollout ({len(recorder.steps)} steps, R={agent0_rollout_reward:.1f}){rel_str} -> {rec_path}")

        # === PPO UPDATE PHASE ===
        metrics = multi_agent.update(buffers)

        # === DIAGNOSTIC: Check actions taken ===
        if update % 50 == 0:
            # Factored action space: directions (0-4) and action_types (0-4)
            directions_taken = buffers[0].directions
            action_types_taken = buffers[0].action_types
            dir_counts = torch.bincount(directions_taken.long(), minlength=5)
            act_counts = torch.bincount(action_types_taken.long(), minlength=5)

            # Direction counts
            dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
            dir_str = ', '.join([f"{dir_names[i]}={dir_counts[i].item()}" for i in range(5)])
            print(f"  [Diag] Directions (128 steps): {dir_str}")

            # Action type counts
            act_names = ['MOVE', 'ATTACK', 'GIVE', 'SIGNAL', 'COOP']
            act_str = ', '.join([f"{act_names[i]}={act_counts[i].item()}" for i in range(5)])
            print(f"  [Diag] Action types: {act_str}")

            # Coop phase diagnostic: show successful cooperations
            if env.get_curriculum_phase() >= 6:
                print(f"  [Coop] Successful rich food eaten: {total_coop_successes}")

        # === DIAGNOSTIC: Check value predictions vs returns ===
        if update % 50 == 0:
            # Only check agent 0 in pretrain mode
            buf = buffers[0]
            mean_value = buf.values.mean().item()
            mean_return = buf.returns.mean().item()
            std_value = buf.values.std().item()
            std_return = buf.returns.std().item()
            print(f"  [Diag] Values: {mean_value:.2f} ± {std_value:.2f} | "
                  f"Returns: {mean_return:.2f} ± {std_return:.2f}")

            # Check hidden feature variance (first batch only)
            with torch.no_grad():
                test_obs = {k: v[:16] for k, v in obs.items()}  # First 16 samples
                net = multi_agent.networks[0]
                # Get hidden features before value head
                if hasattr(net, '_orig_mod'):
                    # Handle compiled module
                    encoder_out = net._orig_mod.encoder(
                        test_obs['entity_tokens'][:1], test_obs['entity_mask'][:1],
                        test_obs['signals'][:1], test_obs['self_hp'][:1],
                        test_obs['self_inventory'][:1], test_obs['agent_id'][:1])
                    hidden = net._orig_mod.shared(encoder_out)
                else:
                    encoder_out = net.encoder(
                        test_obs['entity_tokens'][:1], test_obs['entity_mask'][:1],
                        test_obs['signals'][:1], test_obs['self_hp'][:1],
                        test_obs['self_inventory'][:1], test_obs['agent_id'][:1])
                    hidden = net.shared(encoder_out)
                print(f"  [Diag] Hidden: mean={hidden.mean().item():.3f}, std={hidden.std().item():.3f}, "
                      f"min={hidden.min().item():.3f}, max={hidden.max().item():.3f}")

        # === CURRICULUM DIRECTION TEST (diagnostic only, every 10 updates) ===
        if config.pretrain_mode and config.curriculum_enabled and update % 10 == 0:
            net = multi_agent.networks[0]
            num_correct, details = test_direction_learning(net, config, device)
            phase = env.get_curriculum_phase()

            # Print direction test results (diagnostic, not used for advancement)
            status_chars = ['Y' if d[1] == d[2] else 'N' for d in details]
            print(f"  [Direction Test] Phase {phase} | Score: {num_correct}/4 | "
                  f"R:{status_chars[0]} L:{status_chars[1]} D:{status_chars[2]} U:{status_chars[3]}")

        # === LOGGING ===
        elapsed = time.time() - start_time
        sps = global_step / elapsed if elapsed > 0 else 0
        ep_return = episode_returns[-1] if episode_returns else 0.0
        ep_steps = episode_lengths[-1] if episode_lengths else 0

        if config.pretrain_mode and config.curriculum_enabled:
            phase = env.get_curriculum_phase()
            thresh = THRESHOLDS.get(phase, 0)
            # Determine phase type based on scenario
            if phase >= 9:
                phase_type = "Social"
            elif phase >= 6:
                phase_type = "Coop"
            else:
                phase_type = "Solo"
            phase_str = f" | Ph {phase} {phase_type} ({ep_return:.0f}/{thresh})"

            # Add partner info for phase 9+
            if phase >= 9 and last_partner_info is not None:
                p = last_partner_info
                rel = p.get('rel', 'unknown').upper()[:3]  # 'ALL' or 'ENE'
                partner_str = f" | Partner[{rel}]: food={p['food']:.0f} coop={p['coop']:.0f} def={p['def']:.0f} dmg={p['dmg']:.0f}"
            else:
                partner_str = ""
        else:
            phase_str = ""
            partner_str = ""
        print(f"U{update:4d} | R {ep_return:5.1f} | Steps {ep_steps:5.0f} | SPS {sps:4.0f}{phase_str}{partner_str}")

        # === VISUALIZATION ===
        if visualize:
            # Log metrics (including per-agent returns if available)
            if episode_returns:
                logger.log(global_step, episode_return=episode_returns[-1],
                          agent_returns=last_per_agent_returns, metrics=metrics)
            else:
                logger.log(global_step, metrics=metrics)

            # Render grid snapshot (save to file)
            if update % render_every == 0:
                fig_grid = renderer.render(env, show=False)
                fig_grid.suptitle(f'Update {update} | Step {global_step}')
                grid_path = f"{viz_dir}/grid_update_{update:05d}.png"
                fig_grid.savefig(grid_path, dpi=100, bbox_inches='tight')
                plt.close(fig_grid)
                print(f"  Saved grid -> {grid_path}")

            # Update training curves (save to file, overwrite)
            if update % plot_every == 0 and update > 0:
                fig_curves = logger.plot(window=50, show=False)
                curves_path = f"{viz_dir}/training_curves.png"
                fig_curves.savefig(curves_path, dpi=100, bbox_inches='tight')
                plt.close(fig_curves)


        # === CHECKPOINTING ===
        if update % 10 == 0 and update > 0:
            import os
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = f"{output_dir}/checkpoint_{update}.pt"
            torch.save({
                'update': update,
                'global_step': global_step,
                'network_state_dicts': [net.state_dict() for net in multi_agent.networks],
                'optimizer_state_dicts': [opt.state_dict() for opt in multi_agent.optimizers],
                'curriculum_phase': env.get_curriculum_phase() if hasattr(env, 'get_curriculum_phase') else 6,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final save
    import os
    os.makedirs(output_dir, exist_ok=True)
    final_path = f"{output_dir}/final_model.pt"
    torch.save({
        'network_state_dicts': [net.state_dict() for net in multi_agent.networks],
        'curriculum_phase': env.get_curriculum_phase() if hasattr(env, 'get_curriculum_phase') else config.curriculum_phase,
        'config': config
    }, final_path)
    print(f"Training complete! Saved {final_path}")

    # Save pretrained weights if in pretrain mode
    if pretrain:
        pretrained_path = f"{output_dir}/pretrained.pt"
        curr_phase = env.get_curriculum_phase() if hasattr(env, 'get_curriculum_phase') else config.curriculum_phase
        save_pretrained(multi_agent.networks[0], pretrained_path, curriculum_phase=curr_phase)

    # Save training logs if visualizing
    if visualize and logger is not None:
        logs_path = f"{viz_dir}/training_logs.pt"
        logger.save(logs_path)
        print(f"Saved training logs to {logs_path}")

        # Save final training curves
        fig_final = logger.plot(window=100, show=False)
        final_path = f"{viz_dir}/training_curves_final.png"
        fig_final.savefig(final_path, dpi=150, bbox_inches='tight')
        plt.close(fig_final)
        print(f"Saved final curves to {final_path}")

    return multi_agent, episode_returns


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Multi-Agent RL system')

    # Visualization
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Enable live visualization during training')
    parser.add_argument('--render-every', type=int, default=50,
                        help='Render grid every N updates (default: 50)')
    parser.add_argument('--plot-every', type=int, default=10,
                        help='Update training curves every N updates (default: 10)')
    parser.add_argument('--record-every', type=int, default=100,
                        help='Record episode every N updates (default: 100)')

    # Training
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Discount factor (overrides config)')
    parser.add_argument('--ent-coef', type=float, default=None,
                        help='Entropy coefficient (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')

    # Output
    parser.add_argument('--output-dir', '-o', type=str, default='viz_output',
                        help='Output directory for visualizations and checkpoints (default: viz_output)')

    # Pretraining (curriculum learning)
    parser.add_argument('--pretrain', action='store_true',
                        help='Pretrain single agent on food-finding')
    parser.add_argument('--load-pretrained', type=str, default=None,
                        help='Load pretrained weights into all agents')
    parser.add_argument('--phase', type=int, default=None,
                        help='Starting curriculum phase (1-8, default: 1)')
    parser.add_argument('--enemy-prob', type=float, default=None,
                        help='Probability of enemy in phase 9 (0.0-1.0, default: 0.2)')

    # Environment
    parser.add_argument('--n-agents', type=int, default=None,
                        help='Number of agents (overrides config)')
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Grid size (overrides config)')
    parser.add_argument('--episode-length', type=int, default=None,
                        help='Max steps per episode (overrides config)')
    parser.add_argument('--poor-food-spawn-rate', type=float, default=None,
                        help='Poor food spawn probability per empty cell per tick (overrides config)')
    parser.add_argument('--rich-food-spawn-rate', type=float, default=None,
                        help='Rich food spawn probability per empty cell per tick (overrides config)')
    parser.add_argument('--food-coverage-cap', type=float, default=None,
                        help='Max fraction of grid each food type can cover (overrides config)')

    # Multi-team training
    parser.add_argument('--num-teams', type=int, default=None,
                        help='Number of teams to split agents into (e.g., 2, 3, 4)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create config with any overrides
    config = Config()
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps
    if args.seed is not None:
        config.seed = args.seed
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.ent_coef is not None:
        config.ent_coef = args.ent_coef
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.n_agents is not None:
        config.n_agents = args.n_agents
    if args.grid_size is not None:
        config.grid_size = args.grid_size
    if args.episode_length is not None:
        config.max_steps_per_episode = args.episode_length
        config.num_steps = args.episode_length  # Sync rollout length with episode length
    if args.poor_food_spawn_rate is not None:
        config.poor_food_spawn_rate = args.poor_food_spawn_rate
    if args.rich_food_spawn_rate is not None:
        config.rich_food_spawn_rate = args.rich_food_spawn_rate
    if args.food_coverage_cap is not None:
        config.food_coverage_cap = args.food_coverage_cap
    if args.phase is not None:
        config.curriculum_phase = args.phase

    # Recompute derived values after config modifications
    config.max_entities = config.n_agents + config.max_food_tokens
    config.minibatch_size = config.batch_size // config.num_minibatches

    # Pretraining mode: single agent on food-finding
    if args.pretrain:
        config.pretrain_mode = True
        config.pretrain_spawn_agents = 1
        print("=" * 60)
        print("PRETRAINING MODE: Training single agent on food-finding")
        print("=" * 60)

    # Override enemy probability for phase 9
    if args.enemy_prob is not None:
        from scenarios import CURRICULUM
        if 9 in CURRICULUM:
            CURRICULUM[9].enemy_prob = args.enemy_prob
            print(f"Phase 9 enemy probability: {args.enemy_prob:.0%}")

    # Multi-team training mode
    if args.num_teams is not None and args.num_teams > 1:
        print("=" * 60)
        print(f"MULTI-TEAM MODE: Splitting {config.n_agents} agents into {args.num_teams} teams")
        if args.num_teams >= config.n_agents:
            print(f"  WARNING: num_teams ({args.num_teams}) >= n_agents ({config.n_agents})")
            print(f"           All agents will be enemies (different teams)")
        print("=" * 60)

    train(
        config=config,
        visualize=args.visualize,
        render_every=args.render_every,
        plot_every=args.plot_every,
        record_every=args.record_every,
        output_dir=args.output_dir,
        pretrain=args.pretrain,
        load_pretrained_path=args.load_pretrained,
        num_teams=args.num_teams
    )
