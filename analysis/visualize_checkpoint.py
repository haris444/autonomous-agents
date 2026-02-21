
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
from core.config import Config
from env.environment import GridWorld
from agents.ppo import PPO
from analysis.visualize import EpisodeRecorder, replay_episode

def visualize_checkpoint(checkpoint_path: str, output_name: str = "checkpoint_visualized"):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # Setup config
    config = Config()
    config.n_envs = 1  # Visualizing single env
    config.grid_size = 8
    config.n_agents = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize env and agent
    env = GridWorld(config, device)
    agent = PPO(config, device)

    # Load model weights
    state_dict = checkpoint['network_state_dict']
    agent.network.load_state_dict(state_dict)
    print(f"Model loaded (Episode {checkpoint['episode']}) for all {config.n_agents} agents")

    # Setup recorder
    recorder = EpisodeRecorder(config)

    # Run episode
    print("Running episode...")
    obs = env.reset()
    recorder.reset()

    accumulated_reward = 0
    with torch.no_grad():
        for step in range(config.max_steps_per_episode):
            # Get actions
            direction_mask, action_type_mask = env.get_action_masks()

            # Use single-env get_actions_and_values
            directions, action_types, log_probs, _, values = agent.get_actions_and_values(
                obs, direction_mask=direction_mask, action_type_mask=action_type_mask
            )

            # Calculate probabilities for visualization
            dir_probs, act_probs = agent.get_action_probs(
                obs, direction_mask=direction_mask, action_type_mask=action_type_mask
            )
            action_probs = (dir_probs.cpu().numpy(), act_probs.cpu().numpy())

            # Step env
            next_obs, rewards, dones, infos = env.step(directions, action_types)

            # Record
            # Need to format actions dict for recorder: {agent_id: (dir, act_type)}
            actions_dict = {
                i: (directions[i].item(), action_types[i].item())
                for i in range(config.n_agents)
            }
            rewards_dict = {
                i: rewards[i].item() for i in range(config.n_agents)
            }

            # Get aux values
            aux_values = agent.get_auxiliary_values(obs)
            # aux_values is dict of tensors [n_agents], convert to dict of numpy
            aux_values_np = {k: v.cpu().numpy() for k, v in aux_values.items()}

            recorder.record(
                env=env,
                actions=actions_dict,
                rewards=rewards_dict,
                values=values.cpu().numpy(),
                action_probs=action_probs,
                aux_values=aux_values_np
            )

            accumulated_reward += rewards.sum().item()
            obs = next_obs

            if step % 20 == 0:
                print(f"  Step {step}/{config.max_steps_per_episode}")

            if dones.all():
                break

    print(f"Episode complete. Total Reward: {accumulated_reward:.2f}")

    # Save recording
    save_path = f"{output_name}.pt"
    recorder.save(save_path)
    print(f"Recording saved to {save_path}")

    # Replay as GIF
    gif_path = f"{output_name}.gif"
    print(f"Generating GIF: {gif_path}")
    replay_episode(
        recording=recorder.get_recording(),
        config=config,
        save_gif=True,
        gif_path=gif_path,
        ledger_snapshots=recorder.ledger_snapshots,
        predator_ledger_snapshots=recorder.predator_ledger_snapshots,
        show_ledger=True
    )
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--output", type=str, default=None, help="Output name (default: derived from checkpoint)")
    args = parser.parse_args()

    if args.output is None:
        # Auto-generate output name from checkpoint filename
        base = os.path.basename(args.checkpoint)
        name = os.path.splitext(base)[0]
        args.output = f"replay_{name.replace('checkpoint_', '')}"

    visualize_checkpoint(args.checkpoint, args.output)
