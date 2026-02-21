
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
from core.config import Config
from env.environment import GridWorld
from analysis.visualize import EpisodeRecorder, replay_episode
from analysis.utils import load_auto


def visualize_checkpoint(checkpoint_path: str, output_name: str = "checkpoint_visualized"):
    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent, config, ckpt, agent_type = load_auto(checkpoint_path, device)
    config.n_envs = 1
    print(f"Loaded {agent_type.upper()} (Episode {ckpt.get('episode', '?')}) for {config.n_agents} agents")

    # Get the network for forward passes
    if agent_type == 'sac':
        network = agent.actor
    else:
        network = agent.network

    env = GridWorld(config, device)
    recorder = EpisodeRecorder(config)

    print("Running episode...")
    obs = env.reset()
    recorder.reset()

    accumulated_reward = 0
    with torch.no_grad():
        for step in range(config.max_steps_per_episode):
            direction_mask, action_type_mask = env.get_action_masks()

            # Forward pass through network for all agents
            agent_indices = torch.arange(config.n_agents, device=device)
            dir_logits, act_logits, values, v_surv, v_res, v_soc = network.forward(
                obs, agent_indices, return_aux=True
            )
            aux_vals = {'survival': v_surv, 'resource': v_res, 'social': v_soc}

            # Mask and sample
            if direction_mask is not None:
                dir_logits = dir_logits.masked_fill(~direction_mask, -1e8)
            if action_type_mask is not None:
                act_logits = act_logits.masked_fill(~action_type_mask, -1e8)

            dir_probs = torch.softmax(dir_logits, dim=-1)
            act_probs = torch.softmax(act_logits, dim=-1)

            directions = torch.multinomial(dir_probs, 1).squeeze(-1)
            action_types = torch.multinomial(act_probs, 1).squeeze(-1)

            action_probs = (dir_probs.cpu().numpy(), act_probs.cpu().numpy())

            next_obs, rewards, dones, infos = env.step(directions, action_types)

            actions_dict = {
                i: (directions[i].item(), action_types[i].item())
                for i in range(config.n_agents)
            }
            rewards_dict = {
                i: rewards[i].item() for i in range(config.n_agents)
            }

            aux_values_np = {k: v.cpu().numpy() for k, v in aux_vals.items()}

            recorder.record(
                env=env,
                actions=actions_dict,
                rewards=rewards_dict,
                values=values.squeeze(-1).cpu().numpy(),
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

    save_path = f"{output_name}.pt"
    recorder.save(save_path)
    print(f"Recording saved to {save_path}")

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
        base = os.path.basename(args.checkpoint)
        name = os.path.splitext(base)[0]
        args.output = f"replay_{name.replace('checkpoint_', '')}"

    visualize_checkpoint(args.checkpoint, args.output)
