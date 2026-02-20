
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
from core.config import Config
from env.environment import GridWorld
from agents.ppo import VmapPPO
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
    agent = VmapPPO(config, device)
    
    # Load model weights - apply to ALL agents so they behave intelligently
    state_dict = checkpoint['model_state_dict']
    for i in range(config.n_agents):
        agent.networks[i].load_state_dict(state_dict)
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
            
            # Use single-env get_actions_and_values (not vec)
            directions, action_types, log_probs, _, values = agent.get_actions_and_values(
                obs, direction_mask=direction_mask, action_type_mask=action_type_mask
            )
            
            # Calculate probabilities for visualization
            # We need to run a forward pass for each agent to get the logits
            import numpy as np
            dir_probs_list = []
            act_probs_list = []
            
            for i in range(config.n_agents):
                if env.agent_alive[i]:
                    # Extract single agent obs
                    obs_i = {k: v[i:i+1] for k, v in obs.items()}
                    dir_mask_i = direction_mask[i:i+1]
                    act_mask_i = action_type_mask[i:i+1]
                    
                    net = agent.networks[i]
                    dir_logits, act_logits, _ = net(obs_i)
                    
                    # Apply masks
                    LARGE_NEG = -1e8
                    if dir_mask_i is not None:
                        dir_logits = dir_logits.masked_fill(~dir_mask_i, LARGE_NEG)
                    if act_mask_i is not None:
                        act_logits = act_logits.masked_fill(~act_mask_i, LARGE_NEG)
                    
                    d_probs = torch.softmax(dir_logits, dim=-1)
                    a_probs = torch.softmax(act_logits, dim=-1)
                    
                    dir_probs_list.append(d_probs.cpu().numpy()[0])
                    act_probs_list.append(a_probs.cpu().numpy()[0])
                else:
                    dir_probs_list.append(np.zeros(config.n_directions))
                    act_probs_list.append(np.zeros(config.n_action_types))
            
            action_probs = (np.array(dir_probs_list), np.array(act_probs_list))
            
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
            
            # Get aux values if available
            aux_values = agent.networks[0].get_auxiliary_values(obs)
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
