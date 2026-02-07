
import torch
import torch.nn as nn
from config import Config
from buffer import VecBuffer

def verify_gae_fix():
    print("Verifying GAE Fix in VecBuffer...")
    
    # Setup
    config = Config()
    config.n_envs = 1
    config.n_agents = 1
    config.num_steps = 5
    config.gamma = 0.99
    config.gae_lambda = 0.95
    device = torch.device('cpu')
    
    buffer = VecBuffer(config, device, n_envs=1)
    
    # Fake rollout:
    # Step 0: Alive, Reward = 1, Value = 10
    # Step 1: Alive, Reward = 1, Value = 10
    # Step 2: DIES (Done=True), Reward = -50, Value = 10 (But should be ignored!)
    # Step 3: Alive (New Episode), Reward = 1, Value = 20 (High value from new life)
    # Step 4: Alive, Reward = 1, Value = 20
    
    # We want to make sure Step 2 (Death) does NOT use Step 3's Value (20)
    
    # Store dummy data
    for t in range(5):
        obs = {
            'entity_tokens': torch.zeros(1, 1, config.max_entities, config.entity_token_dim),
            'entity_mask': torch.zeros(1, 1, config.max_entities, dtype=torch.bool),
            'signals': torch.zeros(1, 1, 1),
            'self_hp': torch.zeros(1, 1, 1),
            'self_inventory': torch.zeros(1, 1, 1),
            'agent_id': torch.zeros(1, 1, dtype=torch.long)
        }
        
        reward = 1.0
        done = False
        val = 10.0
        
        if t == 2: # Death step
            reward = -50.0
            done = True
            val = 5.0 # Value estimate AT death
            
        if t == 3 or t == 4: # New episode
            reward = 1.0
            done = False
            val = 20.0 # High value in new life
            
        buffer.store(
            obs, 
            torch.zeros(1, 1, dtype=torch.long), 
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1),
            torch.tensor([[reward]]), # Rewards
            torch.tensor([[done]]),   # Dones
            torch.tensor([[val]]),    # Values
            torch.zeros(1, 1, 5, dtype=torch.bool),
            torch.zeros(1, 1, 5, dtype=torch.bool),
            torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1),
            torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1)
        )
        
    # Compute GAE
    # Next value after step 4 is 0
    buffer.compute_gae(torch.zeros(1, 1), torch.zeros(1, 1))
    
    # Check Advantage/Return at Step 2 (The death step)
    # If correct: Return[2] = Reward[2] (-50) + gamma * 0 (because done=True)
    # If bugged: Return[2] = Reward[2] (-50) + gamma * Value[3] (20) * lambda...
    
    # Approximate math for correct case:
    # Adv[2] = Reward[2] + gamma*V[3]*(1-done[2]) - V[2]
    # Adv[2] = -50 + 0.99 * 20 * (1-1) - 5 = -55
    # Return[2] = Adv[2] + V[2] = -50
    
    # Approximate math for BUGGED case (masked with done[3]=0):
    # Adv[2] = -50 + 0.99 * 20 * (1-0) - 5 = -55 + 19.8 = -35.2
    
    actual_return = buffer.returns[2, 0, 0].item()
    print(f"\nStep 2 (Death) Return: {actual_return:.2f}")
    
    if abs(actual_return - (-50.0)) < 1.0:
        print("SUCCESS: Death step not contaminated by next episode value.")
    else:
        print(f"FAILURE: Return is {actual_return}, expected ~ -50.0. The dead agent is seeing the light!")
        
if __name__ == "__main__":
    verify_gae_fix()
