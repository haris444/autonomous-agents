# Architecture: Neural Network in Multi-Agent RL

This document explains the role of neural networks in our multi-agent reinforcement learning system and justifies the chosen implementation approach.

## 1. Neural Network Role

The neural network acts as a **function approximator** that learns to map observations to actions.

### Think of it as:

- **Input (Eyes):** Numbers representing the world
  - Food location relative to agent
  - Social history from the Global Ledger
  - Current health/energy level
  - Nearby agents and their states

- **Job:** Pattern recognition
  - "Low health + nearby food = eating is good"
  - "This agent shared with me before = cooperation might pay off"
  - "No food nearby + healthy = explore"

- **Output (Decision):** Probability distribution over actions
  - Move up/down/left/right
  - Eat food
  - Share resources
  - Do nothing

### Key Insight

> **You program the inputs (Observation) and incentives (Rewards). The NN figures out the strategy via trial and error.**

The network doesn't know what "cooperation" means. It only knows:
1. What numbers it sees (observation space)
2. What numbers it got after acting (rewards)

Through thousands of episodes, it discovers patterns like "sharing leads to higher long-term rewards" — that's emergent cooperation.

---

## 2. Implementation Options

| Option | Approach | Pros | Cons | Verdict |
|--------|----------|------|------|---------|
| **A: Ray RLLib** | Industry-standard Multi-Agent RL library | Native Dictionary obs support, handles PPO math, scales to 100+ agents | Steep learning curve, complex configs | Alternative |
| **B: CleanRL/PyTorch** | Write training loop from scratch | Full understanding, total control over Ledger processing | Lots of boilerplate, easy to make silent math errors | **Chosen** |
| **C: Stable-Baselines3** | Simple RL library + multi-agent hack | Easy to start (3 lines) | Doesn't support multi-agent natively, messy with Global Ledger | Not recommended |

---

## 3. Chosen Approach: Option B (PyTorch from Scratch)

We will write the training loop in **raw PyTorch**, following the CleanRL style.

### Why This Fits Our Project

1. **Full Understanding** — Every gradient update, every advantage calculation is visible
2. **Total Control Over Ledger** — We decide exactly how social history feeds into the network
3. **No Framework Abstractions** — Nothing hidden; if it breaks, we know where to look
4. **Learning Value** — Deep understanding of PPO internals is the goal

### Trade-offs Accepted

- More boilerplate code (rollout buffer, GAE calculation, policy update loop)
- Must be careful with math correctness (clipping, normalization, entropy bonus)

### High-Level Training Loop

```
for episode in range(num_episodes):
    observations = env.reset()

    for step in range(max_steps):
        # Each agent picks action from its policy
        actions = {agent: policy(obs) for agent, obs in observations.items()}

        # Environment steps, Ledger updates
        next_obs, rewards, dones, infos = env.step(actions)

        # Store transition in buffer
        buffer.store(observations, actions, rewards, dones)

        observations = next_obs

    # After episode: compute advantages, update policy
    policy.update(buffer)
```

---

## 4. Reference

For detailed specifications of:
- **Observation Space** (what the NN sees)
- **Action Space** (what the NN can do)
- **Reward Structure** (how we incentivize behavior)

See: [PROJECT_SPEC.md](./PROJECT_SPEC.md)
