"""
Off-policy Replay Buffer for Discrete SAC.

GPU-resident circular buffer storing individual transitions (obs, action, reward, next_obs, done).
Flattened storage — no env/agent dimensions.
"""
import torch
from config import Config


class ReplayBuffer:
    """Pre-allocated circular replay buffer on GPU."""

    def __init__(self, config: Config, device: torch.device):
        cap = config.sac_buffer_size
        me = config.max_entities
        td = config.entity_token_dim
        n_sig = config.n_agents

        self.capacity = cap
        self.device = device
        self.pos = 0
        self.full = False

        # Current observations
        self.entity_tokens = torch.zeros(cap, me, td, device=device)
        self.entity_mask = torch.zeros(cap, me, device=device, dtype=torch.bool)
        self.signals = torch.zeros(cap, n_sig, device=device)
        self.self_hp = torch.zeros(cap, 1, device=device)
        self.self_inventory = torch.zeros(cap, 1, device=device)
        self.agent_id = torch.zeros(cap, device=device, dtype=torch.long)

        # Next observations
        self.next_entity_tokens = torch.zeros(cap, me, td, device=device)
        self.next_entity_mask = torch.zeros(cap, me, device=device, dtype=torch.bool)
        self.next_signals = torch.zeros(cap, n_sig, device=device)
        self.next_self_hp = torch.zeros(cap, 1, device=device)
        self.next_self_inventory = torch.zeros(cap, 1, device=device)

        # Actions
        self.directions = torch.zeros(cap, device=device, dtype=torch.long)
        self.action_types = torch.zeros(cap, device=device, dtype=torch.long)

        # Scalars
        self.rewards = torch.zeros(cap, device=device)
        self.dones = torch.zeros(cap, device=device)

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add_batch(self, obs, directions, action_types, rewards, next_obs, dones):
        """Add a batch of transitions from vectorized env step.

        All inputs are [n_envs, n_agents, ...] shaped. We flatten to [n_envs*n_agents]
        and insert into the circular buffer.
        """
        ne_na = directions.shape[0] * directions.shape[1]  # n_envs * n_agents

        # Flatten [n_envs, n_agents, ...] → [n_envs*n_agents, ...]
        def flat(t):
            return t.reshape(ne_na, *t.shape[2:])

        et = flat(obs['entity_tokens'])
        em = flat(obs['entity_mask'])
        sg = flat(obs['signals'])
        sh = flat(obs['self_hp'])
        si = flat(obs['self_inventory'])
        ai = flat(obs['agent_id'])

        net = flat(next_obs['entity_tokens'])
        nem = flat(next_obs['entity_mask'])
        nsg = flat(next_obs['signals'])
        nsh = flat(next_obs['self_hp'])
        nsi = flat(next_obs['self_inventory'])

        d = flat(directions)
        at = flat(action_types)
        r = flat(rewards)
        dn = flat(dones.float())

        # Insert into circular buffer
        end = self.pos + ne_na
        if end <= self.capacity:
            idx = slice(self.pos, end)
            self.entity_tokens[idx] = et
            self.entity_mask[idx] = em
            self.signals[idx] = sg
            self.self_hp[idx] = sh
            self.self_inventory[idx] = si
            self.agent_id[idx] = ai
            self.next_entity_tokens[idx] = net
            self.next_entity_mask[idx] = nem
            self.next_signals[idx] = nsg
            self.next_self_hp[idx] = nsh
            self.next_self_inventory[idx] = nsi
            self.directions[idx] = d
            self.action_types[idx] = at
            self.rewards[idx] = r
            self.dones[idx] = dn
            self.pos = end
            if self.pos >= self.capacity:
                self.full = True
                self.pos = 0
        else:
            # Wraps around
            first = self.capacity - self.pos
            second = ne_na - first
            # First chunk
            self.entity_tokens[self.pos:] = et[:first]
            self.entity_mask[self.pos:] = em[:first]
            self.signals[self.pos:] = sg[:first]
            self.self_hp[self.pos:] = sh[:first]
            self.self_inventory[self.pos:] = si[:first]
            self.agent_id[self.pos:] = ai[:first]
            self.next_entity_tokens[self.pos:] = net[:first]
            self.next_entity_mask[self.pos:] = nem[:first]
            self.next_signals[self.pos:] = nsg[:first]
            self.next_self_hp[self.pos:] = nsh[:first]
            self.next_self_inventory[self.pos:] = nsi[:first]
            self.directions[self.pos:] = d[:first]
            self.action_types[self.pos:] = at[:first]
            self.rewards[self.pos:] = r[:first]
            self.dones[self.pos:] = dn[:first]
            # Second chunk (wrap)
            self.entity_tokens[:second] = et[first:]
            self.entity_mask[:second] = em[first:]
            self.signals[:second] = sg[first:]
            self.self_hp[:second] = sh[first:]
            self.self_inventory[:second] = si[first:]
            self.agent_id[:second] = ai[first:]
            self.next_entity_tokens[:second] = net[first:]
            self.next_entity_mask[:second] = nem[first:]
            self.next_signals[:second] = nsg[first:]
            self.next_self_hp[:second] = nsh[first:]
            self.next_self_inventory[:second] = nsi[first:]
            self.directions[:second] = d[first:]
            self.action_types[:second] = at[first:]
            self.rewards[:second] = r[first:]
            self.dones[:second] = dn[first:]
            self.full = True
            self.pos = second

    def sample(self, batch_size):
        """Sample a random batch of transitions.

        Returns dict with 'obs', 'next_obs', 'directions', 'action_types',
        'rewards', 'dones' — all [batch_size, ...] shaped.
        """
        max_idx = len(self)
        idx = torch.randint(0, max_idx, (batch_size,), device=self.device)

        return {
            'obs': {
                'entity_tokens': self.entity_tokens[idx],
                'entity_mask': self.entity_mask[idx],
                'signals': self.signals[idx],
                'self_hp': self.self_hp[idx],
                'self_inventory': self.self_inventory[idx],
                'agent_id': self.agent_id[idx],
            },
            'next_obs': {
                'entity_tokens': self.next_entity_tokens[idx],
                'entity_mask': self.next_entity_mask[idx],
                'signals': self.next_signals[idx],
                'self_hp': self.next_self_hp[idx],
                'self_inventory': self.next_self_inventory[idx],
                'agent_id': self.agent_id[idx],  # agent_id doesn't change
            },
            'directions': self.directions[idx],
            'action_types': self.action_types[idx],
            'rewards': self.rewards[idx],
            'dones': self.dones[idx],
        }
