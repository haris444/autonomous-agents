"""Shared utilities for analysis scripts — checkpoint loading and device setup."""
import torch
from core.config import Config
from agents.ppo import PPO
from agents.sac import SAC
from agents.network import SharedTrunkActorCritic


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_config(ckpt: dict) -> Config:
    """Extract Config from a checkpoint dict."""
    raw = ckpt['config']
    if isinstance(raw, dict):
        return Config.from_dict(raw)
    return raw


def load_ppo(path: str, device: torch.device = None) -> tuple:
    """Load a PPO checkpoint. Returns (ppo, config, ckpt_dict)."""
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = load_config(ckpt)
    ppo = PPO(config, device)
    ppo.network.load_state_dict(ckpt['network_state_dict'])
    ppo.network.eval()
    return ppo, config, ckpt


def load_sac(path: str, device: torch.device = None) -> tuple:
    """Load a SAC checkpoint. Returns (sac, config, ckpt_dict)."""
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = load_config(ckpt)
    sac = SAC(config, device)
    sac.load_state_dict(ckpt['sac_state'])
    return sac, config, ckpt


def load_network(path: str, device: torch.device = None) -> tuple:
    """Load just the SharedTrunkActorCritic network. Returns (network, config, ckpt_dict)."""
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = load_config(ckpt)
    network = SharedTrunkActorCritic(config).to(device)
    network.load_state_dict(ckpt['network_state_dict'])
    network.eval()
    return network, config, ckpt


def load_auto(path: str, device: torch.device = None) -> tuple:
    """Auto-detect PPO vs SAC checkpoint. Returns (agent, config, ckpt_dict, agent_type).

    agent_type is 'ppo' or 'sac'.
    """
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if 'sac_state' in ckpt:
        config = load_config(ckpt)
        sac = SAC(config, device)
        sac.load_state_dict(ckpt['sac_state'])
        return sac, config, ckpt, 'sac'
    else:
        config = load_config(ckpt)
        ppo = PPO(config, device)
        ppo.network.load_state_dict(ckpt['network_state_dict'])
        ppo.network.eval()
        return ppo, config, ckpt, 'ppo'
