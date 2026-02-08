"""
YAML-based experiment configuration system.

Sits on top of the Config dataclass. Precedence: Config defaults < YAML values < CLI overrides.
Supports named reward presets, multi-seed sweeps, and grid searches.
"""
import copy
import itertools
import os
import shutil
from dataclasses import fields
from datetime import datetime

import yaml

from config import Config

# Keys reserved for experiment metadata (not Config fields)
RESERVED_KEYS = frozenset({
    'experiment', 'training', 'reward_presets', 'sweep', 'reward_preset'
})


def load_yaml(path: str) -> dict:
    """Parse a YAML experiment file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def validate_yaml_keys(yaml_dict: dict):
    """Raise ValueError if any non-reserved key isn't a valid Config field."""
    valid_names = {f.name for f in fields(Config)}
    for key in yaml_dict:
        if key in RESERVED_KEYS:
            continue
        if key not in valid_names:
            raise ValueError(
                f"Unknown config key in YAML: '{key}'. "
                f"Valid Config fields: {sorted(valid_names)}"
            )


def resolve_reward_preset(yaml_dict: dict) -> dict:
    """If reward_preset is specified, merge named preset values into the dict.

    Preset values are applied first, then any explicit top-level keys override them.
    Returns a new dict with preset values merged in and preset keys removed.
    """
    preset_name = yaml_dict.get('reward_preset')
    if preset_name is None:
        return yaml_dict

    presets = yaml_dict.get('reward_presets', {})
    if preset_name not in presets:
        raise ValueError(
            f"Reward preset '{preset_name}' not found. "
            f"Available: {list(presets.keys())}"
        )

    result = dict(yaml_dict)
    # Merge preset values (explicit top-level keys take priority)
    preset_values = presets[preset_name]
    for key, val in preset_values.items():
        if key not in result:
            result[key] = val

    # Clean up preset keys
    result.pop('reward_preset', None)
    result.pop('reward_presets', None)
    return result


def build_config(yaml_dict: dict, cli_overrides: dict = None) -> Config:
    """Construct a Config with correct precedence: defaults < YAML < CLI overrides.

    Args:
        yaml_dict: Parsed YAML dict (may contain reserved keys).
        cli_overrides: Dict of CLI overrides to apply on top.

    Returns:
        Fully resolved Config instance.
    """
    validate_yaml_keys(yaml_dict)

    # Resolve reward preset if present
    resolved = resolve_reward_preset(yaml_dict)

    # Extract only Config field values (skip reserved keys)
    valid_names = {f.name for f in fields(Config)} - Config._DERIVED_FIELDS
    config_values = {k: v for k, v in resolved.items() if k in valid_names}

    # Apply CLI overrides on top
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None and k in valid_names:
                config_values[k] = v

    # Convert lists to tuples for tuple-typed fields, coerce types
    field_types = {f.name: f.type for f in fields(Config) if f.name in valid_names}
    for k, v in config_values.items():
        if isinstance(v, list):
            config_values[k] = tuple(v)
        # Coerce string values to the correct type (e.g., YAML parses "3e-4" as str)
        elif isinstance(v, str) and k in field_types:
            ft = field_types[k]
            if ft == 'float' or ft is float:
                config_values[k] = float(v)
            elif ft == 'int' or ft is int:
                config_values[k] = int(v)

    return Config(**config_values) if config_values else Config()


def generate_sweep_configs(yaml_dict: dict) -> list:
    """Produce list of (overrides_dict, label) for grid/seed sweeps.

    Returns:
        List of (dict, str) tuples. Each dict contains the overrides for one run,
        and the str is a human-readable label for the run directory.
    """
    sweep = yaml_dict.get('sweep', {})
    if not sweep:
        # Single run with default seed
        seed = yaml_dict.get('seed', Config.seed)
        return [({'seed': seed}, f'seed_{seed}')]

    seeds = sweep.get('seeds', [yaml_dict.get('seed', Config.seed)])
    grid = sweep.get('grid', {})

    # Resolve reward_preset grid values into actual preset overrides
    presets = yaml_dict.get('reward_presets', {})

    # Build grid combinations
    if grid:
        grid_keys = sorted(grid.keys())
        grid_values = [grid[k] for k in grid_keys]
        grid_combos = list(itertools.product(*grid_values))
    else:
        grid_keys = []
        grid_combos = [()]  # single empty combo

    runs = []
    for combo in grid_combos:
        combo_overrides = dict(zip(grid_keys, combo))

        # Build label from grid params
        if combo_overrides:
            label_parts = []
            for k, v in sorted(combo_overrides.items()):
                label_parts.append(f'{k}={v}')
            combo_label = '_'.join(label_parts)
        else:
            combo_label = None

        # If reward_preset is in the grid, resolve it
        if 'reward_preset' in combo_overrides:
            preset_name = combo_overrides.pop('reward_preset')
            if preset_name in presets:
                for pk, pv in presets[preset_name].items():
                    combo_overrides[pk] = pv
            # Use preset name in label
            combo_label = combo_label.replace(f'reward_preset={preset_name}',
                                              f'preset={preset_name}')

        for seed in seeds:
            run_overrides = dict(combo_overrides)
            run_overrides['seed'] = seed

            if combo_label:
                label = f'{combo_label}/seed_{seed}'
            else:
                label = f'seed_{seed}'

            runs.append((run_overrides, label))

    return runs


def save_resolved_config(config: Config, path: str):
    """Dump the fully resolved Config as a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def create_experiment_dir(base_dir: str, name: str) -> str:
    """Create a timestamped experiment directory.

    Returns:
        Path to the created directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f'{name}_{timestamp}'
    path = os.path.join(base_dir, dir_name)
    os.makedirs(path, exist_ok=True)
    return path


def copy_source_yaml(yaml_path: str, experiment_dir: str):
    """Copy the source YAML into the experiment directory."""
    dest = os.path.join(experiment_dir, 'experiment.yaml')
    shutil.copy2(yaml_path, dest)
