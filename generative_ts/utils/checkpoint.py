#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint utility functions for model loading and resuming training
"""

import os
import sys
import json
import yaml
import torch
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def detect_model_type(config):
    """Detect model type from config"""
    return config.get('model_type', 'Unknown')


def find_latest_checkpoint(base_path):
    """Find the latest checkpoint directory in saves folder"""
    saves_path = Path(base_path) / "generative_ts" / "saves"
    if not saves_path.exists():
        return None

    # Get all directories with timestamp pattern
    checkpoint_dirs = [d for d in saves_path.iterdir() if d.is_dir() and '_' in d.name]
    if not checkpoint_dirs:
        return None

    # Sort by name (timestamp) and return latest
    latest_dir = sorted(checkpoint_dirs, key=lambda x: x.name)[-1]
    return str(latest_dir)


def load_checkpoint_info(checkpoint_path):
    """Load checkpoint information including epoch, optimizer state, etc."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # Load config
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load training info if exists
    training_info_path = checkpoint_path / "training_info.json"
    training_info = {}
    if training_info_path.exists():
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)

    return config, training_info


def load_model_from_checkpoint(checkpoint_path):
    """Load saved model from checkpoint directory"""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # Load config and training info
    config, training_info = load_checkpoint_info(checkpoint_path)
    model_type = detect_model_type(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == "LS4":
        from generative_ts.models.ls4 import LS4_ts, dict2attr

        # Load LS4 yaml config
        yaml_config_path = checkpoint_path / "ls4_config.yaml"
        if yaml_config_path.exists():
            with open(yaml_config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            model_config = dict2attr(yaml_config['model'])
        else:
            yaml_path = config['model']['config_path']
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            model_config = dict2attr(yaml_config['model'])

        model = LS4_ts(model_config)
        model_path = checkpoint_path / "model_LS4.pth"

    elif model_type == "VRNN":
        from generative_ts.models.vrnn import VRNN_ts

        model_config = config['model']
        model = VRNN_ts(
            x_dim=1,
            z_dim=model_config.get('z_dim', 1),
            h_dim=model_config.get('h_dim', 10),
            n_layers=model_config.get('n_layers', 1),
            lmbd=model_config.get('lmbd', 0),
            std_Y=model_config.get('std_Y', config.get('train', {}).get('std_Y', 0.01))
        )
        model_path = checkpoint_path / "model_VRNN.pth"

    elif model_type == "LatentODE":
        from generative_ts.models.latent_ode import LatentODE_ts

        model_config = config['model']
        model = LatentODE_ts(
            x_dim=1,
            z_dim=model_config.get('z_dim', 4),
            h_dim=model_config.get('h_dim', 25),
            std_Y=config['data'].get('std_Y', 0.01)
        )
        model_path = checkpoint_path / "model_LatentODE.pth"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load model weights
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = model.to(device)
    checkpoint_data = torch.load(model_path, map_location=device)

    # Extract model state dict and other info
    if isinstance(checkpoint_data, dict):
        if 'model_state_dict' in checkpoint_data:
            # Load model weights immediately (needed for resume_training_setup)
            model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
            optimizer_state = checkpoint_data.get('optimizer_state_dict', None)
            scheduler_state = checkpoint_data.get('scheduler_state_dict', None)
            # Check actual epoch from checkpoint, but also verify with saved files
            checkpoint_epoch = checkpoint_data.get('epoch', training_info.get('last_epoch', 0))

            # Try to infer actual epoch from saved files (posterior plots)
            post_dir = checkpoint_path / "post"
            if post_dir.exists():
                max_epoch = 0
                for file in post_dir.glob("posterior_*_*.png"):
                    try:
                        # Extract epoch from filename like "posterior_VRNN_490_100samples.png"
                        parts = file.stem.split('_')
                        if len(parts) >= 3 and parts[2].isdigit():
                            epoch_num = int(parts[2])
                            max_epoch = max(max_epoch, epoch_num)
                    except:
                        continue

                if max_epoch > checkpoint_epoch:
                    print(f"🔧 Correcting epoch: checkpoint says {checkpoint_epoch}, but found plots up to epoch {max_epoch}")
                    epoch = max_epoch
                else:
                    epoch = checkpoint_epoch
            else:
                epoch = checkpoint_epoch
            loss = checkpoint_data.get('loss', training_info.get('best_loss', float('inf')))

            # Store full checkpoint data for training resume
            training_info['full_checkpoint'] = checkpoint_data
        else:
            # Old format - just state dict
            model.load_state_dict(checkpoint_data, strict=False)
            optimizer_state = None
            scheduler_state = None
            epoch = training_info.get('last_epoch', 0)
            loss = training_info.get('best_loss', float('inf'))
    else:
        # Very old format
        model.load_state_dict(checkpoint_data, strict=False)
        optimizer_state = None
        scheduler_state = None
        epoch = 0
        loss = float('inf')

    resume_info = {
        'epoch': epoch,
        'best_loss': loss,
        'optimizer_state': optimizer_state,
        'scheduler_state': scheduler_state,
        'save_path': str(checkpoint_path),
        'training_info': training_info
    }

    return model, config, resume_info


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, model_type, additional_info=None):
    """Save checkpoint with all necessary information for resuming"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint data
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }

    if additional_info:
        checkpoint_data.update(additional_info)

    # Save model checkpoint
    model_filename = f"model_{model_type}.pth"
    torch.save(checkpoint_data, save_path / model_filename)

    # Save training info
    training_info = {
        'last_epoch': epoch,
        'best_loss': loss,
        'last_saved': datetime.now().isoformat(),
        'model_type': model_type
    }

    if additional_info:
        training_info.update({k: v for k, v in additional_info.items() if k != 'model_state_dict'})

    with open(save_path / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"✅ Checkpoint saved: {save_path / model_filename}")
    return str(save_path)


def resume_training_setup(checkpoint_path, config_override=None):
    """Setup for resuming training from checkpoint

    Args:
        checkpoint_path: Path to checkpoint directory
        config_override: Optional config overrides for resuming

    Returns:
        model, config, resume_info
    """
    model, config, resume_info = load_model_from_checkpoint(checkpoint_path)

    # Apply config overrides if provided
    if config_override:
        # Merge configs but preserve model architecture settings
        preserved_keys = ['model', 'data']
        for key in preserved_keys:
            if key in config_override and key in config:
                # Deep merge for nested dicts
                for subkey, subvalue in config_override[key].items():
                    if subkey not in ['z_dim', 'h_dim', 'n_layers', 'x_dim']:  # Don't change architecture
                        config[key][subkey] = subvalue

        # Update top-level settings
        for key, value in config_override.items():
            if key not in preserved_keys:
                config[key] = value

    print(f"🔄 Resuming training from epoch {resume_info['epoch']}")
    print(f"📁 Checkpoint path: {resume_info['save_path']}")
    print(f"💾 Best loss so far: {resume_info['best_loss']:.6f}")

    return model, config, resume_info