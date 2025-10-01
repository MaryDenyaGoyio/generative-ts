#!/usr/bin/env python3
import json
import torch
import yaml
from pathlib import Path

def load_model_from_checkpoint(checkpoint_path):
    """Load saved model from checkpoint directory"""
    checkpoint_path = Path(checkpoint_path)

    # Load config
    with open(checkpoint_path / "config.json", 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config.get('model_type', 'Unknown')

    if model_type == "LS4":
        from generative_ts.models.ls4 import LS4_ts, dict2attr
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
            std_Y=model_config.get('std_Y', 0.01)
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

    model = model.to(device)
    checkpoint_data = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
        state_dict = checkpoint_data['model_state_dict']
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys}")

        epoch = checkpoint_data.get('epoch', 0)
        all_losses = checkpoint_data.get('all_losses', {})
        optimizer_state_dict = checkpoint_data.get('optimizer_state_dict', None)
    else:
        model.load_state_dict(checkpoint_data, strict=False)
        epoch = 0
        all_losses = {}
        optimizer_state_dict = None

    # Return a dictionary for clarity and future extension
    resume_data = {
        "model": model,
        "config": config,
        "epoch": epoch,
        "all_losses": all_losses,
        "optimizer_state_dict": optimizer_state_dict
    }
    return resume_data

def save_minimal_checkpoint(model, epoch, all_losses, save_path, model_type):
    """Save minimal checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'all_losses': all_losses
    }, Path(save_path) / f"model_{model_type}.pth")
