#!/usr/bin/env python3
"""
Generate posterior plots for multiple different data samples using a saved VRNN model
"""
import os
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from generative_ts.models.vrnn import VRNN_ts
from generative_ts.eval import plot_posterior


def load_model_and_config(model_path):
    """Load saved model and its configuration"""

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Load config
    config_path = Path(model_path).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model_config = config['model']
    model = VRNN_ts(**model_config)

    # Load state dict (use strict=False to handle mismatched keys)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Move model to CPU to avoid device conflicts
    model = model.cpu()

    # Force all parameters to CPU
    for param in model.parameters():
        param.data = param.data.cpu()

    model.eval()

    return model, config


def main():
    # Model path
    model_path = "generative_ts/saves/250925_142451_VRNN_stdY1_lmbd_dec/model_VRNN.pth"

    print(f"Loading model from: {model_path}")
    model, config = load_model_and_config(model_path)

    dataset_path = config['dataset_path']
    print(f"Dataset path: {dataset_path}")

    # Create output directory
    output_dir = Path("test_multiple_posteriors")
    output_dir.mkdir(exist_ok=True)

    print(f"Generating posterior plots for 10 different data samples...")

    # Generate plots for samples 0-9 (instead of always using sample 0)
    for sample_idx in range(10):
        print(f"  Generating plot for sample {sample_idx}...")

        plot_posterior(
            model=model,
            save_path=str(output_dir),
            epoch=f"sample_{sample_idx}",  # Use sample index as "epoch"
            model_name="VRNN",
            dataset_path=dataset_path,
            fixed_sample_idx=sample_idx,  # Key parameter: use different samples!
            N_samples=100
        )

    print(f"✅ All plots saved to: {output_dir}")
    print(f"   Generated files:")
    for i in range(10):
        print(f"     posterior_VRNN_sample_{i}_100samples.png")


if __name__ == "__main__":
    main()