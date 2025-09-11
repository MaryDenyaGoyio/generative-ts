#!/usr/bin/env python3
"""
Test inference q(z_t | x_<=t) with x ranging from -10 to 10
"""

import sys
import os
sys.path.append('/home/marydenya/Downloads/generative-ts')

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Import the load function from generative_ts.train
from generative_ts.train import load_pretrained_model

def test_inference_range(model, x_range=(-10, 10), num_points=21, seq_len=50):
    """
    Test inference q(z_t | x_<=t) for different x values
    
    Args:
        model: trained LS4 model
        x_range: tuple (min, max) for x values to test
        num_points: number of points to test
        seq_len: sequence length for inference
    """
    device = next(model.parameters()).device
    
    # Create x values to test
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    results = {
        'x_values': x_values,
        'z_means': [],
        'z_stds': [],
        'z_samples': []
    }
    
    with torch.no_grad():
        for x_val in x_values:
            # Create constant sequence with this x value
            # Shape: (batch_size=1, seq_len, d_x=1)
            x_seq = torch.full((1, seq_len, 1), x_val, dtype=torch.float32, device=device)
            
            # Create time vector (uniformly spaced from 0 to 1)
            t_vec = torch.linspace(0, 1, seq_len, device=device).unsqueeze(0)  # (1, seq_len)
            
            # Run encoder to get q(z_t | x_<=t)
            z_sample, z_mean, z_std = model.encode(x_seq, t_vec)
            
            # Store results (take last timestep)
            results['z_means'].append(z_mean[0, -1, 0].cpu().numpy())  # (batch, time, z_dim)
            results['z_stds'].append(z_std[0, -1, 0].cpu().numpy())
            results['z_samples'].append(z_sample[0, -1, 0].cpu().numpy())
    
    return results

def plot_inference_results(results, save_path):
    """Plot inference results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    x_vals = results['x_values']
    z_means = results['z_means']
    z_stds = results['z_stds']
    z_samples = results['z_samples']
    
    # Plot 1: z_mean vs x
    ax1.plot(x_vals, z_means, 'bo-', linewidth=2, markersize=6)
    ax1.fill_between(x_vals, 
                     np.array(z_means) - np.array(z_stds),
                     np.array(z_means) + np.array(z_stds),
                     alpha=0.3, color='blue')
    ax1.set_xlabel('Input x')
    ax1.set_ylabel('z_mean')
    ax1.set_title('Posterior Mean: E[z_t | x_<=t]')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: z_std vs x
    ax2.plot(x_vals, z_stds, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Input x')
    ax2.set_ylabel('z_std')
    ax2.set_title('Posterior Standard Deviation: Sqrt(Var[z_t | x_<=t])')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: z_sample vs x
    ax3.plot(x_vals, z_samples, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Input x')
    ax3.set_ylabel('z_sample')
    ax3.set_title('Sampled z_t ~ q(z_t | x_<=t)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print(f"Input range: {min(x_vals):.1f} to {max(x_vals):.1f}")
    print(f"z_mean range: {min(z_means):.3f} to {max(z_means):.3f}")
    print(f"z_std range: {min(z_stds):.3f} to {max(z_stds):.3f}")
    print(f"z_sample range: {min(z_samples):.3f} to {max(z_samples):.3f}")

if __name__ == "__main__":
    # Model path
    model_dir = "/home/marydenya/Downloads/generative-ts/generative_ts/saves/250911_141705"
    model_path = os.path.join(model_dir, "model_LS4.pth")
    config_path = os.path.join(model_dir, "ls4_config.yaml")
    
    print("Loading model...")
    model, config = load_pretrained_model(model_path, config_path, model_type='LS4')
    
    print("Testing inference for x = -10 to 10...")
    results = test_inference_range(model, x_range=(-10, 10), num_points=21, seq_len=50)
    
    print("Plotting results...")
    save_path = os.path.join(model_dir, "inference_test_x_range.png")
    plot_inference_results(results, save_path)
    
    print(f"Results saved to: {save_path}")