#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple posterior plot generation using existing train.py functions
Usage: python temp.py  # Uses model specified in code
"""

import os
import sys
import json
import yaml
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from generative_ts.data import GP

# =============================================================================
# USER CONFIGURATION: 특정 모델 폴더 이름을 여기서 지정하세요
# =============================================================================
TARGET_MODEL_FOLDER = "250809_121135_LS4"  # 실행할 모델 폴더 이름

def detect_model_type(config):
    """Detect model type from config"""
    return config.get('model_type', 'Unknown')

def load_model_and_config(save_dir):
    """Load saved model and its configuration (supports both LS4 and VRNN)"""
    save_path = Path(save_dir)
    
    if not save_path.exists():
        raise FileNotFoundError(f"Model directory not found: {save_path}")
    
    # Load config
    config_path = save_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_type = detect_model_type(config)
    
    if model_type == "LS4":
        from generative_ts.model.ls4 import LS4_ts, dict2attr
        
        # Load LS4 model
        # Load yaml config
        yaml_config_path = save_path / "ls4_config.yaml"
        if yaml_config_path.exists():
            with open(yaml_config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            model_config = dict2attr(yaml_config['model'])
        else:
            # Fallback to original yaml path
            yaml_path = config['model']['config_path']
            if not Path(yaml_path).exists():
                raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            model_config = dict2attr(yaml_config['model'])
        
        # Create LS4 model
        model = LS4_ts(model_config)
        model_path = save_path / "model_LS4.pth"
        
    elif model_type == "VRNN":
        from generative_ts.model.vrnn import VRNN_ts
        
        # Load VRNN model - following train.py pattern
        model_config = config['model']
        model = VRNN_ts(
            x_dim=1,
            z_dim=model_config.get('z_dim', 1),
            h_dim=model_config.get('h_dim', 10),
            n_layers=model_config.get('n_layers', 1),
            lmbd=model_config.get('lmbd', 0),
            std_Y=config['data'].get('std_Y', 0.01)
        )
        model_path = save_path / "model_VRNN.pth"
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load checkpoint to same device
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Failed to load model_state_dict: {e}")
            print("Trying to load with strict=False...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"Failed to load checkpoint: {e}")
            print("Trying to load with strict=False...")
            model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    return model, config, model_type

def plot_posterior_test(data_generator, model, x_test, t_0, T, save_path, save_name, plot_name):
    """Test version of plot_posterior that saves to test/ directory instead of pred/"""
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        x_given = x_test[:t_0].to(device)
        
        # Model inference using unified posterior method
        x_given_model = x_given.unsqueeze(-1).unsqueeze(-1)  # (T0, 1, 1)
        mean_samples, var_samples, x_samples = model.posterior(x_given_model, T=T, N=20, verbose=0)

        # GP posterior (ground truth)
        x_given_np = x_given.squeeze().detach().cpu().numpy()
        mu_future, sigma_future, x_future = data_generator.posterior(x_given_np, T=T, N=20)
        
        # Plot - now handle full sequence
        t_given = np.arange(t_0)  # Given time range  
        t_pred = np.arange(t_0, T)  # Prediction time range
        plot_file_path = os.path.join(save_path, "test", save_name)  # Save to test/ directory

        plt.figure(figsize=(12, 6))
        
        # Given data (conditioning part)
        plt.plot(t_given, x_given.squeeze().cpu().numpy(), 
                 label=r'$given Y_{\leq t_0}$', color='#1f77b4', linewidth=2, marker='o', markersize=3)

        # GP posterior (ground truth) - future part only
        gp_line, = plt.plot(t_pred, mu_future, 
                           label=r'$Y_{>t_0} | Y_{\leq t_0}$ by GP', 
                           color='#ff7f0e', linewidth=2)
        plt.fill_between(t_pred, 
                        (mu_future - sigma_future), 
                        (mu_future + sigma_future), 
                        alpha=0.2, color=gp_line.get_color())

        # Model posterior - full sequence, but highlight prediction part
        model_mean_given = mean_samples[:t_0].squeeze() if mean_samples.ndim > 1 else mean_samples[:t_0]
        model_var_given = var_samples[:t_0].squeeze() if var_samples.ndim > 1 else var_samples[:t_0]
        model_mean_pred = mean_samples[t_0:].squeeze() if mean_samples.ndim > 1 else mean_samples[t_0:]
        model_var_pred = var_samples[t_0:].squeeze() if var_samples.ndim > 1 else var_samples[t_0:]
        
        # Model reconstruction (conditioning part) - lighter color
        plt.plot(t_given, model_mean_given, 
                color='#2ca02c', alpha=0.7, linewidth=1.5, linestyle='-', 
                label=r"$Y'_{\leq t_0} | Y_{\leq t_0}$ by " + f'{model.__class__.__name__}')
        plt.fill_between(t_given, 
                        model_mean_given - model_var_given, 
                        model_mean_given + model_var_given, 
                        alpha=0.1, color='#2ca02c')
        
        # Model prediction (extrapolation part) - full color
        model_line, = plt.plot(t_pred, model_mean_pred, 
                              label=r'$Y_{>t_0} | Y_{\leq t_0}$ by ' + f'{model.__class__.__name__}', 
                              color='#2ca02c', linewidth=2)
        plt.fill_between(t_pred, 
                        model_mean_pred - model_var_pred, 
                        model_mean_pred + model_var_pred, 
                        alpha=0.2, color=model_line.get_color())

        # Sample trajectories - prediction part only for clarity
        n_traj = min(5, x_future.shape[0], x_samples.shape[0])
        if x_future.shape[0] > 0:
            plt.plot(t_pred, x_future[:n_traj].T, 
                    color=gp_line.get_color(), alpha=0.3, 
                    linestyle='--', linewidth=1.0, label='_nolegend_')
        
        if x_samples.shape[0] > 0:
            # Extract prediction part from samples
            if x_samples.ndim == 3:  # (N, T, D)
                samples_pred = x_samples[:n_traj, t_0:, 0].T  # (T-T0, n_traj)
            else:  # (N, T)
                samples_pred = x_samples[:n_traj, t_0:].T  # (T-T0, n_traj)
            plt.plot(t_pred, samples_pred, 
                    color=model_line.get_color(), alpha=0.3, 
                    linestyle='--', linewidth=1.0, label='_nolegend_')

        plt.axvline(x=t_0, color='black', linestyle='--', linewidth=1.0)
        plt.xlabel(r'step t')
        plt.ylabel(r'Value')
        plt.title(plot_name)
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_file_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate posterior plots for saved models"""
    
    # Use the target model folder specified at the top
    saves_base = "generative_ts/saves"
    save_dir = os.path.join(saves_base, TARGET_MODEL_FOLDER)
    
    if not os.path.exists(save_dir):
        print(f"Error: Model directory not found: {save_dir}")
        print(f"Available models in {saves_base}:")
        if os.path.exists(saves_base):
            for item in os.listdir(saves_base):
                if os.path.isdir(os.path.join(saves_base, item)):
                    print(f"  - {item}")
        return 1
    
    print(f"Using model directory: {save_dir}")
    print(f"\n{'='*60}")
    print(f"Processing: {TARGET_MODEL_FOLDER}")
    print(f"{'='*60}")
    
    try:
        # Load model and config
        model, config, model_type = load_model_and_config(save_dir)
        print(f"{model_type} model loaded successfully")
        
        # Create data generator using same config
        data_generator = GP(**config['data'])
        
        # Generate test data
        n_test = 5
        T = config['data']['T']
        t_0 = 150
        
        # Create test directory
        test_dir = Path(save_dir) / "test" 
        test_dir.mkdir(exist_ok=True)
        
        for i in range(n_test):
            # Generate test sequence
            Y_test, _ = data_generator.data()
            Y_test = torch.from_numpy(Y_test.squeeze()).float()  # (T,)
            
            # Use test version of plot_posterior function
            save_name = f"test_{model_type}_{i}.png"
            plot_name = f"{model_type} vs GP posterior (Test {i})"
            
            plot_posterior_test(
                data_generator=data_generator,
                model=model, 
                x_test=Y_test,
                t_0=t_0,
                T=T,
                save_path=save_dir,
                save_name=save_name,
                plot_name=plot_name
            )
            
            print(f"Generated plot: {save_name}")
        
        print(f"\n✅ Successfully generated {n_test} posterior plots in {test_dir}")
        
    except Exception as e:
        print(f"Error processing {save_dir}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())