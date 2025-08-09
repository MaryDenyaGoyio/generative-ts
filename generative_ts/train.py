import os
import json
import copy
import time
import yaml
from tqdm import trange, tqdm
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .data import GP
from .model import VRNN_ts, LS4_ts, AttrDict, dict2attr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, data_generator, train_config, save_path, model_name):
    """
    Unified training function for both VRNN_ts and LS4_ts models
    
    Args:
        model: VRNN_ts or LS4_ts instance
        data_generator: GP instance
        train_config: training configuration dict
        save_path: directory to save results
        model_name: 'VRNN' or 'LS4' for logging
    """
    
    # Generate dataset
    print(f"Generating {train_config['num_sequences']} GP sequences for {model_name}...")
    Y_N = []
    for _ in range(train_config['num_sequences']):
        Y, _ = data_generator.data()
        Y_N.append(Y.squeeze())  # (T,)
    
    Y_N = torch.from_numpy(np.stack(Y_N, axis=0)).float()  # (N, T)
    
    # Create dataloader
    loader = DataLoader(TensorDataset(Y_N), 
                       batch_size=train_config['batch_size'], 
                       shuffle=True, 
                       num_workers=4)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=train_config['learning_rate'])
    
    # Prepare test data
    test_data, _ = data_generator.data()
    x_test = torch.from_numpy(test_data.squeeze()).float().to(DEVICE)  # (T,)
    t_0 = len(x_test) // 4
    T = len(x_test)
    
    # For LS4: prepare timepoints
    if model_name == 'LS4':
        timepoints = torch.linspace(0.0, 1.0, T, device=DEVICE)
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    all_losses = {}
    
    for epoch in trange(train_config['n_epochs'], desc=f"Training {model_name}", unit="epoch"):
        start = time.time()
        
        # Initialize epoch losses
        for k, v in all_losses.items():
            v.append(0)
        
        num_batches = len(loader)
        
        # Training step
        model.train()
        for batch_idx, (data,) in enumerate(loader):
            
            optimizer.zero_grad()
            
            # Model-specific forward pass
            if model_name == 'VRNN':
                data = data.to(DEVICE).transpose(0, 1)  # (B, T, D) -> (T, B, D)
                loss_dict = model(data)
                
                # Calculate total loss
                if 'total_loss' in loss_dict:
                    train_loss = loss_dict['total_loss']
                else:
                    train_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
                
            elif model_name == 'LS4':
                data = data.to(DEVICE).unsqueeze(-1)  # (B, T) -> (B, T, 1)
                masks = torch.ones(data.shape[0], data.shape[1], 1, device=DEVICE)
                train_loss, loss_dict = model(data, timepoints, masks, plot=False, sum=True)
            
            else:
                raise ValueError(f"Unknown model_name: {model_name}")
            
            train_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # NaN/Inf handling
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(
                        p.grad.data, nan=0.0, posinf=1e6, neginf=-1e6
                    )
            
            optimizer.step()
            
            # Loss logging
            for name, each_loss in loss_dict.items():
                if isinstance(each_loss, torch.Tensor):
                    value = (
                        each_loss.detach().cpu().item()
                        if each_loss.dim() == 0
                        else each_loss.mean().detach().cpu().item()
                    )
                else:
                    value = each_loss  # float
                all_losses.setdefault(name, [0] * (epoch + 1))[epoch] += value / num_batches
        
        elapsed = time.time() - start
        
        # Epoch logging
        log_msg = f"[{epoch}/{train_config['n_epochs']}] " + \
                 "\t".join(f"{k}:{v[epoch]:.2f}" for k, v in all_losses.items()) + \
                 f"  epoch: {elapsed:.2f}s"
        tqdm.write(log_msg)
        
        # Save checkpoints and plots every 10 epochs
        if epoch % 10 == 0:
            # Save model
            torch.save(model.state_dict(), 
                      os.path.join(save_path, f"model_{model_name}_epoch_{epoch}.pth"))
            
            # Loss plot
            plt.figure()
            for key, vals in all_losses.items():
                if len(vals) > 0:
                    plt.plot(vals, label=key.replace('_', ' ').title())
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title(f'{model_name} Loss curves')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_path, f"loss_{model_name}.png"))
            plt.close()
            
            # Extrapolation plot using model's posterior method
            plot_posterior(data_generator, model, x_test, t_0, T, 
                         save_path, epoch, model_name)
    
    return model


def plot_posterior(data_generator, model, x_test, t_0, T, save_path, epoch, model_name):
    """올바른 베이지안 inference로 extrapolation 플롯"""
    
    model.eval()
    with torch.no_grad():
        x_given = x_test[:t_0]
        
        # Model inference using unified posterior method
        x_given_model = x_given.unsqueeze(1)  # (T0, 1, D)
        mean_samples, var_samples, x_samples = model.posterior(x_given_model, T=T, N=100, verbose=0)
        
        # GP posterior (ground truth)
        x_given_np = x_given.squeeze().detach().cpu().numpy()
        mu_future, sigma_future, x_future = data_generator.posterior(x_given_np, T=T, N=10)
        
        # Plot
        geq_t_0 = np.arange(t_0, T)
        plot_name = os.path.join(save_path, "pred", f"test_{model_name}_{epoch}.png")

        plt.figure(figsize=(12, 6))
        
        # Given data
        plt.plot(x_given.squeeze().cpu().numpy(), 
                 label=r'$given Y_{\leq t_0}$', color='#1f77b4', linewidth=2)

        # GP posterior (ground truth)
        gp_line, = plt.plot(geq_t_0, mu_future, 
                           label=r'$Y_{>t_0} | Y_{\leq t_0}$ by GP', 
                           color='#ff7f0e', linewidth=2)
        plt.fill_between(geq_t_0, 
                        (mu_future - sigma_future), 
                        (mu_future + sigma_future), 
                        alpha=0.2, color=gp_line.get_color())

        # Model posterior
        model_line, = plt.plot(geq_t_0, mean_samples.squeeze(), 
                              label=f'$Y_{{>t_0}} | Y_{{\leq t_0}}$ by {model_name}', 
                              color='#2ca02c', linewidth=2)
        plt.fill_between(geq_t_0, 
                        (mean_samples - var_samples).squeeze(), 
                        (mean_samples + var_samples).squeeze(), 
                        alpha=0.2, color=model_line.get_color())

        # Sample trajectories
        n_traj = min(10, x_future.shape[0], x_samples.shape[0])
        if x_future.shape[0] > 0:
            plt.plot(geq_t_0, x_future[:n_traj].T, 
                    color=gp_line.get_color(), alpha=0.3, 
                    linestyle='--', linewidth=1.0, label='_nolegend_')
        
        if x_samples.shape[0] > 0:
            if x_samples.ndim == 3:  # (N, T-T0, D)
                samples_plot = x_samples[:n_traj, :, 0].T  # (T-T0, n_traj)
            else:  # (N, T-T0)
                samples_plot = x_samples[:n_traj].T  # (T-T0, n_traj)
            plt.plot(geq_t_0, samples_plot, 
                    color=model_line.get_color(), alpha=0.3, 
                    linestyle='--', linewidth=1.0, label='_nolegend_')

        plt.axvline(x=t_0, color='black', linestyle='--', linewidth=1.0)
        plt.xlabel(r'step t')
        plt.ylabel(r'Value')
        plt.title(f'{model_name} vs GP extrapolation (Epoch {epoch})')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_name, dpi=150, bbox_inches='tight')
        plt.close()




def train_model(config: Dict[str, Any]) -> nn.Module:
    """
    통합된 훈련 함수
    
    Args:
        config: 훈련 설정 딕셔너리
            - model_type: 'VRNN' or 'LS4'
            - data: GP 파라미터
            - model: 모델별 파라미터
            - train: 훈련 파라미터
    """
    
    # Create save directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    save_path = os.path.join("generative_ts/saves", timestamp)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "pred"), exist_ok=True)
    
    # Save config
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create data generator
    data_config = config['data']
    data_generator = GP(**data_config)
    
    # Create model
    model_type = config['model_type']
    model_config = config['model']
    train_config = config['train']
    
    if model_type == 'VRNN':
        model = VRNN_ts(
            x_dim=1,
            z_dim=model_config.get('z_dim', 1),
            h_dim=model_config.get('h_dim', 10),
            n_layers=model_config.get('n_layers', 1),
            lmbd=model_config.get('lmbd', 0),
            std_Y=train_config.get('std_Y', 0.01)
        ).to(DEVICE)
        
    elif model_type == 'LS4':
        # Load LS4 config
        cfg_path = model_config.get('config_path', 'ls4/configs/monash/vae_nn5daily.yaml')
        cfg_all = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
        cfg_all = dict2attr(cfg_all)
        ls4_config = cfg_all.model

        # Modify config for our task
        ls4_config.n_labels = 1
        ls4_config.classifier = False
        
        model = LS4_ts(ls4_config).to(DEVICE)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Train model using unified train function
    model = train(model, data_generator, train_config, save_path, model_type)
    
    print(f"Training completed! Results saved to: {save_path}")
    return model