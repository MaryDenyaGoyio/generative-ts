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

from .dataset.data import GP
from .eval import plot_posterior, plot_sample

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pretrained_model(model_path, config_path, model_type='LS4'):
    """
    Load a pretrained model from checkpoint
    
    Args:
        model_path: path to model checkpoint (.pth file)
        config_path: path to model config (yaml file for LS4, json for VRNN)
        model_type: 'LS4' or 'VRNN'
    
    Returns:
        model: loaded model
        config: model configuration
    """
    
    if model_type == 'LS4':
        import yaml
        from .models.ls4 import LS4_ts, dict2attr
        
        # Load config
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        model_config = dict2attr(yaml_config['model'])
        
        # Create and load model
        model = LS4_ts(model_config).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint, strict=False)
        model.setup_rnn(mode='dense')  # Setup for inference
        
        return model, yaml_config
        
    elif model_type == 'VRNN':
        import json
        from .models.vrnn import VRNN_ts
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_config = config['model']
        
        # Create and load model
        model = VRNN_ts(
            x_dim=1,
            z_dim=model_config.get('z_dim', 1),
            h_dim=model_config.get('h_dim', 10),
            n_layers=model_config.get('n_layers', 1),
            lmbd=model_config.get('lmbd', 0),
            std_Y=config['train'].get('std_Y', 0.01)
        ).to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint, strict=False)
        
        return model, config
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train(model, data_generator, train_config, save_path, model_name, n_eval=50, dataset_path=None):
    """
    Unified training function for both VRNN_ts and LS4_ts models
    
    Args:
        model: VRNN_ts or LS4_ts instance
        data_generator: GP instance (can be None if dataset_path is provided)
        train_config: training configuration dict (can include 'beta' for KL weighting, default=1.0)
        save_path: directory to save results
        model_name: 'VRNN' or 'LS4' for logging
        dataset_path: if provided, load pre-generated dataset from this path instead of generating new data
    """
    
    # Get beta parameter for KL weighting (default=1.0)
    beta = train_config.get('beta', 1.0)
    if beta != 1.0:
        print(f"🎛️  Using beta={beta} for KL term weighting")
    
    def apply_beta_to_kl_loss(loss_dict, beta):
        """Apply beta weighting to KL terms in loss dictionary"""
        if beta == 1.0:
            return loss_dict  # No change needed
        
        modified_dict = loss_dict.copy()
        for key, value in loss_dict.items():
            if 'kl' in key.lower() or 'kld' in key.lower():
                if isinstance(value, torch.Tensor):
                    modified_dict[key] = value * beta
                else:
                    modified_dict[key] = value * beta  # float case
        
        return modified_dict
    
    # Generate or load dataset
    if dataset_path is not None:
        print(f"Loading pre-generated dataset from: {dataset_path}")
        # Load samples
        samples_path = os.path.join(dataset_path, "samples.npy")
        if not os.path.exists(samples_path):
            raise FileNotFoundError(f"Dataset file not found: {samples_path}")
        
        Y_N_full = np.load(samples_path)  # (N_total, T)
        
        # Load config to verify compatibility
        config_path = os.path.join(dataset_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                dataset_config = json.load(f)
            print(f"Dataset config: T={dataset_config['data']['T']}, total_samples={dataset_config['n_samples']}")
        
        # Sample requested number of sequences
        num_sequences = train_config['num_sequences']
        if num_sequences > Y_N_full.shape[0]:
            print(f"Warning: Requested {num_sequences} sequences but dataset only has {Y_N_full.shape[0]}. Using all available.")
            num_sequences = Y_N_full.shape[0]
        
        # Random sampling from dataset
        indices = np.random.choice(Y_N_full.shape[0], size=num_sequences, replace=False)
        Y_N = Y_N_full[indices]  # (num_sequences, T)
        
        Y_N = torch.from_numpy(Y_N).float()  # (N, T)
        print(f"Loaded {Y_N.shape[0]} sequences of length {Y_N.shape[1]} from pre-generated dataset")
        
    else:
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
    if dataset_path is not None:
        # Use one sample from loaded dataset for testing
        test_data = Y_N[0:1].numpy()  # (1, T)
        x_test = torch.from_numpy(test_data.squeeze()).float().to(DEVICE)  # (T,)
    else:
        test_data, _ = data_generator.data()
        x_test = torch.from_numpy(test_data.squeeze()).float().to(DEVICE)  # (T,)
    
    t_0 = len(x_test) // 4
    T = len(x_test)
    
    # For LS4: prepare timepoints  
    base_model_name = model_name.split('_')[0]
    if base_model_name == 'LS4':
        timepoints = torch.linspace(0.0, 1.0, T, device=DEVICE)
    
    print(f"number of model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training mode message
    print(f"🚀 TRAINING: Full ELBO training")
    
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
            # Handle Phase 2 model names (e.g., 'LS4_Phase2' -> 'LS4')
            base_model_name = model_name.split('_')[0]
            
            if base_model_name == 'VRNN':
                # VRNN expects (T, B, D) where D=1 for our 1D time series
                data = data.to(DEVICE).unsqueeze(-1)  # (B, T) -> (B, T, 1)
                data = data.transpose(0, 1)  # (B, T, 1) -> (T, B, 1)
                loss_dict = model(data)
                
                # Apply beta weighting to KL terms
                loss_dict = apply_beta_to_kl_loss(loss_dict, beta)
                
                # Use full loss (with beta-weighted KL terms)
                if 'total_loss' in loss_dict:
                    train_loss = loss_dict['total_loss']
                else:
                    train_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
                
            elif base_model_name == 'LS4':
                data = data.to(DEVICE).unsqueeze(-1)  # (B, T) -> (B, T, 1)
                
                # FIXED: Train with partial observation using masking
                # This forces the model to learn temporal dynamics for extrapolation
                T_full = data.shape[1]
                T_given = int(0.75 * T_full)  # first 75% observed, last 25% to predict
                
                # Create masks: 1 for observed, 0 for unobserved (to be predicted)
                masks = torch.ones_like(data, device=DEVICE)  # (B, T, 1)
                masks[:, T_given:] = 0.0  # mask out future part during training
                
                # Train model to reconstruct full sequence from partial observation
                train_loss, loss_dict = model(data, timepoints, masks, plot=False, sum=True)
                
                # Apply beta weighting to KL terms in loss_dict for logging
                loss_dict = apply_beta_to_kl_loss(loss_dict, beta)
                
                # Recalculate train_loss with beta-weighted KL terms
                if 'kld_loss' in loss_dict and beta != 1.0:
                    # Get individual loss components
                    nll_loss = loss_dict.get('nll_loss', 0)
                    mse_loss = loss_dict.get('mse_loss', 0)
                    kld_loss_weighted = loss_dict['kld_loss']  # Already beta-weighted
                    
                    # Convert to float values for reconstruction
                    if isinstance(nll_loss, torch.Tensor):
                        nll_val = nll_loss.item() if nll_loss.numel() == 1 else nll_loss.mean().item()
                    else:
                        nll_val = float(nll_loss)
                    
                    if isinstance(mse_loss, torch.Tensor):
                        mse_val = mse_loss.item() if mse_loss.numel() == 1 else mse_loss.mean().item()
                    else:
                        mse_val = float(mse_loss)
                    
                    if isinstance(kld_loss_weighted, torch.Tensor):
                        kld_val = kld_loss_weighted.item() if kld_loss_weighted.numel() == 1 else kld_loss_weighted.mean().item()
                    else:
                        kld_val = float(kld_loss_weighted)
                    
                    # Reconstruct total loss with beta-weighted KL
                    total_loss_val = nll_val + mse_val + kld_val
                    train_loss = torch.tensor(total_loss_val, device=DEVICE, requires_grad=True)
                    
                    # Update loss_dict with recalculated total loss
                    loss_dict['loss'] = total_loss_val
                    # Add extrapolation-specific loss if available
                    if hasattr(model, 'extrapolate_loss'):
                        data_given = data[:, :T_given]  # (B, T_given, 1)
                        timepoints_given = timepoints[:T_given]
                        timepoints_pred = timepoints[T_given:]
                        
                        extrap_loss = model.extrapolate_loss(data_given, timepoints_given, 
                                                           timepoints_pred, data[:, T_given:])
                        train_loss = train_loss + extrap_loss
            
            else:
                raise ValueError(f"Unknown base_model_name: {base_model_name} (from model_name: {model_name})")
            
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

                if name.endswith("loss"):
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
        if epoch % n_eval == 0:
            # Save model
            tqdm.write(f"[{epoch // n_eval}/{train_config['n_epochs'] // n_eval}]", end=" | ")
            start = time.time()

            torch.save(model.state_dict(), 
                      os.path.join(save_path, f"model_{model_name}.pth"))
            
            # Loss plot - filter to show only desired losses
            desired_losses = ['kld_loss', 'nll_loss']  # KLD, NLL (components of total)
            loss_keys = [key for key in desired_losses if key in all_losses and len(all_losses[key]) > 0]
            
            if len(loss_keys) > 0:
                # 2x3 layout: Total Loss + Individual component losses
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                
                # Define colors for consistency
                colors = {'kld_loss': 'red', 'nll_loss': 'green'}
                labels = {'kld_loss': 'KLD', 'nll_loss': 'NLL'}
                
                recent_start = max(0, len(all_losses[loss_keys[0]]) - 100)
                
                # Row 0: All epochs
                # First subplot: Total Loss (showing KLD + NLL components)
                ax = axes[0, 0]
                for key in loss_keys:
                    color = colors.get(key, 'black')
                    label = labels.get(key, key)
                    ax.plot(all_losses[key], color=color, label=label, linewidth=2)
                ax.set_xlabel('epoch')
                ax.set_ylabel('loss')
                ax.set_title('Total Loss (Components) - All Epochs')
                ax.legend()
                ax.grid(True)
                
                # Individual component plots (All epochs)
                for i, key in enumerate(loss_keys):
                    if i < 2:  # KLD and NLL
                        ax = axes[0, i + 1]
                        color = colors.get(key, 'black')
                        title = labels.get(key, key)
                        ax.plot(all_losses[key], color=color, linewidth=2)
                        ax.set_xlabel('epoch')
                        ax.set_ylabel('loss')
                        ax.set_title(f'{title} Loss - All Epochs')
                        ax.grid(True)
                
                # Row 1: Recent 100 epochs
                # Total Loss (Recent 100)
                ax = axes[1, 0]
                for key in loss_keys:
                    color = colors.get(key, 'black')
                    label = labels.get(key, key)
                    recent_vals = all_losses[key][recent_start:]
                    ax.plot(range(recent_start, len(all_losses[key])), recent_vals, 
                           color=color, label=label, linewidth=2)
                ax.set_xlabel('epoch')
                ax.set_ylabel('loss')
                ax.set_title('Total Loss (Components) - Recent 100 Epochs')
                ax.legend()
                ax.grid(True)
                
                # Individual component plots (Recent 100)
                for i, key in enumerate(loss_keys):
                    if i < 2:  # KLD and NLL
                        ax = axes[1, i + 1]
                        color = colors.get(key, 'black')
                        title = labels.get(key, key)
                        recent_vals = all_losses[key][recent_start:]
                        ax.plot(range(recent_start, len(all_losses[key])), recent_vals,
                               color=color, linewidth=2)
                        ax.set_xlabel('epoch')
                        ax.set_ylabel('loss')
                        ax.set_title(f'{title} Loss - Recent 100 Epochs')
                        ax.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"loss_{model_name}.png"))
                plt.close()
            
            # Posterior plot using model's posterior method
            plot_posterior(model, save_path, epoch, model_name=model_name, 
                         dataset_path=dataset_path)
            
            # Prior/Posterior diagnostic plot
            diagnostic_save_path = os.path.join(save_path, "pred")
            plot_sample(model, diagnostic_save_path, epoch, model_name=model_name, 
                       dataset_path=dataset_path)
            
            elapsed = time.time() - start
            # 뭐 이 자리에 나중에 loss도 출력할지도?
            tqdm.write(f"eval: {elapsed:.2f}s")
    
    return model


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
    
    # Save yaml config if specified
    if 'model' in config and 'config_path' in config['model']:
        yaml_config_path = config['model']['config_path']
        if os.path.exists(yaml_config_path):
            import shutil
            shutil.copy2(yaml_config_path, os.path.join(save_path, "ls4_config.yaml"))
    
    # Create data generator
    data_config = config['data']
    
    # Determine data generator type based on config
    if 'phi' in data_config:
        # AR1 data
        from .dataset.data import AR1
        data_generator = AR1(**data_config)
    else:
        # GP data
        data_generator = GP(**data_config)
    
    # Create model
    model_type = config['model_type']
    model_config = config['model']
    train_config = config['train']
    
    if model_type == 'VRNN':
        from .models.vrnn import VRNN_ts

        model = VRNN_ts(
            x_dim=1,
            z_dim=model_config.get('z_dim', 1),
            h_dim=model_config.get('h_dim', 10),
            n_layers=model_config.get('n_layers', 1),
            lmbd=model_config.get('lmbd', 0),
            std_Y=train_config.get('std_Y', 0.01)
        ).to(DEVICE)
        
    elif model_type == 'LS4':
        from .models.ls4 import LS4_ts, AttrDict, dict2attr
        # Load LS4 config
        cfg_path = model_config.get('config_path', 'generative_ts/config/ls4_config.yaml')
        cfg_all = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
        cfg_all = dict2attr(cfg_all)
        ls4_config = cfg_all.model

        # Modify config for our task
        ls4_config.n_labels = 1
        ls4_config.classifier = False
        
        model = LS4_ts(ls4_config).to(DEVICE)
        # Setup RNN for inference/generation
        model.setup_rnn(mode='dense')
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Check if dataset_path is specified in config
    dataset_path = config.get('dataset_path', None)
    
    # Train model using unified train function
    model = train(model, data_generator, train_config, save_path, model_type, dataset_path=dataset_path)
    
    print(f"Training completed! Results saved to: {save_path}")
    return model
