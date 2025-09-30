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
        if 'std_Y' not in model_config:
            raise KeyError("config['model']['std_Y']가 설정되어야 합니다. VRNN 학습 config를 확인하세요.")

        model = VRNN_ts(
            x_dim=model_config.get('x_dim', 1),
            z_dim=model_config.get('z_dim', 1),
            h_dim=model_config.get('h_dim', 10),
            n_layers=model_config.get('n_layers', 1),
            lmbd=model_config.get('lmbd', 0),
            std_Y=model_config['std_Y']
        ).to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint, strict=False)
        
        return model, config
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train(model, train_config, save_path, model_name, dataset_path, n_eval=10, T_0_ratio = 1/4, start_epoch=0, all_losses=None):

    # ---------------- 0) beta VAE ----------------   
    beta = train_config.get('beta', 1.0)
    if beta != 1.0: print(f"[β-VAE] β = {beta} weight for KL")
    
    def apply_beta_to_kl_loss(loss_dict, beta):
        if beta == 1.0: return loss_dict
        modified_dict = loss_dict.copy()
        for key, value in loss_dict.items():
            if 'kl' in key.lower() or 'kld' in key.lower():
                if isinstance(value, torch.Tensor): modified_dict[key] = value * beta
                else:   modified_dict[key] = value * beta
        return modified_dict
    

    # ---------------- 1) load data ---------------- 
    print(f"[Load data] {dataset_path}")

    samples_path = os.path.join(dataset_path, "outcome_Y.npy")
    if not os.path.exists(samples_path):    raise FileNotFoundError(f"Not found: {samples_path}")

    Y_N_full = np.load(samples_path)

    config_path = os.path.join(dataset_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            dataset_config = json.load(f)
        print(f"[Data config]: T = {dataset_config['data']['T']}, N_data = {Y_N_full.shape[0]}")
    
    
    # ---------------- 2) train cfg ---------------- 
    num_sequences = train_config['num_sequences']
    if num_sequences > Y_N_full.shape[0]:
        print(f"Warning: N_train = {num_sequences} > N_data = {Y_N_full.shape[0]}. Using only N_data.")
        num_sequences = Y_N_full.shape[0]
    
    indices = np.random.choice(Y_N_full.shape[0], size=num_sequences, replace=False)
    Y_N = Y_N_full[indices]
    
    Y_N = torch.from_numpy(Y_N).float()  # (N, T, D)
    if Y_N.dim() == 2:  Y_N, D = Y_N.unsqueeze(-1), 1 # (N, T) -> (N, T, 1)
    else:   D = Y_N.shape[-1] # (N, T, D)
    
    N, T, D = Y_N.shape
    print(f"[Train config]: N_train = {N}, T = {T}, D = {D}")
    
    loader = DataLoader(TensorDataset(Y_N), 
                       batch_size=train_config['batch_size'], 
                       shuffle=True, 
                       num_workers=4)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=train_config['learning_rate'])

    # Load optimizer and scheduler state if resuming
    scheduler = None
    # Optimizer and scheduler setup (no resume state for simplicity)

    test_data = Y_N[0:1].numpy()  # (1, T, D)
    x_test = torch.from_numpy(test_data.squeeze(0)).float().to(DEVICE)  # (T, D)

    T, T_0 = x_test.shape[0], int(T * T_0_ratio)

    # For eval: prepare data arrays
    Y_test_eval = Y_N_full  # Full dataset for evaluation
    theta_test_eval = None
    if os.path.exists(os.path.join(dataset_path, "latent_theta.npy")):
        theta_test_eval = np.load(os.path.join(dataset_path, "latent_theta.npy"))
    config_eval = dataset_config if 'dataset_config' in locals() else None
    
    
    # ---------------- 3) load model ----------------  
    base_model_name = model_name.split('_')[0]
    if base_model_name == 'LS4':
        timepoints = torch.linspace(0.0, 1.0, T, device=DEVICE)
    
    print(f"[model config]: {model_name}, num_params = {sum(p.numel() for p in model.parameters())}")
    


    # ================= 4) Training =================  
    print(f"\n===============================================================")
    print(f"=======================     Training     =======================")
    print(f"================================================================\n")
    
    # Load existing loss history if resuming, otherwise start empty
    if all_losses is None:
        all_losses = {}

    # Training loop: continue from start_epoch if resuming, 0 if new training
    actual_start = start_epoch if start_epoch > 0 else start_epoch

    # Resume: train from start_epoch+1 to n_epochs (not adding to existing)
    # New: train from 0 to n_epochs-1
    if start_epoch > 0:
        # Resume mode: train from start_epoch+1 to n_epochs
        total_epochs = train_config['n_epochs']
        actual_start = start_epoch + 1
        remaining_epochs = total_epochs - actual_start
        print(f"Resuming training from epoch {actual_start} to {total_epochs-1} ({remaining_epochs} remaining epochs)")
    else:
        # New training: train from 0 to n_epochs-1
        total_epochs = train_config['n_epochs']
        actual_start = 0
        print(f"Starting new training from epoch 0 to {total_epochs-1} (total {total_epochs} epochs)")
    for epoch in trange(actual_start, total_epochs, desc=f"[{model_name}]", unit="epoch", initial=actual_start, total=total_epochs):
        start = time.time()
        
        num_batches = len(loader)
        
        model.train() # <-> eval

        for batch_idx, (data,) in enumerate(loader):
            optimizer.zero_grad()

            # ---------------- 4-1) VRNN ----------------  
            # VRNN expects (T, B, D) - data is (B, T, D) format

            if base_model_name == 'VRNN':
                data = data.to(DEVICE)  # (B, T, D)
                data = data.transpose(0, 1)  # (B, T, D) -> (T, B, D)
                loss_dict = model(data)
                
                loss_dict = apply_beta_to_kl_loss(loss_dict, beta)
                
                if 'total_loss' in loss_dict:
                    train_loss = loss_dict['total_loss']
                else:
                    train_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            
            # ---------------- 4-2) LS4 ----------------
            # LS4 expects (B, T, D) - data is already in this format

            elif base_model_name == 'LS4':
                data = data.to(DEVICE)  # (B, T, D)
                masks = torch.ones_like(data, device=DEVICE)  # (B, T, D)

                train_loss, loss_dict = model(data, timepoints, masks, plot=False, sum=True)


                loss_dict = apply_beta_to_kl_loss(loss_dict, beta)

                rollout_cfg = train_config.get('rollout') or {}
                if rollout_cfg:
                    segs = rollout_cfg.get('segments', 4)
                    alpha = rollout_cfg.get('alpha', 1.0)
                    rollout_loss = model.rollout_nll(data, timepoints, segments=segs)
                    rollout_term = alpha * rollout_loss
                    loss_dict['rollout_loss'] = rollout_loss.detach()
                    loss_dict['rollout_loss_weighted'] = rollout_term.detach()
                    train_loss = train_loss + rollout_term

            # ---------------- 4-3) LatentODE ----------------
            # LatentODE expects batch_dict format - use existing compute_all_losses

            elif base_model_name == 'LatentODE':
                data = data.to(DEVICE)  # (B, T, D)
                batch_size, seq_len, x_dim = data.shape

                timepoints = torch.linspace(0, 1, seq_len, device=DEVICE)

                # Create explicit mask for latent_ode
                masks = torch.ones_like(data, device=DEVICE)  # (B, T, D)

                # LatentODE expects data with concatenated mask: [data, mask]
                data_with_mask = torch.cat([data, masks], dim=-1)  # (B, T, 2*D)

                # Follow exact latent_ode format with proper mask
                batch_dict = {
                    "observed_data": data_with_mask,  # Data with concatenated mask
                    "observed_tp": timepoints,
                    "data_to_predict": data_with_mask,
                    "tp_to_predict": timepoints,
                    "observed_mask": masks,  # Original mask for likelihood computation
                    "mask_predicted_data": None,
                    "labels": None,
                    "mode": "interpolation"
                }

                loss_dict = model.compute_all_losses(batch_dict)

                if 'loss' in loss_dict:
                    train_loss = loss_dict['loss']
                elif 'likelihood' in loss_dict and 'kl_loss' in loss_dict:
                    train_loss = -loss_dict['likelihood'] + loss_dict['kl_loss']
                else:
                    train_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

                loss_dict = apply_beta_to_kl_loss(loss_dict, beta)

            else:
                raise ValueError(f"Unknown base_model_name: {base_model_name} (from model_name: {model_name})")
            
            train_loss.backward()
            
            # Grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # NaN/Inf handling
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(
                        p.grad.data, nan=0.0, posinf=1e6, neginf=-1e6
                    )
            
            optimizer.step()
            

            # ---------------- 5-1) Loss logging ----------------
            for name, each_loss in loss_dict.items():
                if 'loss' in name:
                    if isinstance(each_loss, torch.Tensor):
                        value = (
                            each_loss.detach().cpu().item()
                            if each_loss.dim() == 0
                            else each_loss.mean().detach().cpu().item()
                        )
                    else:
                        value = each_loss  # float

                    # Initialize loss list if needed (simple approach)
                    if name not in all_losses:
                        all_losses[name] = []

                    # Extend list to current epoch if needed
                    while len(all_losses[name]) <= epoch:
                        all_losses[name].append(0)

                    # Accumulate current epoch loss
                    all_losses[name][epoch] += value
        
        elapsed = time.time() - start
        
        # Average the accumulated losses over number of batches
        for loss_name in all_losses:
            if len(all_losses[loss_name]) > epoch:
                all_losses[loss_name][epoch] /= num_batches



        # Select which losses to display in log (exclude unwanted ones)
        display_losses = []

        def append_loss(loss_key, display_label):
            if loss_key in all_losses and len(all_losses[loss_key]) > epoch:
                display_losses.append(f"{display_label}:{all_losses[loss_key][epoch]:.2f}")
                return True
            return False

        # Prefer kld_loss, fall back to kl_loss if needed
        if not append_loss('kld_loss', 'kld_loss'):
            append_loss('kl_loss', 'kl_loss')

        append_loss('nll_loss', 'nll_loss')
        append_loss('ent_loss', 'ent_loss')

        # Show rollout penalty if available: prefer weighted version, else raw value
        if not append_loss('rollout_loss_weighted', 'rollout_loss'):
            append_loss('rollout_loss', 'rollout_loss')

        log_msg = f"[{epoch}/{total_epochs}] " + "\t".join(display_losses) + f"  epoch: {elapsed:.2f}s"
        tqdm.write(log_msg)
        
        
        # ---------------- 5-2) Model save ----------------  

        if epoch % n_eval == 0 or epoch == total_epochs - 1:
            tqdm.write(f"[{epoch // n_eval}/{total_epochs // n_eval}]", end=" | ")
            start = time.time()

            # Save comprehensive checkpoint including optimizer state
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'loss': all_losses.get('total_loss', [0])[-1] if all_losses else 0,
                'all_losses': all_losses,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(checkpoint_data, os.path.join(save_path, f"model_{model_name}.pth"))

            # Also save loss history separately for easier access
            loss_history_path = os.path.join(save_path, "loss_history.json")
            with open(loss_history_path, 'w') as f:
                json.dump(all_losses, f, indent=2)
            
            # 5-2-1) Loss curve
            plot_loss(all_losses, save_path, model_name)
            
            # 5-2-2) Posterior plot
            posterior_save_path = os.path.join(save_path, "post")
            N_samples_plot = 1000 if epoch % 100 == 0 and epoch > 0 else 100
            plot_posterior(model, posterior_save_path, epoch, model_name=model_name,
                         Y_test=Y_test_eval, theta_test=theta_test_eval, config=config_eval, N_samples=N_samples_plot)

            # 5-2-3) Loss analysis
            analysis_save_path = os.path.join(save_path, "anl")
            plot_sample(model, analysis_save_path, epoch, model_name=model_name,
                       Y_test=Y_test_eval, config=config_eval)
            
            elapsed = time.time() - start
            tqdm.write(f"eval: {elapsed:.2f}s")

    # Final checkpoint already saved in evaluation blocks

    return model


def plot_loss(all_losses, save_path, model_name):
    """
    Plot loss curves in a clean 2x4 layout: Total, KL, NLL, Ent, then same for recent 100 epochs

    Args:
        all_losses: dictionary containing loss history
        save_path: directory to save the plot
        model_name: name of the model for filename
    """
    # Find available losses
    loss_keys = [k for k, v in all_losses.items() if v and len(v) > 0 and 'loss' in k]
    if not loss_keys:
        return


    # Define loss order and colors
    loss_order = ['kl_loss', 'kld_loss', 'nll_loss', 'ent_loss', 'rollout_loss_weighted', 'rollout_loss']
    colors = {'kl_loss': 'red', 'kld_loss': 'red', 'nll_loss': 'green',
              'ent_loss': 'blue', 'rollout_loss_weighted': 'purple', 'rollout_loss': 'purple'}
    titles = {'kl_loss': 'KL', 'kld_loss': 'KL', 'nll_loss': 'NLL',
              'ent_loss': 'Ent', 'rollout_loss_weighted': 'Rollout', 'rollout_loss': 'Rollout'}

    # Get available losses in order
    available_losses = []
    for loss_name in ['kld_loss', 'kl_loss']:
        if loss_name in loss_keys:
            available_losses.append(loss_name)
            break  # Only take first KL if both kl_loss and kld_loss exist

    for loss_name in ['nll_loss', 'ent_loss']:
        if loss_name in loss_keys:
            available_losses.append(loss_name)

    if 'rollout_loss_weighted' in loss_keys:
        available_losses.append('rollout_loss_weighted')
    elif 'rollout_loss' in loss_keys:
        available_losses.append('rollout_loss')

    ref_key = available_losses[0] if available_losses else loss_keys[0]
    recent_start = max(0, len(all_losses[ref_key]) - 100)

    # Create 2x5 subplot layout (Total + up to 4 individual losses)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Row 0: All epochs
    # Col 0: Total (all losses combined)
    ax = axes[0, 0]
    for loss_name in available_losses:
        color = colors.get(loss_name, 'black')
        title = titles.get(loss_name, loss_name)
        ax.plot(all_losses[loss_name], color=color, label=title, linewidth=2)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Total Loss - All Epochs')
    ax.legend()
    ax.grid(True)

    # Cols 1-4: Individual losses (up to 4 losses)
    for i, loss_name in enumerate(available_losses[:4]):
        ax = axes[0, i + 1]
        color = colors.get(loss_name, 'black')
        title = titles.get(loss_name, loss_name)
        ax.plot(all_losses[loss_name], color=color, linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title(f'{title} Loss - All Epochs')
        ax.grid(True)

    # Turn off unused subplots in row 0
    for i in range(min(len(available_losses), 4) + 1, 5):
        axes[0, i].axis('off')

    # Row 1: Recent 100 epochs
    # Col 0: Total (recent)
    ax = axes[1, 0]
    for loss_name in available_losses:
        color = colors.get(loss_name, 'black')
        title = titles.get(loss_name, loss_name)
        recent_vals = all_losses[loss_name][recent_start:]
        ax.plot(range(recent_start, len(all_losses[loss_name])), recent_vals,
               color=color, label=title, linewidth=2)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Total Loss - Recent 100 Epochs')
    ax.legend()
    ax.grid(True)

    # Cols 1-4: Individual losses (recent)
    for i, loss_name in enumerate(available_losses[:4]):
        ax = axes[1, i + 1]
        color = colors.get(loss_name, 'black')
        title = titles.get(loss_name, loss_name)
        recent_vals = all_losses[loss_name][recent_start:]
        ax.plot(range(recent_start, len(all_losses[loss_name])), recent_vals,
               color=color, linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title(f'{title} Loss - Recent 100 Epochs')
        ax.grid(True)

    # Turn off unused subplots in row 1
    for i in range(min(len(available_losses), 4) + 1, 5):
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"loss_{model_name}.png"))
    plt.close()


def train_model(config: Dict[str, Any]) -> nn.Module:


    # ---------------- 1) get config ----------------
    # Check if this is resume mode
    is_resume = config.get('resume', False)

    if is_resume:
        # Resume mode: use existing save_path and info
        save_path = config['resume_path']
        start_epoch = config['resume_epoch']

        # Ensure subdirectories exist
        os.makedirs(os.path.join(save_path, "anl"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "post"), exist_ok=True)
    else:
        # New training mode: create new timestamped directory
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = os.path.join("generative_ts/saves", timestamp)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "anl"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "post"), exist_ok=True)
        start_epoch = 0
        best_loss = float('inf')
    
    # 1-2) save config (exclude non-serializable resume data)
    config_to_save = {k: v for k, v in config.items() if not k.startswith('resume_') and k != 'loaded_model'}
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    if 'model' in config and 'config_path' in config['model']:
        yaml_config_path = config['model']['config_path']
        if os.path.exists(yaml_config_path):
            import shutil
            shutil.copy2(yaml_config_path, os.path.join(save_path, "ls4_config.yaml"))
    
    # ---------------- 2) make model ----------------
    model_type = config['model_type']
    model_config = config['model']
    train_config = config['train']

    if is_resume:
        # Resume mode: Use already loaded model from config
        print(f"Using model from checkpoint")
        model = config['loaded_model']
    else:
        # New training mode: Create new model
        if model_type == 'VRNN':
            from .models.vrnn import VRNN_ts
            model = VRNN_ts(**model_config).to(DEVICE)

        elif model_type == 'LS4':
            from .models.ls4 import LS4_ts, dict2attr
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

        elif model_type == 'LatentODE':
            from .models.latent_ode import LatentODE_ts

            class Config:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            model = LatentODE_ts(Config(**model_config)).to(DEVICE)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # dataset_path is required (move outside of else block)
    dataset_path = config.get('dataset_path')
    if dataset_path is None:
        raise ValueError("dataset_path must be specified in config")
    
    if is_resume:
        all_losses = config.get('resume_losses', [])
        model = train(model, train_config, save_path, model_type, dataset_path,
                    start_epoch=start_epoch, all_losses=all_losses)
    else:
        model = train(model, train_config, save_path, model_type, dataset_path)

    print(f"Training completed! Results saved to: {save_path}")
    return model
