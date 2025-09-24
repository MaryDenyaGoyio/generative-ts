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


def train(model, train_config, save_path, model_name, dataset_path, n_eval=10, T_0_ratio = 1/4, resume_info=None, start_epoch=0):

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

    samples_path = os.path.join(dataset_path, "samples.npy")
    if not os.path.exists(samples_path):    raise FileNotFoundError(f"Not found: {samples_path}")
    
    Y_N_full = np.load(samples_path)

    config_path = os.path.join(dataset_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            dataset_config = json.load(f)
        print(f"[Data config]: T = {dataset_config['data']['T']}, N_data = {dataset_config['n_samples']}")
    
    
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
    if resume_info is not None:
        if resume_info.get('optimizer_state') is not None:
            optimizer.load_state_dict(resume_info['optimizer_state'])
            print(f"🔄 Optimizer state restored")
        if resume_info.get('scheduler_state') is not None:
            # Create scheduler if it was used before
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
            scheduler.load_state_dict(resume_info['scheduler_state'])
            print(f"🔄 Scheduler state restored")

        # Model state is already loaded in resume_training_setup
        print(f"🔄 Model weights already loaded from checkpoint")

    test_data = Y_N[0:1].numpy()  # (1, T, D)
    x_test = torch.from_numpy(test_data.squeeze(0)).float().to(DEVICE)  # (T, D)

    T, T_0 = x_test.shape[0], int(T * T_0_ratio)
    
    
    # ---------------- 3) load model ----------------  
    base_model_name = model_name.split('_')[0]
    if base_model_name == 'LS4':
        timepoints = torch.linspace(0.0, 1.0, T, device=DEVICE)
    
    print(f"[model config]: {model_name}, num_params = {sum(p.numel() for p in model.parameters())}")
    


    # ================= 4) Training =================  
    print(f"\n===============================================================")
    print(f"=======================     Training     =======================")
    print(f"================================================================\n")
    
    all_losses = {}

    # Load existing loss history if resuming
    if resume_info is not None:
        loss_history_path = os.path.join(save_path, "loss_history.json")
        if os.path.exists(loss_history_path):
            with open(loss_history_path, 'r') as f:
                all_losses = json.load(f)
            print(f"🔄 Loaded loss history with {len(list(all_losses.values())[0]) if all_losses else 0} epochs")

    # Training loop: continue from start_epoch + 1 if resuming, 0 if new training
    actual_start = start_epoch + 1 if resume_info else start_epoch
    total_epochs = actual_start + train_config['n_epochs']
    for epoch in trange(actual_start, total_epochs, desc=f"[{model_name}]", unit="epoch", initial=actual_start, total=total_epochs):
        start = time.time()
        
        # Initialize loss entries for this epoch if needed
        current_epoch_idx = epoch - start_epoch
        for k, v in all_losses.items():
            while len(v) <= current_epoch_idx + start_epoch:
                v.append(0)
        
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

                if name.endswith("loss"):
                    if isinstance(each_loss, torch.Tensor):
                        value = (
                            each_loss.detach().cpu().item()
                            if each_loss.dim() == 0
                            else each_loss.mean().detach().cpu().item()
                        )
                    else:
                        value = each_loss  # float


                    # Initialize loss list to proper size if needed
                    if name not in all_losses:
                        all_losses[name] = [0] * (epoch + 1)
                    elif len(all_losses[name]) <= epoch:
                        all_losses[name].extend([0] * (epoch + 1 - len(all_losses[name])))
                    all_losses[name][epoch] += value
        
        elapsed = time.time() - start
        
        # Average the accumulated losses over number of batches
        for loss_name in all_losses:
            all_losses[loss_name][epoch] /= num_batches



        log_msg = f"[{epoch}/{total_epochs}] " + \
                 "\t".join(f"{k}:{v[epoch]:.2f}" for k, v in all_losses.items()) + \
                 f"  epoch: {elapsed:.2f}s"
        tqdm.write(log_msg)
        
        
        # ---------------- 5-2) Model save ----------------  

        if epoch % n_eval == 0:
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
                         dataset_path=dataset_path, N_samples=N_samples_plot)
            
            # 5-2-3) Loss analysis
            analysis_save_path = os.path.join(save_path, "anl")
            plot_sample(model, analysis_save_path, epoch, model_name=model_name, 
                       dataset_path=dataset_path)
            
            elapsed = time.time() - start
            tqdm.write(f"eval: {elapsed:.2f}s")
    
    return model


def plot_loss(all_losses, save_path, model_name):
    """
    Plot loss curves (KLD and NLL components)

    Args:
        all_losses: dictionary containing loss history
        save_path: directory to save the plot
        model_name: name of the model for filename
    """
    # Support both naming conventions for compatibility
    desired_losses = ['kl_loss', 'kld_loss', 'nll_loss']  # KL, NLL (components of total)
    loss_keys = [key for key in desired_losses if key in all_losses and len(all_losses[key]) > 0]

    if len(loss_keys) > 0:
        # 2x3 layout: Total Loss + Individual component losses
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Define colors for consistency
        colors = {'kl_loss': 'red', 'kld_loss': 'red', 'nll_loss': 'green'}
        labels = {'kl_loss': 'KL', 'kld_loss': 'KL', 'nll_loss': 'NLL'}

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


def train_model(config: Dict[str, Any]) -> nn.Module:


    # ---------------- 1) get config ----------------
    # Check if this is resume mode
    is_resume = config.get('resume', False)

    if is_resume:
        # Resume mode: use existing save_path and info
        resume_info = config['resume_info']
        save_path = resume_info['save_path']
        start_epoch = resume_info['epoch']
        best_loss = resume_info['best_loss']
        print(f"🔄 Resuming training from epoch {start_epoch}")
        print(f"📁 Using existing save path: {save_path}")

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
    
    # 1-2) save config
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
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
        # Resume mode: model is already loaded from checkpoint
        print(f"🔄 Model already loaded from checkpoint")
        # The model is already in config['resume_info'] but we need to access it differently
        # We'll create the model and then it will be loaded in the training loop setup
        pass

    if model_type == 'VRNN':
        from .models.vrnn import VRNN_ts

        if not is_resume:
            model = VRNN_ts(
                x_dim=1,
                z_dim=model_config.get('z_dim', 1),
                h_dim=model_config.get('h_dim', 10),
                n_layers=model_config.get('n_layers', 1),
                lmbd=model_config.get('lmbd', 0),
                std_Y=train_config.get('std_Y', 0.01)
            ).to(DEVICE)
        else:
            # Model structure for resume mode
            model = VRNN_ts(
                x_dim=1,
                z_dim=model_config.get('z_dim', 1),
                h_dim=model_config.get('h_dim', 10),
                n_layers=model_config.get('n_layers', 1),
                lmbd=model_config.get('lmbd', 0),
                std_Y=train_config.get('std_Y', 0.01)
            ).to(DEVICE)

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
            pass

        config_obj = Config()
        config_obj.x_dim = 1
        config_obj.z_dim = model_config.get('latent_dim', 6)
        config_obj.std_Y = model_config.get('obsrv_std', 0.01)
        config_obj.rec_dims = model_config.get('rec_dims', 20)
        config_obj.rec_layers = model_config.get('rec_layers', 1)
        config_obj.gen_layers = model_config.get('gen_layers', 1)
        config_obj.units = model_config.get('units', 100)
        config_obj.gru_units = model_config.get('gru_units', 100)
        config_obj.z0_encoder = model_config.get('z0_encoder', 'odernn')

        model = LatentODE_ts(config_obj).to(DEVICE)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # dataset_path is required
    dataset_path = config.get('dataset_path')
    if dataset_path is None:
        raise ValueError("dataset_path must be specified in config")
    
    # Train model using unified train function
    # Pass resume information if available
    if is_resume:
        model = train(model, train_config, save_path, model_type, dataset_path,
                     resume_info=resume_info, start_epoch=start_epoch)
    else:
        model = train(model, train_config, save_path, model_type, dataset_path)

    print(f"Training completed! Results saved to: {save_path}")
    return model
