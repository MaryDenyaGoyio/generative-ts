import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


def plot_posterior(model, save_path, epoch, model_name=None, dataset_path=None, fixed_sample_idx=None):
    """
    Plot posterior: p(x_{1:T} | x_{1:t_0})
    Shows full sequence posterior with data, samples, mean, and ±2σ.
    """
    if hasattr(model, 'eval'):
        model.eval()
    
    # Get model name
    if model_name is None:
        model_name = model.__class__.__name__
    
    # Load dataset - create test data if not provided
    if dataset_path:
        # Check if dataset_path is a directory (new format) or file (old format)
        if os.path.isdir(dataset_path):
            # New format: load from data.pth in the directory
            data_file = os.path.join(dataset_path, 'data.pth')
            data = torch.load(data_file, weights_only=False)
        else:
            # Old format: load directly
            data = torch.load(dataset_path, weights_only=False)
        Y_test = data['Y_test']
        t_test = data.get('t_test', None)
    else:
        # Generate simple test data for plotting
        T = 20  # Match training data length
        t = np.linspace(0, 4, T)
        y = np.sin(t) + 0.1 * np.random.randn(T)
        Y_test = [y]  # Single test sample
        t_test = None
    
    # Select sample (always use sample 0 for consistency across epochs)
    if fixed_sample_idx is not None:
        idx = fixed_sample_idx
    else:
        idx = 0  # Always use first sample for consistency
    
    y_sample = Y_test[idx]
    t_sample = t_test[idx] if t_test is not None else np.arange(len(y_sample))
    
    # Configuration - conditioning and prediction setup
    T_data = len(y_sample)  # Original data length (e.g., 300)
    t_0 = int(0.4 * T_data)  # 40% for conditioning (e.g., 120)
    T = T_data  # Total horizon = data length (300)
    N_samples = 10  # Fewer samples for faster plotting
    
    print(f"Sample {idx}: T_data={T_data}, t_0={t_0}, T={T}")
    
    # Get posterior samples
    with torch.no_grad():
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        y_given = torch.tensor(y_sample[:t_0], dtype=torch.float32, device=device)
        
        # Handle different input shapes for different models
        if y_given.dim() == 1:
            if hasattr(model, 'posterior'):  # LS4, VRNN models
                y_given = y_given.unsqueeze(-1).unsqueeze(-1)  # (t_0, 1, 1)
            else:  # GP_ts model
                pass  # Keep as (t_0,)
        
        result = model.posterior(y_given, T, N=N_samples)
    
    # Plot
    plt.figure(figsize=(12, 6))
    time_steps_conditioning = np.arange(t_0)  # Conditioning data (0 to t_0-1)
    time_steps_prediction = np.arange(t_0, T)  # Prediction region (t_0 to T-1)
    time_steps_full = np.arange(T)  # Full sequence (0 to T-1)
    
    # True data (ONLY conditioning part - what the model sees)
    plt.plot(time_steps_conditioning, y_sample[:t_0], 'b-', linewidth=2, label='Data', alpha=0.8)
    
    # Conditioning boundary (more visible)
    plt.axvline(x=t_0, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Conditioning boundary (t={t_0})')
    
    # Model posterior samples (light gray) - full sequence
    samples = result['x_samples']
    for i in range(min(5, N_samples)):  # Show only first 5 samples
        plt.plot(time_steps_full, samples[i].squeeze(), 'lightgray', alpha=0.3, linewidth=0.5)
    
    # Model posterior mean and ±2σ - full sequence
    mean_traj = result['mean_samples'].squeeze()
    std_traj = result['var_samples'].squeeze()
    
    
    # Model mean (전체 구간)
    plt.plot(time_steps_full, mean_traj, 'r-', linewidth=2, label='Model Mean')
    
    # Model ±2σ (전체 구간)
    plt.fill_between(time_steps_full, 
                     mean_traj - 2*std_traj, 
                     mean_traj + 2*std_traj, 
                     alpha=0.3, color='red', label='Model ±2σ')
    
    # Add AR1 true posterior if applicable
    if dataset_path and 'ar1' in dataset_path.lower():
        try:
            from .dataset.ar1 import AR1_ts
            
            # Load AR1 config
            if os.path.isdir(dataset_path):
                config_file = os.path.join(dataset_path, 'config.json')
            else:
                config_file = dataset_path.replace('.pth', '_config.json')
            
            if os.path.exists(config_file):
                ar1_model = AR1_ts(config_path=config_file)
                ar1_result = ar1_model.posterior(y_sample[:t_0], T, N=1, verbose=0)
                
                ar1_mean = ar1_result['mean_samples'].squeeze()
                ar1_std = ar1_result['var_samples'].squeeze()
                
                # AR1 true posterior - prediction region only (T_0 onwards)
                plt.plot(time_steps_prediction, ar1_mean[t_0:], 'g--', linewidth=2, label='AR1 True Mean', alpha=0.8)
                plt.fill_between(time_steps_prediction, 
                                ar1_mean[t_0:] - 2*ar1_std[t_0:], 
                                ar1_mean[t_0:] + 2*ar1_std[t_0:], 
                                alpha=0.2, color='green', label='AR1 True ±2σ')
                
                print("Added AR1 theoretical posterior")
        except Exception as e:
            print(f"Could not add AR1 theoretical posterior: {e}")
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Posterior plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'posterior_{model_name}_{epoch}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Posterior plot saved: {save_file}")
    

def plot_sample(model, save_path, epoch, model_name=None, dataset_path=None, fixed_sample_idx=None):
    """
    Plot sample diagnostic: θ_t | Y_{≤t} (inference) vs θ_t | θ_{<t} (prior)
    Shows KL divergence and NLL trajectories.
    """
    if hasattr(model, 'eval'):
        model.eval()
    
    # Get model name
    if model_name is None:
        model_name = model.__class__.__name__
    
    # Load test data - create test data if not provided
    if dataset_path:
        # Check if dataset_path is a directory (new format) or file (old format)
        if os.path.isdir(dataset_path):
            # New format: load from data.pth in the directory
            data_file = os.path.join(dataset_path, 'data.pth')
            data = torch.load(data_file, weights_only=False)
        else:
            # Old format: load directly
            data = torch.load(dataset_path, weights_only=False)
        Y_test = data['Y_test']
    else:
        # Generate simple test data for plotting
        T = 20  # Match training data length
        t = np.linspace(0, 4, T)
        y = np.sin(t) + 0.1 * np.random.randn(T)
        Y_test = [y]  # Single test sample
        data = {'Y_test': Y_test}
    
    # Get sigma_Y from dataset config
    if 'config' in data and 'data' in data['config']:
        sigma_Y = data['config']['data'].get('std_Y', 0.01)
    elif hasattr(model, 'sigma_Y'):
        sigma_Y = model.sigma_Y
    elif hasattr(model, 'std_Y'):
        sigma_Y = model.std_Y
    else:
        sigma_Y = 0.01  # fallback
    
    # Select sample
    if fixed_sample_idx is not None:
        idx = min(fixed_sample_idx, len(Y_test) - 1)
    else:
        idx = 0
        
    Y_sample = Y_test[idx]
    T = len(Y_sample)
    
    print(f"Using dataset sample {idx} with mean={np.mean(Y_sample):.2f}, std={np.std(Y_sample):.2f}")
    print(f"\n=== Processing {model_name} ===")
    
    # Convert to tensor
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    Y_test_np = np.array(Y_sample)
    Y_tensor = torch.tensor(Y_test_np, dtype=torch.float32, device=device)
    
    # Extract inference and prior trajectories
    with torch.no_grad():
        if hasattr(model, 'compute_elbo_terms'):
            # GP_ts model - use actual ELBO computation
            print("Using GP_ts inference/prior methods")
            
            nll_trajectory, kl_trajectory, theta_values = model.compute_elbo_terms(Y_tensor, return_trajectory=True)
            
            # Inference trajectory: θ_t | Y_{≤t}
            inference_means = theta_values.squeeze()
            inference_stds = np.full_like(inference_means, sigma_Y)
            
            # Prior trajectory: θ_t | θ_{<t}
            prior_means = np.zeros(T)
            prior_stds = np.zeros(T)
            
            for t in range(T):
                if t == 0:
                    prior_means[t] = model.theta_fixed
                    prior_stds[t] = np.sqrt(model.v)
                else:
                    theta_past = theta_values[:t]
                    prior_mean, prior_var = model.prior(theta_past, t)
                    prior_means[t] = prior_mean
                    prior_stds[t] = np.sqrt(prior_var)
            
            kl_losses = kl_trajectory
            nll_losses = nll_trajectory
            
        elif 'LS4' in model_name:
            # LS4 model - use actual forward pass to get all values
            print("Using LS4 actual forward pass")
            
            # Prepare input
            x_input = Y_tensor.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
            t_vec = torch.arange(T, dtype=torch.float32, device=device)
            masks = torch.ones(1, T, 1, device=device)
            
            try:
                # Get encoder and decoder results separately
                # LS4 forward() returns (loss, log_info), not the trajectories we need
                z_post, z_post_mean, z_post_std = model.encoder.encode(x_input, t_vec, use_forward=True)
                dec_mean, dec_std, z_prior_mean, z_prior_std = model.decoder(x_input, t_vec, z_post)
                
                # Trajectories: 
                # Inference shows q(z_t | x_{1:T}) posterior mean in latent space
                inference_means = z_post_mean.squeeze().cpu().numpy()
                inference_stds = z_post_std.squeeze().cpu().numpy()
                
                
                # Prior shows p(z_t | z_{<t}) in latent space  
                prior_means = z_prior_mean.squeeze().cpu().numpy()
                prior_stds = z_prior_std.squeeze().cpu().numpy()
                
                # Compute step-by-step losses manually
                kl_losses = []
                nll_losses = []
                
                for t in range(T):
                    # KL loss at time t: KL(q(z_t|x_{1:T}) || p(z_t|z_{<t}))
                    kl_t = model._kld_gauss(
                        z_post_mean[:, t:t+1], z_post_std[:, t:t+1], 
                        z_prior_mean[:, t:t+1], z_prior_std[:, t:t+1], 
                        masks[:, t:t+1], sum=False
                    ).squeeze().cpu().numpy()
                    kl_losses.append(kl_t)
                    
                    # NLL loss at time t: -log p(x_t | z_t)
                    nll_t = model._nll_gauss(
                        dec_mean[:, t:t+1], dec_std, 
                        x_input[:, t:t+1], 
                        masks[:, t:t+1], sum=False
                    ).squeeze().cpu().numpy()
                    nll_losses.append(nll_t)
                
                kl_losses = np.array(kl_losses)
                nll_losses = np.array(nll_losses)
                
                print(f"LS4 computation successful: KL range [{np.min(kl_losses):.2f}, {np.max(kl_losses):.2f}], NLL range [{np.min(nll_losses):.2f}, {np.max(nll_losses):.2f}]")
                
            except Exception as e:
                print(f"LS4 forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback
                inference_means = Y_test_np
                inference_stds = np.full(T, sigma_Y)
                prior_means = np.zeros(T)
                prior_stds = np.ones(T)
                kl_losses = np.zeros(T)
                nll_losses = np.full(T, 10)
                
        else:
            # VRNN or other model - use model's forward pass
            print("Using VRNN forward pass")
            
            try:
                # Prepare input for VRNN: (T, 1, D)
                x_input = Y_tensor.unsqueeze(1).unsqueeze(-1)  # (T, 1, 1)
                
                # Use VRNN's forward method
                result = model.forward(x_input)
                
                # Extract losses from forward pass result
                kl_loss_total = result['kld_loss'].item()
                nll_loss_total = result['nll_loss'].item()  
                ent_loss_total = result.get('ent_loss', torch.tensor(0.0)).item()
                
                print(f"VRNN forward successful: total KL={kl_loss_total:.2f}, total NLL={nll_loss_total:.2f}")
                
                # For trajectories, we need step-by-step processing
                # But use the model's actual methods, not manual calculations
                inference_means = np.zeros(T)
                inference_stds = np.zeros(T)
                prior_means = np.zeros(T)
                prior_stds = np.zeros(T)
                kl_losses = np.zeros(T)
                nll_losses = np.zeros(T)
                
                h = torch.zeros(model.n_layers, 1, model.h_dim, device=device)
                
                for t in range(T):
                    x_t = Y_tensor[t:t+1].unsqueeze(0)  # (1, 1)
                    phi_x_t = model.phi_x(x_t)
                    
                    # Inference q(z_t | x_t, h_{t-1})
                    enc_input = torch.cat([phi_x_t, h[-1]], 1)
                    enc_t = model.enc(enc_input)
                    enc_mean = model.enc_mean(enc_t)
                    enc_std = model.enc_std(enc_t)
                    
                    inference_means[t] = enc_mean.item()
                    inference_stds[t] = enc_std.item()
                    
                    # Prior p(z_t | h_{t-1})
                    prior_t = model.prior(h[-1])
                    prior_mean = model.prior_mean(prior_t)
                    prior_std = model.prior_std(prior_t)
                    
                    prior_means[t] = prior_mean.item()
                    prior_stds[t] = prior_std.item()
                    
                    # Use VRNN's internal loss functions (not manual calculation)
                    kl_losses[t] = model._kld_gauss(enc_mean, enc_std, prior_mean, prior_std).item()
                    nll_losses[t] = model._nll_gauss(enc_mean, enc_std, x_t).item()  # VRNN decoder: x = z
                    
                    # Update state
                    z_t = enc_mean + enc_std * torch.randn_like(enc_mean)
                    phi_z_t = model.phi_z(z_t)
                    _, h = model.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
                
            except Exception as e:
                print(f"VRNN forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback
                inference_means = Y_test_np
                inference_stds = np.full(T, sigma_Y)
                prior_means = np.zeros(T)
                prior_stds = np.ones(T)
                kl_losses = np.zeros(T)
                nll_losses = np.full(T, 10)
    
    # Create diagnostic plot
    plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 0.6, 0.6], hspace=0.4)
    
    # Top plot: trajectories
    ax1 = plt.subplot(gs[0])
    time_steps = np.arange(T)
    
    # Data as points
    ax1.plot(time_steps, Y_test_np, 'bo', markersize=2, label='Data $Y_t$', alpha=0.7)
    
    # Inference (green) - line with std band
    ax1.plot(time_steps, inference_means, 'g-', linewidth=2, 
            label=f'LS4 inference $q(\\theta_t | Y_{{\\leq T}})$')
    ax1.fill_between(time_steps, 
                     inference_means - inference_stds, 
                     inference_means + inference_stds, 
                     alpha=0.2, color='green')
    
    # Prior (red) - line with std band
    ax1.plot(time_steps, prior_means, 'r-', linewidth=2,
            label=f'LS4 prior $p(\\theta_t | \\theta_{{<t}})$')
    ax1.fill_between(time_steps, 
                     prior_means - prior_stds, 
                     prior_means + prior_stds, 
                     alpha=0.2, color='red')
    
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'{model_name} Loss analysis (Epoch {epoch})', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Combined losses plot
    ax2 = plt.subplot(gs[1])
    ax2.plot(time_steps, kl_losses, color='purple', linewidth=1.5, label=f'LS4 KL $KL(q || p)$', alpha=0.8)
    ax2.plot(time_steps, nll_losses, color='orange', linewidth=1.5, label=f'LS4 NLL $-\\log p(Y_t | \\theta_t)$', alpha=0.8)
    
    ax2.set_ylabel('Loss Value', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Individual KL plot
    ax3 = plt.subplot(gs[2])
    ax3.plot(time_steps, kl_losses, color='purple', linewidth=1, alpha=0.8)
    ax3.set_ylabel('KL', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    
    # Individual NLL plot
    ax4 = plt.subplot(gs[3])
    ax4.plot(time_steps, nll_losses, color='orange', linewidth=1, alpha=0.8)
    ax4.set_xlabel('Time step t', fontsize=12)
    ax4.set_ylabel('NLL', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'sample_{model_name}_{epoch}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()