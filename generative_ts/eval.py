import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


def plot_posterior_y(model, save_path, epoch, model_name=None, dataset_path=None, idx=0, ratio=0.5, N_samples = 100):


    # ---------------- 0) get model, data ----------------
    if model_name is None:  model_name = model.__class__.__name__
    if hasattr(model, 'eval'):  model.eval()

    data_file = os.path.join(dataset_path, 'data.pth')
    data = torch.load(data_file, weights_only=False)
    Y_test, t_test = data['Y_test'], data.get('t_test', None)

    y_sample = Y_test[idx]
    t_sample = t_test[idx] if t_test is not None else np.arange(len(y_sample))

    T = len(y_sample)
    t_0 = int(ratio * T)

    print(f"Sample {idx}: T_data={T}, t_0={t_0}, T={T}, N={N_samples}")


    # ---------------- 1) call posterior ----------------
    with torch.no_grad():
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        y_given = torch.tensor(y_sample[:t_0], dtype=torch.float32, device=device)

        # Get data posterior model
        config_path = os.path.join(dataset_path, 'config.json')
        if dataset_path and 'ar1' in dataset_path.lower():
            from .dataset.ar1 import AR1_ts
            data_model = AR1_ts(config_path=config_path)
        elif dataset_path and 'gp' in dataset_path.lower():
            from .dataset.gp import GP_ts
            data_model = GP_ts(config_path=config_path)

        
        y_model_input = y_given.unsqueeze(-1).unsqueeze(1) if y_given.dim() == 1 else y_given.unsqueeze(1) if y_given.dim() == 2 else y_given # Model input (T_0, 1, 1)
        y_data_input = y_given.squeeze() if y_given.dim() > 1 else y_given # Data model inpur (T_0,)

        post_model = model.posterior_y(y_model_input, T, N=N_samples)
        post_data = data_model.posterior_y(y_data_input, T, N=N_samples)
    

    # ---------------- 2) make posterior plot ----------------   
    plt.figure(figsize=(12, 6))
    time_t_0 = np.arange(t_0) 
    time_t_0_to_T = np.arange(t_0, T)
    time_T = np.arange(T)
    
    # Y_{:t_0}
    plt.plot(time_t_0, y_sample[:t_0], 'b-', linewidth=2, label='Data', alpha=0.5, zorder=3)
    plt.axvline(x=t_0, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label=f'(t={t_0})')
    
    for c, post in {'r' : post_model, 'g' : post_data}.items():

        # Y'_{:T} | Y_{:t_0}
        samples = post['x_samples']
        for i in range(min(10, N_samples)):
            plt.plot(time_T, samples[i].squeeze(), c, alpha=0.2, linewidth=0.8)
        if N_samples > 0:   plt.plot([], [], c, alpha=0.2, linewidth=0.8, label=f"$Y'_{{\\leq T}} | Y_{{\\leq T_0}}$ ($N={min(10, N_samples)}$)")

        # E[Y_{:T} | Y_{:t_0}]
        mean_traj = post['mean_samples'].squeeze()
        std_traj = post['var_samples'].squeeze()

        plt.plot(time_T, mean_traj, c + '-', linewidth=2, label=f"$\\mathrm{{E}}[Y_{{\\leq T}} | Y_{{\\leq T_0}}]$")

        # Var[Y_{:T} | Y_{:t_0}]
        plt.fill_between(time_T,
                        mean_traj - 2*std_traj,
                        mean_traj + 2*std_traj,
                        alpha=0.3, color=c, label=f"$\\mathrm{{E}}[Y_{{\\leq T}} | Y_{{\\leq T_0}}] \\pm \\text{{Var}}[Y_{{\\leq T}} | Y_{{\\leq T_0}}]$")
    

    plt.xlabel('Time')
    plt.ylabel('Latent State')
    plt.title(f'Posterior plot_{model_name}_{epoch}_{N_samples}samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    try:    plt.tight_layout()
    except Exception:   pass
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'posterior_{model_name}_{epoch}_{N_samples}samples.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()




def plot_posterior(model, save_path, epoch, model_name=None, dataset_path=None, idx=0, ratio=0.5, N_samples=100):

    # ---------------- 0) get model, data ----------------
    if model_name is None:  model_name = model.__class__.__name__
    if hasattr(model, 'eval'):  model.eval()

    data_file = os.path.join(dataset_path, 'data.pth')
    data = torch.load(data_file, weights_only=False)
    Y_test = data['Y_test']
    theta_test = data.get('theta_test', None)
    t_test = data.get('t_test', None)

    y_sample = Y_test[idx]
    theta_sample = theta_test[idx] if theta_test is not None else None
    t_sample = t_test[idx] if t_test is not None else np.arange(len(y_sample))

    T = len(y_sample)
    t_0 = int(ratio * T)

    print(f"Sample {idx}: T_data={T}, t_0={t_0}, T={T}, N={N_samples}")


    # ---------------- 1) call posterior ----------------
    with torch.no_grad():
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        y_given = torch.tensor(y_sample[:t_0], dtype=torch.float32, device=device)

        # Get data posterior model
        config_path = os.path.join(dataset_path, 'config.json')
        if dataset_path and 'ar1' in dataset_path.lower():
            from .dataset.ar1 import AR1_ts
            data_model = AR1_ts(config_path=config_path)
        elif dataset_path and 'gp' in dataset_path.lower():
            from .dataset.gp import GP_ts
            data_model = GP_ts(config_path=config_path)

        # Model input shape: (T_0, 1, 1) for LS4/VRNN/LatentODE
        y_model_input = y_given.unsqueeze(-1).unsqueeze(1) if y_given.dim() == 1 else y_given.unsqueeze(1) if y_given.dim() == 2 else y_given # Model input (T_0, 1, 1)
        y_data_input = y_given.squeeze() if y_given.dim() > 1 else y_given # Data model input (T_0,)

        post_model = model.posterior(y_model_input, T, N=N_samples)
        post_data = data_model.posterior(y_data_input, T, N=N_samples)


    # ---------------- 2) make posterior plot ----------------
    plt.figure(figsize=(12, 6))
    time_t_0 = np.arange(t_0)
    time_t_0_to_T = np.arange(t_0, T)
    time_T = np.arange(T)

    # Conditioning boundary
    plt.axvline(x=t_0, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label=f'(t={t_0})')

    for c, post in {'r': post_model, 'g': post_data}.items():

        # z_{:T} | x_{:t_0}
        samples = post['z_samples']
        for i in range(min(10, N_samples)):
            plt.plot(time_T, samples[i].squeeze(), c, alpha=0.2, linewidth=0.8)
        if N_samples > 0:   plt.plot([], [], c, alpha=0.2, linewidth=0.8, label=f"$z'_{{\\leq T}} | Y_{{\\leq T_0}}$ ($N={min(10, N_samples)}$)")

        # E[z_{:T} | x_{:t_0}]
        mean_traj = post['z_mean'].squeeze()
        std_traj = post['z_std'].squeeze()

        plt.plot(time_T, mean_traj, c + '-', linewidth=2, label=f"$\\mathrm{{E}}[z_{{\\leq T}} | Y_{{\\leq T_0}}]$")

        # Var[z_{:T} | x_{:t_0}]
        plt.fill_between(time_T,
                        mean_traj - 2*std_traj,
                        mean_traj + 2*std_traj,
                        alpha=0.3, color=c, label=f"$\\mathrm{{E}}[z_{{\\leq T}} | Y_{{\\leq T_0}}] \\pm \\text{{Var}}[z_{{\\leq T}} | Y_{{\\leq T_0}}]$")

    # Real theta (cyan solid line) - only up to t_0, behind everything
    if theta_sample is not None:
        plt.plot(time_t_0, theta_sample[:t_0], 'c-', linewidth=2, label='Real theta', alpha=0.7, zorder=1)

    # Y data (blue solid line) - only up to t_0, behind everything
    plt.plot(time_t_0, y_sample[:t_0], 'b-', linewidth=1.5, label='Y data', alpha=0.5, zorder=1)

    plt.xlabel('Time')
    plt.ylabel('Latent State')
    plt.title(f'Latent Posterior plot_{model_name}_{epoch}_{N_samples}samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    try:    plt.tight_layout()
    except Exception:   pass

    # Save
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'latent_posterior_{model_name}_{epoch}_{N_samples}samples.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()




def plot_sample(model, save_path, epoch, model_name=None, dataset_path=None, sample_idx=None):
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
        pass
    
    # Get sigma_Y from dataset config
    if 'config' in data and 'data' in data['config']:
        sigma_Y = data['config']['data'].get('std_Y', 0.01)
    elif hasattr(model, 'sigma_Y'):
        sigma_Y = model.sigma_Y
    elif hasattr(model, 'log_std_y'):
        sigma_Y = torch.exp(model.log_std_y).item()
    else:
        sigma_Y = 0.01  # fallback
    
    # Select sample
    if sample_idx is not None:
        idx = min(sample_idx, len(Y_test) - 1)
    else:
        idx = 0
        
    Y_sample = Y_test[idx]
    T = len(Y_sample)
    

    # Silently process model analysis
    
    # Convert to tensor
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    Y_test_np = np.array(Y_sample)
    Y_tensor = torch.tensor(Y_test_np, dtype=torch.float32, device=device)
    
    # Extract inference and prior trajectories
    with torch.no_grad():
        if hasattr(model, 'compute_elbo_terms'):
            # GP_ts model - use actual ELBO computation
            
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

            # Decoder sigma (observation noise)
            decoder_stds = np.full(T, sigma_Y)

            kl_losses = kl_trajectory
            nll_losses = nll_trajectory
            
        elif 'LS4' in model_name:
            # LS4 model - use actual forward pass to get all values
            
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

                # Decoder sigma (for sigma comparison plot)
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'log_sigma'):
                    decoder_sigma = torch.exp(model.decoder.log_sigma).item()
                    decoder_stds = np.full(T, decoder_sigma)
                else:
                    decoder_stds = dec_std.squeeze().cpu().numpy()
                
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
                
                
            except Exception as e:
                print(f"LS4 forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback
                inference_means = Y_test_np
                inference_stds = np.full(T, sigma_Y)
                prior_means = np.zeros(T)
                prior_stds = np.ones(T)
                decoder_stds = np.full(T, sigma_Y)
                kl_losses = np.zeros(T)
                nll_losses = np.full(T, 10)
                
        elif 'LatentODE' in model_name:
            # LatentODE-specific step-by-step processing
            pass  # Using LatentODE step-by-step analysis

            try:
                # Prepare data with mask (LatentODE expects data+mask concatenated)
                x_data = Y_tensor.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
                mask_data = torch.ones_like(x_data)  # (1, T, 1)
                x_with_mask = torch.cat([x_data, mask_data], dim=-1)  # (1, T, 2)
                time_points = torch.linspace(0, 1, T, device=device)

                # Compute global KL once: KL(q(z_0|x_{1:T}) || p(z_0))
                z0_mu_full, z0_std_full = model.encoder_z0(x_with_mask, time_points, run_backwards=True)
                prior_z0_mu = torch.zeros_like(z0_mu_full)
                prior_z0_std = torch.ones_like(z0_std_full)

                from torch.distributions import Normal, kl_divergence
                q_z0_full = Normal(z0_mu_full, z0_std_full)
                p_z0_full = Normal(prior_z0_mu, prior_z0_std)
                total_kl = kl_divergence(q_z0_full, p_z0_full).mean().item()

                # Sample z_0 from full posterior for trajectory generation
                z0_sample = z0_mu_full + z0_std_full * torch.randn_like(z0_std_full)

                # Generate full trajectory z_1, ..., z_T from z_0
                sol_z = model.diffeq_solver(z0_sample, time_points)  # (1, 1, T, latent_dim)

                # Initialize arrays
                inference_means = np.zeros(T)
                inference_stds = np.zeros(T)
                prior_means = np.zeros(T)
                prior_stds = np.zeros(T)
                nll_losses = np.zeros(T)

                # KL is constant across time (global nature of LatentODE)
                kl_losses = np.full(T, total_kl / T)  # Distribute total KL evenly

                # Compute NLL for each timestep
                for t in range(T):
                    # Extract z_t
                    z_t = sol_z[0, 0, t, :]  # (latent_dim,)

                    # Visualization: inference shows z_t projected to x space
                    inference_means[t] = z_t[:model.x_dim].mean().item()
                    inference_stds[t] = z0_std_full.mean().item()

                    # Prior: z_0 prior projected through ODE to time t
                    # For visualization, show effect of prior z_0
                    prior_z0_sample = torch.randn_like(z0_sample)  # Sample from N(0,1)
                    prior_sol = model.diffeq_solver(prior_z0_sample, time_points[:t+1])
                    prior_z_t = prior_sol[0, 0, -1, :]  # Last timestep
                    prior_means[t] = prior_z_t[:model.x_dim].mean().item()
                    prior_stds[t] = 1.0  # Prior std

                    # NLL: -log p(x_t | z_t) using custom decoder p(x_t | z_t) = N(z_t[:x_dim], std_Y^2)
                    x_t_pred = z_t[:model.x_dim]  # First x_dim dimensions as mean
                    x_t_true = Y_tensor[t]  # Current observation

                    # Gaussian NLL: -log N(x_true | x_pred, std_Y^2)
                    obsrv_std = torch.exp(model.log_obsrv_std).item()
                    nll_t = 0.5 * ((x_t_true - x_t_pred) / obsrv_std) ** 2
                    nll_t += 0.5 * np.log(2 * np.pi * obsrv_std ** 2)
                    nll_losses[t] = nll_t.item()

                # Decoder sigma (learnable for LatentODE)
                decoder_stds = np.full(T, torch.exp(model.log_obsrv_std).item())

                pass  # LatentODE step-by-step analysis completed silently

            except Exception as e:
                print(f"LatentODE step-by-step analysis failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback
                inference_means = Y_test_np
                inference_stds = np.full(T, sigma_Y)
                prior_means = np.zeros(T)
                prior_stds = np.ones(T)
                decoder_stds = np.full(T, sigma_Y)
                kl_losses = np.zeros(T)
                nll_losses = np.full(T, 10)

        else:
            # VRNN or other model - use model's forward pass

            try:
                # Prepare input for VRNN: (T, 1, D)
                x_input = Y_tensor.unsqueeze(1).unsqueeze(-1)  # (T, 1, 1)

                # Use VRNN's forward method
                post = model.forward(x_input)
                
                # Extract losses from forward pass post
                kl_loss_total = post['kld_loss'].item()
                nll_loss_total = post['nll_loss'].item()  
                ent_loss_total = post.get('ent_loss', torch.tensor(0.0)).item()
                
                
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

                # Decoder sigma (learnable observation noise for VRNN)
                decoder_stds = np.full(T, torch.exp(model.log_std_y).item())
                
            except Exception as e:
                print(f"VRNN forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback
                inference_means = Y_test_np
                inference_stds = np.full(T, sigma_Y)
                prior_means = np.zeros(T)
                prior_stds = np.ones(T)
                decoder_stds = np.full(T, sigma_Y)
                kl_losses = np.zeros(T)
                nll_losses = np.full(T, 10)
    
    # Create diagnostic plot
    plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1, 0.6, 0.6, 0.8], hspace=0.4)
    
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
    ax4.set_ylabel('NLL', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=8)

    # Sigma comparison plot (new)
    ax5 = plt.subplot(gs[4])
    ax5.plot(time_steps, prior_stds, 'r-', linewidth=1.5, label='σ_prior (Prior)', alpha=0.8)
    ax5.plot(time_steps, inference_stds, 'g-', linewidth=1.5, label='σ_encoder (Inference)', alpha=0.8)
    ax5.plot(time_steps, decoder_stds, 'b-', linewidth=1.5, label='σ_decoder (Generation)', alpha=0.8)
    ax5.set_xlabel('Time step t', fontsize=12)
    ax5.set_ylabel('σ', fontsize=10)
    ax5.legend(fontsize=9, loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='both', which='major', labelsize=8)
    ax5.set_title('Component-wise Standard Deviations', fontsize=11)

    try:
        plt.tight_layout()
    except Exception:
        pass  # Ignore tight_layout warnings
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'sample_{model_name}_{epoch}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
