import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import sys
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add latent_ode to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
latent_ode_path = os.path.join(project_root, 'latent_ode')
if latent_ode_path not in sys.path:
    sys.path.append(latent_ode_path)

# Import the original LatentODE implementation
from latent_ode.lib.latent_ode import LatentODE
from latent_ode.lib.create_latent_ode_model import create_LatentODE_model


def create_LatentODE_ts_model(args, input_dim, z0_prior, obsrv_std, device,
                             classif_per_tp=False, n_labels=1):
    """
    Custom create function that uses identity decoder for p(x_t | z_t) = N(z_t, std_Y^2).
    Simply calls the original function and replaces the decoder.
    """
    # Use the actual input_dim from the data
    # For 1D time series data (like our AR1 dataset), input_dim should be 1
    # This gives enc_input_dim = 1 * 2 = 2, and GRU input = 20*2 + 2 = 42
    adjusted_input_dim = input_dim  # Use the actual input dimension
    model = create_LatentODE_model(
        args=args,
        input_dim=adjusted_input_dim,
        z0_prior=z0_prior,
        obsrv_std=obsrv_std,
        device=device,
        classif_per_tp=classif_per_tp,
        n_labels=n_labels
    )

    # Replace decoder with identity mapping for p(x_t | z_t) = N(z_t, std_Y^2)
    model.decoder = nn.Identity().to(device)

    return model


class LatentODE_ts(LatentODE):
    """
    LatentODE for time series with custom decoder p(x_t | z_t) = N(z_t, std_Y^2).
    Inherits from original LatentODE and overrides decoder behavior.
    """

    def __init__(self, config):
        # Extract parameters from config
        self.x_dim = getattr(config, 'x_dim', 1)
        self.z_dim = getattr(config, 'z_dim', 2)
        self.std_Y = getattr(config, 'std_Y', 0.1)

        # LatentODE specific params
        rec_dims = getattr(config, 'rec_dims', 20)
        rec_layers = getattr(config, 'rec_layers', 1)
        gen_layers = getattr(config, 'gen_layers', 1)
        units = getattr(config, 'units', 100)
        gru_units = getattr(config, 'gru_units', 100)
        z0_encoder = getattr(config, 'z0_encoder', "odernn")

        # Create args object (same pattern as original)
        class Args:
            def __init__(self, z_dim):
                self.latents = z_dim
                self.rec_dims = rec_dims
                self.rec_layers = rec_layers
                self.gen_layers = gen_layers
                self.units = units
                self.gru_units = gru_units
                self.z0_encoder = z0_encoder
                self.poisson = False
                self.classif = False
                self.linear_classif = False
                self.dataset = None  # Add dataset attribute

        args = Args(self.z_dim)

        # Create z0 prior
        z0_prior = Normal(torch.zeros(self.z_dim).to(DEVICE), torch.ones(self.z_dim).to(DEVICE))

        # Use our custom create function that returns a LatentODE_ts instance
        model = create_LatentODE_ts_model(
            args=args,
            input_dim=self.x_dim,
            z0_prior=z0_prior,
            obsrv_std=self.std_Y,
            device=DEVICE,
            classif_per_tp=False,
            n_labels=1
        )

        # Initialize parent class with the created components
        super().__init__(
            input_dim=model.input_dim,
            latent_dim=model.latent_dim,
            encoder_z0=model.encoder_z0,
            decoder=model.decoder,
            diffeq_solver=model.diffeq_solver,
            z0_prior=model.z0_prior,
            device=model.device,
            obsrv_std=model.obsrv_std,
            use_binary_classif=model.use_binary_classif,
            use_poisson_proc=model.use_poisson_proc,
            linear_classifier=model.linear_classifier,
            classif_per_tp=model.classif_per_tp,
            n_labels=model.n_labels,
            train_classif_w_reconstr=model.train_classif_w_reconstr
        )

    # LatentODE doesn't use forward() method - it uses compute_all_losses() and posterior() instead

    def compute_all_losses(self, batch_dict, n_traj_samples=1, kl_coef=1.0):
        """
        Custom implementation that bypasses the parent's likelihood computation.
        """
        observed_data_with_mask = batch_dict["observed_data"]  # (B, T, 2*D)
        observed_tp = batch_dict["tp_to_predict"]
        if hasattr(observed_tp, 'to'):
            observed_tp = observed_tp.to(self.device)

        # Extract original data without mask for likelihood computation
        n_data_dims = observed_data_with_mask.size(-1) // 2
        observed_data_only = observed_data_with_mask[..., :n_data_dims]  # (B, T, D)

        # Get latent variables by encoding (uses data with mask)
        first_point_mu, first_point_std = self.encoder_z0(
            observed_data_with_mask, observed_tp,
            run_backwards=True)

        # Sample z0 using reparameterization trick
        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = means_z0 + sigma_z0 * torch.randn_like(sigma_z0)

        # KL divergence between encoder and prior
        kldiv_z0 = kl_divergence(Normal(first_point_mu, first_point_std), self.z0_prior)
        kldiv_z0 = torch.mean(kldiv_z0, (1,2))

        # Get latent trajectory using ODE
        sol_y = self.diffeq_solver(first_point_enc, observed_tp)

        # Custom likelihood: p(x_t | z_t) = N(z_t[:x_dim], std_Y^2)
        pred_y_sliced = sol_y[..., :self.x_dim]  # Take first x_dim dimensions

        # Compute likelihood using original data only
        likelihood = self.get_gaussian_likelihood(
            observed_data_only, pred_y_sliced, mask=batch_dict["observed_mask"]
        )

        # Total loss (negative ELBO)
        loss = -torch.mean(likelihood) + kl_coef * torch.mean(kldiv_z0)

        results = {
            "loss": loss,
            "likelihood": torch.mean(likelihood),
            "kl_first_p": torch.mean(kldiv_z0),
            "std_first_p": torch.mean(first_point_std),
            "mse": loss,  # placeholder
            "kl_loss": torch.mean(kldiv_z0),  # KL divergence component
            "nll_loss": -torch.mean(likelihood),  # Negative log-likelihood component
            "ce_loss": torch.tensor(0.0)  # placeholder (not used for LatentODE)
        }

        return results

    def posterior_sample(self, x_given, T):
        """
        Single trajectory sampling with custom decoder p(x_t | z_t) = N(z_t[:x_dim], std_Y^2)
        """
        if x_given.dim() == 3:
            x_given = x_given.squeeze(1)
        elif x_given.dim() == 1:
            x_given = x_given.unsqueeze(1)

        T_0 = x_given.shape[0]

        with torch.no_grad():
            # Prepare data with mask concatenated
            # LatentODE encoder expects data in format (n_traj, n_timepoints, n_dims)
            x_obs = x_given.unsqueeze(0).to(DEVICE)  # (1, T_0, x_dim)
            mask_obs = torch.ones_like(x_obs).to(DEVICE)  # (1, T_0, x_dim)
            x_obs_with_mask = torch.cat([x_obs, mask_obs], dim=-1)  # (1, T_0, 2*x_dim)

            time_obs = torch.linspace(0, float(T_0-1)/(T-1), T_0).to(DEVICE)
            time_full = torch.linspace(0, 1, T).to(DEVICE)

            # Encode initial state using data with mask
            first_point_mu, first_point_std = self.encoder_z0(
                x_obs_with_mask, time_obs, run_backwards=True)

            # Sample z0
            first_point_enc = first_point_mu + first_point_std * torch.randn_like(first_point_std)

            # Get latent trajectory using ODE
            sol_y = self.diffeq_solver(first_point_enc, time_full)

            # Extract predictions: take first x_dim dimensions
            pred_x = sol_y[..., :self.x_dim]  # (1, 1, T, x_dim)
            pred_x = pred_x.squeeze(0).squeeze(0)  # (T, x_dim)

            # Our custom decoder: p(x_t | z_t) = N(z_t[:x_dim], std_Y^2)
            means = pred_x.cpu().numpy()  # z_t[:x_dim] as mean
            stds = np.full_like(means, self.std_Y)  # fixed std_Y
            samples = np.random.normal(means, stds)

            # Use actual observations for conditioning phase
            if T_0 > 0:
                samples[:T_0] = x_given.cpu().numpy()

        return means, stds, samples

    def posterior(self, x_given, T, N=20, verbose=2):
        """
        Posterior inference with multiple samples
        """
        if x_given.dim() == 2 and x_given.shape[1] != 1:
            x_given = x_given.unsqueeze(1)

        T_0 = x_given.shape[0]
        samples = []
        means = []
        stds = []

        for i in range(N):
            if i % 100 == 0 and verbose > 1:
                pass  # Remove noisy sampling output

            sample_mean, sample_std, sample_traj = self.posterior_sample(x_given, T)
            means.append(sample_mean)
            stds.append(sample_std)
            samples.append(sample_traj)

        # Convert to arrays and compute statistics
        means = np.stack(means)      # (N, T, x_dim)
        stds = np.stack(stds)        # (N, T, x_dim)
        samples = np.stack(samples)  # (N, T, x_dim)

        # Compute posterior statistics
        mean_samples = means.mean(axis=0)
        var_alea = (stds ** 2).mean(axis=0)
        var_epis = means.var(axis=0)
        var_samples = np.sqrt(var_alea + var_epis)

        # Reconstruction statistics
        recon_samples = samples[:, :T_0, :]
        recon_mean = np.mean(recon_samples, axis=0)
        recon_std = np.std(recon_samples, axis=0)

        return {
            'mean_samples': mean_samples,
            'var_samples': var_samples,
            'x_samples': samples,
            'recon_mean': recon_mean,
            'recon_std': recon_std,
            'recon_samples': recon_samples
        }