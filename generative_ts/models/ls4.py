import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

# LS4 imports for overrides
from ls4.models.ls4 import VAE as VAE_ls4
from ls4.models.ls4 import Decoder as Decoder_ls4


class AttrDict(dict):
    
    def __getattr__(self, k):
        try:    
            return self[k]
        except KeyError:
            raise AttributeError(f"No attribute '{k}'")

    def __setattr__(self, k, v):
        self[k] = v


def dict2attr(d):
    if isinstance(d, dict):
        return AttrDict({k: dict2attr(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict2attr(v) for v in d]
    else:   
        return d


# ================================================================================
#                                   LS4 for TS
# ================================================================================

class Decoder_ts(Decoder_ls4):
    """
    p( x_t | z_<=t ) = N( x_t, std_Y^2 )
    """
    
    def __init__(self, config, sigma, z_dim, in_channels, bidirectional):
        super().__init__(config, sigma, z_dim, in_channels, bidirectional)
        
        # Override sigmoid activation to identity
        self.act = nn.Identity()
        
        if z_dim != in_channels:
            self.projection = nn.Linear(z_dim, in_channels)
        else:
            self.projection = None

    def decode(self, z, x, t_vec):  
        if self.projection is not None:
            z_mean = self.projection(z)
        else:
            z_mean = z
        
        std = self.sigma * torch.ones_like(z_mean)
        return z_mean, std

    def reconstruct(self, x, t_vec, z_post, z_post_back=None):
        x_input = torch.cat([self.x_prior[None, None].expand(x.shape[0], -1, -1), x[:, :-1]], dim=1)
        
        if self.bidirectional and z_post_back is not None:
            z_post_back = z_post_back.flip(1)
            x, x_std = self.decode(torch.cat([z_post, z_post_back], dim=-1), x_input, t_vec)
        else:
            x, x_std = self.decode(z_post, x_input, t_vec)
        
        return x, x_std


class LS4_ts(VAE_ls4):

    def __init__(self, config):
        if not hasattr(config, 'n_labels'):
            config.n_labels = 1 
        if not hasattr(config, 'classifier'):
            config.classifier = False 
        super().__init__(config)

        self.decoder = Decoder_ts(
            self.config.decoder,
            self.config.sigma,
            self.config.z_dim,
            self.config.in_channels,
            self.config.bidirectional,
        )

    def forward(self, x, timepoints, masks, labels=None, plot=False, sum=False):
        # Adjust masks dim : (B, T) -> (B, T, 1)
        if masks.dim() == 2:
            masks = masks.unsqueeze(-1)
        
        
        # Call parent forward with debug enabled
        result = super().forward(x, timepoints, masks, labels, plot, sum)
        
        return result

    def posterior_sample(self, x_given, T):
        """
        p( x_{1:T} | x_{1:T_0} ) : 1 sample
        Generates full sequence samples conditioned on observations x_given
        
        Uses existing LS4 methods:
        - VAE.reconstruct() for conditional prediction
        - Encoder.encode() for posterior sampling
        """
        # Handle different input shapes
        if x_given.dim() == 3:  # (T_0, 1, D)
            T_0 = x_given.size(0)
            x_given = x_given.squeeze()  # (T_0,)
        elif x_given.dim() == 2:  # (T_0, 1)
            T_0 = x_given.size(0)
            x_given = x_given.squeeze(-1)  # (T_0,)
        else:  # (T_0,)
            T_0 = x_given.size(0)
        
        device = x_given.device
        
        with torch.no_grad():
            try:
                # Setup for LS4 methods
                self.setup_rnn(mode='dense')
                
                # Prepare inputs
                x_obs = x_given.unsqueeze(0).unsqueeze(-1)  # (1, T_0, 1)
                t_vec_obs = torch.arange(T_0, dtype=torch.float32, device=device)  # (T_0,)
                t_vec_pred = torch.arange(T_0, T, dtype=torch.float32, device=device)  # (T-T_0,)
                
                if len(t_vec_pred) == 0:
                    # Only reconstruction, no prediction
                    x_recon = super().reconstruct(x_obs, t_vec_obs)  # Use parent's reconstruct
                    x_mean = x_recon.squeeze(0).cpu().numpy()  # (T_0, 1)
                    x_std = (self.config.sigma * torch.ones_like(x_recon)).squeeze(0).cpu().numpy()
                    
                    # Add noise for sampling
                    eps = np.random.randn(*x_mean.shape) * x_std
                    x_sample = x_mean + eps
                    
                    return x_mean, x_std, x_sample
                else:
                    # Reconstruction + prediction using existing method
                    x_pred = super().reconstruct(x_obs, t_vec_obs, t_vec_pred=t_vec_pred)  # Uses extrapolate internally
                    
                    # For reconstruction part, use posterior mean
                    _, z_post_mean, _ = self.encoder.encode(x_obs, t_vec_obs, use_forward=True)
                    
                    # Decode reconstruction with posterior mean
                    x_recon, _ = self.decoder.decode(z_post_mean, None, t_vec_obs)
                    
                    # Combine reconstruction + prediction
                    x_full = torch.cat([x_recon, x_pred], dim=1)  # (1, T, 1)
                    
                    x_mean = x_full.squeeze(0).cpu().numpy()  # (T, 1)
                    x_std = (self.config.sigma * torch.ones_like(x_full)).squeeze(0).cpu().numpy()
                    
                    # Add noise for sampling
                    eps = np.random.randn(*x_mean.shape) * x_std
                    x_sample = x_mean + eps
                    
                    return x_mean, x_std, x_sample
                    
            except Exception as e:
                print(f"LS4 posterior sampling failed: {e}, using fallback")
                # Fallback: simple extrapolation with noise
                x_full = torch.zeros(T, 1, device=device)
                x_full[:T_0, 0] = x_given
                
                if T_0 >= 2:
                    slope = (x_given[-1] - x_given[-2])
                    for t in range(T_0, T):
                        x_full[t, 0] = x_given[-1] + slope * (t - T_0 + 1)
                else:
                    x_full[T_0:, 0] = x_given[-1]
                
                sigma = self.config.sigma if hasattr(self.config, 'sigma') else 0.1
                x_std = torch.ones_like(x_full) * sigma
                eps = torch.randn_like(x_full) * sigma
                x_sample = x_full + eps
                
                return x_full.cpu().numpy(), x_std.cpu().numpy(), x_sample.cpu().numpy()

    def posterior(self, x_given, T, N=20, verbose=2):
        """
        p( x_<=T | x_<=T_0 ) : N sample, mean, var
        """
        T_0 = x_given.size(0)
        
        with torch.no_grad():
            samples = []
            means = []
            stds = []
            recon_samples = []
            
            for i in range(N):
                if i % 100 == 0 and verbose > 1:
                    print(f"Sampling {i}/{N}")
                
                # Single posterior sample for full sequence
                sample_mean, sample_std, sample_traj = self.posterior_sample(x_given, T)
                
                # Keep entire sequence (T, D)
                means.append(sample_mean)  # (T, D)
                stds.append(sample_std)    # (T, D) 
                samples.append(sample_traj)  # (T, D)
                
                # Extract reconstruction part from full sequence sample
                if T_0 > 0:
                    recon_samples.append(sample_traj[:T_0])  # First T_0 samples are reconstruction
            
            # Convert to arrays and compute statistics
            means = np.stack(means)      # (N, T, D)
            stds = np.stack(stds)        # (N, T, D)
            samples = np.stack(samples)  # (N, T, D)
            recon_samples = np.array(recon_samples)  # (N, T_0, D)
            
            # Bayesian uncertainty decomposition for full sequence
            mean_samples = means.mean(axis=0)                    # (T, D)
            var_alea = (stds ** 2).mean(axis=0)                  # aleatoric
            var_epis = means.var(axis=0)                         # epistemic  
            var_samples = np.sqrt(var_alea + var_epis)           # (T, D)
            
            # Reconstruction uncertainty
            recon_mean = np.mean(recon_samples, axis=0)          # (T_0, D)
            recon_std = np.std(recon_samples, axis=0)            # (T_0, D)
            
            return {
                'mean_samples': mean_samples,
                'var_samples': var_samples,
                'x_samples': samples,
                'recon_mean': recon_mean,
                'recon_std': recon_std,
                'recon_samples': recon_samples
            }
    
