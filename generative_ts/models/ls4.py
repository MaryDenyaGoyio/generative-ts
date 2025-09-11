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
                # Simplified approach: use the parent's reconstruct for full sequence
                # This should properly handle both reconstruction and prediction
                try:
                    # Get full sequence reconstruction + prediction 
                    t_vec_full = torch.arange(T, dtype=torch.float32, device=device)
                    x_full = super().reconstruct(x_obs, t_vec_obs, t_vec_pred=t_vec_pred)
                    
                    # x_full should be (1, T, 1) where T = T_0 + len(t_vec_pred)
                    if x_full.shape[1] != T:
                        # If shapes don't match, manually combine
                        x_recon = super().reconstruct(x_obs, t_vec_obs)  # (1, T_0, 1)
                        x_pred_only = x_full  # This is prediction part
                        x_full = torch.cat([x_recon, x_pred_only], dim=1)
                    
                    x_mean = x_full.squeeze(0).cpu().numpy()  # (T, 1)
                    
                    # Use natural uncertainty from model
                    x_std = (self.config.sigma * torch.ones_like(x_full)).squeeze(0).cpu().numpy()
                    
                    # Add noise for sampling
                    eps = np.random.randn(*x_mean.shape) * x_std
                    x_sample = x_mean + eps
                    
                    return x_mean, x_std, x_sample
                    
                except Exception as e:
                    print(f"Reconstruct failed: {e}, using fallback")
                    # Fallback: original broken method
                    x_pred = super().reconstruct(x_obs, t_vec_obs, t_vec_pred=t_vec_pred)
                    _, z_post_mean, _ = self.encoder.encode(x_obs, t_vec_obs, use_forward=True)
                    x_recon, _ = self.decoder.decode(z_post_mean, None, t_vec_obs)
                    x_full = torch.cat([x_recon, x_pred], dim=1)
                    
                    x_mean = x_full.squeeze(0).cpu().numpy()
                    x_std = (self.config.sigma * torch.ones_like(x_full)).squeeze(0).cpu().numpy()
                    eps = np.random.randn(*x_mean.shape) * x_std
                    x_sample = x_mean + eps
                    
                    return x_mean, x_std, x_sample
                    

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
                

            # Convert to arrays and compute statistics
            means = np.stack(means)      # (N, T, D)
            stds = np.stack(stds)        # (N, T, D)
            samples = np.stack(samples)  # (N, T, D)
            
            # Bayesian uncertainty decomposition for full sequence
            mean_samples = means.mean(axis=0)                    # (T, D)
            var_alea = (stds ** 2).mean(axis=0)                  # aleatoric
            var_epis = means.var(axis=0)                         # epistemic  
            var_samples = np.sqrt(var_alea + var_epis)           # (T, D)
            
            return {
                'mean_samples': mean_samples,
                'var_samples': var_samples,
                'x_samples': samples,
                'recon_samples': recon_samples
            }
    
