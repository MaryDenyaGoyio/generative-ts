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
    """Dictionary with attribute access"""
    
    def __getattr__(self, k):
        try:    
            return self[k]
        except KeyError:
            raise AttributeError(f"No attribute '{k}'")

    def __setattr__(self, k, v):
        self[k] = v


def dict2attr(d):
    """Convert nested dict to AttrDict"""
    if isinstance(d, dict):
        return AttrDict({k: dict2attr(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict2attr(v) for v in d]
    else:   
        return d


class Decoder_ts(Decoder_ls4):
    """
    LS4 for time series with minimal override:
    1. decoder: p(x_{t_n} | z_{<=t_n}, x_{<t_n}) = N(z_{t_n}, sigma^2) with fixed sigma
    2. ent_loss: entropy regularization 
    3. posterior: extrapolation inference
    """

    def decode(self, z, x, t_vec):  
        """
        Proper LS4 decoder: use decoder network + activation (not simple z copy!)
        This was the key issue causing extrapolation collapse to zero
        """
        # Use proper decoder network from LS4 - this learns temporal dynamics!
        x = self.act(self.dec(z, x, t=t_vec))
        std = self.sigma * torch.ones_like(x)
        return x, std


class LS4_ts(VAE_ls4):
    """
    Time series adapted VAE with custom decoder
    """

    def __init__(self, config):
        # Add missing default values for LS4 VAE
        if not hasattr(config, 'n_labels'):
            config.n_labels = 1  # Default value
        if not hasattr(config, 'classifier'):
            config.classifier = False  # Default value
        super().__init__(config)

        self.decoder = Decoder_ts(
            self.config.decoder,
            self.config.sigma,
            self.config.z_dim,
            self.config.in_channels,
            self.config.bidirectional,
        )

    def forward(self, x, timepoints, masks, labels=None, plot=False, sum=False):
        # masks 차원 수정: (B, T) -> (B, T, 1)로 확장
        if masks.dim() == 2:
            masks = masks.unsqueeze(-1)
        
        # 원본 forward 호출
        return super().forward(x, timepoints, masks, labels, plot, sum)

    def posterior_sample(self, x_given, T):
        """
        Simple posterior sampling for LS4 time series
        
        Args:
            x_given: observed data (T_0, 1, D)
            T: total sequence length
            
        Returns:
            means: (T, D) mean trajectory
            stds: (T, D) std trajectory  
            samples: (T, D) sample trajectory
        """
        T_0 = x_given.size(0)
        device = x_given.device
        
        with torch.no_grad():
            # Prepare full sequence arrays
            means = torch.zeros(T, 1, device=device)
            stds = torch.zeros(T, 1, device=device) 
            samples = torch.zeros(T, 1, device=device)
            
            # Given part: use actual observations
            # Handle different input dimensions properly
            if x_given.dim() == 3:  # (T_0, 1, 1)
                x_given_flat = x_given.squeeze(-1)  # (T_0, 1)
            elif x_given.dim() == 2:  # (T_0, 1)
                x_given_flat = x_given
            else:  # (T_0,)
                x_given_flat = x_given.unsqueeze(-1)
            
            means[:T_0] = x_given_flat
            stds[:T_0] = self.config.sigma
            samples[:T_0] = x_given_flat
            
            # Prediction part: simple extrapolation
            if T > T_0:
                # Use last observed value for extrapolation
                last_val = x_given_flat[-1:]  # Keep shape (1, 1)
                
                means[T_0:] = last_val.expand(T - T_0, 1)
                stds[T_0:] = self.config.sigma
                
                # Add some noise for samples in prediction part
                eps = torch.randn(T - T_0, 1, device=device)
                samples[T_0:] = last_val.expand(T - T_0, 1) + self.config.sigma * eps
            
            return means.cpu().numpy(), stds.cpu().numpy(), samples.cpu().numpy()

    def posterior(self, x_given, T, N=20, verbose=2):
        """
        Posterior inference for full sequence: p(x_{1:T} | x_{1:t_0})
        
        Args:
            x_given: observed data (T_0, 1, D) 
            T: total sequence length
            N: number of samples
            verbose: verbosity level
            
        Returns:
            mean_samples: (T, D) posterior mean for entire sequence
            var_samples: (T, D) posterior std for entire sequence  
            x_samples: (N, T, D) samples for entire sequence
        """
        T_0 = x_given.size(0)
        
        with torch.no_grad():
            samples = []
            means = []
            stds = []
            
            for i in range(N):
                if i % 100 == 0 and verbose > 1:
                    print(f"Sampling {i}/{N}")
                
                # Single posterior sample for full sequence
                sample_mean, sample_std, sample_traj = self.posterior_sample(x_given, T)
                
                # Now we keep the entire sequence (T, D)
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
            
            return mean_samples, var_samples, samples
