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

    def decode(self, z, x, t_vec):  # z: (B, L, D or 2D)
        # bidirectional이면 전방 절반 사용
        if self.bidirectional and z.size(-1) == 2 * self.z_dim:
            z = z[..., : self.z_dim]

        # 출력 채널과 z 차원 불일치 시 최소 보정
        if z.size(-1) != self.channels:
            if z.size(-1) > self.channels:
                z = z[..., : self.channels]
            else:
                pad = self.channels - z.size(-1)
                z = torch.nn.functional.pad(z, (0, pad))

        mean = z
        std = self.sigma * torch.ones_like(mean)
        return mean, std


class LS4_ts(VAE_ls4):
    """
    Time series adapted VAE with custom decoder
    """

    def __init__(self, config):
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
        Single posterior sample trajectory - completely unified approach
        
        Args:
            x_given: observed data (T_0, 1, D)
            T: total sequence length
            
        Returns:
            means: (T, D) mean trajectory
            stds: (T, D) std trajectory  
            samples: (T, D) sample trajectory
        """
        T_0 = x_given.size(0)
        
        with torch.no_grad():
            if T <= T_0:
                # Simple reconstruction case
                x_input = x_given.unsqueeze(0)  # (1, T_0, D)
                t_vec = torch.arange(T, dtype=torch.float32, device=x_given.device)
                
                means = self.reconstruct(x_input[:, :T], t_vec).squeeze(0)  # (T, D)
                stds = self.config.sigma * torch.ones_like(means)
                
                x_given_2d = x_given[:T].squeeze()
                if x_given_2d.dim() == 1:
                    x_given_2d = x_given_2d.unsqueeze(-1)
                samples = x_given_2d
                
            else:
                # Extrapolation: create dummy full sequence and use get_full_nll mode
                x_input = x_given.unsqueeze(0)  # (1, T_0, D)
                t_vec_given = torch.arange(T_0, dtype=torch.float32, device=x_given.device)
                t_vec_pred = torch.arange(T_0, T, dtype=torch.float32, device=x_given.device)
                
                # Create dummy full sequence for internal processing
                x_dummy_pred = torch.zeros(1, T-T_0, x_given.shape[-1], device=x_given.device)
                x_full_dummy = torch.cat([x_input, x_dummy_pred], dim=1)  # (1, T, D)
                
                # Try to get full reconstruction + extrapolation
                try:
                    # Method 1: Use get_full_nll interface
                    full_result, _ = self.reconstruct(x_input, t_vec_given, t_vec_pred, 
                                                    x_full=x_full_dummy, get_full_nll=True)
                    pred_mean = full_result.squeeze(0)  # (T-T_0, D)
                    
                    # Also get reconstruction part
                    recon_mean = self.reconstruct(x_input, t_vec_given).squeeze(0)  # (T_0, D)
                    
                    # Combine
                    means = torch.cat([recon_mean, pred_mean], dim=0)  # (T, D)
                    
                except:
                    # Method 2: Fallback to separate calls
                    pred_mean = self.reconstruct(x_input, t_vec_given, t_vec_pred).squeeze(0)  # (T-T_0, D)
                    recon_mean = self.reconstruct(x_input, t_vec_given).squeeze(0)  # (T_0, D)
                    means = torch.cat([recon_mean, pred_mean], dim=0)  # (T, D)
                
                stds = self.config.sigma * torch.ones_like(means)
                
                # Sample only the prediction part
                pred_sample = torch.normal(pred_mean, self.config.sigma * torch.ones_like(pred_mean))
                
                x_given_2d = x_given.squeeze()
                if x_given_2d.dim() == 1:
                    x_given_2d = x_given_2d.unsqueeze(-1)
                samples = torch.cat([x_given_2d, pred_sample], dim=0)  # (T, D)
            
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
