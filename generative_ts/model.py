import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



"""
============================================================================================================
====================================                VRNN                 ===================================
============================================================================================================
"""

from VariationalRecurrentNeuralNetwork.model import VRNN

class VRNN_ts(VRNN):
    """
    VRNN for time series with minimal override:
    1. decoder: p(x_t | z_{<=t}, x_{<t}) = N(z_t, sigma^2) with fixed sigma
    2. ent_loss: entropy regularization 
    3. posterior: extrapolation inference
    """
    
    def __init__(self, x_dim, z_dim, h_dim, n_layers, lmbd=0, std_Y=0.1, verbose=0):
        # Initialize original VRNN with adjusted params
        super().__init__(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers)
        
        # Additional parameters for our modification
        self.lmbd = lmbd  # entropy regularization weight
        self.std_Y = std_Y  # fixed observation noise
        self.verbose = verbose
    
    def forward(self, x):
        """
        Override forward to add entropy loss and use our custom decoder
        """
        # Call original forward but modify decoder behavior
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0
        ent_loss = 0  # Our addition
        
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=DEVICE)
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])
            
            # encoder (unchanged from original)
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)
            
            # prior (unchanged from original)
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            # sampling (unchanged from original)
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)
            
            # Our modification: decoder outputs z_t with fixed sigma
            dec_mean_t = z_t  # p(x_t | z_t) = N(z_t, sigma^2)
            dec_std_t = torch.ones_like(dec_mean_t) * self.std_Y
            
            # recurrence (unchanged from original)
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            
            # losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            ent_loss += self.lmbd * self._entropy_gauss(prior_std_t)  # Our addition
            
            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
        
        # Return format compatible with train.py
        return {
            'kld_loss': kld_loss,
            'nll_loss': nll_loss, 
            'ent_loss': ent_loss
        }
    
    def posterior_sample(self, x_given, T):

        T_0 = x_given.size(0)
        
        means = []
        stds = []
        samples = []
        
        h = torch.zeros(self.n_layers, 1, self.h_dim, device=DEVICE)
        
        for t in range(T):
            if t < T_0:
                # Conditioning phase: use encoder
                phi_x_t = self.phi_x(x_given[t])
                enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)
                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                
                # "Decode" using our modified decoder
                x_mean = z_t
                x_std = torch.ones_like(x_mean) * self.std_Y
                x_sample = x_given[t]  # Use actual observation
            else:
                # Prediction phase: use prior
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)
                z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
                
                # "Decode" using our modified decoder
                x_mean = z_t
                x_std = torch.ones_like(x_mean) * self.std_Y  
                x_sample = self._reparameterized_sample(x_mean, x_std)
            
            # Update RNN state
            phi_x_t = self.phi_x(x_sample)
            phi_z_t = self.phi_z(z_t)
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            
            # Store results
            means.append(x_mean.squeeze().cpu().numpy())
            stds.append(x_std.squeeze().cpu().numpy())
            samples.append(x_sample.squeeze().cpu().numpy())
        
        return np.array(means), np.array(stds), np.array(samples)


    def posterior(self, x_given, T, N=500, verbose=2):
        """
        Posterior inference for extrapolation: p(x_{t_0+1:T} | x_{1:t_0})
        
        Args:
            x_given: observed data (T_0, 1, D) 
            T: total sequence length
            N: number of samples
            verbose: verbosity level
            
        Returns:
            mean_samples: (T-T_0, D) posterior mean
            var_samples: (T-T_0, D) posterior std
            x_samples: (N, T-T_0, D) samples
        """
        T_0 = x_given.size(0)
        
        with torch.no_grad():
            samples = []
            means = []
            stds = []
            
            for i in range(N):
                if i % 100 == 0 and verbose > 1:
                    print(f"Sampling {i}/{N}")
                
                # Single posterior sample
                sample_mean, sample_std, sample_traj = self.posterior_sample(x_given, T)
                
                # Extract future part
                future_mean = sample_mean[T_0:]  # (T-T_0, D)
                future_std = sample_std[T_0:]    # (T-T_0, D) 
                future_traj = sample_traj[T_0:]  # (T-T_0, D)
                
                means.append(future_mean)
                stds.append(future_std)  
                samples.append(future_traj)
            
            # Convert to arrays and compute statistics
            means = np.stack(means)      # (N, T-T_0, D)
            stds = np.stack(stds)        # (N, T-T_0, D)
            samples = np.stack(samples)  # (N, T-T_0, D)
            
            # Bayesian uncertainty decomposition
            mean_samples = means.mean(axis=0)                    # (T-T_0, D)
            var_alea = (stds ** 2).mean(axis=0)                  # aleatoric
            var_epis = means.var(axis=0)                         # epistemic  
            var_samples = np.sqrt(var_alea + var_epis)           # (T-T_0, D)
            
            return mean_samples, var_samples, samples


    def _entropy_gauss(self, std):
        """Gaussian entropy: 0.5 * log(2πe * σ²)"""
        return 0.5 * torch.log(2 * math.pi * math.e * std.pow(2)).sum()



"""
============================================================================================================
====================================                LS4                 ====================================
============================================================================================================
"""

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
        Single posterior sample trajectory using LS4's native extrapolation
        
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
            # Prepare inputs for LS4
            B = 1  # batch size
            x_input = x_given.unsqueeze(0)  # (1, T_0, D)
            
            # Create timepoints
            t_vec_given = torch.arange(T_0, dtype=torch.float32, device=x_given.device)
            t_vec_pred = torch.arange(T_0, T, dtype=torch.float32, device=x_given.device)
            
            # Use LS4's native reconstruct method for extrapolation
            if T > T_0:
                # Extrapolation case
                pred_mean = self.reconstruct(x_input, t_vec_given, t_vec_pred)  # (1, T-T_0, D)
                pred_mean = pred_mean.squeeze(0)  # (T-T_0, D)
                pred_std = self.config.sigma * torch.ones_like(pred_mean)
                
                # Sample from predicted distribution
                pred_sample = torch.normal(pred_mean, pred_std)
                
                # Reconstruct observed part  
                recon_mean = self.reconstruct(x_input, t_vec_given)  # (1, T_0, D)
                recon_mean = recon_mean.squeeze(0)  # (T_0, D)
                recon_std = self.config.sigma * torch.ones_like(recon_mean)
                
                # Combine observed and predicted parts
                means = torch.cat([recon_mean, pred_mean], dim=0)  # (T, D)
                stds = torch.cat([recon_std, pred_std], dim=0)    # (T, D)
                
                # Use actual observations for conditioning part, samples for prediction
                samples = torch.cat([x_given.squeeze(1), pred_sample], dim=0)  # (T, D)
                
            else:
                # Reconstruction only case
                recon_mean = self.reconstruct(x_input, t_vec_given)  # (1, T_0, D)
                means = recon_mean.squeeze(0)  # (T_0, D)
                stds = self.config.sigma * torch.ones_like(means)
                samples = x_given.squeeze(1)  # Use actual observations
            
            return means.cpu().numpy(), stds.cpu().numpy(), samples.cpu().numpy()

    def posterior(self, x_given, T, N=500, verbose=2):
        """
        Posterior inference for extrapolation: p(x_{t_0+1:T} | x_{1:t_0})
        
        Args:
            x_given: observed data (T_0, 1, D) 
            T: total sequence length
            N: number of samples
            verbose: verbosity level
            
        Returns:
            mean_samples: (T-T_0, D) posterior mean
            var_samples: (T-T_0, D) posterior std
            x_samples: (N, T-T_0, D) samples
        """
        T_0 = x_given.size(0)
        
        with torch.no_grad():
            samples = []
            means = []
            stds = []
            
            for i in range(N):
                if i % 100 == 0 and verbose > 1:
                    print(f"Sampling {i}/{N}")
                
                # Single posterior sample
                sample_mean, sample_std, sample_traj = self.posterior_sample(x_given, T)
                
                # Extract future part
                future_mean = sample_mean[T_0:]  # (T-T_0, D)
                future_std = sample_std[T_0:]    # (T-T_0, D) 
                future_traj = sample_traj[T_0:]  # (T-T_0, D)
                
                means.append(future_mean)
                stds.append(future_std)  
                samples.append(future_traj)
            
            # Convert to arrays and compute statistics
            means = np.stack(means)      # (N, T-T_0, D)
            stds = np.stack(stds)        # (N, T-T_0, D)
            samples = np.stack(samples)  # (N, T-T_0, D)
            
            # Bayesian uncertainty decomposition
            mean_samples = means.mean(axis=0)                    # (T-T_0, D)
            var_alea = (stds ** 2).mean(axis=0)                  # aleatoric
            var_epis = means.var(axis=0)                         # epistemic  
            var_samples = np.sqrt(var_alea + var_epis)           # (T-T_0, D)
            
            return mean_samples, var_samples, samples
