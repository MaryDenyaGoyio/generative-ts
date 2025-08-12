import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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
            dec_std_t = torch.ones_like(dec_mean_t) * torch.tensor(self.std_Y, device=dec_mean_t.device, dtype=dec_mean_t.dtype)
            
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
                # Ensure x_given[t] has the correct shape (1, x_dim)
                x_t = x_given[t].unsqueeze(0) if x_given[t].dim() == 1 else x_given[t]
                phi_x_t = self.phi_x(x_t)
                enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)
                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                
                # "Decode" using our modified decoder
                x_mean = z_t
                x_std = torch.ones_like(x_mean) * torch.tensor(self.std_Y, device=x_mean.device, dtype=x_mean.dtype)
                x_sample = x_given[t]  # Use actual observation
            else:
                # Prediction phase: use prior
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)
                z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
                
                # "Decode" using our modified decoder
                x_mean = z_t
                x_std = torch.ones_like(x_mean) * torch.tensor(self.std_Y, device=x_mean.device, dtype=x_mean.dtype)
                x_sample = self._reparameterized_sample(x_mean, x_std)
            
            # Update RNN state
            # Ensure x_sample has the correct shape (1, x_dim) for phi_x
            x_sample_t = x_sample.unsqueeze(0) if x_sample.dim() == 1 else x_sample
            phi_x_t = self.phi_x(x_sample_t)
            phi_z_t = self.phi_z(z_t)
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            
            # Store results
            means.append(x_mean.squeeze().cpu().numpy())
            stds.append(x_std.squeeze().cpu().numpy())
            samples.append(x_sample.squeeze().cpu().numpy())
        
        return np.array(means), np.array(stds), np.array(samples)


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
            
            return mean_samples, var_samples, samples


    def _entropy_gauss(self, std):
        """Gaussian entropy: 0.5 * log(2πe * σ²)"""
        return 0.5 * torch.log(2 * math.pi * math.e * std.pow(2)).sum()
    
    def _nll_gauss(self, mean, std, x):
        """Override _nll_gauss to handle EPS as tensor"""
        EPS = torch.tensor(torch.finfo(std.dtype).eps, device=std.device, dtype=std.dtype)
        pi = torch.tensor(torch.pi, device=std.device, dtype=std.dtype)
        return torch.sum(torch.log(std + EPS) + torch.log(2*pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))