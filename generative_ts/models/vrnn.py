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

    def __init__(self, x_dim, z_dim, h_dim, n_layers, lmbd=0, std_Y=0.01, lr_std=False, verbose=0):
        super().__init__(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers)
        
        self.lmbd = lmbd
        self.log_std_y = nn.Parameter(torch.log(torch.tensor(std_Y))) if lr_std else torch.log(torch.tensor(std_Y))
        self.verbose, self.device = verbose, DEVICE
    

    def forward(self, x):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss, nll_loss, ent_loss = 0, 0, 0
        
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=x.device)
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])
            
            # encoder q( z_t | x_t, h_t-1 )
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)
            
            # prior p( z_t | h_t-1 )
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)
            
            # decoder p( x_t | z_t, h_t-1 ) = N(z_t, std_Y^2)
            dec_mean_t = z_t 
            dec_std_t = torch.ones_like(dec_mean_t) * torch.exp(self.log_std_y)
            
            # recurrence h_t = f(h_t-1, z_t, x_t)
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            
            # losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            ent_loss += self.lmbd * self._entropy_gauss(prior_std_t)
            
            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
        
        return {
            'kld_loss': kld_loss,
            'nll_loss': nll_loss, 
            'ent_loss': ent_loss
        }
    



    def posterior_batch(self, x_given, T, N, verbose=0):

        device = x_given.device
        T_0 = x_given.size(0)

        with torch.no_grad():
            means_batch = torch.zeros(N, T, self.x_dim, device=device)
            stds_batch = torch.zeros(N, T, self.x_dim, device=device)
            samples_batch = torch.zeros(N, T, self.x_dim, device=device)

            means_z_batch = torch.zeros(N, T, self.x_dim, device=device)
            stds_z_batch = torch.zeros(N, T, self.x_dim, device=device)
            samples_z_batch = torch.zeros(N, T, self.x_dim, device=device)

            # h_0 = 0
            h_batch = torch.zeros(self.n_layers, N, self.h_dim, device=device)

            for t in range(T):

                # t <= T_0
                if t < T_0:
                    x_t = x_given[t] 
                    if x_t.dim() == 0:  # scalar
                        x_t = x_t.unsqueeze(0).unsqueeze(0).repeat(N, 1)  # (N, 1)
                    elif x_t.dim() == 1:  # (D,)
                        x_t = x_t.unsqueeze(0).repeat(N, 1)  # (N, D)
                    elif x_t.dim() == 2:  # (1, D)
                        x_t = x_t.repeat(N, 1)  # (N, D)
                    else:   x_t = x_t.unsqueeze(0).repeat(N, 1)  # fallback

                    # encoder q( z_t | x_t, h_t-1 )
                    phi_x_t = self.phi_x(x_t)
                    enc_t = self.enc(torch.cat([phi_x_t, h_batch[-1]], 1))
                    enc_mean_t = self.enc_mean(enc_t)
                    enc_std_t = self.enc_std(enc_t)

                    z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)

                    # decoder p( x_t | z_t, h_t-1 ) = N(z_t, std_Y^2)
                    x_mean = z_t  # (N, x_dim)
                    x_std = torch.ones_like(x_mean) * torch.exp(self.log_std_y)
                    x_sample = self._reparameterized_sample(x_mean, x_std)

                # T_0 < t <= T
                else:

                    # prior p( z_t | h_t-1 )
                    prior_t = self.prior(h_batch[-1])
                    prior_mean_t = self.prior_mean(prior_t)
                    prior_std_t = self.prior_std(prior_t)

                    z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)

                    # decoder p( x_t | z_t, h_t-1 ) = N(z_t, std_Y^2)
                    x_mean = z_t
                    x_std = torch.ones_like(x_mean) * torch.exp(self.log_std_y)
                    x_sample = self._reparameterized_sample(x_mean, x_std)

                    phi_x_t = self.phi_x(x_sample)

                # reccurence h_t = f(h_t-1, z_t, x_t)
                phi_z_t = self.phi_z(z_t)
                h_combined = torch.cat([phi_x_t, phi_z_t], 1)
                _, h_batch = self.rnn(h_combined.unsqueeze(0), h_batch)

                # Store
                samples_batch[:, t] = x_sample
                means_batch[:, t] = x_mean
                stds_batch[:, t] = x_std

                samples_z_batch[:, t] = z_t
                means_z_batch[:, t] = enc_mean_t if t < T_0 else prior_mean_t
                stds_z_batch[:, t] = enc_std_t if t < T_0 else prior_std_t


            # Convert to np
            means_np = means_batch.cpu().numpy()  # (N, T, D)
            stds_np = stds_batch.cpu().numpy()
            y_samples = samples_batch.cpu().numpy()

            means_z_np = means_z_batch.cpu().numpy()  # (N, T, D)
            stds_z_np = stds_z_batch.cpu().numpy()
            z_samples = samples_z_batch.cpu().numpy()

            mean_y_samples = means_np.mean(axis=0)  # (T, D)
            var_alea = (stds_np ** 2).mean(axis=0)
            var_epis = means_np.var(axis=0)
            var_y_samples = np.sqrt(var_alea + var_epis)

            mean_z_samples = means_z_np.mean(axis=0)  # (T, D)
            var_z_alea = (stds_z_np ** 2).mean(axis=0)
            var_z_epis = means_z_np.var(axis=0)
            var_z_samples = np.sqrt(var_z_alea + var_z_epis)

            return {
                'mean_samples': mean_y_samples,
                'var_samples': var_y_samples,
                'x_samples': y_samples,

                'z_mean': mean_z_samples,
                'z_std': var_z_samples,
                'z_samples': z_samples
            }



    def posterior_y(self, x_given, T, N, verbose=2):
        return self.posterior_batch(x_given, T, N, verbose)

    def posterior(self, x_given, T, N, verbose=2):
        return self.posterior_batch(x_given, T, N, verbose)


    

    def _entropy_gauss(self, std):
        return 0.5 * torch.log(2 * math.pi * math.e * std.pow(2)).sum()
    
    def _nll_gauss(self, mean, std, x):
        EPS = torch.tensor(torch.finfo(std.dtype).eps, device=std.device, dtype=std.dtype)
        pi = torch.tensor(torch.pi, device=std.device, dtype=std.dtype)
        return torch.sum(torch.log(std + EPS) + torch.log(2*pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))
