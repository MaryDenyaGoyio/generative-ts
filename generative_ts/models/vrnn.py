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
    
    def __init__(self, x_dim, z_dim, h_dim, n_layers, lmbd=0, std_Y=1.0, verbose=0):
        # Initialize original VRNN with adjusted params
        super().__init__(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, n_layers=n_layers)
        
        # Additional parameters for our modification
        self.lmbd = lmbd  # entropy regularization weight
        # self.log_std_y = nn.Parameter(torch.log(torch.tensor(std_Y))) # learnable std_y, init to std_Y
        self.log_std_y = torch.log(torch.tensor(1.0))
        self.verbose = verbose
        self.device = DEVICE  # Add device attribute
    
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
        
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=x.device)
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
            dec_std_t = torch.ones_like(dec_mean_t) * torch.exp(self.log_std_y)
            
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

    def _rollout_segment(self, x, h, start, end, sigma, teacher=False, accumulate=False):
        loss = torch.tensor(0.0, device=x.device)
        for t in range(start, end):
            if teacher:
                x_t = x[t]
                phi_x_t = self.phi_x(x_t)
                enc_input = torch.cat([phi_x_t, h[-1]], dim=1)
                enc_hidden = self.enc(enc_input)
                enc_mean = self.enc_mean(enc_hidden)
                enc_std = self.enc_std(enc_hidden)
                z_t = enc_mean  # use posterior mean to update state deterministically
                dec_mean_t = z_t
                phi_x_input = phi_x_t
            else:
                prior_hidden = self.prior(h[-1])
                prior_mean = self.prior_mean(prior_hidden)
                prior_std = self.prior_std(prior_hidden)
                z_t = self._reparameterized_sample(prior_mean, prior_std)
                dec_mean_t = z_t
                if accumulate:
                    dec_std_t = torch.ones_like(dec_mean_t) * sigma
                    loss = loss + self._nll_gauss(dec_mean_t, dec_std_t, x[t])
                phi_x_input = self.phi_x(dec_mean_t)

            phi_z_t = self.phi_z(z_t)
            rnn_input = torch.cat([phi_x_input, phi_z_t], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)

        return loss, h

    def _compute_segments(self, T, num_segments):
        num_segments = max(1, min(int(num_segments), T))
        base = T // num_segments
        remainder = T % num_segments
        bounds = []
        start = 0
        for i in range(num_segments):
            seg_len = base + (1 if i < remainder else 0)
            end = start + seg_len
            bounds.append((start, end))
            start = end
        return bounds

    def rollout_nll(self, x, segments=4):
        """Compute additional NLL term using masked prior rollouts.

        Filtering 모델(VRNN)은 과거 은닉 상태만 이용하므로, 각 구간을
        prior rollout으로 예측한 뒤 같은 구간을 teacher-forcing으로 다시
        주입하여 이후 단계에 영향을 주도록 처리한다.
        """
        T, B, _ = x.shape
        sigma = torch.exp(self.log_std_y)
        total_loss = torch.tensor(0.0, device=x.device)
        segments_bounds = self._compute_segments(T, segments)

        h = torch.zeros(self.n_layers, B, self.h_dim, device=x.device)
        for start, end in segments_bounds:
            chunk_loss, h = self._rollout_segment(
                x, h, start, end, sigma, teacher=False, accumulate=True
            )
            total_loss = total_loss + chunk_loss
            _, h = self._rollout_segment(
                x, h, start, end, sigma, teacher=True, accumulate=False
            )

        return total_loss
    
    def posterior_sample(self, x_given, T):

        T_0 = x_given.size(0)
        
        means = []
        stds = []
        samples = []
        
        h = torch.zeros(self.n_layers, 1, self.h_dim, device=x_given.device)
        
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
                x_std = torch.ones_like(x_mean) * torch.exp(self.log_std_y)
                x_sample = x_given[t]  # Use actual observation
            else:
                # Prediction phase: use prior
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)
                z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
                
                # "Decode" using our modified decoder
                x_mean = z_t
                x_std = torch.ones_like(x_mean) * torch.exp(self.log_std_y)
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

    def _sample_reconstruction(self, x_given):
        """
        Sample reconstruction of the given observations using encoder.
        
        Args:
            x_given: observed data (T_0, 1, D)
        
        Returns:
            reconstruction sample (T_0, D)
        """
        x_given = x_given.to(self.device)
        T_0, batch_size, x_dim = x_given.size()
        
        with torch.no_grad():
            # Initialize hidden state with correct dimensions: (n_layers, batch_size, h_dim)
            h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
            reconstructions = []
            
            for t in range(T_0):
                x_t = x_given[t]  # (batch_size, D)
                
                # Encode (same as forward method)
                phi_x_t = self.phi_x(x_t)
                enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)
                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                
                # Reconstruct (decode) - use our custom decoder
                x_mean = z_t
                x_std = torch.ones_like(x_mean) * torch.exp(self.log_std_y)
                x_recon = self._reparameterized_sample(x_mean, x_std)
                
                # Update RNN state (same as forward method)
                phi_z_t = self.phi_z(z_t)
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
                
                reconstructions.append(x_recon.squeeze().cpu().numpy())
        
        return np.array(reconstructions)

    def posterior_batch(self, x_given, T, N, verbose=2):
        """
        배치 처리된 VRNN posterior sampling - 획기적 성능 개선
        """
        T_0 = x_given.size(0)
        device = x_given.device

        with torch.no_grad():
            # 배치 초기화
            means_batch = torch.zeros(N, T, self.x_dim, device=device)
            stds_batch = torch.zeros(N, T, self.x_dim, device=device)
            samples_batch = torch.zeros(N, T, self.x_dim, device=device)

            # 배치 hidden states: (n_layers, N, h_dim)
            h_batch = torch.zeros(self.n_layers, N, self.h_dim, device=device)

            for t in range(T):
                if t < T_0:
                    # Conditioning phase: 배치 encoder
                    x_t = x_given[t]  # 원래 shape
                    if x_t.dim() == 0:  # scalar
                        x_t = x_t.unsqueeze(0).unsqueeze(0).repeat(N, 1)  # (N, 1)
                    elif x_t.dim() == 1:  # (D,)
                        x_t = x_t.unsqueeze(0).repeat(N, 1)  # (N, D)
                    elif x_t.dim() == 2:  # (1, D)
                        x_t = x_t.repeat(N, 1)  # (N, D)
                    else:
                        x_t = x_t.unsqueeze(0).repeat(N, 1)  # fallback
                    phi_x_t = self.phi_x(x_t)
                    enc_t = self.enc(torch.cat([phi_x_t, h_batch[-1]], 1))
                    enc_mean_t = self.enc_mean(enc_t)
                    enc_std_t = self.enc_std(enc_t)
                    z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)

                    # 배치 decoder
                    x_mean = z_t  # (N, x_dim)
                    x_std = torch.ones_like(x_mean) * torch.exp(self.log_std_y)

                    # Conditioning: use actual observations
                    x_sample = x_given[t]
                    if x_sample.dim() == 0:  # scalar
                        x_sample = x_sample.unsqueeze(0).unsqueeze(0).repeat(N, 1)  # (N, 1)
                    elif x_sample.dim() == 1:  # (D,)
                        x_sample = x_sample.unsqueeze(0).repeat(N, 1)  # (N, D)
                    elif x_sample.dim() == 2:  # (1, D)
                        x_sample = x_sample.repeat(N, 1)  # (N, D)
                    else:
                        x_sample = x_sample.unsqueeze(0).repeat(N, 1)  # fallback

                else:
                    # Prediction phase: 배치 prior
                    prior_t = self.prior(h_batch[-1])
                    prior_mean_t = self.prior_mean(prior_t)
                    prior_std_t = self.prior_std(prior_t)
                    z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)

                    # 배치 decoder
                    x_mean = z_t
                    x_std = torch.ones_like(x_mean) * torch.exp(self.log_std_y)

                    # Sample from decoder
                    x_sample = self._reparameterized_sample(x_mean, x_std)

                # Update batch results
                means_batch[:, t] = x_mean
                stds_batch[:, t] = x_std
                samples_batch[:, t] = x_sample

                # Update batch hidden states
                phi_z_t = self.phi_z(z_t)
                phi_x_t = self.phi_x(x_sample)
                h_combined = torch.cat([phi_x_t, phi_z_t], 1)

                # RNN step for all samples in batch
                _, h_batch = self.rnn(h_combined.unsqueeze(0), h_batch)

            # Convert to numpy
            means_np = means_batch.cpu().numpy()  # (N, T, D)
            stds_np = stds_batch.cpu().numpy()
            samples_np = samples_batch.cpu().numpy()

            # Vectorized statistics
            mean_samples = means_np.mean(axis=0)  # (T, D)
            var_alea = (stds_np ** 2).mean(axis=0)
            var_epis = means_np.var(axis=0)
            var_samples = np.sqrt(var_alea + var_epis)

            # Reconstruction statistics
            recon_samples_np = samples_np[:, :T_0]  # (N, T_0, D)
            recon_mean = recon_samples_np.mean(axis=0)
            recon_std = recon_samples_np.std(axis=0)

            return {
                'mean_samples': mean_samples,
                'var_samples': var_samples,
                'x_samples': samples_np,
                'recon_mean': recon_mean,
                'recon_std': recon_std,
                'recon_samples': recon_samples_np
            }

    def posterior_sequential(self, x_given, T, N, verbose=2):
        """
        기존 순차 처리 방식 (fallback용)
        """
        T_0 = x_given.size(0)

        with torch.no_grad():
            samples = []
            means = []
            stds = []
            recon_samples = []

            for i in range(N):
                if i % 100 == 0 and verbose > 1:
                    pass  # Remove noisy sampling output

                # Single posterior sample for full sequence
                sample_mean, sample_std, sample_traj = self.posterior_sample(x_given, T)

                # Keep entire sequence (T, D)
                means.append(sample_mean)  # (T, D)
                stds.append(sample_std)    # (T, D)
                samples.append(sample_traj)  # (T, D)

                # Extract reconstruction part (conditioning on x_given)
                recon_samples.append(self._sample_reconstruction(x_given))

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

    def posterior(self, x_given, T, N, verbose=2):
        """
        Adaptive posterior: 배치 처리 우선, 실패시 순차 처리
        """
        try:
            return self.posterior_batch(x_given, T, N, verbose)
        except Exception as e:
            if verbose > 0:
                print(f"VRNN batch processing failed: {e}, using sequential fallback")
            return self.posterior_sequential(x_given, T, N, verbose)
    
    def _entropy_gauss(self, std):
        """Gaussian entropy: 0.5 * log(2πe * σ²)"""
        return 0.5 * torch.log(2 * math.pi * math.e * std.pow(2)).sum()
    
    def _nll_gauss(self, mean, std, x):
        """Override _nll_gauss to handle EPS as tensor"""
        EPS = torch.tensor(torch.finfo(std.dtype).eps, device=std.device, dtype=std.dtype)
        pi = torch.tensor(torch.pi, device=std.device, dtype=std.dtype)
        return torch.sum(torch.log(std + EPS) + torch.log(2*pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))
