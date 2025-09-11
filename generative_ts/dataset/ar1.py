import numpy as np
import torch
import torch.nn as nn
import json
from scipy import linalg


class AR1_ts():
    """
    Theoretical AR1 model for computing exact posterior predictions
    
    AR1 process: x_t = phi * x_{t-1} + epsilon_t
    where epsilon_t ~ N(0, sigma^2)
    
    Observation model: y_t = x_t + eta_t
    where eta_t ~ N(0, std_Y^2)
    
    Implements:
    - Prior: p(x_t | x_{<t}) = N(phi * x_{t-1}, sigma^2)
    - Posterior: p(x_{>t} | y_{≤t}) using Kalman filter/smoother
    - Prediction: p(y_{>t} | y_{≤t}) with uncertainty quantification
    """
    
    def __init__(self, config_path=None, config_dict=None):
        
        # Load config from file or dict
        if config_path is not None:
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_dict is not None:
            config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        # Extract AR1 parameters
        data_config = config['data']
        self.T = data_config['T']  # Sequence length
        self.phi = data_config['phi']  # AR coefficient
        self.sigma = data_config['sigma']  # Process noise std
        self.std_Y = data_config['std_Y']  # Observation noise std
        
        # State space matrices for AR1
        # x_t = F * x_{t-1} + w_t, w_t ~ N(0, Q)
        # y_t = H * x_t + v_t, v_t ~ N(0, R)
        self.F = self.phi  # Transition matrix (scalar)
        self.H = 1.0  # Observation matrix (scalar)
        self.Q = self.sigma**2  # Process noise variance
        self.R = self.std_Y**2  # Observation noise variance
        
        # Initial state distribution (stationary)
        self.mu_0 = 0.0  # Initial mean
        self.Sigma_0 = self.sigma**2 / (1 - self.phi**2)  # Stationary variance
        
    def _kalman_filter(self, y_obs):
        """
        Forward pass: Kalman filter
        
        Args:
            y_obs: observed sequence (T_obs,)
            
        Returns:
            mu_pred: predicted means (T_obs,)
            Sigma_pred: predicted variances (T_obs,)
            mu_filt: filtered means (T_obs,)
            Sigma_filt: filtered variances (T_obs,)
        """
        T_obs = len(y_obs)
        
        # Storage
        mu_pred = np.zeros(T_obs)
        Sigma_pred = np.zeros(T_obs)
        mu_filt = np.zeros(T_obs)
        Sigma_filt = np.zeros(T_obs)
        
        # Initialize
        mu_t = self.mu_0
        Sigma_t = self.Sigma_0
        
        for t in range(T_obs):
            # Predict step
            if t == 0:
                mu_pred[t] = mu_t
                Sigma_pred[t] = Sigma_t
            else:
                mu_pred[t] = self.F * mu_filt[t-1]
                Sigma_pred[t] = self.F**2 * Sigma_filt[t-1] + self.Q
            
            # Update step
            y_pred = self.H * mu_pred[t]
            S = self.H**2 * Sigma_pred[t] + self.R  # Innovation covariance
            K = Sigma_pred[t] * self.H / S  # Kalman gain
            
            mu_filt[t] = mu_pred[t] + K * (y_obs[t] - y_pred)
            Sigma_filt[t] = Sigma_pred[t] - K * self.H * Sigma_pred[t]
        
        return mu_pred, Sigma_pred, mu_filt, Sigma_filt
    
    def _predict_future(self, mu_T, Sigma_T, T_future):
        """
        Predict future states given final filtered state
        
        Args:
            mu_T: final filtered mean
            Sigma_T: final filtered variance
            T_future: number of future steps
            
        Returns:
            mu_future: future state means (T_future,)
            Sigma_future: future state variances (T_future,)
        """
        mu_future = np.zeros(T_future)
        Sigma_future = np.zeros(T_future)
        
        mu_t = mu_T
        Sigma_t = Sigma_T
        
        for t in range(T_future):
            # Predict next state
            mu_t = self.F * mu_t
            Sigma_t = self.F**2 * Sigma_t + self.Q
            
            mu_future[t] = mu_t
            Sigma_future[t] = Sigma_t
        
        return mu_future, Sigma_future
    
    def posterior(self, y_given, T, N=20, verbose=2):
        """
        AR1 exact posterior: p( y_{1:T} | y_{1:T_0} )
        
        Args:
            y_given: observed sequence (T_0,)
            T: total sequence length for prediction
            N: number of samples (for compatibility)
            
        Returns:
            Dict with posterior statistics
        """
        if isinstance(y_given, torch.Tensor):
            y_given = y_given.cpu().numpy()
        elif isinstance(y_given, list):
            y_given = np.array(y_given)
        
        if y_given.ndim > 1:
            y_given = y_given.squeeze()
        
        T_0 = len(y_given)
        T_future = T - T_0
        
        if verbose > 1:
            print(f"AR1 posterior: T_0={T_0}, T_future={T_future}, T_total={T}")
        
        # Kalman filter on observed data
        mu_pred, Sigma_pred, mu_filt, Sigma_filt = self._kalman_filter(y_given)
        
        # Predict future if needed
        if T_future > 0:
            mu_future, Sigma_future = self._predict_future(mu_filt[-1], Sigma_filt[-1], T_future)
            
            # Combine filtered + future
            mu_full = np.concatenate([mu_filt, mu_future])
            Sigma_full = np.concatenate([Sigma_filt, Sigma_future])
        else:
            mu_full = mu_filt
            Sigma_full = Sigma_filt
        
        # Add observation noise for y predictions
        y_mean = mu_full  # E[y_t] = E[x_t]
        y_var = Sigma_full + self.R  # Var[y_t] = Var[x_t] + Var[eta_t]
        y_std = np.sqrt(y_var)
        
        # Generate samples (for compatibility with other models)
        samples = []
        for i in range(N):
            # Sample states
            x_sample = np.random.normal(mu_full, np.sqrt(Sigma_full))
            # Sample observations
            y_sample = np.random.normal(x_sample, self.std_Y)
            samples.append(y_sample.reshape(-1, 1))
        
        samples = np.stack(samples)  # (N, T, 1)
        
        # Reconstruction part
        recon_mean = y_mean[:T_0].reshape(-1, 1)
        recon_std = y_std[:T_0].reshape(-1, 1)
        recon_samples = samples[:, :T_0, :]
        
        return {
            'mean_samples': y_mean.reshape(-1, 1),  # (T, 1)
            'var_samples': y_std.reshape(-1, 1),    # (T, 1)
            'x_samples': samples,                   # (N, T, 1)
            'recon_mean': recon_mean,               # (T_0, 1)
            'recon_std': recon_std,                 # (T_0, 1)
            'recon_samples': recon_samples,         # (N, T_0, 1)
            # Additional AR1-specific outputs
            'state_mean': mu_full.reshape(-1, 1),  # State means (T, 1)
            'state_std': np.sqrt(Sigma_full).reshape(-1, 1),  # State stds (T, 1)
        }