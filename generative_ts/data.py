import numpy as np
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from typing import Tuple, Dict


class GP():

    def __init__(self, **kwargs): # seed
        self.set_params(**kwargs)
        self._theta_fixed = None  # Store true theta_fixed from data generation

    def set_params(self, **kwargs):
        prms = ["T", "std_Y", "v", "tau", "sigma_f"]
        self.T, self.std_Y, self.v, self.tau, self.sigma_f = [kwargs[prm] for prm in prms]
    
    def data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        # e_t + θ^fixed + θ^GP_t
        noise_Y = np.random.normal(loc=0.0, scale=self.std_Y, size=self.T)

        theta_fixed = np.random.normal(loc=0.0, scale=self.v)
        self._theta_fixed = theta_fixed  # Store for potential use in posterior

        theta_gp = GaussianProcessRegressor(kernel=C(self.sigma_f**2) * RBF(length_scale=self.tau), alpha=1e-10).sample_y(np.arange(self.T).reshape(-1, 1), random_state=None).flatten()

        return (
                    (theta_fixed + theta_gp + noise_Y).reshape(self.T, 1),
                    {
                        r"\epsilon_{a,t}"         : noise_Y,
                        r"\theta^{\text{fixed}}_{a}" : theta_fixed,
                        r"\theta^{\text{GP}}_{a,t}"   : theta_gp
                    }
                )

    def posterior(self, x_past, T=None, N=100, use_true_theta_fixed=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        input : x_past - observed data up to t_0
        output : mu_future, sigma_future, x_future samples
        
        Correctly handles the model: Y_t = θ^fixed + θ^GP_t + ε_t
        
        Args:
            use_true_theta_fixed (bool): If True, use the true θ^fixed from data generation
                                        If False (default), estimate θ^fixed from observed data
        '''

        if T == None:    T = self.T
        if isinstance(x_past, torch.Tensor):    x_past = x_past.detach().cpu().numpy()
        if x_past.ndim == 3:    x_past = x_past.squeeze(1) # (T_x, 1, D) → (T_x, D)
        
        # CRITICAL FIX: Ensure x_past is 1D for sklearn GP
        if x_past.ndim == 2:
            x_past = x_past.flatten()  # (T_x, 1) → (T_x,)

        # Choose θ^fixed: true value or estimate from data
        if use_true_theta_fixed:
            if self._theta_fixed is None:
                raise ValueError("True theta_fixed not available. Call data() first or set use_true_theta_fixed=False")
            theta_fixed_est = self._theta_fixed
        else:
            # MATHEMATICAL FIX: Estimate θ^fixed from observed data
            theta_fixed_est = np.mean(x_past)
        
        # Detrend: remove θ^fixed to get GP component + noise
        x_past_detrended = x_past - theta_fixed_est

        kernel = C(self.sigma_f**2) * RBF(length_scale=self.tau)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=self.std_Y**2)

        # Fit GP to detrended data (should regress to 0)
        gp.fit(np.arange(len(x_past)).reshape(-1, 1), x_past_detrended)

        # Predict GP component for future
        mu_GP_future, sigma_GP_future = gp.predict(np.arange(len(x_past), T).reshape(-1, 1), return_std=True)
        
        # Add back θ^fixed for final prediction (regresses to θ^fixed, not 0)
        mu_future = mu_GP_future + theta_fixed_est

        sigma_future = np.sqrt(sigma_GP_future**2 + self.std_Y**2)

        # Sample future values: GP samples + θ^fixed
        x_GP_future = gp.sample_y(np.arange(len(x_past), T).reshape(-1, 1), N).T
        x_future = x_GP_future + theta_fixed_est

        # (T-T_0), (T-T_0), (N, T-T_0)
        return mu_future, sigma_future, x_future
