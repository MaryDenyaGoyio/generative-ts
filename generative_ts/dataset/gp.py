import numpy as np
import torch
import torch.nn as nn
import json


class GP_ts():
    """
    Theoretical Gaussian Process model for computing exact KL and NLL losses
    
    Implements:
    - Prior: p(θ_t | θ_{<t}) 
    - Inference model: q(θ_t | Y_{≤t})
    - Generation model: p(Y_t | θ_t) = N(θ_t, σ²_Y)
    - Posterior: p(Y_{>t} | Y_{≤t}) - exact GP posterior conditioning
    - KL divergence: KL[q(θ_t | Y_{≤t}) || p(θ_t | θ_{<t})]
    - NLL: -log p(Y_t | θ_t)
    
    Uses Woodbury formula for efficient inversion of K + σ²_fixed·11^T type matrices
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
        
        # Extract GP parameters
        data_config = config['data']
        self.T = data_config['T']  # Sequence length
        self.tau = data_config['tau']  # Length scale
        self.v = data_config['v']  # σ²_fixed (fixed variance component)
        self.sigma_Y = data_config['std_Y']  # Observation noise std
        self.sigma_f = data_config['sigma_f']  # GP variance scale
        
        # Get theta_fixed value from config
        self.theta_fixed = config.get('theta_fixed_value', 0.0)
        
        # Precompute squared exponential kernel matrix K for all timesteps
        self.K = self._compute_kernel_matrix(np.arange(self.T))
        
    def _compute_kernel_matrix(self, t_points):
        """
        Compute squared exponential kernel matrix
        K[i,j] = σ_f² * exp(-0.5 * (t_i - t_j)² / τ²)
        """
        n = len(t_points)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                dist_sq = (t_points[i] - t_points[j]) ** 2
                K[i, j] = self.sigma_f ** 2 * np.exp(-0.5 * dist_sq / (self.tau ** 2))
        
        return K
    
    def _compute_kernel_vector(self, t, t_points):
        """
        Compute kernel vector k(t, t_points)
        k[i] = σ_f² * exp(-0.5 * (t - t_i)² / τ²)
        """
        k = np.zeros(len(t_points))
        for i, t_i in enumerate(t_points):
            dist_sq = (t - t_i) ** 2
            k[i] = self.sigma_f ** 2 * np.exp(-0.5 * dist_sq / (self.tau ** 2))
        
        return k
    
    def _woodbury_solve(self, K, v, y):
        """
        Solve (K + v·11^T)x = y using Woodbury formula
        (K + v·11^T)^{-1} = K^{-1} - (v/(1 + v·1^T K^{-1} 1)) K^{-1} 11^T K^{-1}
        
        Args:
            K: (n, n) kernel matrix
            v: scalar σ²_fixed
            y: (n,) vector to solve for
            
        Returns:
            x: solution to (K + v·11^T)x = y
        """
        n = K.shape[0]
        ones = np.ones(n)
        
        # First solve K·alpha = y and K·beta = 1
        try:
            alpha = np.linalg.solve(K, y)
            beta = np.linalg.solve(K, ones)
        except np.linalg.LinAlgError:
            # Add small regularization if K is singular
            K_reg = K + 1e-6 * np.eye(n)
            alpha = np.linalg.solve(K_reg, y)
            beta = np.linalg.solve(K_reg, ones)
        
        # Apply Woodbury formula
        denominator = 1.0 + v * np.sum(beta)
        correction = (v / denominator) * np.dot(beta, y) * beta
        
        return alpha - correction
    
    def _woodbury_quad_form(self, K, v, x):
        """
        Compute x^T (K + v·11^T)^{-1} x using Woodbury formula
        
        Args:
            K: (n, n) kernel matrix
            v: scalar σ²_fixed  
            x: (n,) vector
            
        Returns:
            scalar: x^T (K + v·11^T)^{-1} x
        """
        n = K.shape[0]
        ones = np.ones(n)
        
        # Solve K·alpha = x and K·beta = 1
        try:
            alpha = np.linalg.solve(K, x)
            beta = np.linalg.solve(K, ones)
        except np.linalg.LinAlgError:
            # Add small regularization if K is singular
            K_reg = K + 1e-6 * np.eye(n)
            alpha = np.linalg.solve(K_reg, x)
            beta = np.linalg.solve(K_reg, ones)
        
        # Compute quadratic form
        term1 = np.dot(x, alpha)
        denominator = 1.0 + v * np.sum(beta)
        term2 = (v / denominator) * (np.dot(x, beta)) ** 2
        
        return term1 - term2
    
    def prior(self, theta_past, t):
        """
        Compute exact GP prior p(θ_t | θ_{<t})
        
        Args:
            theta_past: observed latent states θ_{<t} (t-1,) or (t-1, 1)
            t: current timestep
            
        Returns:
            mean: scalar, prior mean E[θ_t | θ_{<t}]
            var: scalar, prior variance Var[θ_t | θ_{<t}]
        """
        # Handle input shape
        if isinstance(theta_past, torch.Tensor):
            theta_past = theta_past.cpu().numpy()
        
        if theta_past.ndim == 2:
            theta_past = theta_past.squeeze()  # (t-1,)
        
        n_cond = len(theta_past) if t > 0 else 0
        
        if n_cond == 0:
            # No conditioning data (t=0)
            prior_mean = self.theta_fixed
            prior_var = self.sigma_f ** 2 + self.v
        else:
            # Get conditioning latent states θ_{<t}
            # Compute covariance matrices
            K_cond = self.K[:n_cond, :n_cond]  # K_{<t,<t}
            
            # Compute kernel vector k_{<t,t}
            t_points_cond = np.arange(n_cond)
            k_t_cond = self._compute_kernel_vector(t, t_points_cond)  # k(t, 0:t-1)
            
            # Cov(θ_{<t}) = K_{<t,<t} + σ²_fixed·11^T
            # Cov(θ_t, θ_{<t}) = k_{<t,t} + σ²_fixed·1
            cov_theta_past = k_t_cond + self.v * np.ones(n_cond)
            
            # Var(θ_t) = k_{t,t} + σ²_fixed
            var_theta = self.sigma_f ** 2 + self.v  # k(t,t) = σ_f²
            
            
            # Prior mean using Woodbury formula
            # E[θ_t | θ_{<t}] = m_t + Cov(θ_t,θ_{<t})^T Cov(θ_{<t})^{-1} (θ_{<t} - m_{<t})
            m_past = self.theta_fixed * np.ones(n_cond)
            theta_centered = theta_past - m_past
            
            if t < 5:
                print(f"  theta_centered range=[{theta_centered.min():.2e}, {theta_centered.max():.2e}]")
            
            # Check condition number and use appropriate method
            cond_num = np.linalg.cond(K_cond)
            
            if cond_num > 1e6:
                # Use direct solve instead of Woodbury when ill-conditioned
                Cov_theta_cond = K_cond + self.v * np.ones((n_cond, n_cond))
                # Add small regularization for stability
                Cov_theta_cond += 1e-8 * np.eye(n_cond)
                alpha = np.linalg.solve(Cov_theta_cond, theta_centered)
                if t < 5:
                    print(f"  Using direct solve due to high condition number")
            else:
                # Use Woodbury formula to solve (K_{<t,<t} + v·11^T) alpha = (θ_{<t} - m_{<t})
                alpha = self._woodbury_solve(K_cond, self.v, theta_centered)
            
            if t < 5:
                print(f"  alpha range=[{alpha.min():.2e}, {alpha.max():.2e}]")
            
            correction = np.dot(cov_theta_past, alpha)
            prior_mean = self.theta_fixed + correction
            
            if t < 5:
                print(f"  correction={correction:.2e}, prior_mean={prior_mean:.2e}")
            
            # Prior variance using same method as mean
            if cond_num > 1e6:
                # Use direct solve for variance calculation too
                Cov_theta_cond = K_cond + self.v * np.ones((n_cond, n_cond))
                Cov_theta_cond += 1e-8 * np.eye(n_cond)
                beta = np.linalg.solve(Cov_theta_cond, cov_theta_past)
                quad_form = np.dot(cov_theta_past, beta)
            else:
                # Use Woodbury formula
                quad_form = self._woodbury_quad_form(K_cond, self.v, cov_theta_past)
            
            prior_var = var_theta - quad_form
        
        # Ensure positive variance
        prior_var = max(prior_var, 1e-6)
        
        return prior_mean, prior_var
    
    def inference(self, Y_observed, t):
        """
        Compute exact GP inference model q(θ_t | Y_{≤t})
        
        Args:
            Y_observed: observed data Y_{≤t} (t+1,) or (t+1, 1)
            t: current timestep (0-indexed, so Y_{≤t} has t+1 elements)
            
        Returns:
            mean: scalar, inference mean E[θ_t | Y_{≤t}]
            var: scalar, inference variance Var[θ_t | Y_{≤t}]
        """
        # Handle input shape
        if isinstance(Y_observed, torch.Tensor):
            Y_observed = Y_observed.cpu().numpy()
        
        if Y_observed.ndim == 2:
            Y_observed = Y_observed.squeeze()  # (t+1,)
        
        n_obs = min(len(Y_observed), t + 1)  # Use data up to time t
        
        if n_obs == 0:
            # No observations (shouldn't happen)
            infer_mean = self.theta_fixed
            infer_var = self.sigma_f ** 2 + self.v
        else:
            # Get observations Y_{≤t}
            Y_cond = Y_observed[:n_obs]
            
            # Compute covariance matrices
            K_cond = self.K[:n_obs, :n_obs]  # K for Y_{≤t}
            
            # Compute kernel vector k_{t,:t}
            t_points_cond = np.arange(n_obs)
            k_t_cond = self._compute_kernel_vector(t, t_points_cond)
            
            # Cov(θ_t, Y_{≤t}) = k_{:t} + σ²_fixed·1
            cov_theta_Y = k_t_cond + self.v * np.ones(n_obs)
            
            # Var(θ_t) = k_{t,t} + σ²_fixed
            var_theta = self.sigma_f ** 2 + self.v
            
            # For Cov(Y_{≤t}), we have K + σ²_fixed·11^T + σ²_Y·I
            # First handle K + σ²_Y·I, then use Woodbury for the 11^T term
            K_noise = K_cond + (self.sigma_Y ** 2) * np.eye(n_obs)
            
            # Inference mean using Woodbury
            m_Y = self.theta_fixed * np.ones(n_obs)
            Y_centered = Y_cond - m_Y
            
            # Solve (K + σ²_Y·I + v·11^T) alpha = (Y - m)
            alpha = self._woodbury_solve(K_noise, self.v, Y_centered)
            infer_mean = self.theta_fixed + np.dot(cov_theta_Y, alpha)
            
            # Inference variance
            quad_form = self._woodbury_quad_form(K_noise, self.v, cov_theta_Y)
            infer_var = var_theta - quad_form
        
        # Ensure positive variance
        infer_var = max(infer_var, 1e-6)
        
        return infer_mean, infer_var
    
    def compute_kl_divergence(self, Y_observed, theta_past, t):
        """
        Compute KL divergence KL[q(θ_t | Y_{≤t}) || p(θ_t | θ_{<t})]
        For Gaussians: KL = 0.5 * [log(σ²_p/σ²_q) + (σ²_q + (μ_q - μ_p)²)/σ²_p - 1]
        
        Args:
            Y_observed: observed data Y_{≤t} (t+1,)
            theta_past: latent states θ_{<t} (t,) 
            t: current timestep
            
        Returns:
            kl: scalar KL divergence
        """
        # Get inference distribution q(θ_t | Y_{≤t})
        q_mean, q_var = self.inference(Y_observed, t)
        
        # Get prior distribution p(θ_t | θ_{<t})
        p_mean, p_var = self.prior(theta_past, t)
        
        # Compute KL divergence between two Gaussians
        kl = 0.5 * (
            np.log(p_var / q_var) + 
            (q_var + (q_mean - p_mean) ** 2) / p_var - 
            1.0
        )
        
        return kl
    
    def compute_nll(self, Y_t, theta_t):
        """
        Compute negative log likelihood -log p(Y_t | θ_t)
        For Gaussian: NLL = 0.5 * [log(2π σ²_Y) + (Y_t - θ_t)²/σ²_Y]
        
        Args:
            Y_t: observed value at time t (scalar)
            theta_t: latent state at time t (scalar)
            
        Returns:
            nll: scalar negative log likelihood
        """
        # Handle tensor inputs
        if isinstance(Y_t, torch.Tensor):
            Y_t = Y_t.cpu().numpy()
        if isinstance(theta_t, torch.Tensor):
            theta_t = theta_t.cpu().numpy()
        
        # Ensure scalar
        Y_t = float(Y_t)
        theta_t = float(theta_t)
        
        # Compute NLL for Gaussian observation model
        # p(Y_t | θ_t) = N(Y_t | θ_t, σ²_Y)
        nll = 0.5 * (
            np.log(2 * np.pi * self.sigma_Y ** 2) +
            (Y_t - theta_t) ** 2 / (self.sigma_Y ** 2)
        )
        
        return nll
    
    def generation(self, theta, N_samples=1):
        """
        Generation model: p(Y_t | θ_t) = N(θ_t, σ²_Y)
        
        Args:
            theta: latent state value (scalar or array)
            N_samples: number of samples to generate
            
        Returns:
            Y_samples: generated observations
        """
        # Handle tensor inputs
        if isinstance(theta, torch.Tensor):
            theta = theta.cpu().numpy()
        
        # Ensure numpy array
        theta = np.asarray(theta)
        
        # Generate samples from N(θ_t, σ²_Y)
        if N_samples == 1:
            Y_samples = theta + np.random.normal(0, self.sigma_Y, theta.shape)
        else:
            # Multiple samples
            Y_samples = []
            for _ in range(N_samples):
                Y_sample = theta + np.random.normal(0, self.sigma_Y, theta.shape)
                Y_samples.append(Y_sample)
            Y_samples = np.array(Y_samples)
        
        return Y_samples
    
    def posterior(self, Y_given, T, N=20, verbose=2):
        """
        Exact GP posterior: p(Y_{>t} | Y_{≤t}) using the provided mathematical formulation
        
        Based on the joint distribution:
        [Y_{≤t}; Y_{>t}] ~ N([m_{≤t}; m_{>t}], [K_{≤t,≤t}+σ²I+σ²_fix·11ᵀ, K_{>t,≤t}+σ²_fix·11ᵀ; 
                                              (K_{>t,≤t}+σ²_fix·11ᵀ)ᵀ, K_{>t,>t}+σ²I+σ²_fix·11ᵀ])
        
        Args:
            Y_given: observed data Y_{≤t} (t+1,) or (t+1, 1, 1) 
            T: total sequence length for prediction
            N: number of samples (for compatibility with other models)
            verbose: verbosity level
            
        Returns:
            result: dict containing:
                - mean_samples: (T, 1) exact posterior mean for entire sequence
                - var_samples: (T, 1) exact posterior std for entire sequence  
                - x_samples: (N, T, 1) samples for entire sequence
                - recon_mean: (t+1, 1) exact reconstruction mean
                - recon_std: (t+1, 1) exact reconstruction std
                - recon_samples: (N, t+1, 1) reconstruction samples
        """
        # Handle input shapes
        if isinstance(Y_given, torch.Tensor):
            Y_given = Y_given.cpu().numpy()
        
        # Flatten to 1D if needed
        if Y_given.ndim == 3:  # (t+1, 1, 1)
            Y_given = Y_given.squeeze()
        elif Y_given.ndim == 2:  # (t+1, 1)
            Y_given = Y_given.squeeze()
        
        t_obs = len(Y_given)  # Number of observed timesteps
        
        if T <= t_obs:
            # Pure reconstruction case
            return self._exact_reconstruction(Y_given, T, N)
        else:
            # Reconstruction + extrapolation
            return self._exact_posterior_full(Y_given, T, t_obs, N, verbose)
    
    def _exact_reconstruction(self, Y_given, T, N):
        """Handle pure reconstruction case T <= t_obs"""
        # For reconstruction, we compute q(θ_t | Y_{≤t}) for each timestep
        recon_means = []
        recon_stds = []
        
        for t in range(T):
            Y_observed = Y_given[:t+1]
            theta_mean, theta_var = self.inference(Y_observed, t)
            recon_means.append(theta_mean)
            recon_stds.append(np.sqrt(theta_var))
        
        recon_means = np.array(recon_means).reshape(-1, 1)
        recon_stds = np.array(recon_stds).reshape(-1, 1)
        
        # Generate samples
        recon_samples = []
        full_samples = []
        
        for _ in range(N):
            # Sample reconstructions 
            recon_eps = np.random.normal(0, 1, recon_means.shape)
            recon_theta_samples = recon_means + recon_stds * recon_eps
            
            # Generate Y samples from p(Y|θ)
            Y_eps = np.random.normal(0, self.sigma_Y, recon_theta_samples.shape)
            recon_Y_samples = recon_theta_samples + Y_eps
            
            full_samples.append(recon_Y_samples)  # Same as recon for pure reconstruction
        
        full_samples = np.array(full_samples)    # (N, T, 1)
        
        return {
            'mean_samples': recon_means,  # (T, 1) - exact means
            'var_samples': recon_stds,    # (T, 1) - exact stds 
            'x_samples': full_samples,    # (N, T, 1) - samples
        }
    
    def _exact_posterior_full(self, Y_given, T, t_obs, N, verbose):
        """
        Compute exact GP posterior p(Y_{>t} | Y_{≤t}) using the mathematical formulation
        """
        if verbose > 0:
            print(f"Computing exact GP posterior: t_obs={t_obs}, T={T}")
        
        # Time indices
        t_given = np.arange(t_obs)      # 0, 1, ..., t_obs-1
        t_pred = np.arange(t_obs, T)    # t_obs, t_obs+1, ..., T-1
        t_all = np.arange(T)            # 0, 1, ..., T-1
        
        # Compute kernel matrices
        K_given_given = self.K[np.ix_(t_given, t_given)]    # K_{≤t,≤t}
        K_pred_pred = self.K[np.ix_(t_pred, t_pred)]        # K_{>t,>t}  
        K_pred_given = self.K[np.ix_(t_pred, t_given)]      # K_{>t,≤t}
        
        # Mean vectors (all zeros for centered GP)
        m_given = self.theta_fixed * np.ones(t_obs)
        m_pred = self.theta_fixed * np.ones(len(t_pred))
        
        # Covariance components
        ones_given = np.ones((t_obs, 1))
        ones_pred = np.ones((len(t_pred), 1))
        I_given = np.eye(t_obs)
        I_pred = np.eye(len(t_pred))
        
        # Covariance matrices according to the formula
        Cov_given = (K_given_given + 
                    self.v * ones_given @ ones_given.T + 
                    (self.sigma_Y ** 2) * I_given)  # K_{≤t,≤t} + σ²_fixed·11ᵀ + σ²_Y·I
        
        Cov_pred = (K_pred_pred + 
                   self.v * ones_pred @ ones_pred.T + 
                   (self.sigma_Y ** 2) * I_pred)    # K_{>t,>t} + σ²_fixed·11ᵀ + σ²_Y·I
        
        Cov_cross = (K_pred_given + 
                    self.v * ones_pred @ ones_given.T)  # K_{>t,≤t} + σ²_fixed·11ᵀ
        
        # Exact GP posterior conditioning
        Y_centered = Y_given - m_given
        
        try:
            # Use Woodbury formula for numerical stability if possible
            cond_num = np.linalg.cond(Cov_given)
            if cond_num > 1e10:
                print(f"Warning: High condition number {cond_num:.2e}, using regularization")
                Cov_given += 1e-6 * I_given
            
            # Solve Cov_given @ alpha = Y_centered
            alpha = np.linalg.solve(Cov_given, Y_centered)
            
            # Posterior mean: m_{>t} + Cov_cross @ alpha
            posterior_mean_pred = m_pred + Cov_cross @ alpha
            
            # Posterior covariance: Cov_pred - Cov_cross @ Cov_given^{-1} @ Cov_cross^T
            temp = np.linalg.solve(Cov_given, Cov_cross.T)
            posterior_cov_pred = Cov_pred - Cov_cross @ temp
            
            # Extract diagonal for standard deviations
            posterior_std_pred = np.sqrt(np.maximum(np.diag(posterior_cov_pred), 1e-8))
            
        except np.linalg.LinAlgError as e:
            print(f"Matrix inversion failed: {e}, using fallback")
            # Fallback to simple extrapolation
            last_val = Y_given[-1] if len(Y_given) > 0 else self.theta_fixed
            posterior_mean_pred = last_val * np.ones(len(t_pred))
            posterior_std_pred = self.sigma_Y * np.ones(len(t_pred))
        
        # Reconstruction part (exact inference)
        recon_means = []
        recon_stds = []
        
        for t in range(t_obs):
            Y_observed = Y_given[:t+1]
            theta_mean, theta_var = self.inference(Y_observed, t)
            recon_means.append([theta_mean])  # Shape: (1,)
            recon_stds.append([np.sqrt(theta_var)])  # Shape: (1,)

        # Convert to numpy arrays
        if len(recon_means) > 0:
            recon_means = np.array(recon_means)  # (t_obs, 1)
            recon_stds = np.array(recon_stds)    # (t_obs, 1)
        else:
            recon_means = np.array([]).reshape(0, 1)  # (0, 1)
            recon_stds = np.array([]).reshape(0, 1)   # (0, 1)

        # Combine reconstruction + prediction
        full_mean = np.vstack([recon_means, posterior_mean_pred.reshape(-1, 1)])  # (T, 1)
        full_std = np.vstack([recon_stds, posterior_std_pred.reshape(-1, 1)])    # (T, 1)
        
        # Generate samples
        full_samples = []
        
        for _ in range(N):
            # Sample full trajectory
            full_eps = np.random.normal(0, 1, full_mean.shape)
            full_theta_sample = full_mean + full_std * full_eps
            
            # Generate Y samples from p(Y|θ) = N(θ, σ²_Y)
            Y_eps = np.random.normal(0, self.sigma_Y, full_theta_sample.shape)
            full_Y_sample = full_theta_sample + Y_eps
            
            full_samples.append(full_Y_sample)
        
        full_samples = np.array(full_samples)    # (N, T, 1)
        
        if verbose > 0:
            print(f"✅ Exact posterior computed: recon [{recon_means.min():.2f}, {recon_means.max():.2f}], "
                  f"pred [{posterior_mean_pred.min():.2f}, {posterior_mean_pred.max():.2f}]")
        
        return {
            'mean_samples': full_mean,      # (T, 1) - exact means
            'var_samples': full_std,        # (T, 1) - exact stds
            'x_samples': full_samples,      # (N, T, 1) - samples
        }
    
    def compute_elbo_terms(self, Y_sequence, return_trajectory=False):
        """
        Compute ELBO terms (NLL and KL) for entire sequence
        ELBO = -Σ_t [NLL_t + KL_t]
        
        Args:
            Y_sequence: observed sequence (T,) or (T, 1)
            return_trajectory: if True, return per-timestep values
            
        Returns:
            if return_trajectory:
                nll_trajectory: (T,) NLL at each timestep
                kl_trajectory: (T,) KL at each timestep
                theta_trajectory: (T,) inferred θ at each timestep
            else:
                total_nll: scalar sum of NLL
                total_kl: scalar sum of KL
        """
        # Handle input shape
        if isinstance(Y_sequence, torch.Tensor):
            Y_sequence = Y_sequence.cpu().numpy()
        if Y_sequence.ndim == 2:
            Y_sequence = Y_sequence.squeeze()
        
        T = len(Y_sequence)
        nll_values = np.zeros(T)
        kl_values = np.zeros(T)
        theta_values = np.zeros(T)
        
        for t in range(T):
            # Get inference distribution q(θ_t | Y_{≤t})
            Y_observed = Y_sequence[:t+1]
            theta_mean_q, _ = self.inference(Y_observed, t)
            theta_values[t] = theta_mean_q
            
            # Compute NLL: -log p(Y_t | θ_t) using mean of q
            nll_values[t] = self.compute_nll(Y_sequence[t], theta_mean_q)
            
            # Compute KL: KL[q(θ_t | Y_{≤t}) || p(θ_t | θ_{<t})]
            theta_past = theta_values[:t] if t > 0 else np.array([])
            kl_values[t] = self.compute_kl_divergence(Y_observed, theta_past, t)
        
        if return_trajectory:
            return nll_values, kl_values, theta_values
        else:
            return np.sum(nll_values), np.sum(kl_values)