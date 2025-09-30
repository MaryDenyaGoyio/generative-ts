import json
import numpy as np
import torch


class GP_ts:

    def __init__(self, config_path=None, config_dict=None):
        if config_path is not None:
            with open(config_path, "r") as f:   config = json.load(f)
        elif config_dict is not None:   config = config_dict
        else:   raise ValueError("No config provided")

        data_cfg = config["data"]
        self.T = data_cfg["T"]
        self.tau = data_cfg["tau"]
        self.v = data_cfg["v"]
        self.sigma_Y = data_cfg["std_Y"]
        self.sigma_f = data_cfg["sigma_f"]

    @staticmethod
    def _safe_solve(mat, rhs):
        try:    return np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            ridge = 1e-8 * np.eye(mat.shape[0], dtype=mat.dtype)
            return np.linalg.solve(mat + ridge, rhs)

    def _kernel(self, t1, t2=None):
        t1 = np.atleast_1d(t1).astype(float)
        t2 = t1 if t2 is None else np.atleast_1d(t2).astype(float)
        diff = t1[:, None] - t2[None, :]
        return (self.sigma_f ** 2) * np.exp(-0.5 * diff ** 2 / (self.tau ** 2))

    def posterior(self, Y_given, T, N=20, verbose=0):
        '''
        θ_t | Y_<=t_0 ~ N( μ_t, μ_t )

        μ_t = m_t + (k_:t + σ_fixed^2 1)^T (K_:: + σ_fixed^2 11^T + σ_y^2 I)^{-1} (Y_<=t_0 - m_<=t_0)

        Σ_t = k_tt + σ_fixed^2 - (k_:t + σ_fixed^2 1)^T (K_:: + σ_fixed^2 11^T + σ_y^2 I)^{-1}(k_:t + σ_fixed^2 1)
        '''

        Y_T_0 = Y_given.detach().cpu().numpy().reshape(-1) if isinstance(Y_given, torch.Tensor) else np.asarray(Y_given).reshape(-1)
        t_obs = Y_T_0.size
        time_T_0, time_T = np.arange(t_obs), np.arange(T)

        if time_T_0.size == 0:
            theta_mean = np.zeros(time_T.size)
            theta_cov = self._kernel(time_T) + self.v * np.ones((time_T.size, time_T.size))
        
        else:
            # Cov(Y_<=t_0) = K_:: + σ_fixed^2 11^T + σ_y^2 I
            Cov_oo = self._kernel(time_T_0) + self.v * np.ones((time_T_0.size,time_T_0.size)) + self.sigma_Y ** 2 * np.eye(time_T_0.size)
            
            # Cov(θ_t, Y_<=t_0) = k_:t + σ_fixed^2 1
            Cov_to = self._kernel(time_T, time_T_0) + self.v * np.ones((time_T.size,time_T_0.size))

            # Var(θ_t) = k_tt + σ_fixed^2
            Cov_tt = self._kernel(time_T) + self.v* np.ones((time_T.size,time_T.size))

            # μ_t = m_t + Cov(θ_t, Y_<=t_0)^T Cov(Y_<=t_0)^{-1} (Y_<=t_0 - m)
            theta_mean = (Cov_to @ self._safe_solve(Cov_oo, Y_T_0.reshape(-1, 1))).reshape(-1)

            # Σ_t = Var(θ_t) - Cov(θ_t, Y_<=t_0)^T Cov(Y_<=t_0)^{-1} Cov(θ_t, Y_<=t_0)
            theta_cov = Cov_tt - Cov_to @ self._safe_solve(Cov_oo, Cov_to.T)

        # θ ~ N(μ, Σ)
        theta_samples = np.random.multivariate_normal(theta_mean, theta_cov, N)

        # Y ~ N(θ, σ_Y^2)
        y_samples = np.random.normal(theta_samples, self.sigma_Y, theta_samples.shape)

        return {
            "mean_samples": theta_mean.reshape(-1, 1),
            "var_samples": np.diag(theta_cov).reshape(-1, 1),
            "x_samples": y_samples[:, :, None],

            "z_samples": theta_samples,
            "z_mean": theta_mean,
            "z_std": np.sqrt(np.diag(theta_cov)),
        }