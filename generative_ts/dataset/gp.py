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

    def posterior(self, Y_given, T, N=20, mask=None, verbose=0):
        '''
        θ_t | Y_M ~ N( μ_t, Σ_t )

        General form (handles both prefix Y_1:T_0 and arbitrary mask M):
        - mask=None → M = {0, 1, ..., T_0-1} (prefix, original behavior)
        - mask=[...] → M = arbitrary indices

        μ_t = K_(1:T),M (K_M,M + r I)^{-1} y_M
        Σ_t = K_(1:T),(1:T) - K_(1:T),M (K_M,M + r I)^{-1} K_M,(1:T)

        where K includes fixed variance: K + v 11^T
        '''
        Y_arr = Y_given.detach().cpu().numpy().reshape(-1) if isinstance(Y_given, torch.Tensor) else np.asarray(Y_given).reshape(-1)

        # Parse mask
        if mask is None:
            # Original mode: prefix Y_1:T_0
            M_idx = np.arange(len(Y_arr))
        else:
            # Masked mode
            mask_arr = np.asarray(mask)
            M_idx = np.where(mask_arr)[0] if mask_arr.dtype == bool else mask_arr

        y_M = Y_arr
        if len(y_M) != len(M_idx):
            raise ValueError(f"y_obs length {len(y_M)} != mask size {len(M_idx)}")

        time_T = np.arange(T)
        time_M = M_idx

        if len(M_idx) == 0:
            # Prior only
            K_TT = self._kernel(time_T) + self.v * np.ones((T, T))
            theta_mean = np.zeros(T)
            theta_cov = K_TT
        else:
            # K_M,M + r I
            K_MM = self._kernel(time_M) + self.v * np.ones((len(M_idx), len(M_idx))) + self.sigma_Y ** 2 * np.eye(len(M_idx))

            # K_(1:T),M + v 1
            K_TM = self._kernel(time_T, time_M) + self.v * np.ones((T, len(M_idx)))

            # K_(1:T),(1:T) + v 11^T
            K_TT = self._kernel(time_T) + self.v * np.ones((T, T))

            # μ = K_TM (K_MM)^{-1} y_M
            theta_mean = (K_TM @ self._safe_solve(K_MM, y_M.reshape(-1, 1))).ravel()

            # Σ = K_TT - K_TM K_MM^{-1} K_MT
            theta_cov = K_TT - K_TM @ self._safe_solve(K_MM, K_TM.T)

        # Sample θ ~ N(μ, Σ)
        theta_samples = np.random.multivariate_normal(theta_mean, theta_cov, N)

        # Sample y ~ N(θ, σ_Y^2)
        y_samples = np.random.normal(theta_samples, self.sigma_Y, theta_samples.shape)

        return {
            "z_mean": theta_mean,
            "z_std": np.sqrt(np.diag(theta_cov)),
            "z_samples": theta_samples,
            "x_samples": y_samples[:, :, None],
            "mean_samples": theta_mean.reshape(-1, 1),
            "var_samples": np.diag(theta_cov).reshape(-1, 1),
        }