import json
import numpy as np
import torch

class AR1_ts:
    """
    AR(1): θ_t = φ θ_{t-1} + ε_t,  ε_t ~ N(0, q),  q = σ^2
           y_t = θ_t + η_t,        η_t ~ N(0, r),  r = (std_Y)^2
    """
    def __init__(self, config_path=None, config_dict=None):
        if config_path:
            with open(config_path, 'r') as f: config = json.load(f)
        elif config_dict:
            config = config_dict
        else:
            raise ValueError("No config")

        cfg = config['data']
        self.T, self.phi, self.sigma, self.std_Y = cfg['T'], cfg['phi'], cfg['sigma'], cfg['std_Y']

        if abs(self.phi) >= 1.0: raise ValueError("|phi| < 1 required")

        self.Sigma_0 = self.sigma**2 / (1 - self.phi**2)


    def _kf_filter(self, y_obs, mask, T):
        """
        Kalman filter with arbitrary mask M

        Args:
            y_obs: (|M|,) observations
            mask: (T,) boolean or list of indices or None
            T: total length

        Returns:
            m_filt, P_filt: (T,) filtered mean/var
        """
        # Parse mask
        if mask is None:
            M_set = set(range(len(y_obs)))
        else:
            mask_arr = np.asarray(mask)
            M_set = set(np.where(mask_arr)[0].tolist()) if mask_arr.dtype == bool else set(mask_arr.tolist())

        y_M = np.asarray(y_obs).ravel()
        if len(y_M) != len(M_set):
            raise ValueError(f"y_obs length {len(y_M)} != mask size {len(M_set)}")

        # Map t -> y_t for observed times
        M_list = sorted(M_set)
        y_dict = {t: y_M[i] for i, t in enumerate(M_list)}

        m_filt, P_filt = np.zeros(T), np.zeros(T)
        m_pred, P_pred = 0.0, self.Sigma_0

        for t in range(T):
            if t in M_set:
                # Update with observation
                S = P_pred + self.std_Y**2
                K = P_pred / S
                v = y_dict[t] - m_pred

                m_filt[t] = m_pred + K * v
                P_filt[t] = (1.0 - K) * P_pred
            else:
                # No observation: skip update
                m_filt[t] = m_pred
                P_filt[t] = P_pred

            # Predict next
            m_pred = self.phi * m_filt[t]
            P_pred = self.phi**2 * P_filt[t] + self.sigma**2

        return m_filt, P_filt



    def _rts_smooth(self, m_filt, P_filt):
        """
        RTS smoother: backward pass over all t

        Returns:
            m_s, P_s, J: (T,) smoothed mean/var, (T-1,) gain
        """
        T = len(m_filt)
        m_s, P_s, J = np.zeros(T), np.zeros(T), np.zeros(T-1)

        m_s[-1], P_s[-1] = m_filt[-1], P_filt[-1]

        for t in range(T-2, -1, -1):
            # Var(θ_{t+1}|Y<=t) = φ^2 Var(θ_t|Y<=t) + q
            P_pred_next = self.phi**2 * P_filt[t] + self.sigma**2

            # J_t = Var(θ_t|Y<=t) φ (Var(θ_{t+1}|Y<=t))^{-1}
            J[t] = P_filt[t] * self.phi / P_pred_next

            # E[θ_t|Y_<=T] = E[θ_t|Y<=t] + J_t ( E[θ_{t+1}|Y_<=T] - E[θ_{t+1}|Y<=t] )
            m_s[t] = m_filt[t] + J[t] * (m_s[t+1] - self.phi * m_filt[t])

            # Var(θ_t|Y<=T) = Var(θ_t|Y<=t) + J_t^2 ( Var(θ_{t+1}|Y<=T) - Var(θ_{t+1}|Y<=t) )
            P_s[t] = P_filt[t] + J[t]**2 * (P_s[t+1] - P_pred_next)

        return m_s, P_s, J

    def _build_cov_matrix(self, P_s, J):
        """
        Build full Σ_1:T|M from smoothed variance and gain

        Σ_t,t = P_s[t]
        Σ_t,t+1 = J[t] P_s[t+1]
        Σ_i,j = J[i] Σ_{i+1,j}  (i < j, recursive)
        """
        T = len(P_s)
        Cov = np.diag(P_s)

        # Upper diagonal
        for t in range(T - 1):
            Cov[t, t + 1] = J[t] * P_s[t + 1]

        # Recursive fill
        for i in range(T - 2):
            for j in range(i + 2, T):
                Cov[i, j] = J[i] * Cov[i + 1, j]

        # Symmetrize
        Cov = Cov + Cov.T - np.diag(np.diag(Cov))

        return Cov



    def posterior(self, y_given, T, N=20, mask=None, verbose=0):
        """
        Compute p(θ_1:T | y_M)

        General form (handles both prefix Y_1:T_0 and arbitrary mask M):
        - mask=None → M = {0, 1, ..., T_0-1} (prefix, original behavior)
        - mask=[...] → M = arbitrary indices

        Uses Kalman Filter (skip updates at unobserved times) + RTS Smoother
        """
        Y_arr = y_given.detach().cpu().numpy().ravel() if isinstance(y_given, torch.Tensor) else np.asarray(y_given).ravel()

        # Parse mask
        if mask is None:
            # Original mode: prefix Y_1:T_0
            M_idx = np.arange(len(Y_arr))
        else:
            # Masked mode
            mask_arr = np.asarray(mask)
            M_idx = np.where(mask_arr)[0] if mask_arr.dtype == bool else mask_arr

        if len(M_idx) == 0:
            # θ ~ N(0, Σ_0 ϕ^|s-u|) stationary prior
            mu_theta = np.zeros(T)
            idx = np.arange(T)
            Cov_theta = self.Sigma_0 * (self.phi ** np.abs(idx[:, None] - idx[None, :]))

        else:
            # KF + RTS with mask
            m_filt, P_filt = self._kf_filter(Y_arr, mask, T)
            m_s, P_s, J = self._rts_smooth(m_filt, P_filt)

            mu_theta = m_s
            Cov_theta = self._build_cov_matrix(P_s, J)

        # Sample θ ~ N(μ, Σ) with numerical stability
        ridge = 1e-8 * np.eye(T)
        Cov_theta_stable = Cov_theta + ridge
        theta_samples = np.random.multivariate_normal(mu_theta, Cov_theta_stable, N)

        # Sample y ~ N(θ, r)
        y_samples = np.random.normal(theta_samples, self.std_Y, theta_samples.shape)

        return {
            'z_samples': theta_samples,
            'z_mean': mu_theta,
            'z_std': np.sqrt(np.diag(Cov_theta)),
            'x_samples': y_samples[:, :, None],
            'mean_samples': mu_theta.reshape(-1, 1),
            'var_samples': np.diag(Cov_theta).reshape(-1, 1),
        }
