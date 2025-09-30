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


    def _kf_filter(self, y_obs):

        T_0 = len(y_obs)
        m_filt, P_filt = np.zeros(T_0), np.zeros(T_0)

        # θ_0 ~ N(μ_0, Σ_0),  Σ_0 = q / (1 - φ^2)
        m_pred, P_pred = 0, self.Sigma_0

        for t in range(T_0):

            # K_t = Var(θ_t|Y<t) / (Var(θ_t|Y<t) + σ_Y^2)
            K = P_pred / (P_pred + self.std_Y**2)

            # E[θ_t|Y<=t] = E[θ_t|Y<t] + K_t ( y_t - E[θ_t|Y<t] )
            m_filt[t] = m_pred + K * (y_obs[t] - m_pred)

            # Var(θ_t|Y<=t) = (1 - K_t) Var(θ_t|Y<t)
            P_filt[t] = (1.0 - K) * P_pred

            # E[θ_{t+1}|Y<=t] = φ E[θ_t|Y<=t]
            m_pred = self.phi * m_filt[t]

            # Var(θ_{t+1}|Y<=t) = φ^2 Var(θ_t|Y<=t) + q
            P_pred = self.phi**2 * P_filt[t] + self.sigma**2

        return m_filt, P_filt



    def _rts_smooth(self, m_filt, P_filt):

        T_0 = len(m_filt)
        m_s, P_s, J = np.zeros(T_0), np.zeros(T_0), np.zeros(T_0-1)

        # θ_T_0 | Y_<=T_0
        m_s[-1], P_s[-1] = m_filt[-1], P_filt[-1]

        for t in range(T_0-2, -1, -1):

            # Var(θ_{t+1}|Y<=t) = φ^2 Var(θ_t|Y<=t) + q
            P_pred_next = self.phi**2 * P_filt[t] + self.sigma**2

            # J_t = Var(θ_t|Y<=t) φ (Var(θ_{t+1}|Y<=t))^{-1}
            J[t] = P_filt[t] * self.phi / P_pred_next

            # E[θ_t|Y_<=T_0] = E[θ_t|Y<=t] + J_t ( E[θ_{t+1}|Y_<=T_0] - E[θ_{t+1}|Y<=t] )
            m_s[t] = m_filt[t] + J[t] * (m_s[t+1] - self.phi * m_filt[t])

            # Var(θ_t|Y<=T_0) = Var(θ_t|Y<=t) + J_t^2 ( Var(θ_{t+1}|Y<=T_0) - Var(θ_{t+1}|Y<=t) )
            P_s[t] = P_filt[t] + J[t]**2 * (P_s[t+1] - P_pred_next)

        return m_s, P_s, J



    def posterior(self, y_given, T, N=20, verbose=0):

        Y_T_0 = y_given.detach().cpu().numpy().ravel() if isinstance(y_given, torch.Tensor) else np.asarray(y_given).ravel()
        T_0 = len(Y_T_0)

        mu_theta, Cov_theta = np.zeros(T), np.zeros((T, T))

        if T_0 == 0:
            # θ ~ N(0, Σ_0 ϕ^|s-u|) stationary prior
            mu_theta[:] = 0
            idx = np.arange(T)
            Cov_theta = self.Sigma_0 * (self.phi ** np.abs(idx[:, None] - idx[None, :]))
        
        else:
            # θ_<=T_0 | Y_<=T_0
            m_filt, P_filt = self._kf_filter(Y_T_0)
            m_s, P_s, J = self._rts_smooth(m_filt, P_filt)

            # Σ_ij = Cov(θ_i, θ_j | Y_<=T_0)
            Cov_past = np.diag(P_s)
            for j in range(1, T_0):

                # Σ_{j-1,j} = J_{j-1} Var(θ_j|Y<=T_0)
                Cov_past[j-1, j] = J[j-1] * P_s[j]
                for i in range(j-2, -1, -1):

                    # Σ_ij = J_i Σ_{i+1,j}  (i<j)
                    Cov_past[i, j] = J[i] * Cov_past[i+1, j]
            
            Cov_past = Cov_past + Cov_past.T - np.diag(np.diag(Cov_past))


            # μ_<=T_0 = E[θ_<=T_0|Y_<=T_0]
            mu_theta[:T_0] = m_s
            # Σ_{<=T_0,<=T_0} = Cov(θ_<=T_0|Y_<=T_0)
            Cov_theta[:T_0, :T_0] = Cov_past

            if T > T_0:
                # θ_{t_0+h}|Y{1:t_0}:   E[θ_{t_0+h}] = φ^h E[θ_{t_0}],   Var = φ^{2h} Var(θ_{t_0}) + q ∑_{j=0}^{h-1} φ^{2j}
                m_base, P_base = m_s[-1], P_s[-1]
                # Cov(θ_{1:t_0}, θ_{t_0})  →  Cov(θ_{1:t_0}, θ_{t_0+h}) = φ^{h} Cov(θ_{1:t_0}, θ_{t_0})
                cross = Cov_past[:, -1].copy()

                for h in range(T_0, T):
                    # E[θ_h|Y_<=T_0] = φ E[θ_{h-1}|Y_<=T_0]
                    mu_theta[h] = self.phi * m_base

                    # Var(x_h|Y_<=T_0) = φ^2 Var(θ_{h-1}|Y_<=T_0) + q
                    P_h = self.phi**2 * P_base + self.sigma**2
                    Cov_theta[h, h] = P_h

                    # Cov(x_i, x_h|Y_<=T_0) = φ Cov(x_i, θ_{h-1}|Y_<=T_0)  (i < T_0)
                    cross = self.phi * cross
                    Cov_theta[:T_0, h] = cross
                    Cov_theta[h, :T_0] = cross

                    # Cov(x_k, x_h|Y_<=T_0) = φ Cov(x_k, θ_{h-1}|Y_<=T_0)  (T_0 ≤ k < h)
                    Cov_theta[h, T_0:h] = self.phi * Cov_theta[h-1, T_0:h]
                    Cov_theta[T_0:h, h] = Cov_theta[h, T_0:h]

                    # E[θ_h|·], Var(θ_h|·)
                    m_base, P_base = mu_theta[h], P_h

        # y_t|θ_t ~ N(θ_t, std_Y^2)
        mu_y = mu_theta.copy()
        Cov_y = Cov_theta.copy()
        Cov_y[np.diag_indices(T)] += self.std_Y**2

        # θ ~ N(μ, Σ) with numerical stability
        jitter = 1e-9 * np.trace(Cov_theta) / max(T, 1)
        Cov_theta_stable = Cov_theta + jitter * np.eye(T)
        x_samples = np.random.multivariate_normal(mu_theta, Cov_theta_stable, N)
        y_samples = np.random.normal(x_samples, self.std_Y)

        return {
            'z_samples': x_samples,
            'z_mean': mu_theta,
            'z_std': np.sqrt(np.diag(Cov_theta)),

            'y_mean': mu_y[:, None],
            'y_std': np.sqrt(np.diag(Cov_y))[:, None],
            'x_samples': y_samples[:, :, None],
        }
