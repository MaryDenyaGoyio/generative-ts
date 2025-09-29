import numpy as np
import torch
import torch.nn as nn
import json


class AR1_ts():
    """
    Theoretical AR1 model for computing exact posterior predictions

    State-space (scalar):
        x_t = phi * x_{t-1} + epsilon_t,   epsilon_t ~ N(0, sigma^2)
        y_t = 1 * x_t       + eta_t,       eta_t     ~ N(0, std_Y^2)

    Implements (compat-preserving):
    - posterior(y_given, T, N=20, verbose=2):
        Returns the SAME keys/shapes as the original implementation, but now
        computed from the exact joint Gaussian p(y_{1:T} | y_{<T0})
        where T0 = len(y_given), i.e., conditioned on y_1..y_{T0-1} ONLY.

        Returned dict fields (unchanged):
          'mean_samples' : (T,1)  -> here: E[y_t | y_{<T0}]
          'var_samples'  : (T,1)  -> here: std[y_t | y_{<T0}]  (kept as std for compat)
          'x_samples'    : (N,T,1) samples of y (naming preserved for compat)
          'recon_mean'   : (T_0,1) first T0 entries of E[y_t | y_{<T0}]
          'recon_std'    : (T_0,1) first T0 entries of std[y_t | y_{<T0}]
          'recon_samples': (N,T_0,1) first T0 of sampled y
          'state_mean'   : (T,1)   E[x_t | y_{<T0}]   (provided for completeness)
          'state_std'    : (T,1)   std[x_t | y_{<T0}]
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
        self.T = int(data_config['T'])       # default total sequence length
        self.phi = float(data_config['phi']) # AR coefficient
        self.sigma = float(data_config['sigma'])   # process noise std
        self.std_Y = float(data_config['std_Y'])   # observation noise std

        # State space scalars
        self.F = self.phi
        self.H = 1.0
        self.Q = self.sigma**2
        self.R = self.std_Y**2

        # Stationary initial prior for x_1
        if abs(self.F) >= 1.0:
            raise ValueError("For stationarity of the AR(1), require |phi| < 1.")
        self.mu_0 = 0.0
        self.Sigma_0 = self.Q / (1.0 - self.F**2)

    # ----------------------- Core: p(y_{1:T} | y_{<T0}) -----------------------

    def _kf_prefix(self, y_prefix):
        """
        Kalman filter on y_{1:len(y_prefix)} but we RETURN results only up to T0-1,
        i.e., we FILTER using observations y_1..y_{T0-1}. (0-based indices 0..T0-2)
        Returns arrays of length n = T0-1:
          m_pred[t] = E[x_t | y_{1:t-1}], P_pred[t] = Var[x_t | y_{1:t-1}]
          m_filt[t] = E[x_t | y_{1:t}],   P_filt[t] = Var[x_t | y_{1:t}]
        """
        T0 = len(y_prefix)
        n = T0 - 1
        if n <= 0:
            return (np.array([]), np.array([]), np.array([]), np.array([]))

        m_pred = np.zeros(n)
        P_pred = np.zeros(n)
        m_filt = np.zeros(n)
        P_filt = np.zeros(n)

        # Prior for x_1 (before seeing y_1)
        mu = self.mu_0
        P = self.Sigma_0

        for t in range(n):
            # Predict to x_{t+1} in 1-based, which is x_t in 0-based array indexing here
            if t == 0:
                m_pred[t] = mu
                P_pred[t] = P
            else:
                m_pred[t] = self.F * m_filt[t-1]
                P_pred[t] = self.F**2 * P_filt[t-1] + self.Q

            # Update with y_t (0-based)
            S = P_pred[t] + self.R
            K = P_pred[t] / S
            innov = y_prefix[t] - m_pred[t]
            m_filt[t] = m_pred[t] + K * innov
            P_filt[t] = (1.0 - K) * P_pred[t]

        return m_pred, P_pred, m_filt, P_filt

    def _rts_smoother_full_cov(self, m_pred, P_pred, m_filt, P_filt):
        """
        RTS smoothing on indices 0..n-1 (where n = T0-1).
        Build FULL covariance across the past block (size n x n) including all
        cross-time covariances.

        Returns:
          m_s, P_s : smoothed marginals for x_{1:n} (1-based)
          J        : smoother gains (length n-1, if n>=2)
          PastCov  : full covariance Cov(x_{1:n} | y_{<T0}) (n x n)
        """
        n = len(m_filt)
        if n == 0:
            return (np.array([]), np.array([]), np.array([]), np.zeros((0, 0)))

        m_s = np.zeros(n)
        P_s = np.zeros(n)
        J = np.zeros(n-1) if n >= 2 else np.array([])

        # Initialize at last filtered
        m_s[-1] = m_filt[-1]
        P_s[-1] = P_filt[-1]

        # Backward recursions
        for t in range(n-2, -1, -1):
            J[t] = P_filt[t] * self.F / P_pred[t+1]
            m_s[t] = m_filt[t] + J[t] * (m_s[t+1] - m_pred[t+1])
            P_s[t] = P_filt[t] + J[t]**2 * (P_s[t+1] - P_pred[t+1])

        # Build full covariance via recurrence:
        #   Cov(x_{i}, x_{j}) = J_i * Cov(x_{i+1}, x_{j}) for i<j,
        # with diagonals P_s and first off-diagonals Cov(x_j, x_{j-1}) = J_{j-1} * P_s[j].
        PastCov = np.zeros((n, n))
        for j in range(n):
            PastCov[j, j] = P_s[j]
        for j in range(1, n):
            PastCov[j-1, j] = J[j-1] * P_s[j]
            for i in range(j-2, -1, -1):
                PastCov[i, j] = J[i] * PastCov[i+1, j]
        PastCov = PastCov + PastCov.T - np.diag(np.diag(PastCov))

        return m_s, P_s, J, PastCov

    def _assemble_state_joint(self, m_s, P_s, J, PastCov, T0, T):
        """
        Assemble joint mean/cov for x_{1:T} | y_{<T0}.
        Past block covers indices 1..T0-1 (size n_past=T0-1).
        Future starts at t=T0.
        """
        n_past = T0 - 1
        T = int(T)
        mu_x = np.zeros(T)
        Sigma_x = np.zeros((T, T))

        # Fill past block
        if n_past > 0:
            mu_x[:n_past] = m_s
            Sigma_x[:n_past, :n_past] = PastCov
            m_base = m_s[-1]
            P_base = P_s[-1]
            cross_base = PastCov[:n_past, -1]  # Cov(x_{1:n_past}, x_{n_past})
        else:
            # No past used: base is stationary prior for "time 0"
            m_base = self.mu_0
            P_base = self.Sigma_0
            cross_base = np.zeros(0, dtype=float)

        # Future part sizes/indices
        n_future = T - n_past
        if n_future <= 0:
            return mu_x, Sigma_x

        # Future means/vars by propagation
        m_fut = np.zeros(n_future)
        P_fut = np.zeros(n_future)

        for h in range(n_future):
            if h == 0:
                # x_{T0} | y_{<T0}
                m_curr = self.F * m_base
                P_curr = self.F**2 * P_base + self.Q
            else:
                m_curr = self.F * m_fut[h-1]
                P_curr = self.F**2 * P_fut[h-1] + self.Q
            m_fut[h] = m_curr
            P_fut[h] = P_curr

        mu_x[n_past:] = m_fut

        # Cross cov: future vs past via recurrence
        if n_past > 0:
            C = cross_base.copy()
            for h in range(n_future):
                C = self.F * C
                Sigma_x[:n_past, n_past + h] = C
            Sigma_x[n_past:, :n_past] = Sigma_x[:n_past, n_past:].T

        # Future-future block
        for h in range(n_future):
            Sigma_x[n_past + h, n_past + h] = P_fut[h]
            for k in range(h):
                if k == h - 1:
                    cov_hk = self.F * P_fut[k]  # Cov(x_{k+1}, x_k)
                else:
                    cov_hk = self.F * Sigma_x[n_past + h - 1, n_past + k]
                Sigma_x[n_past + h, n_past + k] = cov_hk
                Sigma_x[n_past + k, n_past + h] = cov_hk

        return mu_x, Sigma_x

    def _joint_y_given_prefix(self, y_prefix, T):
        """
        Internal: returns (mu_y, Sigma_y, mu_x, Sigma_x) for p(y_{1:T} | y_{<T0}).
        """
        y_prefix = np.asarray(y_prefix).reshape(-1)
        T0 = len(y_prefix)

        # Filter only up to T0-1 observations
        m_pred, P_pred, m_filt, P_filt = self._kf_prefix(y_prefix)

        # RTS smoother over past block
        m_s, P_s, J, PastCov = self._rts_smoother_full_cov(m_pred, P_pred, m_filt, P_filt)

        # Assemble full state joint
        mu_x, Sigma_x = self._assemble_state_joint(m_s, P_s, J, PastCov, T0, T)

        # Map to y (independent obs noise per time)
        mu_y = mu_x.copy()
        Sigma_y = Sigma_x.copy()
        idx = np.diag_indices_from(Sigma_y)
        Sigma_y[idx] += self.R

        return mu_y, Sigma_y, mu_x, Sigma_x

    # ----------------------- Public API (compat) -----------------------

    def posterior_y(self, y_given, T, N=20, verbose=2):
        """
        COMPATIBLE posterior API: p(y_{1:T} | y_{<T0})
        Now returns statistics derived from the exact joint Gaussian
        p(y_{1:T} | y_{<T0}), where T0 = len(y_given).

        Args:
            y_given: observed sequence (T_0,)  -- we condition on y_{1:T0-1} ONLY
            T: total sequence length for output (>= T_0 recommended)
            N: number of joint-coherent samples to return (as before)
            verbose: prints small info if >1

        Returns: dict with the SAME keys/shapes as the original code.
        """
        # Normalize input
        if isinstance(y_given, torch.Tensor):
            y_given = y_given.detach().cpu().numpy()
        elif isinstance(y_given, list):
            y_given = np.array(y_given)
        if y_given.ndim > 1:
            y_given = y_given.squeeze()

        T0 = int(len(y_given))
        if verbose > 1:
            print(f"[posterior | joint prefix] T0={T0} (condition on y[1..{max(T0-1,0)}]), T={T}")

        # Core joint computation
        mu_y, Sigma_y, mu_x, Sigma_x = self._joint_y_given_prefix(y_given, T)

        # Means/stds for y
        y_mean = mu_y            # (T,)
        y_std  = np.sqrt(np.clip(np.diag(Sigma_y), a_min=0.0, a_max=None))  # (T,)

        # Joint-coherent samples of y: (N,T)
        if N > 0:
            samples_flat = np.random.multivariate_normal(mean=mu_y, cov=Sigma_y, size=N)  # (N,T)
            samples = samples_flat[:, :, None]  # (N,T,1)
        else:
            samples = np.zeros((0, T, 1), dtype=float)

        # Recon blocks (keep original semantics: just first T0 entries)
        recon_mean = y_mean[:T0, None]                    # (T0,1)
        recon_std  = y_std[:T0, None]                     # (T0,1)
        recon_samples = samples[:, :T0, :]                # (N,T0,1)

        # State stats for completeness/compat
        state_mean = mu_x[:, None]                        # (T,1)
        state_std  = np.sqrt(np.clip(np.diag(Sigma_x), a_min=0.0, a_max=None))[:, None]  # (T,1)

        # Package with EXACT same keys as before
        return {
            'mean_samples': y_mean[:, None],   # (T,1)   (E[y])
            'var_samples':  y_std[:, None],    # (T,1)   (std[y])  -- kept as std for compat
            'x_samples':    samples,           # (N,T,1) samples of y (name preserved)
            'recon_mean':   recon_mean,        # (T0,1)
            'recon_std':    recon_std,         # (T0,1)
            'recon_samples':recon_samples,     # (N,T0,1)
            'state_mean':   state_mean,        # (T,1)   E[x]
            'state_std':    state_std,         # (T,1)   std[x]
        }

    def posterior(self, y_given, T, N=20, verbose=2):
        """
        Latent posterior API: p(x_{1:T} | y_{<T0})
        Returns latent state x_t posterior distribution (AR1 hidden states).

        Args:
            y_given: observed sequence (T_0,)
            T: total sequence length for output
            N: number of samples to return
            verbose: prints small info if >1

        Returns: dict with latent z samples/stats
        """
        # Normalize input
        if isinstance(y_given, torch.Tensor):
            y_given = y_given.detach().cpu().numpy()
        elif isinstance(y_given, list):
            y_given = np.array(y_given)
        if y_given.ndim > 1:
            y_given = y_given.flatten()

        T0 = len(y_given)

        # Get joint distribution: both observation and latent
        mu_y, Sigma_y, mu_x, Sigma_x = self._joint_y_given_prefix(y_given, T)

        # Sample from latent posterior p(x_{1:T} | y_{1:T0})
        x_samples = []
        try:
            # Multivariate normal sampling from latent posterior
            for _ in range(N):
                x_sample = np.random.multivariate_normal(mu_x, Sigma_x)
                x_samples.append(x_sample)
            x_samples = np.array(x_samples)  # (N, T)
        except np.linalg.LinAlgError:
            # Fallback: diagonal approximation
            x_std = np.sqrt(np.clip(np.diag(Sigma_x), a_min=1e-8, a_max=None))
            x_samples = np.random.normal(
                mu_x[None, :], x_std[None, :], size=(N, T)
            )

        # Latent state statistics
        x_mean = mu_x  # (T,)
        x_std = np.sqrt(np.clip(np.diag(Sigma_x), a_min=1e-8, a_max=None))  # (T,)

        return {
            'z_samples': x_samples,  # (N, T) - latent state samples
            'z_mean': x_mean,        # (T,) - latent state mean
            'z_std': x_std,          # (T,) - latent state std
        }
