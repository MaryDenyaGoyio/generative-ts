import numpy as np
import torch
import json
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from typing import Tuple, Dict


class GP():

    def __init__(self, **kwargs):
        prms = ["T", "std_Y", "v", "tau", "sigma_f"]
        self.T, self.std_Y, self.v, self.tau, self.sigma_f = [kwargs[prm] for prm in prms]
        self._grid = np.arange(self.T).reshape(-1, 1)
        self._gpr = GaussianProcessRegressor(kernel=C(self.sigma_f**2) * RBF(length_scale=self.tau), alpha=1e-10)
    
    def data(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:

        n = int(n_samples)
        if n < 1:
            raise ValueError("n_samples must be >= 1")

        # e_t ~ N(0, \sigma_Y^2)
        e_t = np.random.normal(0.0, self.std_Y, (n, self.T))

        # \theta_fixed ~ N(0, v^2)
        theta_fixed = np.random.normal(0.0, self.v, n)

        # \theta_gp_t ~ GP(·,·)
        theta_gp_t = self._gpr.sample_y(self._grid, n_samples=n).T  # (n, T)

        theta_t = theta_gp_t + theta_fixed[:, None]  # \theta_t = \theta_gp_t + \theta_fixed
        Y = theta_t + e_t  # Y_t = \theta_t + e_t

        return (
            Y.reshape(n, self.T, 1),
            theta_t.reshape(n, self.T, 1),
            {
                r"\epsilon_{a,t}": e_t,
                r"\theta^{\text{fixed}}_{a}": theta_fixed,
                r"\theta^{\text{GP}}_{a,t}": theta_gp_t
            }
        )


class AR1():
    """
    AR1 process: x_t = phi * x_{t-1} + epsilon_t
    where epsilon_t ~ N(0, sigma^2)
    """

    def __init__(self, **kwargs):
        prms = ["T", "std_Y", "phi", "sigma"]
        self.T, self.std_Y, self.phi, self.sigma = [kwargs[prm] for prm in prms]
        
    def data(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:

        n = int(n_samples)
        if n < 1:
            raise ValueError("n_samples must be >= 1")

        # e_t ~ N(0, \sigma_Y^2)
        e_t = np.random.normal(0, self.std_Y, (n, self.T))

        # init \theta_0
        theta = np.zeros((n, self.T))
        theta[:, 0] = np.random.normal(0, self.sigma / np.sqrt(1 - self.phi**2), n)

        # \theta_t ~ AR(1)
        for t in range(1, self.T):
            theta[:, t] = self.phi * theta[:, t-1] + np.random.normal(0, self.sigma, n)

        # Y_t = \theta_t + e_t
        Y = theta + e_t

        return (
            Y.reshape(n, self.T, 1),
            theta.reshape(n, self.T, 1),
            {
                r"\epsilon_{a,t}": e_t,
                r"x_{AR1,t}": theta,
                "phi": self.phi,
                "sigma": self.sigma
            }
        )


def generate_dataset(config_path, n_samples=10000, save_dir=None, process='gp'):

    with open(config_path, 'r') as f:   config = json.load(f)
    data_config = config['data']

    # set dir
    if process.lower() == 'gp':
        generator = GP(**data_config)
        config_str = f"T{data_config['T']}_tau{data_config['tau']}_v{data_config['v']}_stdY{data_config['std_Y']}_sigmaf{data_config['sigma_f']}_fixed"
    elif process.lower() == 'ar1':
        generator = AR1(**data_config)
        config_str = f"T{data_config['T']}_phi{data_config['phi']}_sigma{data_config['sigma']}_stdY{data_config['std_Y']}"
    else:
        raise ValueError(f"Unsupported process: {process}")
    
    default_dir = Path(__file__).parent / f"{process.lower()}_samples_{config_str}"

    if save_dir is None:
        save_dir = default_dir
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    # generate data
    Y_batch, theta_batch, _ = generator.data(n_samples=n_samples)

    samples_Y, samples_theta = np.asarray(Y_batch), np.asarray(theta_batch)

    np.save(save_dir / "outcome_Y.npy", samples_Y)
    np.save(save_dir / "latent_theta.npy", samples_theta)

    with open(save_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return save_dir, samples_Y, samples_theta



if __name__ == "__main__":
    import sys
    
    dataset_dir, samples_Y, samples_theta = generate_dataset(
        Path(sys.argv[1]),
        n_samples=int(sys.argv[2]),
        process=sys.argv[3]
    )
    print(f"Dataset saved to: {dataset_dir}")
    print(f"Y shape: {samples_Y.shape}, theta shape: {samples_theta.shape}")
