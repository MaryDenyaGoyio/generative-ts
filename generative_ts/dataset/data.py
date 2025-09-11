import numpy as np
import torch
import json
import os
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from typing import Tuple, Dict


class GP():

    def __init__(self, **kwargs): # seed
        self.set_params(**kwargs)
        self._theta_fixed = None  # Store true theta_fixed from data generation
        self._fixed_theta_value = None  # Store fixed theta value for all samples

    def set_params(self, **kwargs):
        prms = ["T", "std_Y", "v", "tau", "sigma_f"]
        self.T, self.std_Y, self.v, self.tau, self.sigma_f = [kwargs[prm] for prm in prms]
        
    def set_fixed_theta(self, theta_value):
        """Set a fixed theta_fixed value to use for all data generation"""
        self._fixed_theta_value = theta_value
    
    def data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        # e_t + θ^fixed + θ^GP_t
        noise_Y = np.random.normal(loc=0.0, scale=self.std_Y, size=self.T)

        # Use fixed theta_fixed if set, otherwise generate new one
        if self._fixed_theta_value is not None:
            theta_fixed = self._fixed_theta_value
        else:
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


class AR1():
    """
    AR1 process: x_t = phi * x_{t-1} + epsilon_t
    where epsilon_t ~ N(0, sigma^2)
    """

    def __init__(self, **kwargs):
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        prms = ["T", "std_Y", "phi", "sigma"]
        self.T, self.std_Y, self.phi, self.sigma = [kwargs[prm] for prm in prms]
        
    def data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate one AR1 sequence"""
        # Generate AR1 process
        x = np.zeros(self.T)
        # Stationary initial condition
        x[0] = np.random.normal(0, self.sigma / np.sqrt(1 - self.phi**2))
        
        for t in range(1, self.T):
            x[t] = self.phi * x[t-1] + np.random.normal(0, self.sigma)
        
        # Add observation noise
        noise_Y = np.random.normal(0, self.std_Y, self.T)
        y = x + noise_Y
        
        return (
            y.reshape(self.T, 1),
            {
                r"\epsilon_{a,t}": noise_Y,
                r"x_{AR1,t}": x,
                "phi": self.phi,
                "sigma": self.sigma
            }
        )


def generate_gp_dataset(config_path, n_samples=10000, save_dir=None):
    """
    Generate GP dataset based on config file
    
    Args:
        config_path: Path to experiment config file
        n_samples: Number of GP samples to generate
        save_dir: Directory to save dataset (auto-generated if None)
        
    Returns:
        save_dir: Path to saved dataset directory
        samples: Generated samples array (n_samples, T)
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_config = config['data']
    
    # Create save directory based on config parameters
    if save_dir is None:
        config_str = f"T{data_config['T']}_tau{data_config['tau']}_v{data_config['v']}_stdY{data_config['std_Y']}_sigmaf{data_config['sigma_f']}_fixed"
        save_dir = Path(__file__).parent / f"gp_samples_{config_str}"
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"Generating {n_samples} GP samples with config:")
    print(f"  T: {data_config['T']}")
    print(f"  tau: {data_config['tau']}")
    print(f"  v: {data_config['v']}")
    print(f"  std_Y: {data_config['std_Y']}")
    print(f"  sigma_f: {data_config['sigma_f']}")
    print(f"  Using FIXED theta_fixed for all samples")
    print(f"  Saving to: {save_dir}")
    
    # Generate ONE theta_fixed for ALL samples
    theta_fixed = np.random.normal(loc=0.0, scale=data_config['v'])
    print(f"  Fixed theta_fixed value: {theta_fixed:.6f}")
    
    # Generate samples
    samples = []
    print(f"Generating samples...")
    
    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples")
        
        # Generate only GP component and noise, use fixed theta_fixed
        noise_Y = np.random.normal(loc=0.0, scale=data_config['std_Y'], size=data_config['T'])
        
        theta_gp = GaussianProcessRegressor(
            kernel=C(data_config['sigma_f']**2) * RBF(length_scale=data_config['tau']), 
            alpha=1e-10
        ).sample_y(np.arange(data_config['T']).reshape(-1, 1), random_state=None).flatten()
        
        Y = (theta_fixed + theta_gp + noise_Y).reshape(data_config['T'], 1)
        samples.append(Y.squeeze())  # (T,)
    
    # Convert to numpy array
    samples = np.stack(samples, axis=0)  # (N, T)
    print(f"Dataset shape: {samples.shape}")
    print(f"Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")
    print(f"Min: {samples.min():.4f}, Max: {samples.max():.4f}")
    
    # Save dataset
    np.save(save_dir / "samples.npy", samples)
    
    # Save config
    with open(save_dir / "config.json", 'w') as f:
        json.dump({
            "data": data_config,
            "n_samples": n_samples,
            "dataset_shape": list(samples.shape),
            "theta_fixed_value": float(theta_fixed),
            "statistics": {
                "mean": float(samples.mean()),
                "std": float(samples.std()),
                "min": float(samples.min()),
                "max": float(samples.max())
            }
        }, f, indent=2)
    
    # Save original experiment config for reference
    with open(save_dir / "original_experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Dataset generated successfully!")
    print(f"   Samples: {save_dir}/samples.npy")
    print(f"   Config: {save_dir}/config.json")
    print(f"   Original config: {save_dir}/original_experiment_config.json")
    
    return save_dir, samples


def generate_ar1_dataset(config_path, n_samples=10000, save_dir=None):
    """
    Generate AR1 dataset based on config file (same structure as GP)
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_config = config['data']
    
    # Create save directory based on config parameters
    if save_dir is None:
        config_str = f"T{data_config['T']}_phi{data_config['phi']}_sigma{data_config['sigma']}_stdY{data_config['std_Y']}"
        save_dir = Path(__file__).parent / f"ar1_samples_{config_str}"
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"Generating {n_samples} AR1 samples with config:")
    print(f"  T: {data_config['T']}")
    print(f"  phi: {data_config['phi']}")
    print(f"  sigma: {data_config['sigma']}")
    print(f"  std_Y: {data_config['std_Y']}")
    print(f"  Saving to: {save_dir}")
    
    # Generate AR1 data generator
    ar1_gen = AR1(**data_config)
    
    # Generate samples
    samples = []
    print(f"Generating samples...")
    
    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples")
        
        Y, _ = ar1_gen.data()
        samples.append(Y.squeeze())  # (T,)
    
    # Convert to numpy array
    samples = np.stack(samples, axis=0)  # (N, T)
    print(f"Dataset shape: {samples.shape}")
    print(f"Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")
    print(f"Min: {samples.min():.4f}, Max: {samples.max():.4f}")
    
    # Save dataset
    np.save(save_dir / "samples.npy", samples)
    
    # Save config
    with open(save_dir / "config.json", 'w') as f:
        json.dump({
            "data": data_config,
            "n_samples": n_samples,
            "dataset_shape": list(samples.shape),
            "statistics": {
                "mean": float(samples.mean()),
                "std": float(samples.std()),
                "min": float(samples.min()),
                "max": float(samples.max())
            }
        }, f, indent=2)
    
    # Create test/train split (same format as GP)
    data_dict = {
        'Y_test': samples.tolist(),
        'config': {"data": data_config}
    }
    
    torch.save(data_dict, save_dir / "data.pth")
    
    # Save original experiment config for reference
    with open(save_dir / "original_experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ AR1 Dataset generated successfully!")
    print(f"   Samples: {save_dir}/samples.npy")
    print(f"   Config: {save_dir}/config.json")
    print(f"   Data: {save_dir}/data.pth")
    print(f"   Original config: {save_dir}/original_experiment_config.json")
    
    return save_dir, samples


if __name__ == "__main__":
    import sys
    
    # Default config path
    config_path = Path(__file__).parent.parent / "config" / "ls4_experiment.json"
    
    # Command line usage: python data.py [config_path] [n_samples]
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    
    n_samples = 10000
    if len(sys.argv) > 2:
        n_samples = int(sys.argv[2])
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Generate dataset
    dataset_dir, samples = generate_gp_dataset(config_path, n_samples=n_samples)
    print(f"Dataset saved to: {dataset_dir}")