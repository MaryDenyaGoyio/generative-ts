"""
Regenerate all datasets with new format (outcome_Y.npy and latent_theta.npy)
"""
import sys
sys.path.insert(0, '/home/marydenya/Downloads/generative-ts')

from pathlib import Path
from generative_ts.dataset.data import generate_ar1_dataset, generate_gp_dataset

# AR1 datasets configurations
ar1_configs = [
    {"T": 300, "phi": 0.99, "sigma": 0.1, "std_Y": 0.01},
    {"T": 300, "phi": 0.99, "sigma": 0.1, "std_Y": 1.0},
    {"T": 300, "phi": 0.9, "sigma": 0.1, "std_Y": 0.01},
]

# GP datasets configurations
gp_configs = [
    {"T": 300, "tau": 10, "v": 10, "std_Y": 0.01, "sigma_f": 1},
]

base_dir = Path('/home/marydenya/Downloads/generative-ts/generative_ts/dataset')

# Create temporary config files and generate datasets
import json
import tempfile

print("=" * 60)
print("Regenerating AR1 datasets...")
print("=" * 60)

for ar1_cfg in ar1_configs:
    print(f"\nGenerating AR1: {ar1_cfg}")

    # Create temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"data": ar1_cfg}, f)
        config_path = f.name

    # Generate dataset
    config_str = f"T{ar1_cfg['T']}_phi{ar1_cfg['phi']}_sigma{ar1_cfg['sigma']}_stdY{ar1_cfg['std_Y']}"
    save_dir = base_dir / f"ar1_samples_{config_str}"

    generate_ar1_dataset(config_path, n_samples=10000, save_dir=str(save_dir))
    Path(config_path).unlink()  # cleanup temp file

print("\n" + "=" * 60)
print("Regenerating GP datasets...")
print("=" * 60)

for gp_cfg in gp_configs:
    print(f"\nGenerating GP: {gp_cfg}")

    # Create temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"data": gp_cfg}, f)
        config_path = f.name

    # Generate dataset
    config_str = f"T{gp_cfg['T']}_tau{gp_cfg['tau']}_v{gp_cfg['v']}_stdY{gp_cfg['std_Y']}_sigmaf{gp_cfg['sigma_f']}_fixed"
    save_dir = base_dir / f"gp_samples_{config_str}"

    generate_gp_dataset(config_path, n_samples=10000, save_dir=str(save_dir))
    Path(config_path).unlink()  # cleanup temp file

print("\n" + "=" * 60)
print("All datasets regenerated successfully!")
print("=" * 60)