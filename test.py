#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-model evaluation test script
Generates both plot_posterior and plot_sample for given models
"""

import os
import sys
import json
import yaml
import torch
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from generative_ts.eval import plot_posterior, plot_sample
from generative_ts.dataset.gp import GP_ts


# =============================================================================
# USER CONFIGURATION
# =============================================================================
MODELS_TO_TEST = [
    "250809_105652_LS4",    # LS4 model
    #"250809_133803_VRNN",   # VRNN model
    "GP_ts"                 # Theoretical GP model
]

# Test parameters
T_TOTAL = 200
T_CONDITIONING = 80
N_SAMPLES = 20


def detect_model_type(config):
    """Detect model type from config"""
    return config.get('model_type', 'Unknown')


def load_saved_model(model_folder):
    """Load saved model from folder"""
    save_path = Path("generative_ts/saves") / model_folder
    
    if not save_path.exists():
        raise FileNotFoundError(f"Model directory not found: {save_path}")
    
    # Load config
    config_path = save_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_type = detect_model_type(config)
    
    if model_type == "LS4":
        from generative_ts.models.ls4 import LS4_ts, dict2attr
        
        # Load LS4 yaml config
        yaml_config_path = save_path / "ls4_config.yaml"
        if yaml_config_path.exists():
            with open(yaml_config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            model_config = dict2attr(yaml_config['model'])
        else:
            yaml_path = config['model']['config_path']
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            model_config = dict2attr(yaml_config['model'])
        
        model = LS4_ts(model_config)
        model_path = save_path / "model_LS4.pth"
        
    elif model_type == "VRNN":
        from generative_ts.models.vrnn import VRNN_ts
        
        model_config = config['model']
        model = VRNN_ts(
            x_dim=1,
            z_dim=model_config.get('z_dim', 1),
            h_dim=model_config.get('h_dim', 10),
            n_layers=model_config.get('n_layers', 1),
            lmbd=model_config.get('lmbd', 0),
            std_Y=config['data'].get('std_Y', 0.01)
        )
        model_path = save_path / "model_VRNN.pth"
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    return model, config, model_type


def load_gp_theoretical():
    """Load theoretical GP_ts model"""
    dataset_name = "gp_samples_T300_tau10_v10_stdY0.01_sigmaf1_fixed"
    dataset_path = Path('generative_ts/dataset') / dataset_name / 'config.json'
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"GP dataset not found: {dataset_path}")
    
    model = GP_ts(config_path=str(dataset_path))
    
    # Load config for compatibility
    with open(dataset_path, 'r') as f:
        config = json.load(f)
    
    return model, config, "GP_ts"


def create_test_session():
    """Create timestamped test session directory"""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    session_dir = Path("generative_ts/tests") / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    
    return session_dir, timestamp


def main():
    """Main evaluation function"""
    print("🧪 Multi-Model Evaluation Test")
    print("=" * 60)
    
    # Create test session
    session_dir, timestamp = create_test_session()
    print(f"Test session: {timestamp}")
    print(f"Results will be saved to: {session_dir}")
    
    # Load models
    models = []
    model_names = []
    configs = []
    
    for model_spec in MODELS_TO_TEST:
        try:
            if model_spec == "GP_ts":
                print(f"\n📊 Loading theoretical GP_ts model...")
                model, config, model_type = load_gp_theoretical()
            else:
                print(f"\n📊 Loading saved model: {model_spec}")
                model, config, model_type = load_saved_model(model_spec)
            
            models.append(model)
            model_names.append(model_type)
            configs.append(config)
            
            print(f"✅ {model_type} loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load {model_spec}: {e}")
            continue
    
    if not models:
        print("❌ No models loaded successfully")
        return 1
    
    print(f"\n🎯 Loaded {len(models)} models: {model_names}")
    
    # Determine dataset path (use first config)
    dataset_name = configs[0].get('dataset_name', 'gp_samples_T300_tau10_v10_stdY0.01_sigmaf1_fixed')
    dataset_path = f"generative_ts/dataset/{dataset_name}"
    
    # Load test data
    samples_path = os.path.join(dataset_path, "samples.npy")
    if not os.path.exists(samples_path):
        print(f"❌ Dataset samples not found: {samples_path}")
        return 1
    
    Y_N_full = np.load(samples_path)
    SAMPLE_IDX = 0  # Use consistent sample index across all plots
    Y_test = torch.from_numpy(Y_N_full[SAMPLE_IDX]).float()
    
    # Load config
    config_path = os.path.join(dataset_path, "config.json")
    with open(config_path, 'r') as f:
        import json
        config = json.load(f)
    
    # Create dataset dict for eval functions
    dataset_dict = {
        'Y_test': Y_N_full,
        'config': config
    }
    dataset_file = os.path.join(dataset_path, "dataset.pth")
    torch.save(dataset_dict, dataset_file)
    dataset_path = dataset_file  # Update to point to torch file
    
    print(f"📈 Test data (sample {SAMPLE_IDX}): T={len(Y_test)}, mean={Y_test.mean():.2f}, std={Y_test.std():.2f}")
    print(f"🎯 Testing: t_0={T_CONDITIONING}, T={T_TOTAL}")
    print(f"📊 Dataset config loaded: sigma_Y={config.get('sigma_Y', 'unknown')}")
    
    # Save session info
    session_info = {
        'timestamp': timestamp,
        'models': model_names,
        'model_folders': MODELS_TO_TEST,
        'dataset': dataset_name,
        'T_total': T_TOTAL,
        'T_conditioning': T_CONDITIONING,
        'n_samples': N_SAMPLES
    }
    
    with open(session_dir / "session_info.json", 'w') as f:
        json.dump(session_info, f, indent=2)
    
    # Create subdirectories
    (session_dir / "posterior").mkdir(exist_ok=True)
    (session_dir / "posterior" / "pred").mkdir(exist_ok=True)
    (session_dir / "sample").mkdir(exist_ok=True)
    
    try:
        # 1. Generate plot_posterior for each model individually
        print(f"\n🔍 Generating individual plot_posterior...")
        
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            print(f"  Processing {model_name}...")
            
            save_name = f"posterior_{model_name}_{timestamp}.png"
            plot_name = f"{model_name} Posterior (t_0={T_CONDITIONING})"
            
            plot_posterior(
                model=model,
                save_path=str(session_dir / "posterior"),
                epoch=timestamp,
                model_name=model_name,
                dataset_path=dataset_path,
                fixed_sample_idx=0
            )
            
            print(f"    ✅ {save_name}")
        
        # 2. Generate plot_sample for each model individually
        print(f"\n📈 Generating individual plot_sample...")
        
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            print(f"  Processing {model_name}...")
            
            individual_save_path = str(session_dir / "sample" / f"sample_{model_name}_{timestamp}.png")
            
            plot_sample(
                model=model,
                save_path=str(session_dir / "sample"),
                epoch=timestamp,
                model_name=model_name,
                dataset_path=dataset_path,
                fixed_sample_idx=SAMPLE_IDX
            )
            
            print(f"    ✅ sample_{model_name}_{timestamp}.png")
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 All results saved to: {session_dir}")
        print(f"📊 Session info: {session_dir / 'session_info.json'}")
        
        # Print summary
        print(f"\n📋 Generated files:")
        print(f"  📊 Individual posteriors: {len(models)} files in posterior/pred/")
        print(f"  🔗 Multi-model comparison: sample/multi_model_comparison_{timestamp}.png")
        print(f"  📈 Individual samples: {len(models)} files in sample/")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())