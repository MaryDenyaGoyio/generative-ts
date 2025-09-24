#!/usr/bin/env python3
"""
실험 실행 스크립트
Usage:
  # New training
  python run.py -c ls4                                         # Smart config resolution
  python run.py --config vrnn                                  # Same as vrnn_experiment.json
  python run.py -c ls4_experiment.json                         # Explicit file name
  python run.py --epochs 1000 --lr 0.001                      # With overrides

  # Resume training
  python run.py -r 250925_032402                              # Short name
  python run.py --resume generative_ts/saves/250925_032402    # Full path
  python run.py -r 250925_032402 --epochs 100                # Resume with overrides

  # Available overrides
  --epochs N        Override number of epochs
  --lr FLOAT        Override learning rate
  --batch-size N    Override batch size
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from generative_ts.train import train_model
from generative_ts.utils import resume_training_setup

def load_config(config_path: str) -> dict:
    """JSON config 파일을 로드합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def resolve_checkpoint_path(checkpoint_name: str) -> str:
    """Resolve checkpoint path from short name or full path"""
    checkpoint_path = Path(checkpoint_name)

    # If it's already a full path and exists, use it
    if checkpoint_path.is_absolute() and checkpoint_path.exists():
        return str(checkpoint_path)

    # If it's a relative path and exists, use it
    if checkpoint_path.exists():
        return str(checkpoint_path)

    # Try to find it in saves directory
    saves_dir = Path("generative_ts/saves")
    potential_path = saves_dir / checkpoint_name
    if potential_path.exists():
        return str(potential_path)

    # If checkpoint_name is just a timestamp, try to find exact match
    if saves_dir.exists():
        for dir_path in saves_dir.iterdir():
            if dir_path.is_dir() and checkpoint_name in dir_path.name:
                return str(dir_path)

    # If nothing found, return original path (will cause error later with helpful message)
    return checkpoint_name

def resolve_config_path(config_name: str) -> str:
    """Resolve config path from short name or full path"""
    config_path = Path(config_name)

    # If it's already a full path and exists, use it
    if config_path.is_absolute() and config_path.exists():
        return str(config_path)

    # If it's a relative path and exists, use it
    if config_path.exists():
        return str(config_path)

    # If it already has .json extension
    if config_name.endswith('.json'):
        # Try to find it in config directory
        config_dir = Path("generative_ts/config")
        potential_path = config_dir / config_name
        if potential_path.exists():
            return str(potential_path)
    else:
        # Try to find it with _experiment.json suffix
        config_dir = Path("generative_ts/config")
        potential_path = config_dir / f"{config_name}.json"
        if potential_path.exists():
            return str(potential_path)

    # If nothing found, return original path (will cause error later with helpful message)
    return config_name

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Training script for generative time series models')

    # Main arguments
    parser.add_argument('--config', '-c', type=str,
                       default="vrnn",
                       help='Configuration name (e.g., "ls4", "vrnn") or full path')

    # Resume options
    parser.add_argument('--resume', '-r', type=str, metavar='CHECKPOINT_NAME',
                       help='Resume training from checkpoint (full path or short name like "250925_032402")')

    # Training overrides
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, help='Override batch size')

    return parser.parse_args()

def main():
    """메인 실행 함수"""
    args = parse_arguments()

    # Resume training mode
    if args.resume:
        # Resolve short checkpoint names
        checkpoint_path = resolve_checkpoint_path(args.resume)
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("💡 Available checkpoints:")
            saves_dir = Path("generative_ts/saves")
            if saves_dir.exists():
                for dir_path in sorted(saves_dir.iterdir()):
                    if dir_path.is_dir():
                        print(f"   {dir_path.name}")
            return 1

        # Prepare config overrides
        config_override = {}
        if args.epochs:
            config_override['training'] = {'n_epochs': args.epochs}
        if args.lr:
            if 'training' not in config_override:
                config_override['training'] = {}
            config_override['training']['lr'] = args.lr
        if getattr(args, 'batch_size', None):
            if 'training' not in config_override:
                config_override['training'] = {}
            config_override['training']['batch_size'] = args.batch_size

        # Load model and config for resuming
        print(f"🔄 Resuming training from: {checkpoint_path}")
        try:
            model, config, resume_info = resume_training_setup(checkpoint_path, config_override)

            # Add resume information to config
            config['resume'] = True
            config['resume_info'] = resume_info

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # New training mode
    else:
        # Resolve config name
        config_file = resolve_config_path(args.config)
        print(f"📁 Loading config from: {config_file}")

        try:
            config = load_config(config_file)
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_file}")
            print("💡 Available config files:")
            config_dir = Path("generative_ts/config")
            if config_dir.exists():
                for config_path in sorted(config_dir.glob("*.json")):
                    print(f"   {config_path.name}")
            return 1

        # Apply command line overrides
        if args.epochs:
            config.setdefault('training', {})['n_epochs'] = args.epochs
        if args.lr:
            config.setdefault('training', {})['lr'] = args.lr
        if getattr(args, 'batch_size', None):
            config.setdefault('training', {})['batch_size'] = args.batch_size

        config['resume'] = False

    # Config 내용 출력
    print("=" * 50)
    print("Experiment Configuration:")
    print("=" * 50)
    if config.get('resume', False):
        print(f"🔄 RESUME MODE: Starting from epoch {config['resume_info']['epoch']}")
        print(f"📁 Checkpoint: {config['resume_info']['save_path']}")
        print(f"💾 Best loss: {config['resume_info']['best_loss']:.6f}")
        print("-" * 50)

    # Don't print sensitive resume_info details
    display_config = {k: v for k, v in config.items() if k != 'resume_info'}
    print(json.dumps(display_config, indent=2))
    print("=" * 50)

    # 실험 실행
    mode_str = "Resuming" if config.get('resume', False) else "Starting"
    print(f"{mode_str} {config['model_type']} training...")
    model = train_model(config)

    print(f"✅ {config['model_type']} training completed successfully!")

    return model

if __name__ == "__main__":
    main()