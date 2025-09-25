#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from generative_ts.train import train_model
from generative_ts.utils.checkpoint import load_model_from_checkpoint


def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_path(name: str, *, base: str = "", suffix: str | None = None, fuzzy: bool = False, files_only: bool = False) -> Path:
    path = Path(name)
    if path.is_absolute() and path.exists():
        if not files_only or path.is_file():
            return path
    if path.exists():
        if not files_only or path.is_file():
            return path

    base_path = Path(base) if base else None
    candidates = []

    if base_path:
        if suffix and not name.endswith(suffix):
            candidates.append(base_path / f"{name}{suffix}")
        candidates.append(base_path / name)
        for candidate in candidates:
            if candidate.exists() and (not files_only or candidate.is_file()):
                return candidate

        if fuzzy:
            for entry in base_path.glob(f"*{name}*"):
                if entry.exists() and (not files_only or entry.is_file()):
                    return entry

    return path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script for generative time series models")
    parser.add_argument('-c', '--config', default='vrnn', help='Config name or path')
    parser.add_argument('-r', '--resume', help='Checkpoint name or path')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch-size', type=int, dest='batch_size')
    return parser.parse_args()


def prepare_resume(checkpoint_name: str):
    ckpt_path = resolve_path(checkpoint_name, base='generative_ts/saves', fuzzy=True)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        saves_dir = Path('generative_ts/saves')
        if saves_dir.exists():
            print("Available checkpoints:")
            for entry in sorted(p for p in saves_dir.iterdir() if p.is_dir()):
                print(f"   {entry.name}")
        sys.exit(1)

    resume = load_model_from_checkpoint(ckpt_path)
    config = resume['config']
    config.update({
        'resume': True,
        'resume_epoch': resume['epoch'],
        'resume_losses': resume.get('all_losses', {}),
        'resume_path': str(ckpt_path),
        'loaded_model': resume['model'],
    })
    return config


def prepare_new(config_name: str, args: argparse.Namespace):
    config_path = resolve_path(config_name, base='generative_ts/config', suffix='.json', files_only=True)
    if not config_path.exists() or not config_path.is_file():
        print(f"Config file not found: {config_path}")
        config_dir = Path('generative_ts/config')
        if config_dir.exists():
            print("Available config files:")
            for entry in sorted(config_dir.glob('*.json')):
                print(f"   {entry.name}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Keep existing behaviour: overrides are stored under 'training'
    overrides = config.setdefault('training', {})
    if args.epochs is not None:
        overrides['n_epochs'] = args.epochs
    if args.lr is not None:
        overrides['lr'] = args.lr
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size

    config['resume'] = False
    return config


def main() -> None:
    args = parse_arguments()

    config = prepare_resume(args.resume) if args.resume else prepare_new(args.config, args)

    print("=" * 50)
    print("Experiment Configuration:")
    print("=" * 50)
    if config.get('resume'):
        print(f"[RESUME] : model {config['resume_path']} epoch {config['resume_epoch']}")
        print("-" * 50)

    filtered = {k: v for k, v in config.items() if not k.startswith('resume_') and k != 'loaded_model'}
    print(json.dumps(filtered, indent=2))
    print("=" * 50)

    train_model(config)
    print(f"✅ {config['model_type']} training completed successfully!")


if __name__ == '__main__':
    main()
