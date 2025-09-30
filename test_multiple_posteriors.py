#!/usr/bin/env python3
"""
Generate posterior plots for multiple different data samples using a saved VRNN model
"""
import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from generative_ts.models.vrnn import VRNN_ts
from generative_ts.eval import plot_posterior


def load_model_and_config(model_path):
    """Load saved model and its configuration"""

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Load config
    config_path = Path(model_path).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model_config = config['model']

    # Ensure std_Y provided in config for reproducible posterior plots
    if 'std_Y' not in model_config:
        raise KeyError("config['model']['std_Y']가 필요합니다. config.json에 std_Y를 추가하세요.")
    model = VRNN_ts(**model_config)

    # Load state dict (use strict=False to handle mismatched keys)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Move model to CPU to avoid device conflicts
    model = model.cpu()

    # Force all parameters to CPU
    for param in model.parameters():
        param.data = param.data.cpu()

    model.eval()

    return model, config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate posterior plots for multiple samples.")
    parser.add_argument("--model-path", type=Path, help="경로가 명시되면 해당 체크포인트 사용")
    parser.add_argument("--run", type=str, help="saves 하위 디렉터리 이름")
    parser.add_argument("--saves-dir", type=Path, default=Path("generative_ts/saves"), help="체크포인트 루트")
    parser.add_argument("--num-samples", type=int, default=10, help="플롯을 생성할 시퀀스 개수")
    parser.add_argument("--mc-samples", type=int, default=100, help="posterior 샘플 수")
    return parser.parse_args()


def resolve_model_path(args):
    if args.model_path:
        model_path = args.model_path
    elif args.run:
        model_path = args.saves_dir / args.run / "model_VRNN.pth"
    else:
        candidates = sorted(args.saves_dir.glob("*/model_VRNN.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"model_VRNN.pth 파일을 {args.saves_dir}에서 찾을 수 없습니다.")
        model_path = candidates[0]

    if not model_path.exists():
        available = sorted(p.parent.name for p in args.saves_dir.glob("*/model_VRNN.pth"))
        hint = "\n".join(available) if available else "(없음)"
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {model_path}\n"
            f"사용 가능한 run 디렉터리:\n{hint}"
        )

    return model_path


def main():
    args = parse_args()
    model_path = resolve_model_path(args)

    print(f"Loading model from: {model_path}")
    model, config = load_model_and_config(model_path)

    dataset_path = config['dataset_path']
    print(f"Dataset path: {dataset_path}")

    output_dir = Path("test_multiple_posteriors")
    output_dir.mkdir(exist_ok=True)

    print(f"Generating posterior plots for {args.num_samples} different data samples...")

    for sample_idx in range(args.num_samples):
        print(f"  Generating plot for sample {sample_idx}...")

        plot_posterior(
            model=model,
            save_path=str(output_dir),
            epoch=f"sample_{sample_idx}",
            model_name="VRNN",
            dataset_path=dataset_path,
            idx=sample_idx,
            N_samples=args.mc_samples
        )

    print(f"✅ All plots saved to: {output_dir}")
    print("   Generated files:")
    for i in range(args.num_samples):
        print(f"     posterior_VRNN_sample_{i}_{args.mc_samples}samples.png")


if __name__ == "__main__":
    main()
