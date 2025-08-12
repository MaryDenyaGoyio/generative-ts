#!/usr/bin/env python3
"""
실험 실행 스크립트
Usage: python run_experiment.py [config_file]
"""

import sys
import json
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from generative_ts.train import train_model

def load_config(config_path: str) -> dict:
    """JSON config 파일을 로드합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def main():
    """메인 실행 함수"""
    
    # Config 파일 경로 설정
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # 기본값: LS4 실험
        config_file = "generative_ts/config/ls4_experiment.json"
    
    # Config 로드
    print(f"Loading config from: {config_file}")
    config = load_config(config_file)
    
    # Config 내용 출력
    print("=" * 50)
    print("Experiment Configuration:")
    print("=" * 50)
    print(json.dumps(config, indent=2))
    print("=" * 50)
    
    # 실험 실행
    print(f"Starting {config['model_type']} training...")
    model = train_model(config)
    
    print(f"✅ {config['model_type']} training completed successfully!")
    
    return model

if __name__ == "__main__":
    main()