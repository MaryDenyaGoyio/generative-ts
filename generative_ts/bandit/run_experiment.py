from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .ts import run_ts


def parse_args():
    parser = argparse.ArgumentParser("Run Thompson Sampling across dataset instances")
    parser.add_argument("--run-dir", type=str, default="251001_025244_VRNN_stdY0.01")
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument("--instance-start", type=int, default=0)
    parser.add_argument("--num-arms", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10, help="number of stochastic runs per instance")
    parser.add_argument("--n-samples", type=int, default=1, help="samples per step for z_t")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save-root", type=str, default="outputs/batch")
    parser.add_argument("--force", action="store_true", help="recompute even if cached results exist")
    parser.add_argument("--post-samples", type=int, default=100)
    parser.add_argument("--arm-idx", type=int, default=0)
    return parser.parse_args()


def run_batch(
    *,
    run_dir: str | Path,
    save_root: str | Path,
    num_instances: int,
    instance_start: int,
    num_arms: int,
    repeats: int,
    n_samples: int,
    verbose: int,
    arm_idx: int,
    post_samples: int,
    force: bool,
):
    save_root = Path(save_root)
    inst_iter = range(instance_start, instance_start + num_instances)

    for inst_idx in tqdm(inst_iter, desc=f"Instances[{run_dir}]", unit="inst"):
        for rep in range(repeats):
            prefix = save_root / f"inst_{inst_idx}" / f"rep_{rep}"
            run_ts(
                run_dir=run_dir,
                num_arms=num_arms,
                n_samples=n_samples,
                verbose=verbose,
                arm_idx=arm_idx,
                post_samples=post_samples,
                save_prefix=prefix,
                recompute=force,
                instance_idx=inst_idx,
            )


def main():
    args = parse_args()
    run_batch(
        run_dir=args.run_dir,
        save_root=args.save_root,
        num_instances=args.num_instances,
        instance_start=args.instance_start,
        num_arms=args.num_arms,
        repeats=args.repeats,
        n_samples=args.n_samples,
        verbose=args.verbose,
        arm_idx=args.arm_idx,
        post_samples=args.post_samples,
        force=args.force,
    )


if __name__ == "__main__":
    main()
