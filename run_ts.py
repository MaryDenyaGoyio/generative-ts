from __future__ import annotations

from pathlib import Path

from generative_ts.bandit.run_experiment import run_batch

RUN_CONFIGS = [
    {
        "run_dir": "250925_114532_VRNN_stdY1",
        "save_root": Path("outputs/batch_stdY1"),
    },
    {
        "run_dir": "250925_142451_VRNN_stdY1_lmbd0.1",
        "save_root": Path("outputs/batch_stdY1_lmbd0p1"),
    },
]

COMMON = dict(
    num_instances=100,
    instance_start=0,
    num_arms=3,
    repeats=1,
    n_samples=1,
    verbose=0,
    arm_idx=0,
    post_samples=1,
    force=False,
)


def main():
    for cfg in RUN_CONFIGS:
        run_batch(
            run_dir=cfg["run_dir"],
            save_root=cfg["save_root"],
            **COMMON,
        )


if __name__ == "__main__":
    main()
