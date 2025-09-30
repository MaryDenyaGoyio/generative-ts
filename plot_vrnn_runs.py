#!/usr/bin/env python3
"""Generate comparison plots for multiple VRNN runs into specified directory."""
import argparse
from pathlib import Path
from test_multiple_posteriors import load_model_and_config
from generative_ts.eval import plot_posterior

DEFAULT_RUNS = {
    "250925_145223_VRNN_stdY0.01_lmbd": "VRNN_stdY0.01_lmbd.png",
    "250925_083034_VRNN_stdY0.01": "VRNN_stdY0.01.png",
    "250925_142451_VRNN_stdY1_lmbd": "VRNN_stdY1_lmbd.png",
    "250925_114532_VRNN_stdY1": "VRNN_stdY1.png",
}

def generate_plot(run_dir: Path, output_file: Path, sample_idx: int, n_samples: int):
    model_path = run_dir / "model_VRNN.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model, config = load_model_and_config(model_path)
    dataset_path = config["dataset_path"]

    output_dir = output_file.parent
    output_dir.mkdir(exist_ok=True)

    epoch = output_file.stem
    plot_posterior(
        model=model,
        save_path=str(output_dir),
        epoch=epoch,
        model_name="VRNN",
        dataset_path=dataset_path,
        idx=sample_idx,
        N_samples=n_samples,
    )

    generated = output_dir / f"latent_posterior_VRNN_{epoch}_{n_samples}samples.png"
    if generated.exists():
        generated.replace(output_file)
    elif not output_file.exists():
        raise FileNotFoundError(f"Expected plot missing: {generated}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate VRNN run plots")
    parser.add_argument("--runs", nargs="*", help="Run directories under saves")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--saves-dir", type=Path, default=Path("generative_ts/saves"))
    parser.add_argument("--output-dir", type=Path, default=Path("post"))
    return parser.parse_args()

def main():
    args = parse_args()
    runs = args.runs if args.runs else list(DEFAULT_RUNS.keys())
    name_map = {run: DEFAULT_RUNS.get(run, f"{run}.png") for run in runs}

    for run in runs:
        run_dir = args.saves_dir / run
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir missing: {run_dir}")
        output_file = args.output_dir / name_map[run]
        print(f"Generating plot for {run} -> {output_file}")
        generate_plot(run_dir, output_file, args.sample_idx, args.mc_samples)

if __name__ == "__main__":
    main()
