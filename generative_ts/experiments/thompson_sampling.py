"""Thompson sampling experiment runner for pretrained generative TS models."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch

from generative_ts.train import load_pretrained_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnsupportedModelError(RuntimeError):
    """Raised when Thompson sampling is requested for an unsupported model."""


def _ensure_tensor(value: float | np.ndarray | torch.Tensor, device: torch.device, dim: int) -> torch.Tensor:
    """Convert scalar/array to a shape (1, dim) float tensor on the target device."""
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
        tensor = torch.from_numpy(arr)
        tensor = tensor.to(device=device)
    tensor = tensor.view(1, -1)
    if tensor.shape[1] != dim:
        if tensor.shape[1] == 1 and dim != 1:
            tensor = tensor.expand(1, dim)
        elif tensor.shape[1] > dim:
            tensor = tensor[:, :dim]
        else:
            tensor = torch.nn.functional.pad(tensor, (0, dim - tensor.shape[1]))
    return tensor


@dataclass
class VRNNArmState:
    """Arm state for VRNN-based Thompson sampling."""

    model: torch.nn.Module
    device: torch.device

    def __post_init__(self) -> None:
        self.h = torch.zeros(self.model.n_layers, 1, self.model.h_dim, device=self.device)

    @torch.no_grad()
    def prior_sample(self) -> dict:
        prior_hidden = self.model.prior(self.h[-1])
        prior_mean = self.model.prior_mean(prior_hidden)
        prior_std = self.model.prior_std(prior_hidden)
        z_sample = self.model._reparameterized_sample(prior_mean, prior_std)
        sigma_y = torch.exp(self.model.log_std_y)
        obs_sample = self.model._reparameterized_sample(z_sample, sigma_y.expand_as(z_sample))
        reward_scalar = float(z_sample.squeeze().item())
        return {
            "z_sample": z_sample.detach(),
            "prior_mean": prior_mean.detach(),
            "prior_std": prior_std.detach(),
            "obs_sample": obs_sample.detach(),
            "reward": reward_scalar,
        }

    @torch.no_grad()
    def advance_with_prediction(self, z_sample: torch.Tensor, obs_sample: torch.Tensor) -> None:
        x_feed = _ensure_tensor(obs_sample, self.device, self.model.x_dim)
        phi_x = self.model.phi_x(x_feed)
        phi_z = self.model.phi_z(z_sample)
        rnn_input = torch.cat([phi_x, phi_z], dim=1).unsqueeze(0)
        _, self.h = self.model.rnn(rnn_input, self.h)

    @torch.no_grad()
    def update_with_observation(self, observation: float) -> dict:
        x_tensor = _ensure_tensor(observation, self.device, self.model.x_dim)
        phi_x = self.model.phi_x(x_tensor)
        enc_input = torch.cat([phi_x, self.h[-1]], dim=1)
        enc_hidden = self.model.enc(enc_input)
        enc_mean = self.model.enc_mean(enc_hidden)
        enc_std = self.model.enc_std(enc_hidden)
        z_post = self.model._reparameterized_sample(enc_mean, enc_std)
        phi_z = self.model.phi_z(z_post)
        rnn_input = torch.cat([phi_x, phi_z], dim=1).unsqueeze(0)
        _, self.h = self.model.rnn(rnn_input, self.h)
        reward_scalar = float(z_post.squeeze().item())
        return {
            "z_post": z_post.detach(),
            "enc_mean": enc_mean.detach(),
            "enc_std": enc_std.detach(),
            "reward": reward_scalar,
        }


@dataclass
class ExperimentConfig:
    checkpoint_dir: Path
    dataset_path: Path
    model_type: str
    config_path: Path
    num_arms: int
    horizon: int | None
    num_experiments: int
    seed: int
    save_dir: Path | None


def parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run Thompson sampling on AR1 arms.")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint directory (contains model_*.pth).")
    parser.add_argument("dataset", type=Path, help="Path to dataset directory containing samples.npy.")
    parser.add_argument("--model-type", choices=["VRNN", "LS4"], help="Override model type if not inferable.")
    parser.add_argument("--config", type=Path, help="Path to model config file if different from checkpoint.")
    parser.add_argument("--num-arms", type=int, default=5, help="Number of AR1 arms (K).")
    parser.add_argument(
        "--horizon",
        type=int,
        default=300,
        help="Planning horizon T (default: 300, clipped to dataset length).",
    )
    parser.add_argument("--experiments", type=int, default=5, help="Number of independent Thompson runs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--save-dir", type=Path, help="Optional directory to save outputs; defaults under checkpoint.")
    args = parser.parse_args(argv)

    checkpoint_dir = args.checkpoint
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    dataset_path = args.dataset
    samples_file = dataset_path / "samples.npy"
    if not samples_file.exists():
        raise FileNotFoundError(f"samples.npy not found under {dataset_path}")

    config_path = args.config
    if config_path is None:
        default_json = checkpoint_dir / "config.json"
        config_path = default_json if default_json.exists() else None

    model_type = args.model_type
    if model_type is None:
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type") or cfg.get("model", {}).get("type")
        if model_type is None:
            raise ValueError("Cannot infer model type; provide --model-type explicitly.")
    model_type = model_type.upper()

    return ExperimentConfig(
        checkpoint_dir=checkpoint_dir,
        dataset_path=dataset_path,
        model_type=model_type,
        config_path=config_path if config_path else Path(),
        num_arms=args.num_arms,
        horizon=args.horizon,
        num_experiments=args.experiments,
        seed=args.seed,
        save_dir=args.save_dir,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(cfg: ExperimentConfig) -> torch.nn.Module:
    model_glob = next(cfg.checkpoint_dir.glob("model_*.pth"), None)
    if model_glob is None:
        raise FileNotFoundError(f"No model_*.pth file found in {cfg.checkpoint_dir}")
    if cfg.config_path and cfg.config_path.exists():
        config_path = cfg.config_path
    else:
        raise FileNotFoundError("Model config path is required to load the checkpoint.")
    model, _ = load_pretrained_model(
        model_path=str(model_glob),
        config_path=str(config_path),
        model_type=cfg.model_type,
    )
    model.eval()
    return model


def load_dataset(cfg: ExperimentConfig) -> np.ndarray:
    samples = np.load(cfg.dataset_path / "samples.npy")
    return samples


def ensure_save_dir(cfg: ExperimentConfig) -> Path:
    if cfg.save_dir:
        save_dir = cfg.save_dir
    else:
        save_dir = cfg.checkpoint_dir / "thompson"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _sample_arm_sequences(samples: np.ndarray, num_arms: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if samples.shape[0] < num_arms:
        raise ValueError(f"Dataset only has {samples.shape[0]} sequences but {num_arms} arms requested.")
    indices = np.random.choice(samples.shape[0], size=num_arms, replace=False)
    effective_horizon = min(horizon, samples.shape[1])
    sliced = samples[indices, :effective_horizon]
    return sliced, indices


def _compute_cum_avg_regret(regrets: List[float]) -> np.ndarray:
    arr = np.array(regrets, dtype=np.float32)
    cumsum = np.cumsum(arr)
    steps = np.arange(1, len(arr) + 1, dtype=np.float32)
    return cumsum / steps


def run_vrnn_thompson(
    model: torch.nn.Module,
    sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    num_arms, horizon = sequences.shape
    arms = [VRNNArmState(model=model, device=device) for _ in range(num_arms)]

    regrets: List[float] = []
    predicted = np.zeros((num_arms, horizon), dtype=np.float32)
    chosen_indices = np.zeros(horizon, dtype=np.int64)
    with torch.no_grad():
        for t in range(horizon):
            prior_info = [arm.prior_sample() for arm in arms]
            rewards = [info["reward"] for info in prior_info]
            chosen = int(np.argmax(rewards))
            predicted[:, t] = np.asarray(rewards, dtype=np.float32)
            chosen_indices[t] = chosen

            actual_rewards = sequences[:, t]
            observed = float(actual_rewards[chosen])
            optimal = float(np.max(actual_rewards))
            regrets.append(optimal - observed)

            for idx, arm in enumerate(arms):
                if idx == chosen:
                    arm.update_with_observation(observed)
                else:
                    arm.advance_with_prediction(
                        prior_info[idx]["z_sample"], prior_info[idx]["obs_sample"]
                    )

    return _compute_cum_avg_regret(regrets), predicted, chosen_indices


def run_single_experiment(
    model: torch.nn.Module,
    model_type: str,
    samples: np.ndarray,
    num_arms: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences, indices = _sample_arm_sequences(samples, num_arms, horizon)
    if model_type == "VRNN":
        path, predicted, chosen = run_vrnn_thompson(model, sequences)
        return path, indices, predicted, chosen, sequences
    raise UnsupportedModelError("Currently only VRNN models are supported for Thompson sampling.")


def run_experiments(
    model: torch.nn.Module,
    model_type: str,
    samples: np.ndarray,
    num_arms: int,
    horizon: int,
    num_runs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runs = []
    all_indices = []
    predicted_runs = []
    chosen_runs = []
    sequence_runs = []
    for _ in range(num_runs):
        path, indices, predicted, chosen, sequences = run_single_experiment(
            model, model_type, samples, num_arms, horizon
        )
        runs.append(path)
        all_indices.append(indices)
        predicted_runs.append(predicted)
        chosen_runs.append(chosen)
        sequence_runs.append(sequences)
    return (
        np.stack(runs, axis=0),
        np.stack(all_indices, axis=0),
        np.stack(predicted_runs, axis=0),
        np.stack(chosen_runs, axis=0),
        np.stack(sequence_runs, axis=0),
    )


def main(argv: Sequence[str] | None = None) -> None:
    cfg = parse_args(argv)
    set_seed(cfg.seed)

    model = load_model(cfg)
    samples = load_dataset(cfg)

    horizon = cfg.horizon or samples.shape[1]
    horizon = min(horizon, samples.shape[1])

    save_dir = ensure_save_dir(cfg)
    print(f"Loaded model {cfg.model_type} from {cfg.checkpoint_dir}")
    print(f"Dataset shape: {samples.shape}, using horizon {horizon}")
    print(f"Saving outputs to: {save_dir}")

    if cfg.model_type != "VRNN":
        raise UnsupportedModelError(
            f"Model type {cfg.model_type} is not yet supported for Thompson sampling."
        )

    (
        regret_paths,
        arm_indices,
        predicted_paths,
        chosen_arms,
        actual_sequences,
    ) = run_experiments(
        model=model,
        model_type=cfg.model_type,
        samples=samples,
        num_arms=cfg.num_arms,
        horizon=horizon,
        num_runs=cfg.num_experiments,
    )

    mean_path = regret_paths.mean(axis=0)
    time_axis = np.arange(1, horizon + 1)

    results_path = save_dir / "thompson_regret_results.npz"
    np.savez(
        results_path,
        regret_paths=regret_paths,
        mean_path=mean_path,
        time=time_axis,
        arm_indices=arm_indices,
        predicted_paths=predicted_paths,
        chosen_arms=chosen_arms,
        actual_sequences=actual_sequences,
        config={
            "num_arms": cfg.num_arms,
            "horizon": horizon,
            "num_experiments": cfg.num_experiments,
            "seed": cfg.seed,
            "model_type": cfg.model_type,
            "checkpoint": str(cfg.checkpoint_dir),
            "dataset": str(cfg.dataset_path),
        },
    )

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    for path in regret_paths:
        ax.plot(time_axis, path, color="tab:gray", alpha=0.4, linewidth=1)

    ax.plot(time_axis, mean_path, color="tab:red", linewidth=2, label="Mean path")

    if cfg.num_experiments > 1:
        stderr = regret_paths.std(axis=0) / math.sqrt(cfg.num_experiments)
        ax.fill_between(
            time_axis,
            mean_path - stderr,
            mean_path + stderr,
            color="tab:red",
            alpha=0.2,
            label="±1 SE",
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative average regret")
    ax.set_title(f"Thompson sampling regret ({cfg.num_experiments} runs, {cfg.num_arms} arms)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = save_dir / "thompson_regret_paths.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    final_mean = float(mean_path[-1])
    print(f"Final cumulative average regret: {final_mean:.6f}")
    print(f"Saved regret data to {results_path}")
    print(f"Saved plot to {plot_path}")

    # Example diagnostic plots using the first experiment run
    if regret_paths.shape[0] > 0:
        example_idx = 0
        arm_order = arm_indices[example_idx]
        actual = actual_sequences[example_idx]
        predicted = predicted_paths[example_idx]
        chosen = chosen_arms[example_idx]

        time_axis = np.arange(1, actual.shape[1] + 1)
        cmap = plt.get_cmap("tab10", actual.shape[0])

        fig, ax = plt.subplots(figsize=(12, 4))
        for k in range(actual.shape[0]):
            color = cmap(k)
            ax.plot(
                time_axis,
                predicted[k],
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.plot(
                time_axis,
                actual[k],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )
            mask = chosen == k
            if np.any(mask):
                ax.scatter(
                    time_axis[mask],
                    actual[k, mask],
                    color=color,
                    edgecolors="k",
                    linewidths=0.6,
                    s=36,
                    zorder=3,
                )

        style_handles = [
            Line2D([], [], color="k", linestyle="-", linewidth=1.5, label="Predicted z_t"),
            Line2D([], [], color="k", linestyle="--", linewidth=1.5, label="Actual reward"),
            Line2D(
                [],
                [],
                color="k",
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="k",
                label="Observed reward (chosen)",
            ),
        ]
        style_legend = ax.legend(handles=style_handles, loc="upper left")
        ax.add_artist(style_legend)

        arm_handles = [
            Patch(facecolor=cmap(k), edgecolor="k", label=f"Arm {k} (id {arm_order[k]})")
            for k in range(actual.shape[0])
        ]
        ax.legend(handles=arm_handles, loc="upper right")

        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward / latent")
        ax.set_title("Example run: actual rewards vs. sampled z_t")
        ax.grid(alpha=0.3)
        combined_plot = save_dir / "example_run_rewards_and_z.png"
        fig.tight_layout()
        fig.savefig(combined_plot, dpi=200)
        plt.close(fig)

        print(f"Saved combined reward/z plot to {combined_plot}")


if __name__ == "__main__":
    main()
