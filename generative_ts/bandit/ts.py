from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_DIR = PROJECT_ROOT / "repos"
if REPO_DIR.exists():
    repo_str = str(REPO_DIR)
    if repo_str not in sys.path:
        sys.path.append(repo_str)

from ..utils.checkpoint import load_model_from_checkpoint


# instance 정보도 저장해야함

def thompson_sampling(model, rewards_np: np.ndarray, latent_np: np.ndarray, n_samples: int = 1):
    device = next(model.parameters()).device

    rewards = torch.from_numpy(rewards_np).float().to(device)
    latent = latent_np[..., 0]

    K, T, D = rewards.shape
    if D != 1:  raise ValueError("Only supports x_dim=1")

    observed = [[] for _ in range(K)]
    regret, choices = [], []
    theta_samples = np.zeros((T, K, n_samples), dtype=float)

    for t in range(T):
        theta_draws = np.zeros(K, dtype=float)

        for k in range(K):
            idx = np.asarray(observed[k], dtype=int)

            if idx.size:    obs = rewards[k, torch.as_tensor(idx, device=device)]
            else:           obs = torch.empty(0, D, device=device)

            z_t = model.sample_z_t(obs, target_t=t, mask=idx if idx.size else None, n_samples=n_samples)
            z_np = z_t[:, 0].cpu().numpy()

            theta_draws[k], theta_samples[t, k] = float(z_np[0]), z_np

        chosen = int(theta_draws.argmax())
        observed[chosen].append(t)
        inst_regret = float(latent[:, t].max()) - float(latent[chosen, t])

        regret.append(inst_regret)
        choices.append(chosen)

    return {
        "choices": np.asarray(choices, dtype=int),
        "regret": np.asarray(regret, dtype=float),
        "theta_samples": theta_samples,
    }



def _plot_run(model, rewards, latent, out, colors, arm_idx=0, post_samples=100, save_path=None):
    device = next(model.parameters()).device

    theta_samples = out.get("theta_samples")
    T = theta_samples.shape[0]
    t = np.arange(T)

    choices = out.get("choices")
    mask_idx = np.sort(np.where(choices == arm_idx)[0] if choices is not None else np.arange(0))

    if mask_idx.size:
        x_obs = torch.from_numpy(rewards[arm_idx, mask_idx]).float().to(device)
        mask_arg = mask_idx
    else:
        x_obs = torch.empty(0, rewards.shape[-1], device=device)
        mask_arg = None

    # plot
    fig, (ax_main, ax_post, ax_reg) = plt.subplots(
        3, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1]}
    )

    # 1st
    for k, c in enumerate(colors):
        y_k = rewards[k, :T, 0]
        ax_main.scatter(t, y_k, s=10, color=c, alpha=0.2)
        ax_main.plot(t, latent[k, :T, 0], color=c, alpha=0.5)
        ax_main.plot(t, theta_samples[:, k, 0], linestyle="--", color=c, alpha=0.7)

        samples_k = theta_samples[:, k]
        if samples_k.shape[1] > 1:
            mean_k = samples_k.mean(axis=1)
            std_k = samples_k.std(axis=1, ddof=0)
            ax_main.fill_between(
                t,
                mean_k - 2 * std_k,
                mean_k + 2 * std_k,
                color=c,
                alpha=0.3,
                linewidth=0,
            )

        if choices is not None:
            picked = np.where(choices == k)[0]
            if picked.size:
                ax_main.scatter(picked, y_k[picked], s=10, color=c, alpha=0.9)

    ax_main.set_ylabel("reward / latent")

    # 2nd
    with torch.no_grad():
        post = model.posterior(x_obs, T, post_samples, mask=mask_arg)

    mu, sigma = post["z_mean"][:T, 0], post["z_std"][:T, 0]

    arm_color = colors[arm_idx]
    arm_rewards = rewards[arm_idx, :T, 0]
    arm_latent = latent[arm_idx, :T, 0]

    ax_post.scatter(t, arm_rewards, s=10, color=arm_color, alpha=0.2, label="outcome")
    if mask_idx.size:   ax_post.scatter(mask_idx, arm_rewards[mask_idx], s=10, color=arm_color, alpha=0.9)

    ax_post.plot(t, arm_latent, color=arm_color, alpha=0.5, label="latent")
    
    ax_post.plot(t, mu, color=arm_color, linestyle="--", linewidth=2, label="posterior $\\mu$")
    ax_post.fill_between(t, mu - 2 * sigma, mu + 2 * sigma, color=arm_color, alpha=0.2)
    
    ax_post.set_ylabel("posterior")
    ax_post.legend(loc="upper right", fontsize=8)

    # 3rd
    ax_reg.plot(t, np.cumsum(out["regret"]), color="black")
    
    ax_reg.set_ylabel("cumulative regret")
    ax_reg.set_xlabel("t")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)

    plt.close(fig)



def run_ts(
    run_dir: str | Path,
    num_arms: int,
    n_samples: int = 1,
    verbose: int = 1,
    arm_idx: int = 0,
    post_samples: int = 100,
    save_path: str | Path | None = None,
    recompute: bool = False,
    instance_idx: int = 0,
):

    # set path
    run_path = Path(run_dir)
    if not run_path.exists():
        candidate = PROJECT_ROOT / "generative_ts" / "saves" / str(run_dir)
        if candidate.exists():  run_path = candidate
    if not run_path.exists():   raise FileNotFoundError(f"run_dir not found: {run_dir}")

    save_dir = run_path / "ts" if save_path is None else Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    # set model & data
    resume = load_model_from_checkpoint(run_path)
    model = resume["model"].eval()

    dataset_path = resume["config"].get("dataset_path")
    if dataset_path is None:    raise KeyError("dataset_path not found")

    data_dir = Path(dataset_path)
    if not data_dir.exists():
        candidate = PROJECT_ROOT / dataset_path
        if candidate.exists():  data_dir = candidate
    if not data_dir.exists():   raise FileNotFoundError(f"dataset not found: {dataset_path}")

    Y = np.load(data_dir / "outcome_Y.npy")
    theta = np.load(data_dir / "latent_theta.npy")

    rewards, latent = Y[start:end], theta[start:end]


    # set results
    result_path = base_dir / f"TS_idx{instance_idx}_K{num_arms}.npz"

    if result_path.exists() and not recompute:  
        with np.load(result_path) as data:
            out = {
                "choices": data["choices"],
                "regret": data["regret"],
                "theta_samples": data["theta_samples"],
            }
    else:
        out = thompson_sampling(model, rewards, latent, n_samples)

        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            result_path,
            choices=out["choices"],
            regret=out["regret"],
            theta_samples=out["theta_samples"],
        )


    # plot
    if verbose:
        cmap = plt.get_cmap("tab10", num_arms)
        colors = [cmap(i) for i in range(num_arms)]
        
        instance_path = save_dir / f"instance_idx{instance_idx}.png"
        run_plot_path = save_dir / f"TS_idx{instance_idx}.png"


        # instance plot
        t = np.arange(rewards.shape[1])

        fig, ax = plt.subplots(figsize=(8, 4))
        for k, c in enumerate(colors):
            ax.scatter(t, rewards[k, :, 0], s=12, color=c, alpha=0.25)
            ax.plot(t, latent[k, :, 0], color=c, alpha=0.5)

        ax.set_title("Bandit instance")
        ax.set_xlabel("t")
        ax.set_ylabel("reward")
        fig.tight_layout()

        instance_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(instance_path, dpi=150)
        plt.close(fig)


        # TS plot
        _plot_run(
            model,
            rewards,
            latent,
            out,
            colors,
            arm_idx=arm_idx,
            post_samples=post_samples,
            save_path=run_plot_path,
        )

    return out
