from . import params, data, model, utils
from .params import DEVICE
from .utils import Experiment, Experiment_p

import os
import numpy as np
import matplotlib.pyplot as plt 

import torch

def plot_posterior(expr : Experiment_p, t_0_ratio=4):
    data_main, model_main, cfg, save_name, load_path = (
        expr.data_main,
        expr.model_main,
        expr.cfg,
        expr.save_name,
        expr.load_path
    )

    _Y, _ = data_main.generate_data()
    Y = torch.from_numpy(_Y).float().to(DEVICE).unsqueeze(1)  # (T, 1, d)

    T = Y.size(0) 
    t_0 = Y.size(0) // t_0_ratio
    Y_l_t0 = Y[:t_0]
    geq_t_0 = np.arange(t_0, Y.size(0))

    mean_samples, var_samples, x_samples = model_main.inference(Y_l_t0, T=T) # type(model_main).__name__
    mu_future, sigma_future, x_future = data_main.posterior_inference(Y_l_t0, T=T)

    plot_name = os.path.join(load_path, f'test {save_name}.png') 

    plt.figure()
    plt.plot(Y_l_t0.squeeze().cpu().numpy(), label=r'$given Y_{\leq t_0}$')

    data_line, = plt.plot(geq_t_0, mu_future, label=r'$Y_{>t_0} | Y_{\leq t_0}$ by GP')
    plt.fill_between(geq_t_0, (mu_future - sigma_future).squeeze(), (mu_future + sigma_future).squeeze(), alpha=0.2, color=data_line.get_color())

    model_line, = plt.plot(geq_t_0, mean_samples, label=r'$Y_{>t_0} | Y_{\leq t_0}$ by VRNN')
    plt.fill_between(geq_t_0, (mean_samples - var_samples).squeeze(), (mean_samples + var_samples).squeeze(), alpha=0.2, color=model_line.get_color())

    n_trajectories = 10
    plt.plot(geq_t_0, x_future[:n_trajectories].T, color=data_line.get_color(), alpha=0.3, linestyle='--', linewidth=1.0,label='_nolegend_')
    plt.plot(geq_t_0, x_samples.squeeze(-1)[:n_trajectories].T, color=model_line.get_color(), alpha=0.3, linestyle='--', linewidth=1.0,label='_nolegend_')


    plt.axvline(x=t_0, color='black', linestyle='--', linewidth=1.0)
    plt.xlabel(r'step t')
    plt.ylabel(r'Value')
    plt.title(f'VRNN λ={model_main.lmbd}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)
    plt.close()

