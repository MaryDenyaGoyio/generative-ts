from generative_ts import params, data, model, utils
from generative_ts.params import DEVICE

import os

import torch

data_func, model_main, save_name, load_path = utils.load_model('250701_142352')

_Y, _ = data_func()
Y = torch.from_numpy(_Y).float().to(DEVICE).unsqueeze(1)  # (T, 1, d)

model_main.inference(Y[:int(Y.size(0)/4)], T=Y.size(0), plot_name=os.path.join(load_path, f'test {save_name}.png') ) # type(model_main).__name__