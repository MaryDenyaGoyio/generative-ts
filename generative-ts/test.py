import params, data, model, test
from params import DEVICE

import os
import json
import glob

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saves')
os.makedirs(SAVE_DIR, exist_ok=True)


target = 'main'
test_dir, test_cfg, test_model_path = None, None, None

for _each_dir in sorted(os.listdir(SAVE_DIR)):
    if target not in _each_dir: continue
    each_dir = os.path.join(SAVE_DIR, _each_dir)

    params_files = [f for f in os.listdir(each_dir) if f.startswith("params") and f.endswith(".json")]
    if not params_files:    continue
    params_path = os.path.join(each_dir, sorted(params_files)[0])

    model_files = [f for f in os.listdir(each_dir) if f.startswith("model") and f.endswith(".pth")]
    if not model_files: continue
    model_path = os.path.join(each_dir, sorted(model_files)[0])

    with open(params_path, 'r') as f:   test_cfg = json.load(f)
    test_model_path, test_dir = model_path, each_dir
    break

save_folders = [name for name in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, name))]




def inference(model, x, pred_path, file_name):

    with torch.no_grad():
        hat_z = torch.zeros(x.size(0), model.z_dim, device=DEVICE)

        for t in range(1, x.size(0)):  hat_z[t] = model.sample(x[:t])[-1]
        
        pred = hat_z.squeeze().cpu().numpy()      # (T,)
        true = x.squeeze().cpu().numpy()   # (T,)

        plt.figure()
        plt.plot(true, label=r'(Y_t)_{\leq T} example')
        plt.plot(pred, label=r'predicted outcome mean (\mu_{t})_{\leq T}')
        # plt.plot(pred, label=r'predicted latent (\theta_{t+1})_{< T}')
        plt.xlabel(r'step t')
        plt.ylabel(r'Value')
        plt.title('prediction by VRNN')
        plt.legend()
        plt.grid(True)
        os.makedirs(pred_path, exist_ok=True)
        plt.savefig(os.path.join(pred_path, file_name))
        plt.close()

        return hat_z


pY_cfg = test_cfg.get("outcome", {})
pY_name  = pY_cfg.pop("name", None)

model_cfg = test_cfg.get("model", {})
model_name = model_cfg.pop("name", None)
try:    model_cl = getattr(model, model_name)
except AttributeError: raise RuntimeError(f"No {model_name} in model.py")
model_main = model_cl(**model_cfg, **pY_cfg).to(DEVICE)
model_main.load_state_dict(torch.load(test_model_path, map_location=DEVICE))
model_main.eval()


_Y, _ = data.generate_data(**test_cfg['data'], **test_cfg['outcome'])
x = torch.from_numpy(_Y).float().to(DEVICE).unsqueeze(1)  # (T, 1, d)

z_sample = torch.zeros(x.size(0), model_main.z_dim, device=DEVICE)

h = torch.zeros(model_main.n_layers, x.size(1), model_main.h_dim, device=DEVICE)

hat_z = torch.zeros(x.size(0), model_main.z_dim, device=DEVICE)

for s in range(1, x.size(0)):

    for t in range(s):

        q_mean_t, q_std_t = model_main.q_enc(x[t], h)

        z_t = torch.randn_like(q_std_t) * q_std_t + q_mean_t

        p_mean_t, p_std_t = model_main.p_dec(z_t, h)

        z_sample[t] = p_mean_t.data

        h = model_main.f_rec(x[t], z_t, h)
      
    hat_z[s] = z_sample[-1]

pred = hat_z.squeeze().cpu().numpy()      # (T,)
true = x.squeeze().cpu().numpy()   # (T,)

pred_path = '/home/marydenya/Downloads/bayesian_generative_TS/saves'

plt.figure()
plt.plot(true, label=r'(Y_t)_{\leq T} example')
plt.plot(pred, label=r'predicted outcome mean (\mu_{t})_{\leq T}')
# plt.plot(pred, label=r'predicted latent (\theta_{t+1})_{< T}')
plt.xlabel(r'step t')
plt.ylabel(r'Value')
plt.title('prediction by VRNN')
plt.legend()
plt.grid(True)
os.makedirs(pred_path, exist_ok=True)
plt.savefig(os.path.join(pred_path, 'plot2.png'))
plt.close()
    

# hat_z = test.inference(model_main, Y, '/home/marydenya/Downloads/bayesian_generative_TS/saves', 'plot.png')


'''
# p_Y
pY_cfg = test_cfg.get("outcome", {})
pY_name  = pY_cfg.pop("name", None)

model_cfg = test_cfg.get("model", {})
model_name = model_cfg.pop("name", None)
try:    model_cl = getattr(model, model_name)
except AttributeError: raise RuntimeError(f"No {model_name} in model.py")
model_main = model_cl(**model_cfg, **pY_cfg).to(DEVICE)
model_main.load_state_dict(torch.load(test_model_path, map_location=DEVICE))
model_main.eval()


_Y, _ = data.generate_data(**test_cfg['data'], **test_cfg['outcome'])
Y = torch.from_numpy(_Y).float().to(DEVICE).unsqueeze(1)  # (T, 1, d)


hat_z = test.inference(model_main, Y)

# ── Plot ──
true = Y.squeeze().cpu().numpy()      # (T,)
pred = hat_z.squeeze().cpu().numpy()  # (T,)

plt.figure()
plt.plot(true, label='True Sequence')
plt.plot(pred, label='Predicted latent (hat_z)')
plt.xlabel('Time step t')
plt.ylabel('Value')
plt.title(f'Inference on {os.path.basename(test_dir)}')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(test_dir, f'inference_{os.path.basename(test_dir)}.png'))
plt.close()
#'''

