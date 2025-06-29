import params, data, model, test
from params import DEVICE

import os
import json
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saves')
os.makedirs(SAVE_DIR, exist_ok=True)

save_folders = [name for name in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, name))]



cfg = copy.deepcopy(params.params)

# p_Y
pY_cfg = cfg.get("outcome", {})
pY_name  = pY_cfg.pop("name", None)

# data.py
data_cfg = cfg.get("data", {})

# model.py
model_cfg = cfg.get("model", {})
model_name = model_cfg.pop("name", None)
try:    model_cl = getattr(model, model_name)
except AttributeError: raise RuntimeError(f"No {model_name} in model.py")
model_main = model_cl(**model_cfg, **pY_cfg).to(DEVICE)



save_path = os.path.join(SAVE_DIR, f"{len(save_folders)+1}")
os.makedirs(save_path, exist_ok=True)

fn = f"{model_name}_" + f"_".join(
        f"{k}_{v}"
        for section in ('model', 'data', 'train', 'outcome')
        for k, v in cfg[section].items()
        if (k in params.save) and not (k == 'name' and v is None)
    )

with open(os.path.join(save_path, f"params_{fn}.json"), 'w') as f:   json.dump(params.params, f, indent=2)



def train(n_epochs, lmbd, batch_size, train_loader, model, optimizer):

    (test_batch,) = next(iter(train_loader))
    test_seq = test_batch[0].to(DEVICE).unsqueeze(1)

    losses = {}

    for epoch in range(1, n_epochs+1):

        sum_loss, num_batches = {}, 0

        for batch_idx, (data,) in enumerate(train_loader):
            num_batches += 1

            data = data.to(DEVICE).transpose(0, 1)
            data = (data - data.min()) / (data.max() - data.min())

            optimizer.zero_grad()
            loss_dict = model(data)
            for name, loss in loss_dict.items():    sum_loss[name] = sum_loss.get(name, 0.0) + loss.item()

            ent = loss_dict.pop('ent_loss')
            kld = loss_dict.pop('kld_loss')
            nll = loss_dict.pop('nll_loss')
            # base = sum(loss for loss in loss_dict.values())
            # * max(epoch/n_epochs-1/2, 0) annealing
            train_loss = nll + kld + lmbd * ent   # base + lmbd * ent
            total = nll + kld + lmbd * ent
            train_loss.backward()
            optimizer.step()
            sum_loss['total_loss'] = sum_loss.get('total_loss', 0.0) + total.item()

        loss_batch = { name.upper() : (lmbd * loss if name == 'ent_loss' else loss) / num_batches for name, loss in sum_loss.items() }
        log_msg = f"{epoch}/{n_epochs} " + "\t".join(f"{k}:{v:.6f}" for k, v in loss_batch.items() )
        print(log_msg)

        for k, v in loss_batch.items(): losses.setdefault(k, []).append(v)



        if epoch % 10 == 1:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{fn}.pth"))

            plt.figure()
            for key, vals in losses.items():
                if key == 'ent_loss':   label = f"λ·ENT Loss"
                else:   label = key.replace('_', ' ').title()
                plt.plot(vals, label=label)

            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title(f'Loss curves (λ = {train_cfg["lmbd"]})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_path, f"loss_{fn}.png"))
            plt.close()

            # test를 import해서 여기서도 그려야한다
            hat_z = test.inference(model, test_seq, os.path.join(save_path, 'pred'), f'prediction_{fn}_{epoch}.png')


train_cfg = cfg.get("train", {})
torch.manual_seed(train_cfg['seed'])

_Y_N = []
for _ in range(train_cfg['num_sequences']):
    Y, _ = data.generate_data(**data_cfg, **pY_cfg)
    _Y_N.append(Y)
                   
Y_N = torch.from_numpy(np.stack(_Y_N, axis=0)).float()    # (N, T, D)

train_loader = torch.utils.data.DataLoader(TensorDataset(Y_N), batch_size=train_cfg['batch_size'], shuffle=True, num_workers=4)
optimizer = torch.optim.Adam(model_main.parameters(), lr=train_cfg['learning_rate'])

train(train_cfg['n_epochs'], train_cfg['lmbd'], train_cfg['batch_size'], train_loader, model_main, optimizer)

    