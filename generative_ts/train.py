from . import params, data, model, utils, test
from .params import DEVICE, BASE_DIR, SAVE_DIR
from .utils import Experiment, Experiment_p

import os
import json
import copy
import time
from tqdm import trange, tqdm

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader


# ============ train ============

torch.autograd.set_detect_anomaly(True)

def train_step(epoch, loader, model, optimizer, all_losses):
    num_batches = len(loader)

    for k, v in all_losses.items():
        v.append(0)

    for batch_idx, (data,) in enumerate(loader):
        data = data.to(DEVICE).transpose(0, 1)  # (B, T, D) -> (T, B, D)

        optimizer.zero_grad()
        loss_dict = model(data, batch_idx == 0)

        # total_loss 유무에 따라 train_loss 결정
        if 'total_loss' in loss_dict:
            train_loss = loss_dict['total_loss']
        else:
            train_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

        train_loss.backward()

        # 1) 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 2) NaN/Inf 잘라내기
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(
                    p.grad.data,
                    nan=0.0,         # NaN → 0
                    posinf=1e6,      # +Inf → 큰 양수
                    neginf=-1e6      # -Inf → 큰 음수
                )

        # 3) 파라미터 업데이트
        optimizer.step()

        # 로깅
        for name, each_loss in loss_dict.items():
            if isinstance(each_loss, torch.Tensor):
                value = (
                    each_loss.detach().cpu().item()
                    if each_loss.dim() == 0
                    else each_loss.mean().detach().cpu().item()
                )
            else:
                value = each_loss  # float
            all_losses.setdefault(name, [0] * (epoch + 1))[epoch] += value / num_batches




def train(expr : Experiment, verbose=1):
    data_main, model_main, cfg, save_name = (
        expr.data_main,
        expr.model_main,
        expr.cfg,
        expr.save_name
    )

    os.makedirs(save_path := os.path.join(SAVE_DIR, f"{datetime.now().strftime('%y%m%d_%H%M%S')}"), exist_ok=True)
    with open(os.path.join(save_path, f"params_{save_name}.json"), 'w') as f:   json.dump(cfg, f, indent=2)

    # ------ set data ------
    train_cfg = cfg.get("train", {})

    N_epochs = train_cfg['n_epochs']

    _Y_N = []
    for _ in range(train_cfg['num_sequences']):
        Y, _ = data_main.generate_data()
        _Y_N.append(Y)
                    
    Y_N = torch.from_numpy(np.stack(_Y_N, axis=0)).float()    # (N, T, D)

    loader = torch.utils.data.DataLoader(TensorDataset(Y_N), batch_size=train_cfg['batch_size'], shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model_main.parameters(), lr=train_cfg['learning_rate'])


    # ------ train ------

    all_losses = {}

    for epoch in trange(N_epochs, desc="Training", unit="epoch"): # 1 ~ n_epoch

        start = time.time()
        train_step(epoch, loader, model_main, optimizer, all_losses)
        elapsed = time.time() - start

        log_msg = f"[{epoch}/{N_epochs}] " + "\t".join(f"{k}:{v[epoch]:.2f}" for k, v in all_losses.items()) + f"  epoch: {elapsed:.2f}s"
        tqdm.write(log_msg)

        # ------ save ------

        if epoch % 10 == 0:

            torch.save(model_main.state_dict(), os.path.join(save_path, f"model_{save_name}.pth"))

            # save plot
            if verbose>0:
                plt.figure()
                for key, vals in all_losses.items():    plt.plot(vals, label=key.replace('_', ' ').title())

                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.title(f'Loss curves (λ = {model_main.lmbd})')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(save_path, f"loss_{save_name}.png"))
                plt.close()

                os.makedirs(os.path.join(save_path, 'pred'), exist_ok=True)
                expr_p = Experiment_p(data_main, model_main, None, f'{save_name}_{epoch}', os.path.join(save_path, 'pred'))
                test.plot_posterior(expr_p, t_0_ratio=4)
