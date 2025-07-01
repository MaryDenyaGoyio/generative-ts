from . import params, data, model, utils
from .params import DEVICE, BASE_DIR, SAVE_DIR

import os
import json
import copy

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader


# ------ load & save ------

data_func, model_main, train_cfg, save_name = utils.load_params()

save_folders = [name for name in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, name))]
os.makedirs(save_path := os.path.join(SAVE_DIR, f"{datetime.now().strftime("%y%m%d_%H%M%S")}"), exist_ok=True)
with open(os.path.join(save_path, f"params_{save_name}.json"), 'w') as f:   json.dump(params.params, f, indent=2)


# ============ train ============

def train(N_epochs, train_loader, model, optimizer, test_seq = None, verbose=1):

    num_batches = len(train_loader)
    batch_size = train_loader.batch_size

    all_losses = {}

    for epoch in range(0, N_epochs): # 1 ~ n_epoch

        for k, v in all_losses.items(): v.append(0)

        for batch_idx, (data,) in enumerate(train_loader): # 1 ~ int(N/B)

            data = data.to(DEVICE).transpose(0, 1) # (B, T, D) -> (T, B, D)

            # ------ loss ------
            optimizer.zero_grad()
            loss = model(data, batch_idx==0)

            train_loss = sum(loss for loss in loss.values())
            train_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # grads = {n: p.grad.data.norm(2) for n,p in model.named_parameters()}
            # top5 = sorted(grads.items(), key=lambda kv: kv[1], reverse=True)[:5]

            for name, each_loss in loss.items():    all_losses.setdefault(name, [ 0 for _ in range(epoch+1) ])[epoch] += each_loss.item() / num_batches # Tensor -> float


        log_msg = f"[{epoch}/{N_epochs}] " + "\t".join(f"{k}:{v[epoch]:.2f}" for k, v in all_losses.items() )
        print(log_msg)


        # ------ save ------

        if epoch % 10 == 0:
            # save model
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{save_name}.pth"))

            # save plot
            if verbose>0:
                plt.figure()
                for key, vals in all_losses.items():    plt.plot(vals, label=key.replace('_', ' ').title())

                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.title(f'Loss curves (λ = {model.lmbd})')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(save_path, f"loss_{save_name}.png"))
                plt.close()

                if test_seq is not None:
                    os.makedirs(os.path.join(save_path, 'pred'), exist_ok=True)
                    model_main.inference(test_seq, T=test_seq.size(0), N=1, online=False, plot_name=os.path.join(save_path, 'pred', f'prediction_{save_name}_{epoch}.png'))



# ============ run ============

torch.manual_seed(train_cfg['seed'])

# ------ set data ------
_Y_N = []
for _ in range(train_cfg['num_sequences']):
    Y, _ = data_func()
    _Y_N.append(Y)
                   
Y_N = torch.from_numpy(np.stack(_Y_N, axis=0)).float()    # (N, T, D)

train_loader = torch.utils.data.DataLoader(TensorDataset(Y_N), batch_size=train_cfg['batch_size'], shuffle=True, num_workers=4)
optimizer = torch.optim.Adam(model_main.parameters(), lr=train_cfg['learning_rate'])

# ------ test data ------
(test_batch,) = next(iter(train_loader)) # (B, T, D)
test_seq = test_batch[0].to(DEVICE).unsqueeze(1) # (T, D) -> (T, 1, D)

train(train_cfg['n_epochs'], train_loader, model_main, optimizer, test_seq)

    