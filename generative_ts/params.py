import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saves')
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {
    "data": {
        "name": 'GP',
        "T": 500,
        "v": 10,
        "tau": 1,
        "sigma_f": 1
    },

    "model": {
        "name": 'VRNN',
        "h_dim": 10,
        "n_layers": 1,
        "lmbd": 0
    },

    "outcome": {
        "name": None, # 나중에 reward function 이름으로 바뀌어야함
        "x_dim": 1,
        "z_dim": 1,
        "std_Y": 1
    },

    "train": {
        "n_epochs": 10,
        "num_sequences": 10,
        "batch_size": 10,

        "learning_rate": 1e-3,

        "seed": 128 #전체 공통으로 바뀌어야함
    }
}

save = ["v", "tau", "std_Y", "lmbd"]

'''
    "train": {
        "n_epochs": 300,
        "num_sequences": 1000,
        "batch_size": 100,

        "learning_rate": 1e-3,

        "seed": 128 #전체 공통으로 바뀌어야함
    }
'''