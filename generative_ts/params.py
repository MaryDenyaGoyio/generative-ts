import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saves')
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"=======================")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {DEVICE}")
print(f"=======================")

params = {
    "data": {
        "name": 'example_1',
        "T": 100,
        "v": 10,
        "tau": 10
    },

    "model": {
        "name": 'VRNN',
        "h_dim": 10,
        "n_layers": 1,
        "lmbd": 10
    },

    "outcome": {
        "name": None, # 나중에 reward function 이름으로 바뀌어야함
        "x_dim": 1,
        "z_dim": 1,
        "std_Y": 0.01
    },

    "train": {
        "n_epochs": 500,
        "num_sequences": 500,
        "batch_size": 100,

        "learning_rate": 1e-3,

        "seed": 128
    }
}

save = ["v", "tau", "std_Y", "lmbd"]
