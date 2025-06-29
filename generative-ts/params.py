import torch

print(f"=======================")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {DEVICE}")
print(f"=======================")

params = {
    "data": {
        "name": 'example_1',
        "T": 300,
        "v": 10,
        "tau": 10
    },

    "model": {
        "name": 'VRNN',
        "h_dim": 1,
        "n_layers": 10
    },

    "outcome": {
        "name": None, # 나중에 reward function 이름으로 바뀌어야함
        "x_dim": 1,
        "z_dim": 1,
        "std_Y": 0.5
    },

    "train": {
        "lmbd": 0,

        "n_epochs": 100,
        "num_sequences": 1000,
        "batch_size": 100,

        "learning_rate": 1e-3,

        "seed": 128
    }
}

save = ["v", "tau", "std_Y"]
