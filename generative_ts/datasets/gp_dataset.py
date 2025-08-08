# generative_ts/datasets/gp_dataset.py
import torch
from torch.utils.data import Dataset
from generative_ts.data import GP
import numpy as np

class GPSyntheticDataset(Dataset):
    """
    GP generator 로부터 시퀀스를 on-the-fly로 생성합니다.
    각 샘플의 shape = (T, x_dim)
    """

    def __init__(self,
                 T: int,
                 std_Y: float,
                 v: float,
                 tau: float,
                 sigma_f: float,
                 num_sequences: int):
        self.num_sequences = num_sequences
        self.generator = GP(T=T, std_Y=std_Y, v=v, tau=tau, sigma_f=sigma_f)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        x_np, _ = self.generator.generate_data()    # x_np shape = (T, 1)
        x = torch.from_numpy(x_np).float().squeeze(-1)  # (T,)
        return x
