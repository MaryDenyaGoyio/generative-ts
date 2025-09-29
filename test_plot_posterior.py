import torch
import sys
sys.path.insert(0, '/home/marydenya/Downloads/generative-ts')

from generative_ts.models.vrnn import VRNN_ts
from generative_ts.eval import plot_posterior

# Load model
save_path = '/home/marydenya/Downloads/generative-ts/generative_ts/saves/250925_114532_VRNN_stdY1'
model_path = f'{save_path}/model_VRNN.pth'
dataset_path = '/home/marydenya/Downloads/generative-ts/generative_ts/dataset/ar1_samples_T300_phi0.99_sigma0.1_stdY1.0'

# Create model
device = torch.device('cpu')
model = VRNN_ts(x_dim=1, z_dim=1, h_dim=10, n_layers=1, lmbd=0, std_Y=1.0)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Plot
plot_posterior(
    model=model,
    save_path=f'{save_path}/test_posterior',
    epoch='test',
    model_name='VRNN',
    dataset_path=dataset_path,
    idx=0,
    ratio=0.5,
    N_samples=100
)

print(f"Plot saved to {save_path}/test_posterior/")