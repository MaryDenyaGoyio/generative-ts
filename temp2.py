import math
import numpy as np
import matplotlib.pyplot as plt
from generative_ts.data import GP  

def plot_one_step_nll(y_seq: np.ndarray, gp: GP):
    """
    한 시퀀스 y_seq에 대해 샘플-플러그인 θ_t, NLL을 계산하고 플롯
    """
    T = gp.T
    theta_samples = []
    nll_per_step = []

    for t in range(1, T):
        x_past = y_seq[:t].reshape(-1,1)
        _, _, x_samples = gp.posterior_inference(x_past, T=t+1, N=1)
        theta_t = x_samples[0, -1]
        theta_samples.append(theta_t)

        var = gp.std_Y**2
        nll = 0.5 * ((y_seq[t] - theta_t)**2 / var + math.log(2*math.pi*var))
        nll_per_step.append(nll)

    print(np.sum(nll_per_step))

    steps = np.arange(1, T)

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(steps, y_seq[1:], label='True Y')
    plt.plot(steps, theta_samples, '--', label='Sampled θ')
    plt.xlabel('Time step t')
    plt.ylabel('Value')
    plt.title('True Y vs. Sampled θ')
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(steps, nll_per_step, label='One-step NLL')
    plt.xlabel('Time step t')
    plt.ylabel('NLL')
    plt.title('Per-step One-step Sample-and-Plug NLL')
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # GP 인스턴스
    gp = GP(T=100, std_Y=0.01, v=10.0, tau=10.0, sigma_f=1.0)
    # 데이터 생성
    y_seq = gp.generate_data()[0].flatten()
    # 플롯
    plot_one_step_nll(y_seq, gp)

