from .params import DEVICE

import os
import math
import numpy as np
import matplotlib.pyplot as plt 
from typing import Optional, Tuple, Dict
from torchtyping import TensorType

import torch
import torch.nn as nn



class VRNN(nn.Module):

    def __init__(self, x_dim, z_dim, h_dim, n_layers, lmbd, std_Y = 0.1, verbose=0):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.h_dim = h_dim
        self.n_layers = n_layers

        self.lmbd = lmbd


        # known outcome dist Y_t | θ_t ~ p_Y(·) = N(θ_t, σ_Y^2)
        self.std_Y = std_Y

        # feature model
        self.phi_x = nn.Identity()  # 원래는 x_dim -> h_dim
        self.phi_z = nn.Identity()  # 원래는 z_dim -> h_dim


        # ------ Inference model ------

        # encoder z_t | x_t, h_t-1 = x_t, z_<t ~ q_enc(·) = N( μ_enc(x_t, h_t), σ_enc(x_t, h_t) )
        # self.enc = nn.Linear(h_dim + x_dim, h_dim) # 원래는 x_dim -> h_dim
        self.enc = nn.Sequential(
            nn.Linear(h_dim + x_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim)
        )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())


        # ------ Generation model ------

        # prior z_t | h_t-1 = z_<t ~ p_pr(·) = N( μ_pr(h_t), σ_pr(h_t) )
        # self.pr = nn.Linear(h_dim, h_dim)
        self.pr = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim)
        )
        self.pr_mean = nn.Linear(h_dim, z_dim)
        self.pr_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        
        # decoder x_t | z_t, h_t-1 = z_<=t ~ p_dec(·) = N( μ_dec(h_t), σ_dec(h_t) )
        # self.dec = nn.Linear(h_dim + z_dim, h_dim) # 원래는 z_dim -> h_dim
        self.dec = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim)
        )
        self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        

        # ------ autoregressive model ------

        # Reccurence h_t = GRU( z_t, h_t-1 )
        # input, output : R^(L, B, x_dim + z_dim)
        # hidden : R^(n_layers, B, h_dim)
        self.rnn = nn.GRU(z_dim, h_dim, n_layers)

        for weight in self.parameters():    weight.data.normal_(0, 1e-1)
        bias_init = math.log(math.exp(1e-1) - 1)
        nn.init.constant_(self.enc_std[0].bias, bias_init)
        nn.init.constant_(self.pr_std[0].bias, bias_init)

        self.verbose = verbose



    # encoder μ_enc( enc(Y_t, h_t) ), σ_enc( enc(Y_t, h_t) )
    def q_enc(self, x_t, h) -> Tuple[torch.Tensor, torch.Tensor]:
        
        phi_x_t = self.phi_x(x_t)
        enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
        q_mean_t = self.enc_mean(enc_t)
        q_std_t = self.enc_std(enc_t) 
        # q_std_t = torch.ones_like(q_mean_t) * 0.1

        return q_mean_t, q_std_t
    
    # prior μ_pr( pr(h_t) ), σ_pr( pr(h_t) )
    def p_pr(self, h) -> Tuple[torch.Tensor, torch.Tensor]:

        pr_t = self.pr(h[-1])
        pr_mean_t = self.pr_mean(pr_t)
        pr_std_t = self.pr_std(pr_t)
        # pr_std_t = torch.ones_like(pr_mean_t) * 0.1
        
        return pr_mean_t, pr_std_t
 
    # decoder μ_dec( dec(θ_t, h_t) ), σ_dec( dec(θ_t, h_t) )
    def p_dec(self, z_t, h) -> Tuple[torch.Tensor, torch.Tensor]:

        phi_z_t = self.phi_z(z_t)
        dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
        # p_mean_t = self.dec_mean(dec_t)
        p_mean_t = z_t
        # p_std_t = self.dec_std(dec_t)
        p_std_t = torch.ones_like(p_mean_t) * self.std_Y
        
        return p_mean_t, p_std_t

    # recurrence h_t+1 = GRU(θ_t, h_t)
    def f_rec(self, z_t, h) -> torch.Tensor:

        phi_z_t = self.phi_z(z_t)
        # input : (B, z_dim) -> (1, B, z_dim)
        _, h = self.rnn(phi_z_t.unsqueeze(0), h)

        return h



    # ============ forward ============

    def forward(self, x : TensorType["T", "B", "D", torch.float32], plot=False) -> Dict[str, torch.Tensor]:
        '''
        x : R ^ (T * B * D)
        '''

        kld_loss, nll_loss = 0, 0

        # E_q
        # hidden : R^(n_layers, B, h_dim)
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=DEVICE) # x.size(1) = B

        data_x = []
        mean_x = []
        mean_z = []
        mean_pr = []
        sample_z = []
        std_pr = []
        std_z = []
        std_x = []

        for t in range(x.size(0)): # x.size(0) = T

            q_mean_t, q_std_t = self.q_enc(x[t], h) # (B, z_dim), (B, z_dim) z_t | x_t, h_t-1

            pr_mean_t, pr_std_t = self.p_pr(h) # (B, z_dim), (B, z_dim) z_t | h_t-1

            kld_loss += self._kld_gauss(pr_mean_t, pr_std_t, q_mean_t, q_std_t)


            z_t = torch.randn_like(q_std_t) * q_std_t + q_mean_t # z_t | x_t, h_t-1

            p_mean_t, p_std_t = self.p_dec(z_t, h) # (B, x_dim), (B, x_dim) x_t | z_t, h_t-1

            h = self.f_rec(z_t, h) # h_t | z_t, h_t-1

            nll_loss += self._nll_gauss(x[t], p_mean_t, p_std_t)


            # plot
            data_x.append(x[t][0].item())
            mean_z.append(q_mean_t[0].item())
            sample_z.append(z_t[0].item())
            mean_pr.append(pr_mean_t[0].item())
            mean_x.append(p_mean_t[0].item())
            std_pr.append(pr_std_t[0].item())
            std_z.append(q_std_t[0].item())
            std_x.append(p_std_t[0].item())

        ent_loss = 0

        # E_p (given x)
        # hidden : R^(n_layers, B, h_dim)
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=DEVICE) # x.size(1) = B
        for t in range(x.size(0)): # x.size(0) = T
            
            pr_mean_t, pr_std_t = self.p_pr(h)

            z_t = torch.randn_like(pr_std_t) * pr_std_t + pr_mean_t

            h = self.f_rec(z_t, h)

            ent_loss += self._ent_gauss(pr_std_t)


        if plot:
            plt.figure()
            plt.plot(data_x, label=r'$Y_t$ example')
            line_z, = plt.plot(mean_z, label=r'$q$ : $\theta_t | Y_{\leq t}, \theta_{<t}$')
            line_pr, = plt.plot(mean_pr, label=r'$pr$ : $\theta_t | Y_{<t}, \theta_{<t}$')
            plt.plot(sample_z, label=r'sampled $\theta_t | Y_{\leq t}$')
            line_x, = plt.plot(mean_x, label=r'$p$ : $Y_t | Y_{< t}, \theta_{\leq t}$')

            plt.fill_between(np.arange(x.size(0)), np.array(mean_x) - np.array(std_x), np.array(mean_x) + np.array(std_x), alpha=0.2, color=line_x.get_color())
            plt.fill_between(np.arange(x.size(0)), np.array(mean_z) - np.array(std_z), np.array(mean_z) + np.array(std_z), alpha=0.2, color=line_z.get_color())
            plt.fill_between(np.arange(x.size(0)), np.array(mean_pr) - np.array(std_pr), np.array(mean_pr) + np.array(std_pr), alpha=0.2, color=line_pr.get_color())

            plt.xlabel(r'step t')
            plt.ylabel(r'Value')
            plt.title(f'VRNN λ={self.lmbd} : forward')
            plt.legend()
            plt.grid(True)
            plt.savefig('/home/marydenya/Downloads/generative-ts/generative_ts/saves/now.png')
            plt.close()

        return {
            'kld_loss': kld_loss,
            'nll_loss': nll_loss,
            'ent_loss': ent_loss
        }
    

    # ============ test ============

    def sample(self, x : TensorType["T_x", 1, "D", torch.float32], T, plot_name = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        x : R ^ (T_x * 1 * D)

        output : R ^ (T * D_z)   hat{z}_t given x_t-1
        '''

        with torch.no_grad():
            # (T * D_z)
            z_t_with_x_t = np.zeros((T, self.z_dim))
            z_t_with_x_tm1 = np.zeros((T, self.z_dim))
            x_t_with_x_tm1 = np.zeros((T, self.x_dim))
            x_mean_t_with_x_tm1 = np.zeros((T, self.x_dim))
            x_std_t_with_x_tm1 = np.zeros((T, self.x_dim))

            # hidden : R^(n_layers, B, h_dim)
            h = torch.zeros(self.n_layers, 1, self.h_dim, device=DEVICE)

            for t in range(T): 

                pr_mean_t, pr_std_t = self.p_pr(h) # z_t+1 | h_t from (sample z_t, h_t-1)
                z_tp1 = torch.randn_like(pr_std_t) * pr_std_t + pr_mean_t # sample z_t+1 by p

                z_t_with_x_tm1[t] = z_tp1.squeeze().cpu().numpy() 

                p_mean_t, p_std_t = self.p_dec(z_tp1, h) # x_t+1 | z_t+1, h_t
                x_tp1 = torch.randn_like(p_std_t) * p_std_t + p_mean_t # sample x_t+1 by p

                x_t_with_x_tm1[t] = x_tp1.squeeze().cpu().numpy()
                x_mean_t_with_x_tm1[t] = p_mean_t.squeeze().cpu().numpy()
                x_std_t_with_x_tm1[t] = p_std_t.squeeze().cpu().numpy()

                if t < x.size(0):
                    q_mean_t, q_std_t = self.q_enc(x[t], h) # z_t | x_t, h_t-1
                    z_t = torch.randn_like(q_std_t) * q_std_t + q_mean_t # sample z_t by q

                else:
                    pr_mean_t, pr_std_t = self.p_pr(h) # z_t | h_t-1
                    z_t = torch.randn_like(pr_std_t) * pr_std_t + pr_mean_t # sample z_t by p

                z_t_with_x_t[t] = z_t.squeeze().cpu().numpy() 

                h = self.f_rec(z_t, h) # h_t = f(z_t, h_t-1)

            
            return z_t_with_x_t, z_t_with_x_tm1, x_t_with_x_tm1, x_mean_t_with_x_tm1, x_std_t_with_x_tm1
        


    def inference(self, x: TensorType["T_x", 1, "D", torch.float32], T, N=500, verbose=2) -> np.ndarray:
        '''
        x : R ^ (T_x * 1 * D)
        '''
        # (T * D_z)
        x_samples = np.zeros((N, T, self.x_dim))
        x_means = np.zeros((N, T, self.x_dim))
        x_stds = np.zeros((N, T, self.x_dim))

        for i in range(N):
            if i%100==0 and verbose>1:    print(i)
            _, _, x_samples[i], x_means[i], x_stds[i] = self.sample(x, T)
        
        # for t in range(x.size(0)): z_sample[t] = self.sample(x[:t+1])[t] # (T * D_z)

        mean_samples = x_means.mean(axis=0)

        var_alea = (x_stds ** 2).mean(axis=0)  # aleatoric
        var_epis = x_means.var(axis=0)    # epistemic
        var_samples = np.sqrt(var_alea + var_epis)

        # (T-T_0 * D), (T-T_0 * D?), (N * T-T_0 * D)
        return mean_samples[x.size(0):], var_samples[x.size(0):], x_samples[:, x.size(0):]

        



    # ============ loss ============

    def _kld_gauss(self, pr_mean, pr_std, q_mean, q_std) -> torch.Tensor:
        return ( torch.log(pr_std) - torch.log(q_std) + (q_std.pow(2) + (q_mean - pr_mean).pow(2)) / (2 * pr_std.pow(2)) - 1/2 ).mean()
    
    def _nll_gauss(self, x, p_mean_t, p_std_t)-> torch.Tensor:
        return ( torch.log(p_std_t) + math.log(2 * math.pi)/2 + (x - p_mean_t).pow(2)/(2*p_std_t.pow(2)) ).mean()
    
    def _ent_gauss(self, pr_std_t) -> torch.Tensor:
        return self.lmbd * ( torch.log(2*math.pi*math.e*(pr_std_t**2))/2 ).mean()





"""
============================================================================================================
============================================================================================================
============================================================================================================
====================================                                    ====================================
====================================                LS4                 ====================================
====================================                                    ====================================
============================================================================================================
============================================================================================================
============================================================================================================
"""

from ls4.models.ls4 import VAE as LS4Model


class LS4(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config 는 params.py 에서 OmegaConf 등으로 읽은 LS4용 config
        self.model = LS4Model(config)
        # autoregressive sampling 속도 조정을 위해
        self.model.setup_rnn(mode=getattr(config, 'rnn_mode', 'dense'))
        self.lmbd = 0

    def forward(self, x: torch.Tensor, plot=False):
        T, B, D = x.shape
        x_ls4 = x.permute(1, 0, 2)
        timepoints = torch.arange(T, device=x.device)
        masks = torch.ones(B, T, 1, device=x.device)

        total_loss, log_info = self.model(x_ls4, timepoints, masks, plot=plot)

        # log_info에는 kld_loss, nll_loss 등 (float or tensor) 있음
        # total_loss는 항상 torch.Tensor임
        # dict로 합쳐서 반환
        result = {'total_loss': total_loss}
        # log_info의 모든 값 추가 (float이나 tensor 상관없이)
        result.update(log_info)
        # 필요하면 ent_loss도 넣기
        result['ent_loss'] = torch.tensor(0., device=x.device)
        return result

    def sample(self,
               x: torch.Tensor,     # shape (T₀, 1, D)
               T: int):
        # LS4 는 generate(B, timepoints) 만 제공 → x_t0 까지는 posterior, 이후 p(z), p(x)
        T0, B, D = x.shape
        assert B == 1, "샘플링 배치 크기는 1만 지원"
        # 이후 T0+T 길이의 timepoints 생성
        tp = torch.arange(T0 + T, device=x.device)
        # autoregressive 모드로 바꿔야 generate() 가 recurrence 를 사용
        self.model.setup_rnn(mode='diagonal')
        x_full = self.model.generate(1, tp)  # (1, T0+T, D)
        x_full = x_full.squeeze(0).cpu().numpy()  # (T0+T, D)
        # 돌려줄 값:  
        #  - z_t_with_x_t 처럼 세 가지 배열 대신, x_full[T0:] 만 반환해도 무방
        return x_full[:T0], x_full[T0:]

    def inference(self,
                  x: torch.Tensor,  # (T₀, 1, D)
                  T: int,
                  N: int = 500,
                  verbose: int = 2):
        # VRNN inference: 평균, 분산, 샘플 N개
        T0, B, D = x.shape
        assert B == 1
        # LS4 predict 는 분류용 → 여기서는 posterior 샘플링을 N번 반복
        means = []
        vars_ = []
        samples = []
        for i in range(N):
            if i % 100 == 0 and verbose > 1: print(i)
            x0, x_pred = self.sample(x, T)
            means.append(x_pred)
            # 분산 계산은 샘플링 결과로 직접
            samples.append(x_pred)
        arr = np.stack(samples, axis=0)  # (N, T, D)
        mean = arr.mean(0)
        var = arr.var(0)
        return mean, np.sqrt(var), arr
