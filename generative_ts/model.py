from .params import DEVICE

import os
import math
import numpy as np
import matplotlib.pyplot as plt 

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
    def q_enc(self, x_t, h):
        
        phi_x_t = self.phi_x(x_t)
        enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
        q_mean_t = self.enc_mean(enc_t)
        q_std_t = self.enc_std(enc_t) 
        # q_std_t = torch.ones_like(q_mean_t) * 0.1

        return q_mean_t, q_std_t
    
    # prior μ_pr( pr(h_t) ), σ_pr( pr(h_t) )
    def p_pr(self, h):

        pr_t = self.pr(h[-1])
        pr_mean_t = self.pr_mean(pr_t)
        pr_std_t = self.pr_std(pr_t)
        # pr_std_t = torch.ones_like(pr_mean_t) * 0.1
        
        return pr_mean_t, pr_std_t
 
    # decoder μ_dec( dec(θ_t, h_t) ), σ_dec( dec(θ_t, h_t) )
    def p_dec(self, z_t, h):

        phi_z_t = self.phi_z(z_t)
        dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
        # p_mean_t = self.dec_mean(dec_t)
        p_mean_t = z_t
        # p_std_t = self.dec_std(dec_t)
        p_std_t = torch.ones_like(p_mean_t) * self.std_Y
        
        return p_mean_t, p_std_t

    # recurrence h_t+1 = GRU(θ_t, h_t)
    def f_rec(self, z_t, h):

        phi_z_t = self.phi_z(z_t)
        # input : (B, z_dim) -> (1, B, z_dim)
        _, h = self.rnn(phi_z_t.unsqueeze(0), h)

        return h



    # ============ forward ============

    def forward(self, x, plot=False):
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

    def sample(self, x, T, plot_name = None):
        '''
        x : R ^ (T_x * 1 * D)

        output : R ^ (T * D_z)   hat{z}_t given x_t-1
        '''

        with torch.no_grad():
            # (T * D_z) # 이거 다 다시 np array로 바꾸기
            z_sample = np.zeros((T, self.z_dim))
            z_mean = np.zeros((T, self.z_dim))
            z_std = np.zeros((T, self.z_dim))

            # hidden : R^(n_layers, B, h_dim)
            h = torch.zeros(self.n_layers, 1, self.h_dim, device=DEVICE)

            for t in range(T): 

                pr_mean_t, pr_std_t = self.p_pr(h) # get mean z_t | h_t 나중에 std_t도 사용

                # z_sample[t] = 직접 뽑기
                z_mean[t] = pr_mean_t.squeeze().cpu().numpy()
                z_std[t] = pr_std_t.squeeze().cpu().numpy()


                if t < x.size(0): # x.size(0) = T_x
                    q_mean_t, q_std_t = self.q_enc(x[t], h)
                    z_t = torch.randn_like(q_std_t) * q_std_t + q_mean_t # sample z_t | x_t, h_t-1

                else:
                    pr_mean_t, pr_std_t = self.p_pr(h)
                    z_t = torch.randn_like(pr_std_t) * pr_std_t + pr_mean_t # sample z_t | h_t-1            
                

                h = self.f_rec(z_t, h) # h_t = f(z_t, h_t-1)
            
            return z_sample, z_mean, z_std
        


    def inference(self, x, T, N=100, online = True, plot_name = None):
        '''
        x : R ^ (T_x * 1 * D)
        '''
        # (T * D_z)
        z_samples = np.zeros((N, T, self.z_dim))
        z_means = np.zeros((N, T, self.z_dim))
        z_stds = np.zeros((N, T, self.z_dim))

        for i in range(N):
            z_samples[i], z_means[i], z_stds[i] = self.sample(x, T)
        
        # for t in range(x.size(0)): z_sample[t] = self.sample(x[:t+1])[t] # (T * D_z)

        z_mean_avg = z_means.mean(axis=0)[:, 0]  # shape (T, D_z)
        z_std_avg = z_stds.mean(axis=0)[:, 0]    # shape (T, D_z)


        if plot_name is not None:
            plt.figure()
            plt.plot(x.squeeze().cpu().numpy(), label=r'$Y_t$ example')
            z_line, = plt.plot(z_mean_avg, label=r'$\theta_t | Y_{< t}, \theta_{< t}$')
            plt.fill_between(np.arange(T)[(x.size(0) if online else 0):], (z_mean_avg - z_std_avg)[(x.size(0) if online else 0):], (z_mean_avg + z_std_avg)[(x.size(0) if online else 0):], alpha=0.2, color=z_line.get_color())
            if online:  plt.axvline(x=x.size(0)-1, color='black', linestyle='--', linewidth=1.0)
            plt.xlabel(r'step t')
            plt.ylabel(r'Value')
            plt.title(f'VRNN λ={self.lmbd}')
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_name)
            plt.close()

        return z_samples

        



    # ============ loss ============

    def _kld_gauss(self, pr_mean, pr_std, q_mean, q_std):
        return ( torch.log(pr_std) - torch.log(q_std) + (q_std.pow(2) + (q_mean - pr_mean).pow(2)) / (2 * pr_std.pow(2)) - 1/2 ).mean()
  
    def _nll_gauss(self, x, p_mean_t, p_std_t):
        return ( torch.log(p_std_t) + math.log(2 * math.pi)/2 + (x - p_mean_t).pow(2)/(2*p_std_t.pow(2)) ).mean()
    
    def _ent_gauss(self, pr_std_t):
        return self.lmbd * ( torch.log(2*math.pi*math.e*(pr_std_t**2))/2 ).mean()





