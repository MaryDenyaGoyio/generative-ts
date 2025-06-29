from params import DEVICE

import math
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn



class VRNN(nn.Module):

    def __init__(self, x_dim, z_dim, std_Y, h_dim, n_layers, verbose=0):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.h_dim = h_dim
        self.n_layers = n_layers


        # known outcome dist Y_t | θ_t ~ p_Y(·) = N(θ_t, σ_Y^2)
        self.std_Y = std_Y

        # feature-extracting transformations
        self.phi_x = nn.Identity()  # 원래는 x_dim -> h_dim
        self.phi_z = nn.Identity()  # 원래는 z_dim -> h_dim


        # Inference model

        # encoder z_t | x_t, h_t = x_<=t, z_<t ~ q_enc(·) = N( μ_enc(x_t, h_t), σ_enc(x_t, h_t) )
        # encoder z_t | x_t, h_t = x_t, z_<t ~ q_enc(·) = N( μ_enc(x_t, h_t), σ_enc(x_t, h_t) )
        self.enc = nn.Linear(h_dim + x_dim, h_dim) # 원래는 x_dim -> h_dim
        self.enc_mean = nn.Linear(h_dim, x_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())


        # Generation model

        # prior z_t | h_t = z_<t, x_<t () ~ p_pr(·) = N( μ_pr(h_t), σ_pr(h_t) )
        # prior z_t | h_t = z_<t () ~ p_pr(·) = N( μ_pr(h_t), σ_pr(h_t) )
        self.pr = nn.Linear(h_dim, h_dim)
        self.pr_mean = nn.Linear(h_dim, z_dim)
        self.pr_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        
        # decoder x_t | z_t, h_t = z_<=t, x<t ~ p_dec(·) = N( μ_dec(h_t), σ_dec(h_t) )
        # decoder x_t | z_t, h_t = z_<=t ~ p_dec(·) = N( μ_dec(h_t), σ_dec(h_t) )
        self.dec = nn.Linear(h_dim + z_dim, h_dim) # 원래는 z_dim -> h_dim
        self.dec_mean = nn.Linear(h_dim, z_dim)
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        

        # autoregressive model

        # Reccurence h_t = GRU( x_t, z_t, h_t-1 )
        # Reccurence h_t = GRU( z_t, h_t-1 )
        self.rnn = nn.GRU(x_dim + z_dim, h_dim, n_layers)

        self.verbose = verbose



    # encoder μ_enc( enc(Y_t, h_t) ), σ_enc( enc(Y_t, h_t) )
    def q_enc(self, x_t, h):
        
        phi_x_t = self.phi_x(x_t)
        enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
        q_mean_t = self.enc_mean(enc_t)
        q_std_t = self.enc_std(enc_t) 

        return q_mean_t, q_std_t
    
    # prior μ_pr( pr(h_t) ), σ_pr( pr(h_t) )
    def p_pr(self, h):

        pr_t = self.pr(h[-1])
        pr_mean_t = self.dec_mean(pr_t)
        pr_std_t = self.dec_std(pr_t)
        
        return pr_mean_t, pr_std_t

    # decoder μ_dec( dec(z_t, h_t) ), σ_dec( dec(z_t, h_t) )
    def p_dec(self, z_t, h):

        phi_z_t = self.phi_z(z_t)
        dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
        p_mean_t = self.dec_mean(dec_t)
        p_std_t = self.dec_std(dec_t)
        
        return p_mean_t, p_std_t


    # recurrence h_t+1 = GRU(θ_t, h_t)
    def f_rec(self, x_t, z_t, h):

        phi_x_t = self.phi_x(x_t)
        phi_z_t = self.phi_z(z_t)
        _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

        return h



    def forward(self, x):

        kld_loss, nll_loss = 0, 0

        # E_q
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=DEVICE)

        for t in range(x.size(0)):

            q_mean_t, q_std_t = self.q_enc(x[t], h)

            pr_mean_t, pr_std_t = self.p_pr(h)
            z_t = torch.randn_like(q_std_t) * q_std_t + q_mean_t
            p_mean_t, p_std_t = self.p_dec(z_t, h)

            h = self.f_rec(x[t], z_t, h)

            kld_loss += self._kld_gauss(pr_mean_t, pr_std_t, q_mean_t, q_std_t)
            nll_loss += self._nll_gauss(x[t], p_mean_t, p_std_t)
            


        ent_loss = 0

        # E_p (given x)
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=DEVICE)
        for t in range(x.size(0)):
            
            pr_mean_t, pr_std_t = self.p_pr(h)

            z_t = torch.randn_like(pr_std_t) * pr_std_t + pr_mean_t

            h = self.f_rec(x[t], z_t, h)

            ent_loss += self._ent_gauss(pr_std_t)

        return {
            'kld_loss': kld_loss,
            'nll_loss': nll_loss,
            'ent_loss': ent_loss
        }
    


    def sample(self, x):

        z_sample = torch.zeros(x.size(0), self.z_dim, device=DEVICE)

        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=DEVICE)
        
        for t in range(x.size(0)):

            q_mean_t, q_std_t = self.q_enc(x[t], h)

            z_t = torch.randn_like(q_std_t) * q_std_t + q_mean_t

            p_mean_t, p_std_t = self.p_dec(z_t, h)

            z_sample[t] = p_mean_t.data

            h = self.f_rec(x[t], z_t, h)
        
        return z_sample



    def _kld_gauss(self, p_mean, p_std, q_mean, q_std):
        return torch.sum((torch.log(q_std) - torch.log(p_std) + (p_std**2+ (p_mean - q_mean)**2) / (2 * q_std**2) - 1/2))
    
    def _nll_gauss(self, x, p_mean_t, p_std_t):
        return torch.sum( torch.log(p_std_t) + math.log(2 * math.pi)/2 + (x - p_mean_t).pow(2)/(2*p_std_t.pow(2)) )
    
    def _ent_gauss(self, std):
        return torch.sum( torch.log(2*math.pi*math.e*(std**2))/2 )