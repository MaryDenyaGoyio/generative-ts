# train_gp_ls4.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
import types
import numpy as np

class AttrDict(dict):

    def __getattr__(self, k):
        try:    return self[k]
        except KeyError:
            raise AttributeError(f"No attribute '{k}'")

    def __setattr__(self, k, v):
        self[k] = v

def dict2attr(d):

    if isinstance(d, dict):
        return AttrDict({k: dict2attr(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict2attr(v) for v in d]
    else:   return d






# ============ LS4 DEBUG ============

import ls4.models.s4 as _s4
from ls4.models.s4 import SSKernelNPLR, S4


# ------ debug forward ------

_orig_ss_fwd = SSKernelNPLR.forward

def debug_ss_fwd(self, state=None, rate=1.0, L=None):
    print(
        f"[SSK.debug] L={L}, rate="
        f"{'tensor'+str(tuple(rate.shape)) if torch.is_tensor(rate) else rate}, "
        f"state={'None' if state is None else tuple(state.shape)}"
    )

    k, k_state = _orig_ss_fwd(self, state=state, rate=rate, L=L)

    print(
        f"[SSK.debug] → k NaN? {k.isnan().any().item()}, "
        f"k_state NaN? { (k_state.isnan().any().item() if torch.is_tensor(k_state) else False) }"
    )
    return k, k_state

SSKernelNPLR.forward = debug_ss_fwd


# ------ debug cauchy ------

_orig_cauchy_conj = getattr(_s4, 'cauchy_conj', None)
if _orig_cauchy_conj is not None:
    def cauchy_conj_stable(v, z, w, eps: float = 1e-6):
        # 복소 안정화: 분모가 0에 가까워지는 것을 피하기 위해 z에 작은 허수 성분 추가
        if z.dtype == torch.cfloat:
            z = z + 1j * eps
        elif z.dtype == torch.cdouble:
            z = z + 1j * (eps)

        # 고정밀 계산 후 반환을 입력 정밀도로 복귀
        use_double = (v.dtype == torch.cfloat)
        if use_double:
            v_d = v.to(torch.cdouble)
            z_d = z.to(torch.cdouble)
            w_d = w.to(torch.cdouble)
            r = _orig_cauchy_conj(v_d, z_d, w_d)
            return r.to(torch.cfloat)
        else:
            return _orig_cauchy_conj(v, z, w)

    _s4.cauchy_conj = cauchy_conj_stable


# ------ debug nan ------

def nan_logger_hook(module, inp, out):
    def check(t: torch.Tensor, name: str):
        # NaN 이 아닌 부분만 뽑아서 min/max 계산
        mask = ~torch.isnan(t)
        if mask.any():
            t_valid = t[mask]
            t_min, t_max = t_valid.min().item(), t_valid.max().item()
        else:
            t_min, t_max = float('nan'), float('nan')
        if torch.isnan(t).any():
            print(f"[NaN] {module.__class__.__name__}.{name} 에 NaN 발견! valid min/max:", t_min, t_max)
    if isinstance(out, torch.Tensor):
        check(out, "output")
    elif isinstance(out, (tuple, list)):
        for i, o in enumerate(out):
            if torch.is_tensor(o):
                check(o, f"output[{i}]")









# ============ LS4 override ============

from ls4.models.ls4 import VAE as _BaseVAE

class VAE(_BaseVAE):
    """
    원본 VAE에서 decoder 출력(dec_mean, dec_std)만 고정된 p(x|z)=N(z, σ^2) 형태로 덮어씌웁니다.
    encoder, prior, kld, nll 계산 등 나머지는 원본 그대로 사용됩니다.
    """
    def forward(self, x, timepoints, masks, labels=None, plot=False, sum=False):
        # 1) posterior sample
        z_post, z_post_mean, z_post_std = self.encoder.encode(x, timepoints, use_forward=True)

        # (bidirectional 여부에 따라 뒤쪽도 처리—but 여기선 생략)
        # 2) prior 계산만 원본대로 받고
        if self.bidirectional:
            # bidirectional 일 때 prior 계산
            dec_stats = self.decoder(x, timepoints, z_post, *self.encoder.encode(x, timepoints, use_forward=False))
            # dec_stats = (dec_mean, dec_std, z_prior_mean, z_prior_std, z_prior_mean_back, z_prior_std_back)
        else:
            # 단방향일 때 prior 계산
            # 원래 self.decoder(x, timepoints, z_post) 이 반환하는 튜플을 받되
            *_, z_prior_mean, z_prior_std = self.decoder(x, timepoints, z_post)

        # 3) 오리지널 디코더 mean/std 대신 내가 고정한 p(x|z)=N(z_post, σ^2)
        σ = self.config.sigma
        dec_mean = z_post
        dec_std  = torch.ones_like(dec_mean) * σ

        # 4) 나머지 ELBO 조립은 원본대로
        #    kld_loss 계산
        if self.bidirectional:
            # 뒤쪽 prior mean/std 은 dec_stats 에서 가져와야 하지만…
            # (bidirectional 상황에 맞게 z_prior_mean_back, z_prior_std_back 넣어주세요)
            z_prior_mean_back, z_prior_std_back = dec_stats[-2], dec_stats[-1]
            kld_f = self._kld_gauss(z_post_mean, z_post_std, z_prior_mean, z_prior_std, masks, sum=sum)
            kld_b = self._kld_gauss(*self.encoder.encode(x, timepoints, use_forward=False)[1:], 
                                     z_prior_mean_back, z_prior_std_back, masks, sum=sum)
            kld_loss = kld_f + kld_b
        else:
            kld_loss = self._kld_gauss(z_post_mean, z_post_std, z_prior_mean, z_prior_std, masks, sum=sum)

        #    nll_loss 계산
        nll_loss = self._nll_gauss(dec_mean, dec_std, x, masks, sum=sum)

        #    total ELBO
        loss = (kld_loss + nll_loss).mean()

        #    (이후 classification, logging 등은 그대로 복사)
        log_info = {
            'dec_mean': dec_mean.mean().item(),
            'kld_loss': kld_loss.mean().item(),
            'nll_loss': nll_loss.mean().item(),
            'loss': loss.item(),
        }
        return loss, log_info



EPS = torch.finfo(torch.float).eps

def patched_kld_gauss(self, mean_1, std_1, mean_2, std_2, masks=None, sum=False):
    # ---- debug print 시작 ----
    if torch.any(std_1 <= 0) or torch.any(std_2 <= 0):
        print("⚠️ [kld] std_1.min/max:", std_1.min().item(), std_1.max().item())
        print("⚠️ [kld] std_2.min/max:", std_2.min().item(), std_2.max().item())
    # ---- 원본 계산 ----
    kld_elem = (
        2 * torch.log(std_2 + EPS)
      - 2 * torch.log(std_1 + EPS)
      + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2)
      - 1
    )  # (B, L, D)
    # ---- debug nan 체크 ----
    if torch.isnan(kld_elem).any():
        print("🔍 nan in kld_elem!")
        # 한 배치, 한 시퀀스 샘플만 보여줍니다 (detach() 추가)
        print(" mean_1[0,0,:5]:", mean_1[0,0,:5].detach().cpu().numpy())
        print(" std_1[0,0,:5]: ", std_1[0,0,:5].detach().cpu().numpy())
        print(" mean_2[0,0,:5]:", mean_2[0,0,:5].detach().cpu().numpy())
        print(" std_2[0,0,:5]: ", std_2[0,0,:5].detach().cpu().numpy())
    # ---- aggregated return ----
    if masks is None:
        flat = kld_elem.view(kld_elem.size(0), -1)
        return 0.5 * (flat.sum(dim=1) if sum else flat.mean(dim=1))
    m = masks.unsqueeze(-1)  # (B, L, 1)
    if sum:
        return 0.5 * (kld_elem * m).sum(dim=1).sum(dim=1)
    else:
        lens = masks.sum(dim=1).clamp(min=1)            # (B,)
        per_t = (kld_elem * m).sum(dim=2)               # (B, L)
        return 0.5 * (per_t.sum(dim=1) / lens)          # (B,)

def patched_nll_gauss(self, mean, std, x, masks=None, sum=False):
    # ---- debug print 시작 ----
    if torch.any(std <= 0):
        print("⚠️ [nll] std.min/max:", std.min().item(), std.max().item())
    # ---- 원본 계산 ----
    nll = (
        torch.log(std + EPS)
      + np.log(2*np.pi)/2
      + (x - mean).pow(2) / (2 * std.pow(2))
    )  # (B, L, D)
    # ---- debug nan 체크 ----
    if torch.isnan(nll).any():
        print("🔍 nan in nll!")
        print(" mean[0,0]:", mean[0,0,0].detach().cpu().item())
        print(" std[0,0]: ", std[0,0,0].detach().cpu().item())
        print(" x[0,0]:   ",   x[0,0,0].detach().cpu().item())
    # ---- aggregated return ----
    if masks is None:
        flat = nll.view(nll.size(0), -1)
        return flat.sum(dim=1) if sum else flat.mean(dim=1)
    m = masks.unsqueeze(-1)
    if sum:
        return (nll * m).sum(dim=1).sum(dim=1)
    else:
        lens = masks.sum(dim=1).clamp(min=1)
        per_t = (nll * m).sum(dim=2)  # (B, L)
        return per_t.sum(dim=1) / lens

def patched_mse(self, mean, x, masks=None, sum=False):
    err = (x - mean).pow(2)  # (B, L, D)
    if torch.isnan(err).any():
        print("🔍 nan in mse err!")
    if masks is None:
        flat = err.view(err.size(0), -1)
        return flat.sum(dim=1) if sum else flat.mean(dim=1)
    m = masks.unsqueeze(-1)
    if sum:
        return (err * m).sum(dim=1).sum(dim=1)
    else:
        lens = masks.sum(dim=1).clamp(min=1)
        per_t = (err * m).sum(dim=2)  # (B, L)
        return per_t.sum(dim=1) / lens

# 메서드 교체
VAE._kld_gauss = types.MethodType(patched_kld_gauss, VAE)
VAE._nll_gauss = types.MethodType(patched_nll_gauss, VAE)
VAE._mse      = types.MethodType(patched_mse, VAE)



# 3) 모델·데이터셋 import
from generative_ts.datasets.gp_dataset import GPSyntheticDataset

def main():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs   = 100
    batch_size = 64
    lr         = 1e-3

    # GP 데이터 파라미터
    T, std_Y, v, tau, sigma_f = 500, 1.0, 10.0, 1.0, 1.0
    timepoints = torch.linspace(0.0, 1.0, T, device=device)
    num_seq = 2000
    dataset = GPSyntheticDataset(T, std_Y, v, tau, sigma_f, num_sequences=num_seq)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # YAML 직접 로드
    cfg_all = yaml.safe_load(open(
        "ls4/configs/monash/vae_nn5daily.yaml", "r", encoding="utf-8"
    ))
    cfg_all = dict2attr(cfg_all)
    model_cfg = cfg_all.model

    # **필수 설정 보충**
    model_cfg.n_labels   = 1      # GP엔 레이블 없으니 1
    model_cfg.classifier = False  # 분류기 사용 안 함

    # 모델 생성
    model     = VAE(model_cfg).to(device)
    model.setup_rnn(mode='dense')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for name, module in model.encoder.named_modules():
        if isinstance(module, SSKernelNPLR):
            module.register_forward_hook(
                lambda mod, inp, out, name=name:
                print(f"[HOOK] {name} (SSK) → k NaN? {out[0].isnan().any().item()}")
            )
        if isinstance(module, S4):
            def _safe_nan_report(mod, inp, out, name=name):
                def tensor_has_nan(t: torch.Tensor) -> bool:
                    return bool(torch.isnan(t).any().item())
                flag = None
                if torch.is_tensor(out):
                    flag = tensor_has_nan(out)
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    flag = tensor_has_nan(out[0])
                else:
                    flag = False
                print(f"[HOOK] {name} (S4) → output NaN? {flag}")

            module.register_forward_hook(_safe_nan_report)

    # 학습 루프
    print("================================================")
    print("모델 파라미터 수:", sum(p.numel() for p in model.parameters()))
    print("================================================")
    
    for epoch in range(1, n_epochs+1):
        model.train()
        total_loss = 0.0
        for i, x in enumerate(loader):
            x     = x.to(device).unsqueeze(-1)  # (B, T, 1)
            masks = torch.ones(x.shape[0], x.shape[1], device=device)

            optimizer.zero_grad()
            loss, log_info = model(x, timepoints, masks, plot=False, sum=True)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:  
                print(f"{i}th : ")

                print(loss.item(), end=", ")
                for k,v in log_info.items():
                    # 스칼라 값만 출력
                    try:
                        print(f"  {k}:", float(v) if torch.is_tensor(v) else v)
                    except:
                        print(f"  {k}: (unprintable)")

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch:03d}] avg ELBO = {avg_loss:.4f}")


        os.makedirs("trained_models", exist_ok=True)
        torch.save(model.state_dict(), "trained_models/ls4_gp.pth")
        print("✅ 모델 파라미터를 'trained_models/ls4_gp.pth'에 저장했습니다.")

if __name__ == "__main__":
    main()

