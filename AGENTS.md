# AGENTS.md

This file provides guidance to Codex when working with code in this repository.
Put this file at the repository root. Unless noted, all paths are relative to the repo root.

---

## Project Structure

This is a generative time series modeling research project that compares multiple deep learning architectures:

- **VariationalRecurrentNeuralNetwork/**: VRNN implementation from Chung et al. (2015) for sequential data modeling with MNIST experiments
- **latent_ode/**: Latent ODE models for irregular time series from Rubanova et al. (2019) supporting multiple datasets (MuJoCo, Physionet, Human Activity)
- **ls4/**: Deep Latent State Space Models (LS4) from Zhou et al. (2023) for time series generation on Monash forecasting datasets
- **_generative_ts/**: Custom implementation integrating VRNN with Gaussian Process synthetic datasets

The main development occurs in `_generative_ts/` and `train_gp_ls4.py`, which combines LS4 models with GP synthetic data.

---

## Data & Experiments

- All runs save checkpoints/logs/visualizations to timestamped subfolders under `saves/`
- Every model should produce both **training loss curves** and **predictive performance plots** (e.g., MSE/MAE, NLL) under `saves/.../plots/`
- For reproducibility, dump **hyperparameters/seed/git commit** as JSON alongside results

---

## Environment & Execution (Conda)

> Always activate the correct conda environment **before** running any script to avoid dependency/version mismatches.

### Environment ↔ Module mapping
- `ls4/`, `train_gp_ls4.py` → conda env: **`ls4`**
- `VariationalRecurrentNeuralNetwork/` → conda env: **`vrnn`**


## Development Guidelines

**IMPORTANT: Follow these rules when making changes:**

0) **Always leverage existing implementations before creating new ones.**

1) **Do not modify `ls4/`, `VariationalRecurrentNeuralNetwork`, `latent_ode` repository code directly (principle + patterns).**  
   Treat `ls4/` and other models as **read-only (submodule-like)**. Do **not** add patches/PRs/edits inside `ls4/`. Extend **externally**:
   - **(a) Inheritance/Composition**: Define wrapper classes in `_generative_ts/` (or top-level scripts) to extend behavior.
   - **(b) Minimal monkey-patch (last resort)**: Only for urgent bug workarounds; keep it runtime-only with comments and an issue link.

   **Suggested layout**
   ~~~text
   _generative_ts/
     ls4_overrides.py     # all LS4 wrappers/inheritance live here
     trainers/
       train_gp_ls4.py    # experiment script (current name OK)
   ~~~

   **Canonical pattern** (see `train_gp_ls4.py`’s `Decoder_ts` / `VAE_ts`)
   - Preserve **public interfaces** of original LS4 components (input/output shapes, dtype, device).
   - Inject custom logic **inside** `forward`/`step` only (e.g., output latent states directly, integrate GP terms).
   - Keep parent initialization/registration calls intact.

   **Forbidden**
   - Editing `ls4/` source files (`__init__.py`, model defs, config files)
   - Global rebinding that collides with LS4 symbols

   **Allowed**
   - External **mixin/adapter** classes
   - Config flag or class-path toggle to switch between original vs extended behavior
   - Prefer reusing and extending existing implementations wherever possible.

2) **Environment pinning**
   - Always `conda activate ls4` for LS4, `conda activate vrnn` for VRNN (or `conda run -n <env> ...`).

---

## Model Architecture Notes

### Variational Recurrent Neural Network (VRNN)

**Generative / Inference Definitions**

Decoder:  
  p(x_t | z_<=t, x_<t) = p(x_t | z_t, h_{t-1})  
  = N(mu_x(z_t, h_{t-1}), sigma_x^2(z_t, h_{t-1}))
  → `VRNN.dec` + `VRNN.dec_mean` + `VRNN.dec_std` (model.py:60-71)

Prior:  
  p(z_t | z_<t, x_<t) = p(z_t | h_{t-1})  
  = N(mu_0(h_{t-1}), sigma_0^2(h_{t-1}))
  → `VRNN.prior` + `VRNN.prior_mean` + `VRNN.prior_std` (model.py:51-57)

Encoder (Approx. Posterior):  
  q(z_t | x_<=t, z_<t) = N(mu_z(x_t, h_{t-1}), sigma_z^2(x_t, h_{t-1}))  
  q(z_<=T | x_<=T) = Prod_{t=1..T} q(z_t | x_<=t, z_<t)
  → `VRNN.enc` + `VRNN.enc_mean` + `VRNN.enc_std` (model.py:40-48)

Transition (RNN):  
  h_t = f_theta(h_{t-1}, z_t, x_t)
  → `VRNN.rnn` (GRU) (model.py:74, 109)

Joint (Generation factorization):  
  p(x_<=T, z_<=T) = Prod_{t=1..T} p(x_t | z_<=t, x_<t) * p(z_t | z_<t, x_<t)

**Training (ELBO)**

L = E_{q(z_<=T | x_<=T)} [  
  Sum_{t=1..T} {  
    - KL( q(z_t | x_<=t, z_<t) || p(z_t | z_<t, x_<t) )  
    + log p(x_t | z_<=t, x_<t)  
  }  
]
→ `VRNN._kld_gauss` (KL divergence) (model.py:172-178)
→ `VRNN._nll_bernoulli` or `VRNN._nll_gauss` (reconstruction) (model.py:181-186)
→ `VRNN.forward` (full training loop) (model.py:77-123)

**Posterior Sampling Procedure**

1) Sample z_<=T0 from q(z_<=T0 | x_<=T0) for observed interval x_<=T0
2) Sample z_t from prior p(z_t | h_{t-1}) for t = T0+1..T
3) Sample x_t from decoder p(x_t | z_t, h_{t-1})
4) Generate x_{T0+1:T} by repeating steps 2-3
→ `VRNN.sample` (generation method) (model.py:126-154)
→ `VRNN._reparameterized_sample` (reparameterization trick) (model.py:166-169)  

---

### Latent State Space Sequence Model (LS4)

**State-Space Backbone**

d/dt h_t = A h_t + B x_t + E z_t  
y_t      = C h_t + D x_t + F z_t  

→ Implemented with FFT-based convolution for efficient long sequence processing (O(L log L))
→ `S4Model`, `Model` (s4models.py:40-160, 161-380)

**Generative / Inference Definitions**

Prior:  
  p(z_t | z_<t) = N(mu_{z,t}(z_<t), sigma_{z,t}^2(z_<t))
  → `Decoder.latent` (Model instance) (ls4.py:17-24)
  → `Decoder.forward` computes prior_mean, prior_std (ls4.py:213)

Decoder:  
  p(x_t | z_<=t, x_<t) = N(mu_{x,t}(z_<=t, x_<t), sigma_x^2)
  → `Decoder.dec` (Model instance) (ls4.py:27-29)
  → `Decoder.decode` (ls4.py:197-201)
  → `Decoder.reconstruct` (ls4.py:180-195)

Inference:  
  q(z_t | x_<=T) = N(hat_mu_{z,t}(x_<=T), hat_sigma_{z,t}^2(x_<=T))  
  q(z_<=T | x_<=T) = Prod_{t=1..T} q(z_t | x_<=T)
  → `Encoder.latent` (Model instance) (ls4.py:242-244)
  → `Encoder.encode` (ls4.py:256-266)
  → `Encoder._reparameterized_sample` (ls4.py:251-254)

**Block Architecture**

- LS4prior: Takes z_<t as input, outputs mu_{z,t}, sigma_{z,t}  
  → `Decoder.latent` + `Model.forward` (config.prior settings)
- LS4gen: Takes (x_<t, z_<=t) as input, outputs mu_{x,t}  
  → `Decoder.dec` + `Model.forward` (config.decoder settings)
- LS4inf: Takes full x_<=T as input, outputs hat_mu_{z,t}, hat_sigma_{z,t} (parallel FFT inference)
  → `Encoder.latent` + `Model.forward` (config.posterior settings)

**Training (ELBO)**

L = E_{q(z_<=T | x_<=T)} [  
  Sum_{t=1..T} {  
    - KL( q(z_t | x_<=t) || p(z_t | z_<t) )  
    + log p(x_t | z_<=t, x_<t)  
  }  
]
→ `VAE._kld_gauss` (KL divergence computation) (ls4.py:501-520)
→ `VAE._nll_gauss` (negative log-likelihood) (ls4.py:522-532)
→ `VAE.forward` (full training loop) (ls4.py:298-379)

**Posterior Sampling Procedure**

1) Sample z_<=T0 from LS4inf: q(z_<=T0 | x_<=T0) for observed interval x_<=T0
   → `VAE.encode` → `Encoder.encode` (ls4.py:495-499, 256-266)
2) Sample z_t from LS4prior: p(z_t | z_<t) for t = T0+1..T ( prior rollout )
   → `Decoder.sample` using `latent.step` (ls4.py:74)
3) Sample x_t from LS4gen: p(x_t | z_<=t, x_<t)
   → `Decoder.sample` using `dec.step` or `dec` call (ls4.py:93-104)
4) Generate x_{T0+1:T} by repeating steps 2-3
   → `VAE.generate` → `Decoder.sample` (ls4.py:397-402, 53-105)
   → `VAE.reconstruct` + `Decoder.extrapolate` (for prediction) (ls4.py:428-460, 108-178)


**Bidirectional (Causality)**

- Encoder: computes forward/backward posteriors z_t^fwd ~ q_phi(z_t | x_<=t) and z_t^bwd ~ q_phi(z_t | x_>=t) using x and x.flip(1) with time indices aligned back; → Encoder.encode(..., use_forward=True/False) (ls4.py:256–266)

- Decoder (training/reconstruction): concatenates [ z_t^fwd , (z_t^bwd aligned via flip(1)) ] so each output depends on past+future (smoothing, non-causal); → Decoder.reconstruct → Decoder.decode (ls4.py:180–201)

- Causal prediction/evaluation: set bidirectional=False to use only forward posterior q_phi(z_t | x_<=t) for observed prefix, then generate t>T0 via prior rollout p(z_t | z_<t) in Decoder.extrapolate; → VAE.reconstruct(..., t_vec_pred) → Decoder.extrapolate


**Key Generation Methods: Mathematical Formulation (Rigorous Definitions)**

### 1. `VAE.generate` → `Decoder.sample` (Unconditional Generation)
**Input**: batch size bs, sequence length T  
**Output**: generated sequence x ∈ ℝ^{bs×T×d_x}

**Algorithm**:
- Initialize: z_0 = `self.z_prior` (learned parameter ∈ ℝ^{d_z})
- For t = 1 to T:
  - Execute `Decoder.latent.step`:
    - Input: z_{t-1}, hidden_state_{t-1}
    - Output: z_t (sample), μ_z^t, σ_z^t where z_t ~ N(μ_z^t, (σ_z^t)²)
  - Execute `Decoder.dec` or `Decoder.dec.step`:
    - Input: z_{1:t} (accumulated latent variables)
    - Output: x_t ∈ ℝ^{d_x}
- Return: x = [x_1, ..., x_T]

### 2. `VAE.reconstruct(t_vec_pred=None)` → `Decoder.reconstruct` (Reconstruction)
**Input**: observed sequence x ∈ ℝ^{bs×T₀×d_x}, time vector t_vec  
**Output**: reconstructed sequence x̂ ∈ ℝ^{bs×T₀×d_x}

**Algorithm**:
- Execute `Encoder.encode`:
  - Input: x_{1:T₀}
  - Output: z_{1:T₀}, μ_z^{post}, σ_z^{post} where z_{1:T₀} ~ q(z|x) = N(μ_z^{post}, (σ_z^{post})²)
- Execute `Decoder.decode`:
  - Input: z_{1:T₀}, x_{0:T₀-1} (shifted input)
  - Output: x̂_{1:T₀} where each x̂_t = μ_x^t (deterministic output)
- Return: x̂ = [x̂_1, ..., x̂_{T₀}]

### 3. `VAE.reconstruct(t_vec_pred≠None)` → `Decoder.extrapolate` (Conditional Prediction)
**Input**: observations x_{1:T₀} ∈ ℝ^{bs×T₀×d_x}, observation times t_vec, prediction times t_vec_pred  
**Output**: predicted sequence x̂_{T₀+1:T} ∈ ℝ^{bs×(T-T₀)×d_x}, prior statistics μ_z^{prior}, σ_z^{prior}

**Algorithm**:
- **Process observation interval** (t = 1 to T₀):
  - Execute `Encoder.encode`:
    - Input: x_{1:T₀}
    - Output: z_{1:T₀}^{post} ~ q(z|x_{1:T₀})
  - `Decoder.latent.step` (for hidden state update):
    - Input: [z_prior, z_1^{post}, ..., z_{T₀-1}^{post}]
    - Output: updated hidden_state_{T₀} (prior's internal state)
    
- **Generate prediction interval** (t = T₀+1 to T):
  - For t = T₀+1 to T:
    - Execute `Decoder.latent.step`:
      - Input: z_{t-1}, hidden_state_{t-1}
      - Output: z_t, μ_z^t, σ_z^t where z_t ~ N(μ_z^t, (σ_z^t)²)
  - Execute `Decoder.dec`:
    - Input: [z_{1:T₀}^{post}, z_{T₀+1:T}^{prior}] (concatenated full latent sequence)
    - Output: x̂_{T₀+1:T} (extract prediction interval only)
- Return: x̂_{T₀+1:T}, σ_x (fixed noise), μ_z^{prior}, σ_z^{prior}


## Model Architecture Notes

### Latent Ordinary Differential Equation (Latent ODE)

**Generative / Inference Definitions**

Prior (Initial latent state):  
  p(z_0) ~ N(μ, σ^2)  
  → z_0 ∼ p(z_0)  (eq. 11)

Dynamics (Latent trajectory via ODE):  
  z_{t1}, …, z_{tN} = ODESolve(z_0, f, θ_f, t_0, …, t_N)  (eq. 12)  
  with latent dynamics defined as:  
  d z(t) / d t = f(z(t), θ_f)  (time-invariant neural net)  
  → ODESolve (black-box solver) applies f to produce z_{t_i} on requested times  
  → f parameterized by an MLP (time-invariant)

Decoder:  
  x_{t_i} ∼ p(x | z_{t_i}, θ_x)  (eq. 13)  
  → decoder MLP: p(x | z)

Encoder (Approx. Posterior):  
  q(z_0 | {x_{t_i}, t_i}_i) = N( μ_{z0}, σ_{z0}^2 )  
  → Recognition net = RNN run **backwards in time** to output (μ_{z0}, σ_{z0})  
  → Trained as a VAE: RNN posterior + ODESolve generative path

Joint (Generation factorization):  
  p(x_{1:N}, z_{0:N}) = p(z_0) · ∏_{i=1}^N p(z_{t_i} | z_0, f) · p(x_{t_i} | z_{t_i})

**Training (ELBO)**

L = E_{q(z_0 | x_{1:N})} [  ∑_{i=1}^N log p(x_{t_i} | z_{t_i})  −  KL( q(z_0 | x_{1:N}) || p(z_0) ) ]  
→ Encoder: RNN (backward over {x_{t_i}, t_i}) → (μ_{z0}, σ_{z0})  
→ Latent dynamics: compute z_{t_i} via ODESolve(z_0, f, θ_f, t_0…t_N)  
→ Decoder likelihood: log p(x_{t_i} | z_{t_i}) via decoder MLP

**Posterior Sampling Procedure**

1) Run RNN encoder on {x_{t_i}, t_i} (backward in time) → infer q(z_0 | {x_{t_i}, t_i})  
2) Sample z_0 ~ q(z_0 | {x_{t_i}, t_i})  
3) Solve latent trajectory: z_{t_1:t_N} = ODESolve(z_0, f, θ_f, t_0…t_N)  
4) Decode: for each i, compute x̂_{t_i} from p(x_{t_i} | z_{t_i}) (mean or sample)

**Notes / Options**

- Continuous-time latent dynamics → unique trajectory from z_0; supports interpolation & long-horizon extrapolation across arbitrary time points.  
- Naturally handles **irregularly-sampled** sequences (no binning): evaluate latent states exactly at observed timestamps t_i via ODESolve.  
- Optional event-time likelihood: inhomogeneous Poisson process on observation times  
  log p(t_1…t_N | [t_start, t_end]) = ∑_i log λ(z(t_i)) − ∫_{t_start}^{t_end} λ(z(t)) dt  
  → λ(z(t)) parameterized by a NN; evaluate jointly with latent trajectory in one ODE call.

---

## Communication Preferences
- 모든 답변은 한국어로 작성합니다.
- 이 문서의 규칙은 Codex의 일반 프롬프트보다 **우선** 적용됩니다.