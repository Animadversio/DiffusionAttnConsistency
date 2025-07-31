import torch
import math
import numpy as np
from torch.optim import Adam
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

def marginal_prob_std(t, sigma):
  """Note that this std -> 0, when t->0
  So it's not numerically stable to sample t=0 in the dataset
  Note an earlier version missed the sqrt...
  """
  return torch.sqrt( (sigma**(2*t) - 1) / 2 / torch.log(torch.tensor(sigma)) ) # sqrt fixed Jun.19


def denoise_loss_fn(model, x, marginal_prob_std_f, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability, sample t uniformly from [eps, 1.0]
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std_f(random_t,)
  perturbed_x = x + z * std[:, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=(1)))
  return loss


def train_score_td(X_train_tsr, score_model_td=None,
                   sigma=25,
                   lr=0.005,
                   nepochs=750,
                   eps=1E-3,
                   batch_size=None,
                   device="cpu"):
    ndim = X_train_tsr.shape[1]
    if score_model_td is None:
        score_model_td = ScoreModel_Time(sigma=sigma, ndim=ndim)
    score_model_td.to(device)
    X_train_tsr = X_train_tsr.to(device)
    marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)
    optim = Adam(score_model_td.parameters(), lr=lr)
    pbar = trange(nepochs)
    score_model_td.train()
    loss_traj = []
    for ep in pbar:
        if batch_size is None:
            loss = denoise_loss_fn(score_model_td, X_train_tsr, marginal_prob_std_f, eps=eps)
        else:
            idx = torch.randint(0, X_train_tsr.shape[0], (batch_size,))
            loss = denoise_loss_fn(score_model_td, X_train_tsr[idx], marginal_prob_std_f, eps=eps)

        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"step {ep} loss {loss.item():.3f}")
        if ep == 0:
            print(f"step {ep} loss {loss.item():.3f}")
        loss_traj.append(loss.item())
    return score_model_td, loss_traj



def reverse_diffusion_time_dep(score_model_td, sampN=500, sigma=5, nsteps=200, ndim=2, exact=False, device="cpu"):
  """
  score_model_td: if `exact` is True, use a gmm of class GaussianMixture
                  if `exact` is False. use a torch neural network that takes vectorized x and t as input.
  """
  lambdaT = (sigma**2 - 1) / (2 * np.log(sigma))
  xT = np.sqrt(lambdaT) * np.random.randn(sampN, ndim)
  x_traj_rev = np.zeros((*xT.shape, nsteps, ))
  x_traj_rev[:,:,0] = xT
  dt = 1 / nsteps
  for i in range(1, nsteps):
    t = 1 - i * dt
    tvec = torch.ones((sampN)) * t
    eps_z = np.random.randn(*xT.shape)
    if exact:
      gmm_t = diffuse_gmm(score_model_td, t, sigma)
      score_xt = gmm_t.score(x_traj_rev[:,:,i-1])
    else:
      with torch.no_grad():
        score_xt = score_model_td(torch.tensor(x_traj_rev[:,:,i-1]).float(), tvec).numpy()
    # simple Euler-Maryama integration of SGD
    x_traj_rev[:,:,i] = x_traj_rev[:,:,i-1] + eps_z * (sigma ** t) * np.sqrt(dt) + score_xt * dt * sigma**(2*t)
  return x_traj_rev


def reverse_diffusion_time_dep_torch(score_model_td, sampN=500, sigma=5, nsteps=200, ndim=2, exact=False, device="cpu"):
  """More efficient version that run solely on device
  score_model_td: if `exact` is True, use a gmm of class GaussianMixture
                  if `exact` is False. use a torch neural network that takes vectorized x and t as input.
  """
  lambdaT = (sigma**2 - 1) / (2 * math.log(sigma))
  xT = math.sqrt(lambdaT) * torch.randn(sampN, ndim, device=device)
  x_traj_rev = torch.zeros((sampN, ndim, nsteps), device="cpu")
  x_traj_rev[:, :, 0] = xT.cpu()
  dt = 1 / nsteps
  x_next = xT
  for i in range(1, nsteps):
      t = 1 - i * dt
      tvec = torch.ones((sampN,), device=device) * t
      eps_z = torch.randn_like(xT)
      with torch.no_grad():
        score_xt = score_model_td(x_next, tvec)
      # if exact:
      #     gmm_t = diffuse_gmm(score_model_td, t, sigma, device)
      #     score_xt = gmm_t.score(x_traj_rev[:, :, i-1])
      # else:
      x_next = x_next + eps_z * (sigma ** t) * math.sqrt(dt) + score_xt * dt * sigma**(2*t)
      x_traj_rev[:, :, i] = x_next.cpu()

  return x_traj_rev


def reverse_diffusion_deterministic_torch(score_model_td, sampN=500, sigma=5, nsteps=200, ndim=2, exact=False, device="cpu"):
  """More efficient version that run solely on device
  score_model_td: if `exact` is True, use a gmm of class GaussianMixture
                  if `exact` is False. use a torch neural network that takes vectorized x and t as input.
  """
  lambdaT = (sigma**2 - 1) / (2 * math.log(sigma))
  xT = math.sqrt(lambdaT) * torch.randn(sampN, ndim, device=device)
  x_traj_rev = torch.zeros((sampN, ndim, nsteps), device="cpu")
  x_traj_rev[:, :, 0] = xT.cpu()
  dt = 1 / nsteps
  x_next = xT
  for i in range(1, nsteps):
      t = 1 - i * dt
      tvec = torch.ones((sampN,), device=device) * t
      # eps_z = torch.randn_like(xT)
      with torch.no_grad():
        score_xt = score_model_td(x_next, tvec)
      # if exact:
      #     gmm_t = diffuse_gmm(score_model_td, t, sigma, device)
      #     score_xt = gmm_t.score(x_traj_rev[:, :, i-1])
      # else:
      # x_next = x_next + eps_z * (sigma ** t) * math.sqrt(dt) + score_xt * dt * sigma**(2*t)
      # sigma_t = math.sqrt((sigma**(2*t) - 1) / (2 * math.log(sigma)))
      x_next = x_next + score_xt * dt * sigma**(2*t) / 2
      x_traj_rev[:, :, i] = x_next.cpu()

  return x_traj_rev


def visualize_diffusion_distr(x_traj_rev, leftT=0, rightT=-1, explabel=""):
  if rightT == -1:
    rightT = x_traj_rev.shape[2]-1
  figh, axs = plt.subplots(1,2,figsize=[12,6])
  sns.kdeplot(x=x_traj_rev[:,0,leftT], y=x_traj_rev[:,1,leftT], ax=axs[0])
  axs[0].set_title("Density of Gaussian Prior of $x_T$\n before reverse diffusion")
  plt.axis("equal")
  sns.kdeplot(x=x_traj_rev[:,0,rightT], y=x_traj_rev[:,1,rightT], ax=axs[1])
  axs[1].set_title(f"Density of $x_0$ samples after {rightT} step reverse diffusion")
  plt.axis("equal")
  plt.suptitle(explabel)
  return figh