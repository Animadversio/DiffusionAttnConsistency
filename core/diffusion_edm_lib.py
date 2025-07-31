import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import trange, tqdm

# EDM loss
# https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L66
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, X, labels=None, ):
        rnd_normal = torch.randn([X.shape[0],] + [1, ] * (X.ndim - 1), device=X.device)
        # unsqueeze to match the ndim of X
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # maybe augment
        n = torch.randn_like(X) * sigma
        D_yn = net(X + n, sigma, cond=labels, )
        loss = weight * ((D_yn - X) ** 2)
        return loss
    
    
class EDMPrecondWrapper(nn.Module):
    def __init__(self, model, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
    def forward(self, X, sigma, cond=None, ):
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        ## https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        # unsqueze sigma to have same dimension as X (which may have 2-4 dim) 
        sigma_vec = sigma.view([-1, ] + [1, ] * (X.ndim - 1))
        c_skip = self.sigma_data ** 2 / (sigma_vec ** 2 + self.sigma_data ** 2)
        c_out = sigma_vec * self.sigma_data / (sigma_vec ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma_vec ** 2).sqrt()
        c_noise = sigma.log() / 4
        model_out = self.model(c_in * X, c_noise, cond=cond)
        return c_skip * X + c_out * model_out


class EDMCNNPrecondWrapper(nn.Module):
    def __init__(self, model, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
    def forward(self, X, sigma, cond=None, ):
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        ## https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        # unsqueze sigma to have same dimension as X (which may have 2-4 dim) 
        sigma_vec = sigma.view([-1, ] + [1, ] * (X.ndim - 1))
        c_skip = self.sigma_data ** 2 / (sigma_vec ** 2 + self.sigma_data ** 2)
        c_out = sigma_vec * self.sigma_data / (sigma_vec ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma_vec ** 2).sqrt()
        c_noise = sigma.log() / 4
        model_out = self.model(c_in * X, c_noise.view(-1), cond=cond) # this is required for EDM Unet model. 
        return c_skip * X + c_out * model_out


class EDMDiTPrecondWrapper(nn.Module):
    def __init__(self, model, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
    def forward(self, X, sigma, cond=None, ):
        if cond is None:
            cond = torch.zeros(X.shape[0], device=X.device, dtype=torch.long)
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        ## https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        # unsqueze sigma to have same dimension as X (which may have 2-4 dim) 
        sigma_vec = sigma.view([-1, ] + [1, ] * (X.ndim - 1))
        c_skip = self.sigma_data ** 2 / (sigma_vec ** 2 + self.sigma_data ** 2)
        c_out = sigma_vec * self.sigma_data / (sigma_vec ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma_vec ** 2).sqrt()
        c_noise = sigma.log() / 4
        model_out = self.model(c_in * X, c_noise.view(-1), y=cond) # this is required for EDM Unet model. 
        return c_skip * X + c_out * model_out


import sys
sys.path.append("..")
from core.diffusion_nn_lib import UNetMLPBlock, GaussianFourierProjection, UNetBlockStyleMLP_backbone

class UNetBlockStyleMLP_backbone(nn.Module):
    """A time-dependent score-based model."""
    
    def __init__(self, ndim=2, nlayers=5, nhidden=64, time_embed_dim=64,):
        super().__init__()
        self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
        layers = nn.ModuleList()
        layers.append(UNetMLPBlock(ndim, nhidden, time_embed_dim))
        for _ in range(nlayers-2):
            layers.append(UNetMLPBlock(nhidden, nhidden, time_embed_dim))
        layers.append(nn.Linear(nhidden, ndim))
        self.net = layers

    def forward(self, x, t_enc, cond=None):
        # t_enc : preconditioned version of sigma, usually 
        # ln_std_vec = torch.log(std_vec) / 4
        if cond is not None:
            raise NotImplementedError("Conditional training is not implemented")
        t_embed = self.embed(t_enc)
        for layer in self.net[:-1]:
            x = layer(x, t_embed)
        pred = self.net[-1](x)
        return pred


def train_score_model_custom_loss(X_train_tsr, score_model_td, loss_fn,
                   lr=0.005,
                   nepochs=750,
                   batch_size=None,
                   device="cpu",
                   callback=None,
                   callback_freq=100,
                   callback_step_list=[],
                   save_ckpts=False,
                   save_ckpt_step_list=[],
                   ckpt_dir=None):
    """
    Trains a score model using a custom loss function with an optional callback.

    Args:
        X_train_tsr (torch.Tensor): Training data tensor of shape [N, ndim].
        score_model_td (nn.Module): The score model to be trained.
        loss_fn (callable): The loss function to compute the loss.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.005.
        nepochs (int, optional): Number of training epochs. Defaults to 750.
        batch_size (int, optional): Batch size for training. If None, uses the entire dataset. Defaults to None.
        device (str, optional): Device to train on ("cpu" or "cuda"). Defaults to "cpu".
        callback (callable, optional): A function to call at specified intervals. Should accept arguments (epoch, loss, model).
        callback_freq (int, optional): Frequency (in epochs) to call the callback function. Defaults to 100. Set to None or 0 or negative to disable -> can use callback_step_list instead.
        callback_step_list (list, optional): List of steps to call the callback function. Defaults to []. 
    Returns:
        tuple: Trained score model and a list of loss values per epoch.
    """
    def check_callback_step(ep):
        if ep == nepochs - 1 or ep == 0:
            # always call callback for the first and last epoch
            return True
        if callback_freq is not None and callback_freq > 0 and (ep + 1) % callback_freq == 0:
            # If callback_freq is set, call callback every callback_freq epochs
            return True
        if callback_step_list is not None and (ep + 1) in callback_step_list:
            # If callback_step_list is set, call callback for the list of steps in callback_step_list
            return True
        if save_ckpts and (ep in save_ckpt_step_list):
            return True
        return False

    ndim = X_train_tsr.shape[1]
    score_model_td.to(device)
    X_train_tsr = X_train_tsr.to(device)
    optim = Adam(score_model_td.parameters(), lr=lr)
    pbar = trange(nepochs)
    score_model_td.train()
    loss_traj = []
    for ep in pbar:
        if batch_size is None:
            loss = loss_fn(score_model_td, X_train_tsr, )
        else:
            idx = torch.randint(0, X_train_tsr.shape[0], (batch_size,))
            loss = loss_fn(score_model_td, X_train_tsr[idx], )
        loss = loss.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"step {ep} loss {loss.item():.3f}")
        if ep == 0:
            print(f"step {ep} loss {loss.item():.3f}")
        loss_traj.append(loss.item())

        # Invoke callback if specified and if epoch matches the frequency
        if callback is not None and check_callback_step(ep):
            callback(epoch=ep + 1, loss=loss.item(), model=score_model_td)
        
        if save_ckpts and (ep in save_ckpt_step_list or ep == nepochs - 1 or ep == 0):
            torch.save(score_model_td.state_dict(), f"{ckpt_dir}/model_epoch_{ep:06d}.pth")
        
    return score_model_td, loss_traj


@torch.no_grad()
def edm_sampler(
    edm, latents, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, 
    dtype=torch.float32, return_traj=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # t_steps = torch.cat([edm.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    x_traj = []
    denoiser_traj = []
    # Main sampling loop.
    x_next = latents.to(dtype) * t_steps[0]
    x_traj.append(x_next)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_hat = x_next
        t_hat = t_cur
        
        # Euler step.
        denoised = edm(x_hat, t_hat, class_labels, ).to(dtype)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = edm(x_next, t_next, class_labels, ).to(dtype)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        x_traj.append(x_next)
        denoiser_traj.append(denoised)
    if return_traj:
        x_traj = torch.stack(x_traj, dim=0)
        denoiser_traj = torch.stack(denoiser_traj, dim=0)
        return x_next, x_traj, denoiser_traj, t_steps
    else:
        return x_next
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    sys.path.append("/Users/binxuwang/Github/DiffusionMemorization")
    from core.diffusion_nn_lib import UNetBlockStyleMLP_backbone
    from core.toy_shape_dataset_lib import generate_random_star_shape_torch
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    pnts, radius_fun, amplitudes, phases = generate_random_star_shape_torch(1000, num_modes=10)
    pnts = pnts.float()
    device = get_device()
    model = UNetBlockStyleMLP_backbone(ndim=2, nlayers=5, nhidden=64, time_embed_dim=64,)
    model_precd = EDMPrecondWrapper(model, sigma_data=0.5, sigma_min=0.002, sigma_max=80, rho=7.0)
    edm_loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=0.5)
    model_precd, loss_traj = train_score_model_custom_loss(pnts, model_precd, edm_loss_fn, 
                                        lr=0.001, nepochs=2000, batch_size=1024, device=device)
    
    noise_init = torch.randn(1000, 2).to(device)
    x_out, x_traj, x0hat_traj, t_steps = edm_sampler(model_precd, noise_init, 
                    num_steps=20, sigma_min=0.002, sigma_max=80, rho=7, return_traj=True)
    
    scaling = 1 / (t_steps ** 2 + 1).sqrt()
    scaled_x_traj = (scaling[:, None, None] * x_traj).cpu()
    plt.figure(figsize=[8,8])
    plt.plot(scaled_x_traj[:, ::4, 0].numpy(), 
            scaled_x_traj[:, ::4, 1].numpy(), lw=0.5, color="k", alpha=0.5)
    plt.scatter(scaled_x_traj[-1,:,0], scaled_x_traj[-1,:,1], c="red", s=4, marker='o', alpha=0.3)
    plt.scatter(scaled_x_traj[ 0,:,0], scaled_x_traj[ 0,:,1], c="blue", s=9, marker='o', alpha=0.3)
    plt.axis("equal")
    plt.show()