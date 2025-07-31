import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GroupNorm, Linear
from core.diffusion_basics_lib import marginal_prob_std


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps.
  Basically it multiplexes a scalar `t` into a vector of `sin(2 pi k t)` and `cos(2 pi k t)` features.
  """
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, t):
    t_proj = t.view(-1, 1) * self.W[None, :] * 2 * math.pi
    return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class ScoreModel_Time(nn.Module):
  """A time-dependent score-based model."""

  def __init__(self, sigma, ndim=2, nlayers=5, nhidden=50, time_embed_dim=10,
               act_fun=nn.Tanh):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    layers = []
    layers.extend([nn.Linear(time_embed_dim + ndim, nhidden),
                   act_fun()])
    for _ in range(nlayers-2):
        layers.extend([nn.Linear(nhidden, nhidden),
                         act_fun()])
    layers.extend([nn.Linear(nhidden, ndim)])
    self.net = nn.Sequential(*layers)
    self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

  def forward(self, x, t):
    std_vec = self.marginal_prob_std_f(t)[:, None,]
    t_embed = self.embed(t)
    pred = self.net(torch.cat((x / (1 + std_vec ** 2).sqrt(),
                               t_embed), dim=1))
    # this additional steps provides an inductive bias.
    # the neural network output on the same scale,
    pred = pred / std_vec
    return pred


class ScoreModel_Time_edm(nn.Module):
  """A time-dependent score-based model."""

  def __init__(self, sigma, ndim=2, nlayers=5, nhidden=50, time_embed_dim=10,
               act_fun=nn.Tanh):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    layers = []
    layers.extend([nn.Linear(time_embed_dim + ndim, nhidden), act_fun()])
    for _ in range(nlayers - 2):
        layers.extend([nn.Linear(nhidden, nhidden), act_fun()])
    layers.extend([nn.Linear(nhidden, ndim)])
    self.net = nn.Sequential(*layers)
    self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

  def forward(self, x, t):
    std_vec = self.marginal_prob_std_f(t)[:, None,]
    ln_std_vec = torch.log(std_vec) / 4
    t_embed = self.embed(ln_std_vec)
    pred = self.net(torch.cat((x / (1 + std_vec ** 2).sqrt(),
                               t_embed), dim=1))
    # this additional steps provides an inductive bias.
    # the neural network output on the same scale,
    pred = pred / std_vec - x / (1 + std_vec ** 2)
    return pred


class MLPResBlock(nn.Module):
    def __init__(self, in_features, out_features=None, activation=nn.ReLU):
        super(MLPResBlock, self).__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.activation = activation()
        self.norm = nn.LayerNorm(out_features)
        # Ensure the input can be added to the output
        self.shortcut = nn.Identity() if in_features == out_features \
                                    else nn.Linear(in_features, out_features)

    def forward(self, x):
        # Save the input for residual connection
        identity = self.shortcut(x)
        # Linear layer followed by activation
        out = self.norm(x)
        out = self.fc(out)
        out = self.activation(out)
        # Residual connection
        out = out + identity
        return out
      


class ScoreModel_Time_resnet_edm(nn.Module):
  """A time-dependent score-based model."""

  def __init__(self, sigma, ndim=2, nlayers=5, nhidden=50, time_embed_dim=10,
               act_fun=nn.Tanh):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    layers = []
    layers.extend([nn.Linear(time_embed_dim + ndim, nhidden), act_fun()])
    for _ in range(nlayers - 2):
        layers.extend([MLPResBlock(nhidden, activation=act_fun)])
    layers.extend([nn.Linear(nhidden, ndim)])
    self.net = nn.Sequential(*layers)
    self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

  def forward(self, x, t):
    std_vec = self.marginal_prob_std_f(t)[:, None,]
    ln_std_vec = torch.log(std_vec) / 4
    t_embed = self.embed(ln_std_vec)
    pred = self.net(torch.cat((x / (1 + std_vec ** 2).sqrt(),
                               t_embed), dim=1))
    # this additional steps provides an inductive bias.
    # the neural network output on the same scale,
    pred = pred / std_vec - x / (1 + std_vec ** 2)
    return pred


class UNetMLPBlock(torch.nn.Module):
    def __init__(self,
        in_features, out_features, emb_features, dropout=0, skip_scale=1, eps=1e-5,
        adaptive_scale=True, init=dict(), init_zero=dict(),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.emb_features = emb_features
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = nn.LayerNorm(in_features, eps=eps) #GroupNorm(num_channels=in_features, eps=eps)
        self.fc0 = Linear(in_features=in_features, out_features=out_features, **init)
        self.affine = Linear(in_features=emb_features, out_features=out_features*(2 if adaptive_scale else 1), **init)
        self.norm1 = nn.LayerNorm(out_features, eps=eps) #GroupNorm(num_channels=out_features, eps=eps)
        self.fc1 = Linear(in_features=out_features, out_features=out_features, **init_zero)

        self.skip = None
        if out_features != in_features:
            self.skip = Linear(in_features=in_features, out_features=out_features, **init)

    def forward(self, x, emb):
        orig = x
        x = self.fc0(F.silu(self.norm0(x)))

        params = self.affine(emb).to(x.dtype) # .unsqueeze(1)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))

        x = self.fc1(F.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        return x
    
    
class ScoreModel_Time_UNetlike_edm(nn.Module):
  """A time-dependent score-based model."""

  def __init__(self, sigma, ndim=2, nlayers=5, nhidden=50, time_embed_dim=10,
               act_fun=nn.Tanh):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    layers = nn.ModuleList()
    layers.append(UNetMLPBlock(ndim, nhidden, time_embed_dim))
    for _ in range(nlayers-2):
        layers.append(UNetMLPBlock(nhidden, nhidden, time_embed_dim))
    # layers.extend([nn.Linear(time_embed_dim + ndim, nhidden), act_fun()])
    # for _ in range(nlayers - 2):
    #     layers.extend([MLPResBlock(nhidden, activation=act_fun)])
    layers.append(nn.Linear(nhidden, ndim))
    self.net = layers
    self.marginal_prob_std_f = lambda t: marginal_prob_std(t, sigma)

  def forward(self, x, t):
    std_vec = self.marginal_prob_std_f(t)[:, None,]
    ln_std_vec = torch.log(std_vec) / 4
    t_embed = self.embed(ln_std_vec)
    orig = nn.Identity()(x)
    x = x / (1 + std_vec ** 2).sqrt()
    for layer in self.net[:-1]:
        x = layer(x, t_embed)
    pred = self.net[-1](x)
    # pred = self.net(torch.cat((x / (1 + std_vec ** 2).sqrt(),
    #                            t_embed), dim=1))
    # this additional steps provides an inductive bias.
    # the neural network output on the same scale,
    # print(pred.shape)
    # print(std_vec.shape)
    pred = pred / std_vec - orig / (1 + std_vec ** 2)
    return pred
  
  

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
    t_embed = self.embed(t_enc)
    for layer in self.net[:-1]:
        x = layer(x, t_embed)
    pred = self.net[-1](x)
    return pred


class UNetBlockStyleMLP_backbone_NoFirstNorm(nn.Module):
  """A time-dependent score-based model with improved handling for low-dim inputs.
  
  Adds a pre-projection layer to avoid LayerNorm issues with very low dimensional data.
  """

  def __init__(self, ndim=2, nlayers=5, nhidden=64, time_embed_dim=64,):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    layers = nn.ModuleList()
    
    # Add pre-projection to avoid LayerNorm on low-dim input
    if ndim < nhidden:
        layers.append(nn.Linear(ndim, nhidden))
        first_block_input_dim = nhidden
    else:
        first_block_input_dim = ndim
    
    layers.append(UNetMLPBlock(first_block_input_dim, nhidden, time_embed_dim))
    for _ in range(nlayers-2):
        layers.append(UNetMLPBlock(nhidden, nhidden, time_embed_dim))
    layers.append(nn.Linear(nhidden, ndim))
    self.net = layers

  def forward(self, x, t_enc, cond=None):
    # t_enc : preconditioned version of sigma, usually 
    # ln_std_vec = torch.log(std_vec) / 4
    t_embed = self.embed(t_enc)
    for layer in self.net[:-1]:
        if isinstance(layer, nn.Linear):
            # Apply pre-projection without time embedding
            x = layer(x)
        else:
            # Apply UNetMLPBlock with time embedding
            x = layer(x, t_embed)
    pred = self.net[-1](x)
    return pred