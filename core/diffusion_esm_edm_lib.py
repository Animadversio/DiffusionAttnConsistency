import torch
import torch.nn.functional as F

def delta_GMM_score(Xt, train_Xmat, sigma, return_weights=False):
    # get squared distance matrix
    sqdist = torch.cdist(Xt.flatten(1), train_Xmat.flatten(1), p=2) ** 2
    weights = F.softmax(-sqdist / (2 * sigma**2), dim=1)
    score = (torch.matmul(weights, train_Xmat) - Xt) / sigma**2
    if return_weights:
        return score, weights
    else:
        return score


def delta_GMM_denoiser(Xt, train_Xmat, sigma):
    # get squared distance matrix
    sqdist = torch.cdist(Xt.flatten(1), train_Xmat.flatten(1), p=2) ** 2
    # fixed.
    if isinstance(sigma, torch.Tensor):
        # Ensure sigma has the right shape for broadcasting
        if sigma.dim() == 0:
            # sigma is already a scalar, no change needed
            pass
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(1)  # Shape becomes (nXt, 1)
        else:
            # For higher dimensional tensors, flatten to 1D then unsqueeze
            sigma = sigma.squeeze().unsqueeze(1)
    # the sigma is either a scalar or a tensor of shape (nXt, 1)
    weights = F.softmax(-sqdist / (2 * sigma**2), dim=1)
    # denoised = torch.matmul(weights, train_Xmat) # this is the original implementation
    # this is the more general implementation for tensors. 
    denoised = torch.matmul(weights, train_Xmat.flatten(1)).reshape(Xt.shape)
    return denoised


# EDMDeltaGMMScoreLoss
class EDMDeltaGMMScoreLoss:
    def __init__(self, train_Xmat, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.train_Xmat = train_Xmat
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
        # D_gmm = delta_GMM_denoiser(X, self.train_Xmat, sigma)
        # fixed July27
        D_gmm = delta_GMM_denoiser(X + n, self.train_Xmat, sigma)
        # loss = weight * ((D_yn - X) ** 2)
        loss = weight * ((D_yn - D_gmm) ** 2)
        return loss