import torch
import torch.nn.functional as F
import numpy as np


def get_similarity_matrix(x, rbf_scale):
    b, c = x.size()
    sq_dist = ((x.view(b, 1, c) - x.view(1, b, c)) ** 2).sum(-1) / np.sqrt(c)
    mask = sq_dist != 0
    sq_dist = sq_dist / sq_dist[mask].std()
    weights = torch.exp(-sq_dist * rbf_scale)
    mask = torch.eye(weights.size(1), dtype=torch.bool, device=weights.device)
    weights = weights * (~mask).float()
    return weights


def embedding_propagation(x, alpha, rbf_scale, norm_prop, propagator=None):
    if propagator is None:
        weights = get_similarity_matrix(x, rbf_scale)
        propagator = global_consistency(weights, alpha=alpha, norm_prop=norm_prop)
    return torch.mm(propagator, x)


def global_consistency(weights, alpha=1, norm_prop=False):
    """Implements D. Zhou et al. "Learning with local and global consistency". (Same as in TPN paper but without bug)

    Args:
        weights: Tensor of shape (n, n). Expected to be exp( -d^2/s^2 ), where d is the euclidean distance and
            s the scale parameter.
        labels: Tensor of shape (n, n_classes)
        alpha: Scaler, acts as a smoothing factor
    Returns:
        Tensor of shape (n, n_classes) representing the logits of each classes
    """
    n = weights.shape[1]
    identity = torch.eye(n, dtype=weights.dtype, device=weights.device)
    isqrt_diag = 1.0 / torch.sqrt(1e-4 + torch.sum(weights, dim=-1))
    # checknan(laplacian=isqrt_diag)
    S = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
    # checknan(normalizedlaplacian=S)
    propagator = identity - alpha * S
    propagator = torch.inverse(propagator[None, ...])[0]
    # checknan(propagator=propagator)
    if norm_prop:
        propagator = F.normalize(propagator, p=1, dim=-1)
    return propagator
