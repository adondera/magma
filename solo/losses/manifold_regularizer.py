import torch
import matplotlib.pyplot as plt
import wandb
from solo.utils.embedding_propagation import get_similarity_matrix, get_laplacian
from matplotlib import figure


def manifold_regularizer_loss(x: torch.Tensor, y: torch.Tensor, log_laplacian: bool = False):
    weights_matrix = get_similarity_matrix(x, rbf_scale=1.0, scaling_factor=False)
    laplacian = get_laplacian(weights_matrix, normalized=True)
    if log_laplacian:
        fig = figure.Figure()
        ax = fig.subplots(1)
        ax.imshow(-laplacian.detach().cpu().numpy())
        wandb.log({"laplacian": fig})

    # D = torch.diag(weights_matrix.sum(dim=-1))
    # subspace = y.T @ D @ y

    regularizer_loss_term = torch.trace(y.T @ laplacian @ y) / (x.shape[0] ** 2)
    # collapse_loss_term = (
    #     (subspace - torch.eye(subspace.shape[0], device=x.device)).pow(2).mean()
    # )

    return regularizer_loss_term #, collapse_loss_term
