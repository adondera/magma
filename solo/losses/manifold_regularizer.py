import torch
from solo.utils.embedding_propagation import get_similarity_matrix, get_laplacian


def manifold_regularizer_loss(x: torch.Tensor, y: torch.Tensor):
    weights_matrix = get_similarity_matrix(x, rbf_scale=1.0, scaling_factor=False)
    laplacian = get_laplacian(weights_matrix, normalized=True)
    D = torch.diag(weights_matrix.sum(dim=-1))
    subspace = y.T @ D @ y

    regularizer_loss_term = torch.trace(y.T @ laplacian @ y)
    collapse_loss_term = (
        (subspace - torch.eye(subspace.shape[0], device=x.device)).pow(2).sum()
    )

    return regularizer_loss_term, collapse_loss_term
