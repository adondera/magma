import numpy as np
import torch
from solo.utils.embedding_propagation import get_similarity_matrix, get_laplacian
from matplotlib import figure

class ManifoldRegularizer():
    def __init__(self, return_metrics: bool = False):
        self.return_metrics = return_metrics
        self.last_laplacian_matrix = None
        self.last_similarity_matrix = None

    def manifold_regularizer_loss(self, x: torch.Tensor, y: torch.Tensor):
        weights_matrix = get_similarity_matrix(x, rbf_scale=1.0, scaling_factor=False)
        laplacian = get_laplacian(weights_matrix, normalized=True)
        if self.return_metrics:
            with torch.no_grad():
                sorted_eigvals = torch.linalg.eigvals(laplacian).real.cpu().numpy()
                sorted_eigvals.sort()
                zero_eigvals = sorted_eigvals.size - np.count_nonzero(sorted_eigvals)
                second_smallest_eigenvalue = sorted_eigvals[1]
                spectral_gap = sorted_eigvals[1] - sorted_eigvals[0]
                laplacian_energy = sum(abs(x - 1) for x in sorted_eigvals)

                metrics = {
                    "Number of zero eigenvalues": zero_eigvals,
                    "Second smallest eigenvalue": second_smallest_eigenvalue,
                    "Spectral gap": spectral_gap,
                    "Laplacian energy": laplacian_energy,
                }

                if self.last_laplacian_matrix is not None:
                    laplacian_diff = torch.linalg.norm(
                        laplacian - self.last_laplacian_matrix
                    )
                    metrics["Laplacian difference"] = laplacian_diff

                if self.last_similarity_matrix is not None:
                    similarity_diff = torch.linalg.norm(
                        weights_matrix - self.last_similarity_matrix
                    )
                    metrics["Similarity difference"] = similarity_diff

        # D = torch.diag(weights_matrix.sum(dim=-1))
        # subspace = y.T @ D @ y

        regularizer_loss_term = torch.trace(y.T @ laplacian @ y) / (x.shape[0] ** 2)
        # collapse_loss_term = (
        #     (subspace - torch.eye(subspace.shape[0], device=x.device)).pow(2).mean()
        # )

        self.last_laplacian_matrix = laplacian
        self.last_similarity_matrix = weights_matrix

        return regularizer_loss_term, metrics #, collapse_loss_term
