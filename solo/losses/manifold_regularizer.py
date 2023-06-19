import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from solo.utils.embedding_propagation import get_similarity_matrix, get_laplacian
from matplotlib import figure

class ManifoldRegularizer():
    def __init__(self, log_metrics: bool = False):
        self.log_metrics = log_metrics
        self.last_laplacian_matrix = None
        self.last_similarity_matrix = None

    def manifold_regularizer_loss(self, x: torch.Tensor, y: torch.Tensor):
        weights_matrix = get_similarity_matrix(x, rbf_scale=1.0, scaling_factor=False)
        laplacian = get_laplacian(weights_matrix, normalized=True)
        if self.log_metrics:
            with torch.no_grad():
                sorted_eigvals = torch.linalg.eigvals(laplacian).real.cpu().numpy()
                sorted_eigvals.sort()
                zero_eigvals = sorted_eigvals.size - np.count_nonzero(sorted_eigvals)
                second_smallest_eigenvalue = sorted_eigvals[1]
                spectral_gap = sorted_eigvals[1] - sorted_eigvals[0]
                laplacian_energy = sum(abs(x - 1) for x in sorted_eigvals)

                fig = figure.Figure()
                ax = fig.subplots(1)
                ax.imshow(-laplacian.detach().cpu().numpy(), norm="log")

                fig2 = figure.Figure()
                ax2 = fig2.subplots(1)
                ax2.imshow(weights_matrix.detach().cpu().numpy())

                metrics = {
                    "Number of zero eigenvalues": zero_eigvals,
                    "Second smallest eigenvalue": second_smallest_eigenvalue,
                    "Spectral gap": spectral_gap,
                    "Laplacian energy": laplacian_energy,
                    "Laplacian": fig,
                    "Similarity matrix": fig2,
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

                wandb.log(metrics)

        # D = torch.diag(weights_matrix.sum(dim=-1))
        # subspace = y.T @ D @ y

        regularizer_loss_term = torch.trace(y.T @ laplacian @ y) / (x.shape[0] ** 2)
        # collapse_loss_term = (
        #     (subspace - torch.eye(subspace.shape[0], device=x.device)).pow(2).mean()
        # )

        self.last_laplacian_matrix = laplacian
        self.last_similarity_matrix = weights_matrix

        return regularizer_loss_term #, collapse_loss_term
