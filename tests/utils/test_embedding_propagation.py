import torch
import math
from solo.utils.embedding_propagation import get_laplacian, get_similarity_matrix, get_distance_matrix

def test_laplacian():
    weights = torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    computed_laplacian = get_laplacian(weights)
    correct_laplacian = torch.Tensor(
        [[1, -math.sqrt(1 / 2), 0], [-math.sqrt(1/2), 1, -math.sqrt(1/2)], [0, -math.sqrt(1/2), 1]]
    )
    assert torch.allclose(computed_laplacian, correct_laplacian, atol=1e-4)


def test_manifold_regularization():
    z = torch.randn(3, 2)
    weights_matrix = get_similarity_matrix(z, rbf_scale=1.0, scaling_factor=False)
    squared_dist = get_distance_matrix(z)
    sum_loss = (weights_matrix * squared_dist).sum()
    laplacian = get_laplacian(weights_matrix, normalized=False)
    laplacian_loss = torch.trace(z.T @ laplacian @ z)
    assert math.isclose(sum_loss, laplacian_loss * 2, abs_tol=1e-4)
