from typing import Tuple
from collections import namedtuple
import torch
from torchmetrics import Metric
from sklearn import metrics

ClusterMetrics = namedtuple("ClusterMetrics", ["silhouette", "chi", "dbi"])

class ClusterEmbeddings(Metric):
    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.

        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    @torch.no_grad()
    def compute(self) -> Tuple[ClusterMetrics]:
        train_features = torch.cat(self.train_features).cpu()
        train_targets = torch.cat(self.train_targets).cpu()
        test_features = torch.cat(self.test_features).cpu()
        test_targets = torch.cat(self.test_targets).cpu()

        train_silhouette = metrics.silhouette_score(train_features, train_targets)
        test_silhouette = metrics.silhouette_score(test_features, test_targets)

        train_chi = metrics.calinski_harabasz_score(train_features, train_targets)
        test_chi = metrics.calinski_harabasz_score(test_features, test_targets)

        train_dbi = metrics.davies_bouldin_score(train_features, train_targets)
        test_dbi = metrics.davies_bouldin_score(test_features, test_targets)

        return ClusterMetrics(train_silhouette, train_chi, train_dbi), ClusterMetrics(test_silhouette, test_chi, test_dbi)
