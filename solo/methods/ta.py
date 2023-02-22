# TODO: Check copyright
# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence, Tuple

import math
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params


class BYOLWithTA(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # TODO: Add multi head attention
        # Initialize student TA network parameters
        self.query_matrix = nn.Linear(proj_output_dim, proj_output_dim, bias=False)
        self.key_matrix = nn.Linear(proj_output_dim, proj_output_dim, bias=False)
        self.value_matrix = nn.Linear(proj_output_dim, proj_output_dim, bias=False)

        # Initialize momentum teacher TA network parameters
        self.momentum_query_matrix = nn.Linear(
            proj_output_dim, proj_output_dim, bias=False
        )
        self.momentum_key_matrix = nn.Linear(
            proj_output_dim, proj_output_dim, bias=False
        )
        self.momentum_value_matrix = nn.Linear(
            proj_output_dim, proj_output_dim, bias=False
        )

        initialize_momentum_params(self.projector, self.momentum_projector)
        initialize_momentum_params(self.query_matrix, self.momentum_query_matrix)
        initialize_momentum_params(self.key_matrix, self.momentum_key_matrix)
        initialize_momentum_params(self.value_matrix, self.momentum_value_matrix)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(BYOLWithTA, BYOLWithTA).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector, predictor and TA parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
            {"name": "query_matrix", "params": self.query_matrix.parameters()},
            {"name": "key_matrix", "params": self.key_matrix.parameters()},
            {"name": "value_matrix", "params": self.value_matrix.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [
            (self.projector, self.momentum_projector),
            (self.query_matrix, self.momentum_query_matrix),
            (self.key_matrix, self.momentum_key_matrix),
            (self.value_matrix, self.momentum_value_matrix),
        ]
        return super().momentum_pairs + extra_momentum_pairs

    # def forward(self, X: torch.Tensor) -> Dict[str, Any]:
    #     """Performs forward pass of the online backbone, projector and predictor.

    #     Args:
    #         X (torch.Tensor): batch of images in tensor format.

    #     Returns:
    #         Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
    #     """

    #     out = super().forward(X)
    #     z = self.projector(out["feats"])
    #     p = self.predictor(z)
    #     out.update({"z": z, "p": p})
    #     return out

    # def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
    #     """Performs the forward pass for the multicrop views.

    #     Args:
    #         X (torch.Tensor): batch of images in tensor format.

    #     Returns:
    #         Dict[]: a dict containing the outputs of the parent
    #             and the projected features.
    #     """

    #     out = super().multicrop_forward(X)
    #     z = self.projector(out["feats"])
    #     p = self.predictor(z)
    #     out.update({"z": z, "p": p})
    #     return out

    # @torch.no_grad()
    # def momentum_forward(self, X: torch.Tensor) -> Dict:
    #     """Performs the forward pass of the momentum backbone and projector.

    #     Args:
    #         X (torch.Tensor): batch of images in tensor format.

    #     Returns:
    #         Dict[str, Any]: a dict containing the outputs of
    #             the parent and the momentum projected features.
    #     """

    #     out = super().momentum_forward(X)
    #     z = self.momentum_projector(out["feats"])
    #     out.update({"z": z})
    #     return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)

        # out['feats'] has two tensors of size [batch_size, dimension]
        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        # ------- projection -------
        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        with torch.no_grad():
            momentum_z1 = self.momentum_projector(momentum_feats1)
            momentum_z2 = self.momentum_projector(momentum_feats2)
        
        # attention mechanism
        student_representations = torch.cat([z1, z2])
        student_queries = self.query_matrix(student_representations)
        student_keys = self.key_matrix(student_representations)
        student_values = self.value_matrix(student_representations)

        with torch.no_grad():
            teacher_representations = torch.cat([momentum_z1, momentum_z2])
            teacher_queries = self.momentum_query_matrix(teacher_representations)
            teacher_keys = self.momentum_key_matrix(teacher_representations)
            teacher_values = self.momentum_value_matrix(teacher_representations)

        d = student_queries.shape[-1]
        student_scores = torch.mm(student_queries, torch.cat([student_keys, teacher_keys]).T) / math.sqrt(d)
        student_attention_weights = torch.nn.functional.softmax(student_scores, dim=-1)
        student_y = torch.mm(student_attention_weights, torch.cat([student_values, teacher_values]))
        # calculate y1 and y2 for student
        student_y = [student_y[:student_y.shape[0]//2], student_y[student_y.shape[0]//2:]]
        # pass y1 and y2 to the predictor
        p1, p2 = [self.predictor(y) for y in student_y]

        with torch.no_grad():
            teacher_scores = torch.mm(teacher_queries, torch.cat([student_keys, teacher_keys]).T) / math.sqrt(d)
            teacher_attention_weights = torch.nn.functional.softmax(teacher_scores, dim=-1)
            teacher_y = torch.mm(teacher_attention_weights, torch.cat([student_values, teacher_values]))
            teacher_y1, teacher_y2 = teacher_y[:teacher_y.shape[0]//2], teacher_y[teacher_y.shape[0]//2:]

        class_loss = out["loss"]
        # Z = out["z"]
        # P = out["p"]
        # Z_momentum = out["momentum_z"]

        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        neg_cos_sim += byol_loss_func(p1, teacher_y2)
        neg_cos_sim += byol_loss_func(p2, teacher_y1)

        # for v1 in range(self.num_large_crops):
        #     for v2 in np.delete(range(self.num_crops), v1):
        #         neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        # calculate std of features
        with torch.no_grad():
            z_std = (
                F.normalize(torch.stack(student_y[: self.num_large_crops]), dim=-1)
                .std(dim=1)
                .mean()
            )

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
