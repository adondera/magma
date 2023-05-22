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

import numpy as np
import omegaconf
import torch
import torch.nn as nn

import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.misc import omegaconf_select
from solo.utils.ta_attention import TA_Attention
from solo.utils.embedding_propagation import embedding_propagation


class BYOLWithTA(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements a TA network on top of BYOL

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(cfg)

        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim
        num_heads: int = cfg.method_kwargs.num_heads
        attn_dropout = cfg.method_kwargs.attn_dropout
        proj_dropout = cfg.method_kwargs.proj_dropout
        qkv_hidden_dim = cfg.method_kwargs.qkv_hidden_dim
        query_dim = cfg.method_kwargs.query_dim
        value_dim = cfg.method_kwargs.value_dim
        self.gamma = cfg.method_kwargs.gamma
        self.regularizer_weight = cfg.method_kwargs.regularizer_weight

        self.ta_lr = cfg.optimizer.ta_lr

        # assert (
        #     self.features_dim % num_heads == 0
        # ), "features_dim must be divisible by num_heads"

        self.student_TA = TA_Attention(
            value_dim=value_dim,
            query_dim=query_dim,
            input_dim=proj_output_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            hidden_dim=qkv_hidden_dim,
        )

        self.teacher_TA = TA_Attention(
            value_dim=value_dim,
            query_dim=query_dim,
            input_dim=proj_output_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            hidden_dim=qkv_hidden_dim,
        )

        initialize_momentum_params(self.student_TA, self.teacher_TA)

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
        initialize_momentum_params(self.projector, self.momentum_projector)

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
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.num_heads")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.attn_dropout")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_dropout")

        cfg.optimizer.ta_lr = omegaconf_select(cfg, "optimizer.ta_lr", cfg.optimizer.lr)
        cfg.method_kwargs.qkv_hidden_dim = omegaconf_select(
            cfg, "method_kwargs.qkv_hidden_dim", None
        )
        cfg.method_kwargs.query_dim = omegaconf_select(
            cfg, "method_kwargs.query_dim", cfg.method_kwargs.proj_output_dim
        )
        cfg.method_kwargs.value_dim = omegaconf_select(
            cfg, "method_kwargs.value_dim", cfg.method_kwargs.proj_output_dim
        )
        cfg.method_kwargs.gamma = omegaconf_select(cfg, "method_kwargs.gamma", 0)
        cfg.method_kwargs.regularizer_weight = omegaconf_select(
            cfg, "method_kwargs.regularizer_weight", 0.0
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector, predictor and TA parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "predictor", "params": self.predictor.parameters()},
            {"name": "projector", "params": self.projector.parameters()},
            {
                "name": "student_TA",
                "params": self.student_TA.parameters(),
                "lr": self.ta_lr,
            },
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
            (self.student_TA, self.teacher_TA),
        ]
        return super().momentum_pairs + extra_momentum_pairs

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

        neg_cos_sim = 0
        # barlow_residual_loss = 0

        ta_output = []
        residuals = []
        attn_weights = []

        for idx1 in range(self.num_large_crops):
            for idx2 in np.delete(range(self.num_crops), idx1):
                z = self.projector(out["feats"][idx1])

                student_q, student_k, student_v = self.student_TA(z)

                with torch.no_grad():
                    momentum_z = self.momentum_projector(out["momentum_feats"][idx2])
                    teacher_q, teacher_k, teacher_v = self.teacher_TA(momentum_z)
                
                key_pool = torch.cat([student_k, teacher_k.detach()], dim=1)
                value_pool = torch.cat([student_v, teacher_v.detach()], dim=1)

                student_residual, student_attention_weights = self.student_TA.attention(
                    student_q, key_pool, value_pool
                )
                z = z + student_residual

                with torch.no_grad():
                    teacher_residual, _ = self.teacher_TA.attention(
                        teacher_q, key_pool, value_pool
                    )
                    momentum_z = momentum_z + teacher_residual

                ta_output.append(z)
                residuals.append(student_residual)
                attn_weights.append(student_attention_weights)

                p = self.predictor(z)

                neg_cos_sim += byol_loss_func(p, momentum_z)
                # barlow_residual_loss += barlow_loss_func(z, momentum_z.detach(), lamb=0.0051, scale_loss=0.1)
                # b, c = z.shape
                # all_z = torch.cat([z, momentum_z])
                # all_z = embedding_propagation(all_z, alpha=0.5, rbf_scale=1.0, norm_prop=False)

                # z = all_z[:b]
                # momentum_z = all_z[b:]

        # barlow_residual_loss *= self.regularizer_weight
        class_loss = out["loss"]

        with torch.no_grad():
            z_std = (
                F.normalize(torch.stack(ta_output[: self.num_large_crops]), dim=-1)
                .std(dim=1)
                .mean()
            )
            z_unnormalized_std = (
                torch.stack(ta_output[: self.num_large_crops]).std(dim=1).mean()
            )
            residual_unnormalized_std = torch.stack(residuals).std(dim=1).mean()
            residual_std = (
                F.normalize(torch.stack(residuals[: self.num_large_crops]), dim=-1)
                .std(dim=1)
                .mean()
            )
            attention_entropy = (
                torch.special.entr(torch.stack(attn_weights)).sum(dim=-1).mean()
            )
            residual_svd_entropy = (
                torch.special.entr(
                    F.normalize(
                        torch.linalg.svdvals(torch.stack(residuals).float()),
                        dim=1,
                        p=1.0,
                    )
                )
                .sum(dim=-1)
                .mean()
            )
            ta_svd_entropy = (
                torch.special.entr(
                    F.normalize(
                        torch.linalg.svdvals(torch.stack(ta_output).float()),
                        dim=1,
                        p=1.0,
                    )
                )
                .sum(dim=-1)
                .mean()
            )

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            # "train_barlow_residual_loss": barlow_residual_loss,
            "train_z_unnormalized_std": z_unnormalized_std,
            "train_z_std": z_std,
            "train_residual_std": residual_std,
            "train_residual_unnormalized_std": residual_unnormalized_std,
            "attention_entropy": attention_entropy,
            "residual_svd_entropy": residual_svd_entropy,
            "ta_svd_entropy": ta_svd_entropy,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss #+ barlow_residual_loss
