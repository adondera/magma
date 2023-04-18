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

from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn.functional as F
import torch.nn as nn
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
from solo.utils.ta_attention import TA_Attention


class TA_BarlowTwins(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements TA on top of Barlow Twins (https://arxiv.org/abs/2103.03230)

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
                scale_loss (float): scaling factor of the loss.
        """

        super().__init__(cfg)

        self.lamb: float = cfg.method_kwargs.lamb
        self.scale_loss: float = cfg.method_kwargs.scale_loss

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        num_heads: int = cfg.method_kwargs.num_heads
        attn_dropout = cfg.method_kwargs.attn_dropout
        proj_dropout = cfg.method_kwargs.proj_dropout
        qkv_hidden_dim = cfg.method_kwargs.qkv_hidden_dim
        query_dim = cfg.method_kwargs.query_dim
        value_dim = cfg.method_kwargs.value_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # TODO: Add a number of hidden layers param to TA?
        self.ta = TA_Attention(
            value_dim=value_dim,
            query_dim=query_dim,
            input_dim=proj_output_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            hidden_dim=qkv_hidden_dim,
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(TA_BarlowTwins, TA_BarlowTwins).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.num_heads")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.attn_dropout")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_dropout")

        cfg.method_kwargs.lamb = omegaconf_select(cfg, "method_kwargs.lamb", 0.0051)
        cfg.method_kwargs.scale_loss = omegaconf_select(
            cfg, "method_kwargs.scale_loss", 0.024
        )
        cfg.method_kwargs.qkv_hidden_dim = omegaconf_select(
            cfg, "method_kwargs.qkv_hidden_dim", None
        )
        cfg.method_kwargs.query_dim = omegaconf_select(
            cfg, "method_kwargs.query_dim", cfg.method_kwargs.proj_output_dim
        )
        cfg.method_kwargs.value_dim = omegaconf_select(
            cfg, "method_kwargs.value_dim", cfg.method_kwargs.proj_output_dim
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "ta", "params": self.ta.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        query1, key1, value1 = self.ta(z1)
        query2, key2, value2 = self.ta(z2)

        key_pool = torch.cat([key1, key2], dim=1)
        value_pool = torch.cat([value1, value2], dim=1)

        z1, attn_weights1 = self.ta.attention(query1, key_pool, value_pool)
        z2, attn_weights2 = self.ta.attention(query2, key_pool, value_pool)

        # ------- barlow twins loss -------
        barlow_loss = barlow_loss_func(
            z1, z2, lamb=self.lamb, scale_loss=self.scale_loss
        )

        # TODO: Check that these metrics are computed correctly
        with torch.no_grad():
            z_std = F.normalize(torch.stack([z1, z2]), dim=-1).std(dim=1).mean()
            z_unnormalized_std = torch.stack([z1, z2]).std(dim=1).mean()
            residual_unnormalized_std = torch.stack([z1, z2]).std(dim=1).mean()
            residual_std = F.normalize(torch.stack([z1, z2]), dim=-1).std(dim=1).mean()
            attention_entropy = (
                torch.special.entr(torch.stack([attn_weights1, attn_weights2]))
                .sum(dim=-1)
                .mean()
            )
            residual_svd_entropy = (
                torch.special.entr(
                    F.normalize(
                        torch.linalg.svdvals(
                            torch.stack([attn_weights1, attn_weights2]).float()
                        ),
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
                        torch.linalg.svdvals(torch.stack([z1, z2]).float()),
                        dim=1,
                        p=1.0,
                    )
                )
                .sum(dim=-1)
                .mean()
            )

        metrics = {
            "train_barlow_loss": barlow_loss,
            "train_z_unnormalized_std": z_unnormalized_std,
            "train_z_std": z_std,
            "train_residual_std": residual_std,
            "train_residual_unnormalized_std": residual_unnormalized_std,
            "attention_entropy": attention_entropy,
            "residual_svd_entropy": residual_svd_entropy,
            "ta_svd_entropy": ta_svd_entropy,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return barlow_loss + class_loss
