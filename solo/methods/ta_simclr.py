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

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simclr import simclr_loss_func
from solo.losses.koleo import KoLeoLoss
from solo.methods.base import BaseMethod
from solo.utils.ta_attention import TA_Attention
from solo.utils.misc import omegaconf_select


class TA_SimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature
        self.regularizer_weight = cfg.method_kwargs.regularizer_weight

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
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.batch_norm = nn.BatchNorm1d(proj_output_dim)

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

        cfg = super(TA_SimCLR, TA_SimCLR).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.num_heads")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.attn_dropout")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_dropout")

        cfg.method_kwargs.qkv_hidden_dim = omegaconf_select(
            cfg, "method_kwargs.qkv_hidden_dim", None
        )
        cfg.method_kwargs.query_dim = omegaconf_select(
            cfg, "method_kwargs.query_dim", cfg.method_kwargs.proj_output_dim
        )
        cfg.method_kwargs.value_dim = omegaconf_select(
            cfg, "method_kwargs.value_dim", cfg.method_kwargs.proj_output_dim
        )

        cfg.method_kwargs.regularizer_weight = omegaconf_select(
            cfg, "method_kwargs.regularizer_weight", 0.0
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "ta", "params": self.ta.parameters()},
            {"name": "batch_norm", "params": self.batch_norm.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])
        queries, keys, values = self.ta(self.batch_norm(z))
        residual, attn_weights = self.ta.attention(queries, keys, values)
        # regularizer_loss = manifold_regularizer_loss(z, residual)
        z = residual

        # regularizer_loss = self.koleo(residual) * self.regularizer_weight
        # z = embedding_propagation(z, alpha=0.5, rbf_scale=1.0, norm_prop=False)

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )

        with torch.no_grad():
            residual_std = F.normalize(residual, dim=-1).std(dim=1).mean()
            unnormalized_residual_std = residual.std(dim=1).mean()
            z_std = F.normalize(z, dim=-1).std(dim=1).mean()
            unnormalized_z_std = z.std(dim=1).mean()
            attention_entropy = torch.special.entr(attn_weights).sum(dim=-1).mean()

        metrics = {
            "train_nce_loss": nce_loss,
            "train_residual_std": residual_std,
            "train_residual_unnormalized_std": unnormalized_residual_std,
            "train_z_std": z_std,
            "train_z_unnormalized_std": unnormalized_z_std,
            "attention_entropy": attention_entropy,
            # "regularizer_loss": regularizer_loss,
            # "collapse_loss": collapse_loss,
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss #+ regularizer_loss * self.regularizer_weight #+ collapse_loss
